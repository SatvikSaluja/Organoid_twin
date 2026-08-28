"""
FastAPI app entrypoint: routes, WebSocket endpoint, lifespan hooks.

Run from the repo root with:
    uvicorn backend.main:app --reload --port 8000

The full pipeline (biology -> sensors -> GNN -> bifurcation detector ->
recommendation engine -> narrator) runs behind backend/ws/plate_stream.py's
PlateStreamManager; this module just exposes it over REST + WebSocket, plus
the four newer research features: closed-loop control arena, the
interactive what-if/attention control panel, drug-screening dose-response,
and database-backed cohort analytics.
"""
import asyncio
import dataclasses
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.analysis.cohort import get_experiment_summary, list_experiments, run_and_persist_experiment
from backend.analysis.csv_ingest import CsvValidationError, analyze_csv, generate_sample_csv
from backend.analysis.dose_response import run_dose_response_plate
from backend.config import PLATE_COLS, PLATE_ROWS, WELL_IDS
from backend.control.whatif import DEFAULT_HORIZON_STEPS, run_whatif
from backend.models.db import get_or_create_plate, get_session, init_db
from backend.models.schemas import HealthResponse, LiveFrame, WellState
from backend.ws.plate_stream import manager, plate_ws_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with get_session() as session:
        get_or_create_plate(session, WELL_IDS, PLATE_ROWS, PLATE_COLS)

    broadcast_task = asyncio.create_task(manager.broadcast_loop())
    try:
        yield
    finally:
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="OrganoidTwin API", lifespan=lifespan)

# Vite's default dev server port (5173) plus the alternate port used in local
# testing, always allowed. Add the deployed frontend's exact origin via the
# CORS_EXTRA_ORIGINS env var (comma-separated) once it's known; any Netlify
# subdomain (preview or prod) is allowed automatically via the regex below so
# a first deploy doesn't need this env var set at all.
_default_origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5180", "http://127.0.0.1:5180",
]
_extra_origins = [o.strip() for o in os.environ.get("CORS_EXTRA_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(num_wells=len(WELL_IDS))


@app.get("/api/plate", response_model=LiveFrame)
def get_plate_state() -> LiveFrame:
    return manager.current_frame()


@app.get("/api/wells/{well_id}", response_model=WellState)
def get_well_state(well_id: str) -> WellState:
    frame = manager.current_frame()
    for well in frame.plate_state.wells:
        if well.well_id == well_id:
            return well
    raise HTTPException(status_code=404, detail=f"Unknown well_id '{well_id}'")


@app.websocket("/ws/plate")
async def ws_plate(websocket: WebSocket) -> None:
    try:
        await plate_ws_endpoint(websocket)
    except WebSocketDisconnect:
        pass


# -- Interactive control panel: live attention graph + what-if preview --

@app.get("/api/attention/{well_id}")
def get_attention(well_id: str) -> dict:
    if well_id not in WELL_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown well_id '{well_id}'")
    return manager.get_attention(well_id)


class WhatIfRequest(BaseModel):
    well_id: str
    o2_boost: float = 0.0
    refill_glucose: bool = False
    horizon_steps: int = DEFAULT_HORIZON_STEPS


@app.post("/api/whatif")
def post_whatif(req: WhatIfRequest) -> dict:
    if req.well_id not in WELL_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown well_id '{req.well_id}'")
    result = run_whatif(
        well_id=req.well_id,
        live_sim=manager.plate.sims[req.well_id],
        live_sensor_state=manager.plate.sensors[req.well_id],
        live_history=manager.plate.history[req.well_id],
        other_wells_history=manager.plate.history,
        model=manager.model,
        o2_boost=req.o2_boost,
        refill_glucose=req.refill_glucose,
        horizon_steps=req.horizon_steps,
    )
    return dataclasses.asdict(result)


# -- Closed-loop control arena: three-arm experiment, persisted for cohort analytics --

class ControlExperimentRequest(BaseModel):
    n_plates: int = 5
    base_seed: int = 1000


@app.post("/api/control/run")
def post_control_run(req: ControlExperimentRequest) -> dict:
    experiment_id = run_and_persist_experiment(WELL_IDS, n_plates=req.n_plates, base_seed=req.base_seed, model=manager.model)
    summary = get_experiment_summary(experiment_id)
    return _summary_to_dict(summary)


@app.get("/api/control/experiments")
def get_experiments() -> list[dict]:
    return list_experiments()


@app.get("/api/control/experiments/{experiment_id}")
def get_experiment(experiment_id: int) -> dict:
    try:
        summary = get_experiment_summary(experiment_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Unknown experiment_id {experiment_id}")
    return _summary_to_dict(summary)


def _summary_to_dict(summary) -> dict:
    return {
        "experiment_id": summary.experiment_id,
        "created_at": summary.created_at,
        "n_plates": summary.n_plates,
        "n_steps": summary.n_steps,
        "arm_stats": summary.arm_stats,
        "survival_curves": {arm: dataclasses.asdict(curve) for arm, curve in summary.survival_curves.items()},
        "log_rank": summary.log_rank,
        "cause_breakdown": summary.cause_breakdown,
    }


# -- Drug-screening dose-response mode --

class DoseResponseRequest(BaseModel):
    doses: list[float] = [0, 2, 5, 10, 20, 50]
    base_seed: int = 7
    true_ec50: float = 10.0
    hill_slope: float = 2.0
    n_replicate_plates: int = 3


@app.post("/api/dose_response/run")
def post_dose_response_run(req: DoseResponseRequest) -> dict:
    if len(req.doses) != PLATE_COLS:
        raise HTTPException(status_code=400, detail=f"Expected {PLATE_COLS} doses (one per plate column), got {len(req.doses)}")
    result = run_dose_response_plate(
        WELL_IDS, base_seed=req.base_seed, doses=req.doses, model=manager.model,
        true_ec50=req.true_ec50, hill_slope=req.hill_slope, n_replicate_plates=req.n_replicate_plates,
    )
    return {
        "well_doses": result.well_doses,
        "well_true_response": result.well_true_response,
        "well_inferred_response": result.well_inferred_response,
        "true_ec50": result.true_ec50,
        "fit_true": dataclasses.asdict(result.fit_true) if result.fit_true else None,
        "fit_inferred": dataclasses.asdict(result.fit_inferred) if result.fit_inferred else None,
    }


# -- Real-data ingestion: upload a CSV of actual (or externally-generated)
# electrochemical readings and run it through the same trained pipeline --

@app.get("/api/csv/sample", response_class=PlainTextResponse)
def get_sample_csv() -> str:
    """A ready-to-download example in the exact expected format."""
    return generate_sample_csv()


@app.post("/api/csv/analyze")
async def post_csv_analyze(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        return analyze_csv(content, model=manager.model)
    except CsvValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
