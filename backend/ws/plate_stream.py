"""
Live plate-state streaming — the full pipeline wired together.

Each tick advances one simulated timestep (organoid_trajectory.py's
DT_HOURS = 30 minutes of culture time) for every well's WellSimulator,
generates a sensor reading from the result, and runs the trained GNN, the
bifurcation detector, the recommendation engine, and the narrator on the
resulting window — plus periodically runs one EWC fine-tune step so the
CalibrationPanel has something real to show. When a plate finishes its
n_steps, a fresh one is simulated (new seed) so the demo runs indefinitely.

Built on the same steppable WellSimulator + StreamingSensorState primitives
as backend/control/ (closed-loop arena, what-if preview) rather than a
separate precomputed-batch path, so "what would happen if I intervened on
this well right now" can clone the *actual* live state rather than an
approximation of it.
"""
import asyncio
import json
from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import WebSocket

from backend.biology.organoid_trajectory import N_STEPS, WellSimulator
from backend.config import CHECKPOINT_DIR, PLATE_COLS, PLATE_ROWS, SENSOR_TYPES, WELL_IDS, WS_BROADCAST_INTERVAL_SEC
from backend.gnn.architecture import CAUSE_CLASSES, DEFAULT_WINDOW, PlateGNN
from backend.gnn.bifurcation import DEFAULT_COOLDOWN_STEPS, DEFAULT_CONSEC_REQUIRED, BifurcationDetector, compute_jacobian_norms
from backend.gnn.coevolution import EWCState, compute_fisher_information, online_finetune_step
from backend.gnn.plate_graph import PLATE_ADJACENCY
from backend.gnn.train import build_dataset, calibrate_stoichiometry, load_checkpoint
from backend.gnn.uncertainty import predict_with_uncertainty
from backend.models.schemas import (
    CalibrationPoint,
    EventType,
    HealthLabel,
    LiveFrame,
    PlateEvent,
    PlateStateMessage,
    Recommendation,
    SensorReading,
    SensorType,
    WellState,
    health_label_from_score,
)
from backend.recommend.engine import (
    DO2_DROP_THRESHOLD,
    GLUCOSE_LACTATE_DROP_THRESHOLD,
    PH_DROP_THRESHOLD,
    SensorDeltas,
    compute_sensor_deltas,
    recommend_for_well,
)
from backend.explain.narrator import narrate_well
from backend.sensors.sensor_model import StreamingSensorState, true_signals_from_state

TREND_WINDOW = DEFAULT_WINDOW
EWC_FINETUNE_INTERVAL_STEPS = 20
EWC_REFERENCE_PLATES = 2

IMPEDANCE_DROP_THRESHOLD = -0.10  # no dedicated recommend/engine.py rule uses this; just a "notable move" scale

# how many normalizer-units a sensor moved by, for picking a single "driving" sensor
_SENSOR_SIGNIFICANCE_NORM = {
    "do2": abs(DO2_DROP_THRESHOLD),
    "ph": abs(PH_DROP_THRESHOLD),
    "glucose_lactate": abs(GLUCOSE_LACTATE_DROP_THRESHOLD),
    "impedance": abs(IMPEDANCE_DROP_THRESHOLD),
}


class LivePlate:
    """One live-running plate: a WellSimulator + StreamingSensorState + reading
    history per well, advanced one tick at a time."""

    def __init__(self, seed: int):
        self.seed = seed
        self.sims = {wid: WellSimulator(wid, seed=seed * 1000 + idx) for idx, wid in enumerate(WELL_IDS)}
        self.sensors = {wid: StreamingSensorState(seed=seed * 7919 + idx) for idx, wid in enumerate(WELL_IDS)}
        self.history: dict[str, list[list[float]]] = {wid: [] for wid in WELL_IDS}
        self.t = -1

    def tick(self) -> None:
        self.t += 1
        for wid in WELL_IDS:
            state = self.sims[wid].step(None)
            true_signals = true_signals_from_state(
                glucose=state["glucose"], oxygen=state["oxygen"], lactate=state["lactate"],
                stress_level=state["stress_level"], demand=state["demand"],
            )
            reading = self.sensors[wid].step(true_signals)
            self.history[wid].append([reading[s] for s in SENSOR_TYPES])

    def window_tensor(self, window: int = TREND_WINDOW) -> torch.Tensor:
        stack = np.array([self.history[wid][-window:] for wid in WELL_IDS])
        return torch.tensor(stack, dtype=torch.float32).unsqueeze(0)  # (1, N, T, 4)


class PlateStreamManager:
    """Owns the live model, the current live plate, connections, and the broadcast loop."""

    def __init__(self, well_ids: list[str], rows: int, cols: int):
        self.well_ids = well_ids
        self.rows = rows
        self.cols = cols
        self.connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

        self.model, ckpt = load_checkpoint("constrained")
        self.fit = calibrate_stoichiometry()

        threshold_path = CHECKPOINT_DIR / "bifurcation_threshold.json"
        if threshold_path.exists():
            threshold = json.load(open(threshold_path))["threshold"]
        else:
            threshold = 1.0  # conservative fallback if calibration hasn't been run yet
        self.detector = BifurcationDetector(
            well_ids, threshold=threshold,
            consec_required=DEFAULT_CONSEC_REQUIRED, cooldown_steps=DEFAULT_COOLDOWN_STEPS,
        )

        ref_x, ref_h, ref_o2, ref_lac, _ = build_dataset(EWC_REFERENCE_PLATES, seed_offset=9_000_000, fit=self.fit)
        self._reference_dataset = torch.utils.data.TensorDataset(ref_x, ref_h, ref_o2, ref_lac)
        self.ewc = compute_fisher_information(self.model, self._reference_dataset)
        self.finetune_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.finetune_count = 0
        self.calibration_history: list[CalibrationPoint] = []

        self.plate = LivePlate(seed=1)
        self._warm_up()
        self.event_log: list[PlateEvent] = []
        self.active_recommendations: dict[str, Recommendation] = {}

    def _warm_up(self) -> None:
        """Advance past the window warm-up period once at startup so the
        first broadcast tick already has a full window (rather than the
        frontend waiting TREND_WINDOW ticks for its first real frame)."""
        for _ in range(TREND_WINDOW):
            self.plate.tick()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.connections:
                self.connections.remove(websocket)

    def _advance_time(self) -> None:
        self.plate.tick()
        if self.plate.t >= N_STEPS:
            self.plate = LivePlate(seed=self.plate.seed + 1)
            self._warm_up()

    def _deltas_for_well(self, well_id: str) -> SensorDeltas:
        series = self.plate.history[well_id]
        past_idx = max(0, len(series) - TREND_WINDOW)
        return compute_sensor_deltas(np.array(series[-1]), np.array(series[past_idx]))

    def _driving_sensor(self, deltas: SensorDeltas) -> SensorType | None:
        ratios = {
            "do2": abs(deltas.do2) / _SENSOR_SIGNIFICANCE_NORM["do2"],
            "ph": abs(deltas.ph) / _SENSOR_SIGNIFICANCE_NORM["ph"],
            "glucose_lactate": abs(deltas.glucose_lactate) / _SENSOR_SIGNIFICANCE_NORM["glucose_lactate"],
            "impedance": abs(deltas.impedance) / _SENSOR_SIGNIFICANCE_NORM["impedance"],
        }
        best = max(ratios, key=ratios.get)
        return SensorType(best) if ratios[best] >= 1.0 else None

    def _ground_truth_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stress = np.array([self.plate.sims[wid].decline.stress_level for wid in self.well_ids])
        warburg = np.array([self.plate.sims[wid].last_warburg_index for wid in self.well_ids])
        o2_consumption = np.array([self.plate.sims[wid].last_o2_consumption for wid in self.well_ids])
        lactate_production = np.array([self.plate.sims[wid].last_lactate_production for wid in self.well_ids])

        health_true = np.clip(1.0 - (0.7 * stress + 0.3 * warburg), 0.0, 1.0)
        o2_norm = np.clip((o2_consumption - self.fit.o2_min) / (self.fit.o2_max - self.fit.o2_min + 1e-9), 0, 1)
        lac_norm = np.clip((lactate_production - self.fit.lactate_min) / (self.fit.lactate_max - self.fit.lactate_min + 1e-9), 0, 1)
        return (
            torch.tensor(health_true, dtype=torch.float32),
            torch.tensor(o2_norm, dtype=torch.float32),
            torch.tensor(lac_norm, dtype=torch.float32),
        )

    def _maybe_run_ewc_finetune(self) -> None:
        if self.plate.t % EWC_FINETUNE_INTERVAL_STEPS != 0:
            return
        x = self.plate.window_tensor()
        health_true, o2_norm, lac_norm = self._ground_truth_batch()

        batch = (x, health_true.unsqueeze(0), o2_norm.unsqueeze(0), lac_norm.unsqueeze(0))
        online_finetune_step(self.model, self.finetune_optimizer, batch, self.ewc)
        self.finetune_count += 1

        self.model.eval()
        with torch.no_grad():
            # Chunked rather than one pass over the whole reference set: dense
            # GATv2 attention materializes a (batch, N, N, heads) tensor, so
            # memory scales with batch size -- a single unbatched forward
            # over all ~300 reference samples permanently grows the process's
            # RSS by ~200MB (PyTorch's CPU allocator caches the peak rather
            # than releasing it), which is enough on its own to OOM a 512MB
            # container. A small chunk size keeps that peak bounded no matter
            # how large EWC_REFERENCE_PLATES gets.
            ref_x, ref_h, _, _ = self._reference_dataset.tensors
            REF_VAL_CHUNK = 16
            total_sq_err, n_seen = 0.0, 0
            for start in range(0, ref_x.shape[0], REF_VAL_CHUNK):
                xb, hb = ref_x[start:start + REF_VAL_CHUNK], ref_h[start:start + REF_VAL_CHUNK]
                pred, _ = self.model(xb, PLATE_ADJACENCY)
                total_sq_err += torch.nn.functional.mse_loss(pred, hb, reduction="sum").item()
                n_seen += hb.numel()
            val_loss = total_sq_err / max(1, n_seen)
        self.calibration_history.append(CalibrationPoint(
            timestamp=datetime.now(timezone.utc),
            sim_hours=self.plate.t * 0.5,
            reference_val_loss=val_loss,
            finetune_count=self.finetune_count,
        ))
        if len(self.calibration_history) > 200:
            self.calibration_history = self.calibration_history[-200:]

    def _build_wells(self, x_window: torch.Tensor, now: datetime, allow_fire: bool) -> tuple[list[WellState], list[PlateEvent]]:
        self.model.eval()
        with torch.no_grad():
            health_scores, _, cause_logits = self.model(x_window, PLATE_ADJACENCY, return_cause=True)
        mean_scores, std_scores = predict_with_uncertainty(self.model, x_window, PLATE_ADJACENCY, n_samples=8)
        health_scores = health_scores.squeeze(0).tolist()
        std_scores = std_scores.squeeze(0).tolist()
        predicted_causes = [CAUSE_CLASSES[c] for c in cause_logits.squeeze(0).argmax(dim=-1).tolist()]

        jac_norms = compute_jacobian_norms(self.model, x_window, PLATE_ADJACENCY, self.well_ids) if allow_fire else {}

        wells: list[WellState] = []
        new_events: list[PlateEvent] = []
        for i, wid in enumerate(self.well_ids):
            raw = self.plate.history[wid][-1]
            reading = SensorReading(
                well_id=wid, timestamp=now,
                ph=float(raw[SENSOR_TYPES.index("ph")]),
                do2=float(raw[SENSOR_TYPES.index("do2")]),
                glucose_lactate=float(raw[SENSOR_TYPES.index("glucose_lactate")]),
                impedance=float(raw[SENSOR_TYPES.index("impedance")]),
            )
            score = float(health_scores[i])
            label = health_label_from_score(score)
            deltas = self._deltas_for_well(wid)
            driving = self._driving_sensor(deltas)

            fired = self.detector.update(wid, jac_norms[wid]) if allow_fire else False
            if fired:
                event = PlateEvent(
                    well_id=wid, timestamp=now, event_type=EventType.regime_shift,
                    description=f"Well {wid}: regime shift detected (Jacobian norm {jac_norms[wid]:.2f} above baseline).",
                )
                new_events.append(event)
                self.event_log.append(event)
                if len(self.event_log) > 200:
                    self.event_log = self.event_log[-200:]

            cause = predicted_causes[i]
            rec = recommend_for_well(wid, deltas, score, cause=cause)
            if rec is not None:
                self.active_recommendations[wid] = rec
            elif wid in self.active_recommendations:
                del self.active_recommendations[wid]

            narration = narrate_well(wid, deltas, score, label, bifurcation_fired=fired, cause=cause)

            wells.append(WellState(
                well_id=wid, timestamp=now, reading=reading,
                health_score=score, health_label=label, driving_sensor=driving,
                narration=narration, health_std=float(std_scores[i]),
            ))
        return wells, new_events

    def step(self) -> LiveFrame:
        self._advance_time()
        now = datetime.now(timezone.utc)
        x_window = self.plate.window_tensor()

        wells, new_events = self._build_wells(x_window, now, allow_fire=True)
        self._maybe_run_ewc_finetune()

        plate_state = PlateStateMessage(timestamp=now, plate_rows=self.rows, plate_cols=self.cols, wells=wells)
        return LiveFrame(
            plate_state=plate_state,
            new_events=new_events,
            recommendations=list(self.active_recommendations.values()),
            calibration=self.calibration_history[-1] if self.calibration_history else None,
        )

    def current_frame(self) -> LiveFrame:
        """Non-advancing snapshot for REST GET (doesn't step the simulation)."""
        now = datetime.now(timezone.utc)
        x_window = self.plate.window_tensor()
        wells, _ = self._build_wells(x_window, now, allow_fire=False)
        plate_state = PlateStateMessage(timestamp=now, plate_rows=self.rows, plate_cols=self.cols, wells=wells)
        return LiveFrame(
            plate_state=plate_state,
            new_events=[],
            recommendations=list(self.active_recommendations.values()),
            calibration=self.calibration_history[-1] if self.calibration_history else None,
        )

    def get_attention(self, well_id: str) -> dict:
        """Last GAT layer's attention weights, averaged over heads, for the
        interactive control panel's live graph view. Rows/cols in well_ids order."""
        x_window = self.plate.window_tensor()
        self.model.eval()
        with torch.no_grad():
            _, _, attn = self.model(x_window, PLATE_ADJACENCY, return_attention=True)
        attn_mean = attn.squeeze(0).mean(dim=-1)  # (N, N), averaged over heads
        focus_idx = self.well_ids.index(well_id)
        return {
            "well_id": well_id,
            "weights": {wid: float(attn_mean[focus_idx, j]) for j, wid in enumerate(self.well_ids)},
        }

    async def broadcast_loop(self) -> None:
        while True:
            frame = self.step()
            payload = frame.model_dump_json()
            async with self._lock:
                dead = []
                for ws in self.connections:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self.connections.remove(ws)
            await asyncio.sleep(WS_BROADCAST_INTERVAL_SEC)


manager = PlateStreamManager(well_ids=WELL_IDS, rows=PLATE_ROWS, cols=PLATE_COLS)


async def plate_ws_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        await manager.disconnect(websocket)
