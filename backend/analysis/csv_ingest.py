"""
Ingests real (or externally-generated) electrochemical sensor readings from
a CSV file and runs them through the exact same trained pipeline the live
simulation uses -- the bridge from "this only works on synthetic data" to
"this could plug into an actual assay's data export."

Expected CSV columns (header row required):
    well_id, step, ph, do2, glucose_lactate, impedance

  - well_id: "A1".."D6" for the standard 4x6 layout (24 wells). A subset of
    wells is fine; missing wells just don't appear in the plate graph, which
    degrades gracefully (see `_build_adjacency_for_subset`) rather than
    failing.
  - step: an integer timestep index, consistent across wells (0, 1, 2, ...)
    -- doesn't need to be real time, just a fixed ordering. Each well needs
    at least DEFAULT_WINDOW rows to get a first prediction.
  - ph, do2, glucose_lactate, impedance: raw sensor values in the same
    units the simulation uses (pH unitless ~6.5-7.8; do2 % air saturation
    0-100; glucose_lactate mM; impedance ohms). Values well outside those
    ranges will still run through the model but its predictions are only as
    meaningful as how close the input distribution is to what it was
    trained on -- this is a real limitation of a model trained entirely on
    synthetic data, stated plainly rather than glossed over.

`generate_sample_csv()` produces a valid example (from a fresh simulated
plate) so a user knows the exact expected shape before preparing real data.
"""
import csv
import io
from dataclasses import dataclass

import numpy as np
import torch

from backend.config import SENSOR_TYPES, WELL_IDS
from backend.gnn.architecture import CAUSE_CLASSES, DEFAULT_WINDOW
from backend.gnn.uncertainty import predict_with_uncertainty
from backend.models.schemas import health_label_from_score
from backend.recommend.engine import compute_sensor_deltas, recommend_for_well
from backend.explain.narrator import narrate_well

REQUIRED_COLUMNS = ["well_id", "step", "ph", "do2", "glucose_lactate", "impedance"]


class CsvValidationError(Exception):
    pass


@dataclass
class WellAnalysisResult:
    well_id: str
    steps: list[int]
    health_scores: list[float]
    health_std: list[float]
    final_health_label: str
    final_cause: str
    recommendation_action: str | None
    recommendation_reasoning: str | None
    narration: str


def parse_csv(file_bytes: bytes) -> dict[str, dict[int, dict[str, float]]]:
    """Returns {well_id: {step: {sensor: value}}}."""
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise CsvValidationError("CSV appears to be empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise CsvValidationError(f"Missing required column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED_COLUMNS)}")

    data: dict[str, dict[int, dict[str, float]]] = {}
    for row_num, row in enumerate(reader, start=2):
        wid = row["well_id"].strip()
        try:
            step = int(row["step"])
            values = {s: float(row[s]) for s in SENSOR_TYPES}
        except (ValueError, KeyError) as e:
            raise CsvValidationError(f"Row {row_num}: couldn't parse values ({e}).")
        data.setdefault(wid, {})[step] = values

    if not data:
        raise CsvValidationError("No data rows found.")
    return data


def _build_adjacency_for_wells(well_ids: list[str]) -> torch.Tensor:
    """
    Degrades gracefully for a non-standard well subset: reuses the standard
    4x6 grid adjacency where all wells are on it, otherwise falls back to
    identity (independent per-well inference, no cross-well coupling) --
    stated to the caller via `used_standard_layout` in analyze_csv's result.
    """
    from backend.gnn.plate_graph import PLATE_ADJACENCY, WELL_INDEX

    n = len(well_ids)
    if all(w in WELL_INDEX for w in well_ids):
        idx = [WELL_INDEX[w] for w in well_ids]
        return PLATE_ADJACENCY[idx][:, idx], True
    return torch.eye(n), False


def analyze_csv(file_bytes: bytes, model) -> dict:
    parsed = parse_csv(file_bytes)
    well_ids = sorted(parsed.keys())
    adjacency, used_standard_layout = _build_adjacency_for_wells(well_ids)

    # Build a dense (N, T, 4) array; wells with gaps get forward-filled from
    # the last known reading (real data collection isn't always perfectly
    # regular) so a few missing rows don't just break everything.
    all_steps = sorted({s for steps in parsed.values() for s in steps})
    n_steps = len(all_steps)
    step_to_idx = {s: i for i, s in enumerate(all_steps)}

    sensor_array = np.full((len(well_ids), n_steps, len(SENSOR_TYPES)), np.nan)
    for wi, wid in enumerate(well_ids):
        for step, values in parsed[wid].items():
            sensor_array[wi, step_to_idx[step], :] = [values[s] for s in SENSOR_TYPES]
        # forward-fill, then back-fill any leading gaps
        for t in range(1, n_steps):
            if np.isnan(sensor_array[wi, t, 0]):
                sensor_array[wi, t, :] = sensor_array[wi, t - 1, :]
        for t in range(n_steps - 2, -1, -1):
            if np.isnan(sensor_array[wi, t, 0]):
                sensor_array[wi, t, :] = sensor_array[wi, t + 1, :]

    if n_steps < DEFAULT_WINDOW:
        raise CsvValidationError(f"Need at least {DEFAULT_WINDOW} timesteps per well; got {n_steps}.")

    results: dict[str, WellAnalysisResult] = {}
    model.eval()

    health_series = {wid: [] for wid in well_ids}
    std_series = {wid: [] for wid in well_ids}
    eval_steps = list(range(DEFAULT_WINDOW, n_steps + 1))

    for t in eval_steps:
        x = torch.tensor(sensor_array[:, t - DEFAULT_WINDOW:t, :], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            health_scores, _, cause_logits = model(x, adjacency, return_cause=True)
        for i, wid in enumerate(well_ids):
            health_series[wid].append(float(health_scores[0, i]))
            std_series[wid].append(0.0)  # per-tick MC-dropout skipped here for speed; final tick gets it below

    for i, wid in enumerate(well_ids):
        final_t = eval_steps[-1]
        x_final = torch.tensor(sensor_array[:, final_t - DEFAULT_WINDOW:final_t, :], dtype=torch.float32).unsqueeze(0)
        mean, std = predict_with_uncertainty(model, x_final, adjacency, n_samples=20)
        with torch.no_grad():
            _, _, cause_logits = model(x_final, adjacency, return_cause=True)
        final_cause = CAUSE_CLASSES[int(cause_logits[0, i].argmax())]
        final_score = float(mean[0, i])
        final_std = float(std[0, i])
        label = health_label_from_score(final_score)

        now_vec = sensor_array[i, -1, :]
        past_vec = sensor_array[i, max(0, n_steps - DEFAULT_WINDOW), :]
        deltas = compute_sensor_deltas(now_vec, past_vec)
        rec = recommend_for_well(wid, deltas, final_score, cause=final_cause)
        narration = narrate_well(wid, deltas, final_score, label, bifurcation_fired=False, cause=final_cause)

        results[wid] = WellAnalysisResult(
            well_id=wid,
            steps=all_steps,
            health_scores=health_series[wid],
            health_std=[0.0] * (len(health_series[wid]) - 1) + [final_std],
            final_health_label=label.value,
            final_cause=final_cause,
            recommendation_action=rec.action if rec else None,
            recommendation_reasoning=rec.reasoning if rec else None,
            narration=narration,
        )

    return {
        "well_ids": well_ids,
        "n_steps": n_steps,
        "used_standard_layout": used_standard_layout,
        "results": {wid: r.__dict__ for wid, r in results.items()},
    }


def generate_sample_csv(n_steps: int = 48, seed: int = 42) -> str:
    """A ready-to-download example in the exact expected format, from a
    freshly simulated plate -- shows a real user the shape to match."""
    from backend.biology.organoid_trajectory import simulate_plate
    from backend.sensors.sensor_model import trajectory_to_readings

    plate = simulate_plate(WELL_IDS, base_seed=seed)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(REQUIRED_COLUMNS)
    for i, wid in enumerate(WELL_IDS):
        readings = trajectory_to_readings(plate[wid], seed=seed * 13 + i)
        for t in range(min(n_steps, len(plate[wid].t))):
            writer.writerow([
                wid, t,
                f"{readings.ph[t]:.4f}", f"{readings.do2[t]:.3f}",
                f"{readings.glucose_lactate[t]:.4f}", f"{readings.impedance[t]:.2f}",
            ])
    return buf.getvalue()
