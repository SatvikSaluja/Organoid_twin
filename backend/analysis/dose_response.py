"""
Fits a 4-parameter Hill (logistic) dose-response curve to per-well health
scores inferred from noisy sensor data -- the same curve pharmacology
assays report as an EC50. This is the analysis half of the drug-screening
demo whose data side lives in organoid_trajectory.WellSimulator's
toxin_dose/ec50/hill_slope.

    response(dose) = bottom + (top - bottom) / (1 + (dose / EC50) ** hill_slope)

Fit purely from the model's inferred health scores (never the ground-truth
enzyme/flux state), so a good fit here is evidence the whole pipeline --
biology -> noisy sensors -> GNN -> curve fit -- recovers a real
pharmacological parameter, not just that the simulator's own formula is
self-consistent.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import curve_fit

from backend.biology.organoid_trajectory import N_STEPS, WellSimulator
from backend.config import PLATE_COLS, SENSOR_TYPES
from backend.gnn.architecture import DEFAULT_WINDOW
from backend.gnn.plate_graph import PLATE_ADJACENCY
from backend.sensors.sensor_model import StreamingSensorState, true_signals_from_state


def hill_equation(dose: np.ndarray, top: float, bottom: float, ec50: float, hill_slope: float) -> np.ndarray:
    dose = np.clip(dose, 1e-6, None)  # avoid 0**negative_slope
    return bottom + (top - bottom) / (1.0 + (dose / ec50) ** hill_slope)


@dataclass
class DoseResponseFit:
    top: float
    bottom: float
    ec50: float
    hill_slope: float
    r_squared: float
    doses: list[float]
    responses: list[float]  # mean response per dose (replicate-averaged)
    response_std: list[float]


def fit_dose_response(doses_per_well: list[float], responses_per_well: list[float]) -> DoseResponseFit | None:
    """
    doses_per_well / responses_per_well: one entry per well (replicates of
    the same dose repeated). Returns None if the fit fails to converge
    (e.g. too few distinct doses).
    """
    doses = np.array(doses_per_well, dtype=float)
    responses = np.array(responses_per_well, dtype=float)

    unique_doses = sorted(set(doses.tolist()))
    if len(unique_doses) < 3:
        return None

    mean_by_dose = [responses[doses == d].mean() for d in unique_doses]
    std_by_dose = [responses[doses == d].std() for d in unique_doses]

    top_guess = max(mean_by_dose)
    bottom_guess = min(mean_by_dose)
    ec50_guess = float(np.median(unique_doses)) or 1.0

    try:
        popt, _ = curve_fit(
            hill_equation, doses, responses,
            p0=[top_guess, bottom_guess, ec50_guess, 1.5],
            bounds=([0, 0, 1e-3, 0.1], [2, 2, 1e4, 10]),
            maxfev=5000,
        )
    except RuntimeError:
        return None

    top, bottom, ec50, hill_slope = popt
    predicted = hill_equation(doses, *popt)
    ss_res = float(np.sum((responses - predicted) ** 2))
    ss_tot = float(np.sum((responses - responses.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DoseResponseFit(
        top=float(top), bottom=float(bottom), ec50=float(ec50), hill_slope=float(hill_slope),
        r_squared=r_squared, doses=unique_doses, responses=mean_by_dose, response_std=std_by_dose,
    )


@dataclass
class DoseResponsePlateResult:
    well_doses: dict[str, float]
    well_true_response: dict[str, float]      # ground-truth Warburg index (direct toxicity readout)
    well_inferred_response: dict[str, float]  # model's own predicted lactate-production head -- what a real deployment would fit against
    fit_true: DoseResponseFit | None
    fit_inferred: DoseResponseFit | None
    true_ec50: float


def assign_column_doses(well_ids: list[str], doses: list[float], cols: int = PLATE_COLS) -> dict[str, float]:
    """One dose per plate column, replicated down each row -- standard
    dose-response plate layout."""
    assert len(doses) == cols, f"expected {cols} doses (one per column), got {len(doses)}"
    out = {}
    for idx, wid in enumerate(well_ids):
        col = idx % cols
        out[wid] = doses[col]
    return out


def _simulate_one_dose_plate(well_ids: list[str], base_seed: int, well_doses: dict[str, float], model,
                              true_ec50: float, hill_slope: float, n_steps: int):
    """One plate's worth of (dose, true_response, inferred_response) triples.

    Response variables are chosen to directly track the drug's mechanism
    (it caps mitochondrial capacity -- see WellSimulator.drug_ceiling) rather
    than the composite health score, whose stress term is driven by
    *nutrient scarcity* and can even move the "wrong" way under a pure
    mitochondrial poison (less O2 consumed looks like *more* O2 available).
    Wells run with control=True (adverse events suppressed) -- a real
    dose-response assay is designed to isolate the compound's effect too,
    not have it confounded by an unrelated contamination event.
    """
    sims = {
        wid: WellSimulator(wid, seed=base_seed * 1000 + idx, control=True,
                            toxin_dose=well_doses[wid], ec50=true_ec50, hill_slope=hill_slope)
        for idx, wid in enumerate(well_ids)
    }
    sensors = {wid: StreamingSensorState(seed=base_seed * 7919 + idx) for idx, wid in enumerate(well_ids)}
    history: dict[str, list[list[float]]] = {wid: [] for wid in well_ids}
    true_response_series = {wid: [] for wid in well_ids}
    inferred_response_series = {wid: [] for wid in well_ids}

    for t in range(n_steps):
        for wid in well_ids:
            state = sims[wid].step(None)
            true_response_series[wid].append(state["warburg_index"])
            true_signals = true_signals_from_state(
                glucose=state["glucose"], oxygen=state["oxygen"], lactate=state["lactate"],
                stress_level=state["stress_level"], demand=state["demand"],
            )
            reading = sensors[wid].step(true_signals)
            history[wid].append([reading[s] for s in SENSOR_TYPES])

        if t >= DEFAULT_WINDOW:
            x = torch.tensor(
                np.stack([history[wid][-DEFAULT_WINDOW:] for wid in well_ids]), dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                _, aux = model(x, PLATE_ADJACENCY)
            for i, wid in enumerate(well_ids):
                inferred_response_series[wid].append(float(aux[0, i, 1]))  # lactate_production_norm head

    tail = max(1, n_steps // 4)
    return (
        {wid: float(np.mean(true_response_series[wid][-tail:])) for wid in well_ids},
        {wid: float(np.mean(inferred_response_series[wid][-tail:])) for wid in well_ids},
    )


def run_dose_response_plate(
    well_ids: list[str], base_seed: int, doses: list[float], model, true_ec50: float = 10.0,
    hill_slope: float = 2.0, n_steps: int = N_STEPS, n_replicate_plates: int = 3,
) -> DoseResponsePlateResult:
    """
    Simulates `n_replicate_plates` plates with the same dose gradient across
    columns (pooling replicates the way a real assay runs the same layout
    several times), runs the trained GNN over the resulting noisy sensor
    streams, and fits a Hill curve to both the ground-truth toxicity signal
    (sanity check) and the model's own predicted lactate head (the actual
    pipeline result a real deployment would see).
    """
    well_doses = assign_column_doses(well_ids, doses)

    all_true, all_inferred, all_doses, well_true_first, well_inferred_first = [], [], [], {}, {}
    for p in range(n_replicate_plates):
        true_resp, inferred_resp = _simulate_one_dose_plate(
            well_ids, base_seed + p, well_doses, model, true_ec50, hill_slope, n_steps
        )
        if p == 0:
            well_true_first, well_inferred_first = true_resp, inferred_resp
        for wid in well_ids:
            all_true.append(true_resp[wid])
            all_inferred.append(inferred_resp[wid])
            all_doses.append(well_doses[wid])

    fit_true = fit_dose_response(all_doses, all_true)
    fit_inferred = fit_dose_response(all_doses, all_inferred)

    return DoseResponsePlateResult(
        well_doses=well_doses, well_true_response=well_true_first, well_inferred_response=well_inferred_first,
        fit_true=fit_true, fit_inferred=fit_inferred, true_ec50=true_ec50,
    )
