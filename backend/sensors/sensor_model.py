"""
Converts a true per-well metabolic state (biology/organoid_trajectory.py)
into what a real sensor would report: noisy, indirect, multimodal, and
lagged by physical probe response time (noise_profiles.py).

  - pH               <- lactate accumulation (saturating decrease)
  - dissolved O2      <- oxygen pool, directly, but lagged (real probes aren't instant)
  - glucose/lactate   <- net substrate proxy: glucose pool minus a lactate term
  - impedance         <- density/viability proxy: scales with organoid growth
                         (demand) and drops with accumulated stress (viability loss)

This is what the GNN model actually sees; ground truth (glucose, oxygen,
lactate, stress_level, warburg_index, ...) stays in the biology layer and is
only used for training labels and eval, never fed to the model directly.

`StreamingSensorState` is the causal, one-step-at-a-time form (a real probe
only ever sees the past) -- backend/control/closed_loop.py drives it
interleaved with WellSimulator.step(). `trajectory_to_readings` is a thin
batch wrapper around it for the existing training/eval code, which needs a
whole precomputed trajectory converted at once.
"""
from dataclasses import dataclass

import numpy as np

from backend.biology.organoid_trajectory import WellTrajectory
from backend.sensors.noise_profiles import NOISE_PROFILES, lag_filter_step

PH_BASELINE = 7.6
PH_LACTATE_SPAN = 1.1
PH_LACTATE_HALF_SAT = 8.0

GLUCOSE_LACTATE_LACTATE_WEIGHT = 0.3

IMPEDANCE_BASELINE = 500.0
IMPEDANCE_VIABILITY_SLOPE = 0.6  # fraction of baseline lost at stress_level=1

SENSOR_KEYS = ("ph", "do2", "glucose_lactate", "impedance")


@dataclass
class SensorTimeSeries:
    well_id: str
    ph: np.ndarray
    do2: np.ndarray
    glucose_lactate: np.ndarray
    impedance: np.ndarray


def true_signals_from_state(glucose: float, oxygen: float, lactate: float, stress_level: float, demand: float) -> dict[str, float]:
    """The un-lagged, un-noised sensor targets implied by one instant of true state."""
    ph = PH_BASELINE - PH_LACTATE_SPAN * lactate / (lactate + PH_LACTATE_HALF_SAT)
    do2 = oxygen
    glucose_lactate = max(0.0, glucose - GLUCOSE_LACTATE_LACTATE_WEIGHT * lactate)
    viability_factor = 1.0 - IMPEDANCE_VIABILITY_SLOPE * stress_level
    impedance = IMPEDANCE_BASELINE * demand * viability_factor
    return {"ph": ph, "do2": do2, "glucose_lactate": glucose_lactate, "impedance": impedance}


class StreamingSensorState:
    """Causal per-well sensor state: one true-state dict in, one noisy/lagged reading out."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self._filtered_prev: dict[str, float] | None = None

    def step(self, true_signals: dict[str, float]) -> dict[str, float]:
        if self._filtered_prev is None:
            self._filtered_prev = dict(true_signals)

        out = {}
        for sensor in SENSOR_KEYS:
            profile = NOISE_PROFILES[sensor]
            filtered = lag_filter_step(self._filtered_prev[sensor], true_signals[sensor], profile.lag_tau_steps)
            self._filtered_prev[sensor] = filtered
            out[sensor] = filtered + self.rng.normal(0, profile.noise_std)

        out["glucose_lactate"] = max(0.0, out["glucose_lactate"])
        out["impedance"] = max(0.0, out["impedance"])
        out["do2"] = min(100.0, max(0.0, out["do2"]))
        return out


def trajectory_to_readings(traj: WellTrajectory, seed: int) -> SensorTimeSeries:
    state = StreamingSensorState(seed)
    n = len(traj.t)
    out = {k: np.zeros(n) for k in SENSOR_KEYS}
    for i in range(n):
        true_signals = true_signals_from_state(
            glucose=traj.glucose[i], oxygen=traj.oxygen[i], lactate=traj.lactate[i],
            stress_level=traj.stress_level[i], demand=traj.demand[i],
        )
        reading = state.step(true_signals)
        for k in SENSOR_KEYS:
            out[k][i] = reading[k]
    return SensorTimeSeries(well_id=traj.well_id, **out)
