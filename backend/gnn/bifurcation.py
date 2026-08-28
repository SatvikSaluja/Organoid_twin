"""
Jacobian-norm regime-shift ("bifurcation") detector.

Idea: the model's predicted health score is a smooth, slowly-changing
function of the recent sensor window during ordinary drift, but right
around the onset of a genuine regime shift (aerobic -> Warburg-like decline
"setting in"), that function's local sensitivity rises well above its normal
operating range -- a small nudge to the recent readings moves the
prediction much more than it did during quiet periods. That sensitivity is
||d(health_score_i)/d(window_i)||, computed via autograd for each well's own
recent window (holding the rest of the plate fixed).

Calibration note (see calibrate_threshold()): an earlier version of this
detector used a live, per-well EWMA baseline updated on the fly. Empirically
that was unstable -- a well's own history is short and noisy, so the
baseline chased recent values instead of characterizing "normal". What
works much better is calibrating one *fixed* threshold offline, from the
distribution of Jacobian norms observed during known-healthy stretches
across many simulated plates (a percentile of that distribution), then
firing when a well's norm exceeds it for several consecutive steps (to
reject single-tick sensor noise) with a cooldown (to avoid re-firing every
tick while a well stays elevated). On a held-out plate this reaches 100%
recall on ground-truth decline onsets with roughly one false positive per
healthy well over a full simulated week -- see eval/run_benchmark.py for
the full numbers this claim is based on.
"""
from dataclasses import dataclass

import numpy as np
import torch

from backend.gnn.architecture import PlateGNN

DEFAULT_PERCENTILE = 97.5
DEFAULT_CONSEC_REQUIRED = 3
DEFAULT_COOLDOWN_STEPS = 16
CALIBRATION_MARGIN_STEPS = 30  # exclude this many steps immediately before a known onset from the "normal" pool


def compute_jacobian_norms(model: PlateGNN, x_window: torch.Tensor, adj: torch.Tensor, well_ids: list[str]) -> dict[str, float]:
    """
    x_window: (1, N, T, 4) raw sensor window, batch size 1 (live single-plate use).
    Returns {well_id: ||d(health_score_i)/d(window_i)||} for every well i.
    """
    x = x_window.clone().requires_grad_(True)
    health_score, _ = model(x, adj)  # (1, N)

    norms = {}
    n = len(well_ids)
    for i, wid in enumerate(well_ids):
        grad = torch.autograd.grad(health_score[0, i], x, retain_graph=(i < n - 1), create_graph=False)[0]
        norms[wid] = grad[0, i].norm().item()
    return norms


def calibrate_threshold(
    model: PlateGNN,
    adj: torch.Tensor,
    well_ids: list[str],
    window: int,
    n_calibration_plates: int = 6,
    percentile: float = DEFAULT_PERCENTILE,
    base_seed: int = 8_000_000,
) -> float:
    """
    Simulate a handful of fresh plates, compute the Jacobian-norm series for
    every well, and pool together only the steps that are either from a
    well that never declines or from well before CALIBRATION_MARGIN_STEPS
    prior to that well's true decline onset. Returns the requested
    percentile of that pooled "normal operation" distribution.
    """
    # Local imports to avoid a biology/sensors -> gnn import cycle at module load time.
    from backend.biology.organoid_trajectory import N_STEPS, simulate_plate
    from backend.config import SENSOR_TYPES
    from backend.sensors.sensor_model import trajectory_to_readings

    pool = []
    for p in range(n_calibration_plates):
        seed = base_seed + p
        plate = simulate_plate(well_ids, base_seed=seed)
        readings = {
            wid: trajectory_to_readings(traj, seed=seed * 13 + i)
            for i, (wid, traj) in enumerate(plate.items())
        }
        sensor_stack = np.stack(
            [np.stack([getattr(readings[wid], s) for s in SENSOR_TYPES], axis=-1) for wid in well_ids],
            axis=0,
        )  # (N, T, 4)

        series_by_well = {wid: [] for wid in well_ids}
        for t in range(window, N_STEPS):
            x = torch.tensor(sensor_stack[:, t - window:t, :], dtype=torch.float32).unsqueeze(0)
            norms = compute_jacobian_norms(model, x, adj, well_ids)
            for wid, v in norms.items():
                series_by_well[wid].append(v)

        for wid in well_ids:
            onset = plate[wid].decline_onset_step
            series = np.array(series_by_well[wid])
            if onset is None:
                pool.append(series)
            else:
                cutoff = max(0, onset - window - CALIBRATION_MARGIN_STEPS)
                if cutoff > 0:
                    pool.append(series[:cutoff])

    pooled = np.concatenate(pool)
    return float(np.percentile(pooled, percentile))


@dataclass
class WellBifurcationState:
    cooldown: int = 0
    consec_count: int = 0


class BifurcationDetector:
    """Fixed-threshold + consecutive-ticks + cooldown. See module docstring for why."""

    def __init__(
        self,
        well_ids: list[str],
        threshold: float,
        consec_required: int = DEFAULT_CONSEC_REQUIRED,
        cooldown_steps: int = DEFAULT_COOLDOWN_STEPS,
    ):
        self.threshold = threshold
        self.consec_required = consec_required
        self.cooldown_steps = cooldown_steps
        self.states = {wid: WellBifurcationState() for wid in well_ids}

    def update(self, well_id: str, jac_norm: float) -> bool:
        """Feed one new Jacobian-norm reading for a well; returns True if a regime shift fires now."""
        s = self.states[well_id]

        if s.cooldown > 0:
            s.cooldown -= 1
            s.consec_count = 0
            return False

        if jac_norm > self.threshold:
            s.consec_count += 1
        else:
            s.consec_count = 0

        if s.consec_count >= self.consec_required:
            s.consec_count = 0
            s.cooldown = self.cooldown_steps
            return True
        return False
