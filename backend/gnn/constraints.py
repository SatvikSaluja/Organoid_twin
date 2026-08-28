"""
Hard biological-consistency constraint.

The model's two auxiliary outputs -- normalized O2-consumption rate and
normalized lactate-production rate -- are not free to drift independently:
metabolic_sim.py ties them together through the same aerobic_ceiling
mechanism that generates the ground truth (more O2 consumption implies more
oxidative routing, which implies proportionally less fermentation/lactate).
`calibrate_stoichiometry()` fits that relationship once, empirically, by
sweeping metabolic_sim across an oxygen range; train.py then either:

  - adds `stoichiometric_consistency_penalty(...)` to the training loss
    (soft constraint), or
  - calls `project_lactate(...)` to overwrite the model's own lactate
    prediction with the value implied by its O2 prediction (hard
    projection) -- the two ablation arms eval/run_benchmark.py compares.
"""
from dataclasses import dataclass

import numpy as np
import torch

from backend.biology.metabolic_sim import compute_fluxes


@dataclass
class StoichiometryFit:
    slope: float
    intercept: float
    o2_min: float
    o2_max: float
    lactate_min: float
    lactate_max: float


def calibrate_stoichiometry(
    glucose: float = 15.0,
    oxygen_range: tuple[float, float] = (0.5, 90.0),
    n_points: int = 200,
) -> StoichiometryFit:
    """
    Sweep oxygen at fixed glucose/enzyme_activity/temperature, computing the
    (o2_consumption, lactate_production) pairs metabolic_sim.py implies, then
    fit lactate_norm ~= slope * (1 - o2_norm) + intercept via least squares.
    This is the same relationship the biology layer uses to generate ground
    truth, so a model whose two heads are consistent with it is consistent
    with the underlying stoichiometry -- not just self-consistent.
    """
    oxygens = np.linspace(*oxygen_range, n_points)
    o2c, lac = [], []
    for o2 in oxygens:
        f = compute_fluxes(glucose=glucose, oxygen=o2, enzyme_activity=1.0, temperature_c=37.0)
        o2c.append(f.o2_consumption)
        lac.append(f.lactate_production)
    o2c, lac = np.array(o2c), np.array(lac)

    o2_min, o2_max = float(o2c.min()), float(o2c.max())
    lac_min, lac_max = float(lac.min()), float(lac.max())
    o2_norm = (o2c - o2_min) / (o2_max - o2_min + 1e-9)
    lac_norm = (lac - lac_min) / (lac_max - lac_min + 1e-9)

    slope, intercept = np.polyfit(1.0 - o2_norm, lac_norm, deg=1)
    return StoichiometryFit(
        slope=float(slope), intercept=float(intercept),
        o2_min=o2_min, o2_max=o2_max, lactate_min=lac_min, lactate_max=lac_max,
    )


def stoichiometric_consistency_penalty(
    pred_o2_norm: torch.Tensor, pred_lactate_norm: torch.Tensor, fit: StoichiometryFit
) -> torch.Tensor:
    """Mean-squared deviation of the model's two heads from the fitted stoichiometric line."""
    expected_lactate = fit.slope * (1.0 - pred_o2_norm) + fit.intercept
    return torch.mean((pred_lactate_norm - expected_lactate) ** 2)


def project_lactate(pred_o2_norm: torch.Tensor, fit: StoichiometryFit) -> torch.Tensor:
    """Hard-projection variant: replace the lactate head's output entirely."""
    return torch.clamp(fit.slope * (1.0 - pred_o2_norm) + fit.intercept, 0.0, 1.0)
