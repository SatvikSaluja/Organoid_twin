"""
Metabolic simulator — extends the aerobic/anaerobic flux heuristic
prototyped in ../../cell-digital-twin/backend/app/ml_model.py::predict_fluxes.

That prototype took (glucose, oxygen) and returned a flux dict using a
Michaelis-Menten-like aerobic/anaerobic split:

    aerobic_factor = oxygen / (oxygen + 5)
    anaerobic_factor = 1 - aerobic_factor
    Glycolysis = glucose * 1.2
    Lactate Fermentation = glucose * anaerobic_factor * 1.3
    ...

This module ports that exact split and extends it with:
  - enzyme_activity and temperature dependence (Gaussian around 37C), so a
    stressed/dysfunctional organoid (decline_dynamics.py) can express itself
    as reduced flux without changing glucose/oxygen inputs directly.
  - explicit consumption-rate outputs (glucose_consumption, o2_consumption)
    that organoid_trajectory.py uses to deplete each well's nutrient pools.
  - a `warburg_index` diagnostic: the fraction of pyruvate handling going to
    fermentation rather than oxidative routes — 0 = fully aerobic, 1 = fully
    Warburg/fermentative. This is the quantity decline_dynamics.py drives
    up during a stress/decline event, and what the GNN constraint layer
    ties pH/O2 predictions together through.
"""
from dataclasses import dataclass
import math

OPTIMAL_TEMP_C = 37.0
TEMP_SIGMA_C = 5.0
O2_HALF_SAT = 5.0  # same constant as cell-digital-twin's aerobic_factor


@dataclass
class FluxState:
    glycolysis: float
    pyruvate_oxidation: float
    tca_cycle: float
    oxidative_phosphorylation: float
    lactate_fermentation: float
    atp_synthase: float
    glucose_consumption: float
    o2_consumption: float
    lactate_production: float
    warburg_index: float


def temperature_factor(temperature_c: float) -> float:
    """Gaussian falloff in metabolic efficiency away from 37C."""
    return math.exp(-((temperature_c - OPTIMAL_TEMP_C) ** 2) / (2 * TEMP_SIGMA_C ** 2))


def compute_fluxes(
    glucose: float,
    oxygen: float,
    enzyme_activity: float = 1.0,
    temperature_c: float = OPTIMAL_TEMP_C,
) -> FluxState:
    """
    glucose, oxygen: current pool concentrations (arbitrary consistent units;
        organoid_trajectory.py treats them as mM and %-air-saturation resp.).
    enzyme_activity: 0-1 scalar, reduced by decline_dynamics.py to model
        organoid dysfunction (mitochondrial/enzymatic impairment) independent
        of substrate availability.
    temperature_c: incubator temperature; deviation from 37C (e.g. during a
        simulated temperature-excursion adverse event) suppresses flux.
    """
    glucose = max(0.0, glucose)
    oxygen = max(0.0, oxygen)
    temp_efficiency = temperature_factor(temperature_c)

    # Oxygen sets the *ceiling* on aerobic routing (Michaelis-Menten-like,
    # same constant as cell-digital-twin's aerobic_factor); enzyme_activity
    # then gates how much of that ceiling the mitochondria can actually use.
    # This decoupling is what makes mitochondrial dysfunction (falling
    # enzyme_activity, from decline_dynamics.py) shunt flux from oxidative
    # to fermentative routes even when oxygen is still plentiful -- the
    # real aerobic -> Warburg-like shift, rather than a uniform flux scale-
    # down that would leave the fermentation/oxidation ratio unchanged.
    aerobic_ceiling = oxygen / (oxygen + O2_HALF_SAT)
    effective_aerobic_fraction = aerobic_ceiling * enzyme_activity
    effective_anaerobic_fraction = 1.0 - effective_aerobic_fraction

    glycolysis = glucose * 1.2 * temp_efficiency
    pyruvate_oxidation = glycolysis * effective_aerobic_fraction
    tca_cycle = pyruvate_oxidation * 0.8
    oxidative_phosphorylation = min(oxygen * 1.5 * temp_efficiency * enzyme_activity, tca_cycle * 2.5)
    lactate_fermentation = glycolysis * effective_anaerobic_fraction * 1.3
    atp_synthase = oxidative_phosphorylation * 0.9

    glucose_consumption = glycolysis
    o2_consumption = oxidative_phosphorylation
    lactate_production = lactate_fermentation

    denom = lactate_fermentation + tca_cycle
    warburg_index = lactate_fermentation / denom if denom > 1e-9 else 0.0

    return FluxState(
        glycolysis=glycolysis,
        pyruvate_oxidation=pyruvate_oxidation,
        tca_cycle=tca_cycle,
        oxidative_phosphorylation=oxidative_phosphorylation,
        lactate_fermentation=lactate_fermentation,
        atp_synthase=atp_synthase,
        glucose_consumption=glucose_consumption,
        o2_consumption=o2_consumption,
        lactate_production=lactate_production,
        warburg_index=warburg_index,
    )
