"""
Recommendation engine: rule-based reasoning on top of a learned root-cause
classification, kept interpretable rather than a black box.

The decision of *which* action to recommend is driven by the GNN's own
learned `cause_head` (architecture.py) when available -- a hand-tuned
sensor-delta-threshold heuristic (`_recommend_from_heuristic` below) topped
out around 24%/4% accuracy on oxygen/glucose attribution, because by the
time a well's health score crosses the action threshold, growth-driven
nutrient demand has often already shifted which pool is most limiting since
onset; a handful of fixed thresholds can't track that, but the model
learned directly on the same labels can (see eval/run_benchmark.py's
before/after comparison). The heuristic is kept, not deleted, as the
fallback for callers with no model cause available and as the "before"
side of that ablation.

Either way, a PI evaluating this should be able to read *why* an action was
suggested, not just trust a class index -- so the reasoning text always
cites the actual sensor deltas (fractional change over a recent trend
window, computed by the caller from raw readings), regardless of which path
picked the action.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from backend.config import SENSOR_TYPES
from backend.models.schemas import Recommendation

DO2_DROP_THRESHOLD = -0.04
PH_DROP_THRESHOLD = -0.05           # well above the pH sensor's noise floor (std 0.015, see noise_profiles.py)
GLUCOSE_LACTATE_DROP_THRESHOLD = -0.10
HEALTH_SCORE_ACTION_THRESHOLD = 0.6  # only recommend something if health looks mild-stress-or-worse

CAUSE_TO_ACTION = {
    "oxygen": "Increase media oxygenation",
    "glucose": "Increase glucose feed rate",
    "adverse_event": "Flag for manual inspection",
}


@dataclass
class SensorDeltas:
    """Fractional (or, for pH, absolute) change over a recent trend window."""
    do2: float
    ph: float
    glucose_lactate: float
    impedance: float


_SENSOR_IDX = {s: i for i, s in enumerate(SENSOR_TYPES)}


def compute_sensor_deltas(now: np.ndarray, past: np.ndarray) -> SensorDeltas:
    """
    now, past: raw 4-vectors in SENSOR_TYPES order (one well, two timesteps).
    Shared by ws/plate_stream.py (live) and eval/run_benchmark.py (offline)
    so both score recommendations against the exact same trend definition.
    """
    def frac(n: float, p: float) -> float:
        return (n - p) / max(abs(p), 1e-6)

    return SensorDeltas(
        do2=frac(now[_SENSOR_IDX["do2"]], past[_SENSOR_IDX["do2"]]),
        ph=now[_SENSOR_IDX["ph"]] - past[_SENSOR_IDX["ph"]],
        glucose_lactate=frac(now[_SENSOR_IDX["glucose_lactate"]], past[_SENSOR_IDX["glucose_lactate"]]),
        impedance=frac(now[_SENSOR_IDX["impedance"]], past[_SENSOR_IDX["impedance"]]),
    )


def recommend_for_well(well_id: str, deltas: SensorDeltas, health_score: float, cause: str | None = None) -> Recommendation | None:
    """
    cause: the GNN cause_head's predicted class ("none" / "oxygen" /
    "glucose" / "adverse_event"), if the caller has run the model with
    return_cause=True. When omitted, falls back to the sensor-delta
    heuristic below -- kept for callers without model access and as the
    ablation baseline.
    """
    if health_score >= HEALTH_SCORE_ACTION_THRESHOLD:
        return None
    if cause is not None:
        return _recommend_from_ml_cause(well_id, cause, deltas)
    return _recommend_from_heuristic(well_id, deltas)


def _recommend_from_ml_cause(well_id: str, cause: str, deltas: SensorDeltas) -> Recommendation | None:
    now = datetime.now(timezone.utc)
    action = CAUSE_TO_ACTION.get(cause)

    if action is None:  # "none" -- the classifier itself doesn't see a specific cause
        return Recommendation(
            well_id=well_id, timestamp=now,
            action="Flag for manual inspection",
            reasoning="Health score has dropped but the model's root-cause classifier doesn't attribute it to a known correctable pattern.",
        )

    reasoning = {
        "oxygen": (
            f"Root-cause classifier: oxygen-limited. Dissolved O2 change {deltas.do2:+.0%}, pH change "
            f"{deltas.ph:+.2f} over the trend window -- consistent with oxygen supply falling while lactate accumulates."
        ),
        "glucose": (
            f"Root-cause classifier: glucose-limited. Glucose/lactate proxy change {deltas.glucose_lactate:+.0%} "
            "over the trend window, without the lactate-driven pH drop an oxygen-limited state would show -- "
            "consistent with substrate running low rather than a fermentation shift."
        ),
        "adverse_event": (
            f"Root-cause classifier: metabolic/mitochondrial damage. pH change {deltas.ph:+.2f} with dissolved O2 "
            f"change {deltas.do2:+.0%} -- lactate production up independent of oxygen availability, which a feed "
            "or O2 adjustment would not fix."
        ),
    }[cause]

    return Recommendation(well_id=well_id, timestamp=now, action=action, reasoning=reasoning)


def _recommend_from_heuristic(well_id: str, deltas: SensorDeltas) -> Recommendation | None:
    """
    Hand-tuned sensor-delta thresholds -- the original approach, kept as a
    fallback and as the eval ablation's "before" baseline. See module
    docstring for why the learned classifier now leads.
    """
    now = datetime.now(timezone.utc)
    ph_falling = deltas.ph < PH_DROP_THRESHOLD
    do2_falling = deltas.do2 < DO2_DROP_THRESHOLD
    do2_flat_or_rising = deltas.do2 >= 0.0
    glucose_falling = deltas.glucose_lactate < GLUCOSE_LACTATE_DROP_THRESHOLD

    # pH-falling (lactate rising) is checked before the glucose rule below,
    # deliberately: glucose depletes on essentially every feed cycle in every
    # well (healthy or not), so "glucose falling" alone is a poor signal for
    # *which* well is actually in trouble. A rising lactate signature (pH
    # falling) is what actually distinguishes an active fermentation-driven
    # decline -- both the oxygen-limited and the mitochondrial-dysfunction
    # cases below produce it; true substrate starvation (glucose branch)
    # doesn't, because with less glucose there's also less to ferment.
    if do2_falling and ph_falling:
        return Recommendation(
            well_id=well_id, timestamp=now,
            action="Increase media oxygenation",
            reasoning=(
                f"Dissolved O2 down {abs(deltas.do2):.0%} and pH down {abs(deltas.ph):.2f} units over the trend "
                "window -- oxygen supply is falling while lactate accumulates, consistent with an "
                "oxygen-limited, fermentation-shifted state."
            ),
        )

    if ph_falling and do2_flat_or_rising:
        return Recommendation(
            well_id=well_id, timestamp=now,
            action="Flag for manual inspection",
            reasoning=(
                f"pH down {abs(deltas.ph):.2f} units (lactate rising) but dissolved O2 is flat or rising, "
                "not falling -- lactate production is up even though oxygen is available, which points to "
                "a metabolic/mitochondrial issue rather than a supply problem a feed or O2 adjustment would fix."
            ),
        )

    if glucose_falling and not ph_falling:
        return Recommendation(
            well_id=well_id, timestamp=now,
            action="Increase glucose feed rate",
            reasoning=(
                f"Glucose/lactate proxy down {abs(deltas.glucose_lactate):.0%} with no corresponding rise in "
                "lactate (pH stable) -- consistent with substrate (glucose) running low rather than a "
                "fermentation shift."
            ),
        )

    return Recommendation(
        well_id=well_id, timestamp=now,
        action="Flag for manual inspection",
        reasoning="Health score has dropped but the sensor pattern doesn't match a known correctable cause.",
    )
