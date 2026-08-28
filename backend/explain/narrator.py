"""
Template-based plain-language narration -- no LLM. Reuses the exact same
signal-attribution logic recommend/engine.py already computes (which sensor
moved, in which direction, by how much) and turns it into a sentence a
non-specialist can read at a glance, e.g.:

    "Well B4: regime shift detected. Dissolved O2 down 18% and pH down 0.31
    units over the last 6 hours -- oxygen supply is falling while lactate
    accumulates, consistent with an oxygen-limited, fermentation-shifted
    state. Recommended action: increase media oxygenation."
"""
from backend.models.schemas import HealthLabel
from backend.recommend.engine import SensorDeltas, recommend_for_well


def narrate_well(
    well_id: str,
    deltas: SensorDeltas,
    health_score: float,
    health_label: HealthLabel,
    bifurcation_fired: bool,
    cause: str | None = None,
) -> str:
    if health_label == HealthLabel.healthy and not bifurcation_fired:
        return f"Well {well_id}: stable. Health score {health_score:.2f}, no concerning sensor trends."

    prefix = f"Well {well_id}: "
    if bifurcation_fired:
        prefix += "regime shift detected. "

    rec = recommend_for_well(well_id, deltas, health_score, cause=cause)
    if rec is None:
        return f"{prefix}Health score {health_score:.2f} ({health_label.value})."

    return f"{prefix}{rec.reasoning} Recommended action: {rec.action.lower()}."
