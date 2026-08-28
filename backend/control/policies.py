"""
Translates a decision (ground truth, for the oracle; or a Recommendation,
for the real model-driven system) into an Intervention, with cooldown
bookkeeping shared by both so repeated firings don't re-trigger every tick.

NoInterventionPolicy needs no state at all -- it's the "do nothing, just
watch it decline" baseline arm.
"""
from backend.biology.decline_dynamics import DeclineState
from backend.biology.organoid_trajectory import Intervention
from backend.models.schemas import Recommendation

O2_BOOST_MAGNITUDE = 0.15
INTERVENTION_COOLDOWN_STEPS = 12  # ~6h before the same well can trigger again


class CooldownGate:
    def __init__(self):
        self._remaining: dict[str, int] = {}

    def ready(self, well_id: str) -> bool:
        return self._remaining.get(well_id, 0) <= 0

    def trigger(self, well_id: str) -> None:
        self._remaining[well_id] = INTERVENTION_COOLDOWN_STEPS

    def tick(self, well_id: str) -> None:
        if well_id in self._remaining and self._remaining[well_id] > 0:
            self._remaining[well_id] -= 1


class NoInterventionPolicy:
    name = "no_control"

    def decide(self, well_id: str) -> Intervention | None:
        return None


class OraclePolicy:
    """Upper bound: acts on ground truth (decline_active + recorded root
    cause), never on noisy sensor inference. Won't waste an intervention on
    adverse-event damage, which no amount of feed/O2 fixes -- see
    decline_dynamics.py. Note this checks the *recorded* cause, not whether
    an event is currently mid-fire: the event itself only lasts a few steps,
    but the mitochondrial damage it leaves behind is permanent, so a well
    must be excluded for the rest of the run, not just while the event is
    literally active."""
    name = "oracle"

    def __init__(self):
        self.gate = CooldownGate()

    def decide(self, well_id: str, decline: DeclineState, limiting_factor: str | None) -> Intervention | None:
        self.gate.tick(well_id)
        if not self.gate.ready(well_id):
            return None
        if decline.decline_active and limiting_factor != "adverse_event":
            self.gate.trigger(well_id)
            return Intervention(o2_boost=O2_BOOST_MAGNITUDE, refill_glucose=True)
        return None


class ModelDrivenPolicy:
    """The real system: only ever sees recommend/engine.py's output, which is
    itself built purely from sensor readings + the GNN's health score --
    exactly what a real deployment has access to."""
    name = "model_driven"

    def __init__(self):
        self.gate = CooldownGate()

    def decide(self, well_id: str, recommendation: Recommendation | None) -> Intervention | None:
        self.gate.tick(well_id)
        if not self.gate.ready(well_id):
            return None
        if recommendation is not None and recommendation.action != "Flag for manual inspection":
            self.gate.trigger(well_id)
            return Intervention(o2_boost=O2_BOOST_MAGNITUDE, refill_glucose=True)
        return None
