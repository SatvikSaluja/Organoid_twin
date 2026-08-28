"""
Latent stress/decline process for a single well, driven by nutrient
depletion and randomly-injected adverse events.

Design: a scalar `stress_level` in [0, 1] accumulates when glucose/oxygen
pools run low, and jumps sharply on an adverse event. Once stress_level
crosses DECLINE_THRESHOLD for the first time, `decline_active` latches on.

Two distinct decline mechanisms are modeled, deliberately kept separate so
the recommendation engine has a real "correctable vs. not" distinction to
learn, not just one flavor of decline:

  - Pure substrate limitation (glucose- or oxygen-driven): enzyme_activity
    stays near 1.0. The Warburg-like shift comes entirely from
    metabolic_sim's aerobic_ceiling term (low O2 forces fermentation even
    with intact mitochondria) -- correctable by feeding/oxygenating more.
  - Adverse-event damage (contamination / temperature excursion):
    `enzyme_activity` (mitochondrial function) permanently decays toward a
    floor once an event has fired -- lactate rises while O2 consumption
    *drops* (damaged mitochondria can't use the O2 that's there), which is
    NOT fixed by adding more oxygen or glucose -- the ground truth for the
    recommendation engine's "flag for manual inspection" branch.

The recorded onset step is the ground truth eval/run_benchmark.py scores
detection lead-time against.
"""
from dataclasses import dataclass, field
from enum import Enum
import random


DECLINE_THRESHOLD = 0.5
NUTRIENT_DEFICIT_TRIGGER = 0.28  # below this deficit, treated as normal feed-cycle dip
STRESS_RISE_RATE = 0.06     # per step, scaled by nutrient deficit past the trigger
STRESS_DECAY_RATE = 0.025   # per step, when nutrients are adequate
ADVERSE_EVENT_STRESS_JUMP = 0.45
ENZYME_ACTIVITY_FLOOR = 0.35
ENZYME_DECAY_RATE = 0.05    # per step once decline_active, toward floor
ADVERSE_EVENT_PROB_PER_STEP = 0.0018  # ~ a handful of events over a full trajectory
ADVERSE_EVENT_DURATION_STEPS = 4


class AdverseEventType(str, Enum):
    contamination = "contamination"
    temperature_excursion = "temperature_excursion"


@dataclass
class AdverseEvent:
    event_type: AdverseEventType
    remaining_steps: int


@dataclass
class DeclineState:
    stress_level: float = 0.0
    enzyme_activity: float = 1.0
    temperature_c: float = 37.0
    decline_active: bool = False
    decline_onset_step: int | None = None
    active_event: AdverseEvent | None = None
    mitochondrial_damage: bool = False  # latches True once any adverse event has fired

    def step(
        self,
        step_idx: int,
        glucose_frac: float,
        oxygen_frac: float,
        rng: random.Random,
        allow_events: bool = True,
    ) -> None:
        """
        glucose_frac / oxygen_frac: current pool level as a fraction of a
        healthy baseline (1.0 = fully fed, 0.0 = depleted). Mutates state
        in place; call once per simulated timestep.

        allow_events: set False for "control" wells in the false-positive-
        rate eval — decline can still occur from pure nutrient stress, but
        never from an injected adverse-event shock.
        """
        # -- maybe start a new adverse event --
        if allow_events and self.active_event is None and rng.random() < ADVERSE_EVENT_PROB_PER_STEP:
            event_type = rng.choice(list(AdverseEventType))
            self.active_event = AdverseEvent(event_type, ADVERSE_EVENT_DURATION_STEPS)
            self.stress_level = min(1.0, self.stress_level + ADVERSE_EVENT_STRESS_JUMP)
            self.mitochondrial_damage = True

        # -- adverse event effects while active --
        if self.active_event is not None:
            if self.active_event.event_type == AdverseEventType.temperature_excursion:
                self.temperature_c = 37.0 + rng.uniform(3.0, 6.0) * rng.choice([-1, 1])
            # contamination doesn't directly change temperature; its damage
            # is entirely through the stress_level jump + resulting enzyme decay.
            self.active_event.remaining_steps -= 1
            if self.active_event.remaining_steps <= 0:
                self.active_event = None
                self.temperature_c = 37.0
        else:
            self.temperature_c = 37.0

        # -- nutrient-driven stress accumulation --
        nutrient_deficit = max(0.0, 1.0 - min(glucose_frac, oxygen_frac))
        if nutrient_deficit > NUTRIENT_DEFICIT_TRIGGER:
            self.stress_level = min(1.0, self.stress_level + STRESS_RISE_RATE * (nutrient_deficit - NUTRIENT_DEFICIT_TRIGGER))
        else:
            self.stress_level = max(0.0, self.stress_level - STRESS_DECAY_RATE)

        # -- decline latch --
        if not self.decline_active and self.stress_level >= DECLINE_THRESHOLD:
            self.decline_active = True
            self.decline_onset_step = step_idx

        # -- enzyme activity dynamics: only adverse-event damage impairs
        # mitochondria; pure substrate-limited decline leaves enzyme_activity
        # near 1.0 so the O2/glucose signal isn't masked (see module docstring) --
        if self.mitochondrial_damage:
            self.enzyme_activity = max(
                ENZYME_ACTIVITY_FLOOR,
                self.enzyme_activity - ENZYME_DECAY_RATE * (self.enzyme_activity - ENZYME_ACTIVITY_FLOOR + 0.05),
            )
        else:
            self.enzyme_activity = min(1.0, self.enzyme_activity + 0.01)
