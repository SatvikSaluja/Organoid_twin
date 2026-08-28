"""
Simulates a multi-day culture period for a single well: nutrient pools
(glucose, oxygen) are consumed by metabolic_sim.compute_fluxes and
replenished on a feed schedule (glucose, lactate clearance) or by continuous
gas exchange (oxygen), while decline_dynamics.py tracks the latent
stress/decline process on top. Each well gets its own RNG stream so wells
started "identically" still diverge (biological variability).

One simulated step = DT_HOURS of culture time. Defaults give a 7-day
culture at 30-minute resolution (336 steps) — long enough for nutrient
depletion and decline to play out, short enough to replay quickly.

`WellSimulator` holds one well's mutable state and exposes `.step()` for one
timestep at a time -- this is what backend/control/closed_loop.py drives
interactively (interleaved with sensor generation + a policy that can call
`.step(intervention=...)` to actually change the trajectory going forward,
not just observe it). `simulate_well()` below is a thin batch wrapper around
it for the existing training/eval code, which never intervenes.
"""
from dataclasses import dataclass, field
import random

import numpy as np

from backend.biology.decline_dynamics import DeclineState
from backend.biology.metabolic_sim import compute_fluxes

N_STEPS = 336
DT_HOURS = 0.5
FEED_INTERVAL_STEPS = 48  # every 24h

BASELINE_GLUCOSE = 25.0   # mM, fresh media
AMBIENT_O2 = 90.0         # % air saturation, incubator headspace equilibrium

# Gas exchange efficiency varies by plate position (edge wells vent heat/gas
# faster than center wells) -- jittered per well so a genuine oxygen-limited
# decline pathway exists (poor diffusion + rising demand -> real O2 deficit),
# distinct from the metabolic-dysfunction pathway triggered by adverse events.
O2_DIFFUSION_RATE_RANGE = (0.06, 0.35)

GLUCOSE_CONSUMPTION_SCALE = 0.012
O2_CONSUMPTION_SCALE = 0.12
LACTATE_ACCUMULATION_SCALE = 0.03

# Organoid growth over the culture period increases nutrient demand
# (more cells consuming the same feed), which is what turns an initially
# comfortable feed schedule into a late-culture nutrient squeeze -- the
# main non-adverse-event route to decline. `growth_rate` and `growth_cap`
# are jittered per well (biological variability in growth rate).
GROWTH_RATE_RANGE = (0.010, 0.028)   # per hour
GROWTH_CAP_RANGE = (1.3, 2.6)        # fold-increase in demand at saturation


@dataclass
class Intervention:
    """A corrective action applied for one step by backend/control/closed_loop.py.

    o2_boost temporarily raises this well's gas-exchange rate (modeling
    increased aeration); refill_glucose immediately tops up the glucose pool
    (an off-schedule feed). Both are undone automatically once the
    WellSimulator moves past `active_until_step` -- see `_active_o2_boost`.
    """
    o2_boost: float = 0.0
    refill_glucose: bool = False


@dataclass
class WellTrajectory:
    well_id: str
    seed: int
    control: bool  # if True, adverse events are suppressed (for FP-rate eval)

    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    glucose: np.ndarray = field(default_factory=lambda: np.zeros(0))
    oxygen: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lactate: np.ndarray = field(default_factory=lambda: np.zeros(0))
    enzyme_activity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temperature_c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stress_level: np.ndarray = field(default_factory=lambda: np.zeros(0))
    decline_active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    warburg_index: np.ndarray = field(default_factory=lambda: np.zeros(0))
    o2_consumption: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lactate_production: np.ndarray = field(default_factory=lambda: np.zeros(0))
    glucose_consumption: np.ndarray = field(default_factory=lambda: np.zeros(0))
    demand: np.ndarray = field(default_factory=lambda: np.zeros(0))  # organoid growth-driven demand multiplier

    decline_onset_step: int | None = None
    limiting_factor: str | None = None  # "oxygen" | "glucose" | "adverse_event", ground truth for eval


class WellSimulator:
    """One well's mutable simulation state, steppable one timestep at a time."""

    INTERVENTION_DURATION_STEPS = 12  # ~6h of boosted aeration per triggered intervention

    def __init__(
        self, well_id: str, seed: int, dt_hours: float = DT_HOURS, control: bool = False,
        toxin_dose: float = 0.0, ec50: float = 10.0, hill_slope: float = 2.0,
    ):
        self.well_id = well_id
        self.seed = seed
        self.dt_hours = dt_hours
        self.control = control

        self.rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        self.glucose = BASELINE_GLUCOSE * (1.0 + np_rng.normal(0, 0.08))
        self.oxygen = AMBIENT_O2 * (1.0 + np_rng.normal(0, 0.05))
        self.lactate = 0.0
        self._np_rng = np_rng

        self.growth_rate = np_rng.uniform(*GROWTH_RATE_RANGE)
        self.growth_cap = np_rng.uniform(*GROWTH_CAP_RANGE)
        self.o2_diffusion_rate_base = np_rng.uniform(*O2_DIFFUSION_RATE_RANGE)
        self._o2_boost_remaining_steps = 0
        self._o2_boost_value = 0.0

        # Drug/toxin dose-response: a Hill-equation ceiling on mitochondrial
        # capacity, independent of (and multiplicative with) the natural
        # decline process below -- see backend/analysis/dose_response.py,
        # which fits this exact relationship back out from noisy sensor data.
        self.toxin_dose = toxin_dose
        self.ec50 = ec50
        self.hill_slope = hill_slope
        self.drug_ceiling = 1.0 / (1.0 + (toxin_dose / ec50) ** hill_slope) if toxin_dose > 0 else 1.0

        self.decline = DeclineState()
        self.step_idx = -1
        self.limiting_factor: str | None = None
        self.last_warburg_index = 0.0
        self.last_o2_consumption = 0.0
        self.last_lactate_production = 0.0

    def clone(self) -> "WellSimulator":
        """Deep copy for what-if exploration: run the clone forward under a
        hypothetical intervention without touching the live simulator."""
        import copy
        return copy.deepcopy(self)

    def step(self, intervention: Intervention | None = None) -> dict:
        """Advance one timestep, optionally applying a correction. Returns
        the true (ground-truth) state at the new step_idx as a plain dict --
        this is what feeds both WellTrajectory (batch mode) and the sensor
        layer (closed-loop mode)."""
        self.step_idx += 1
        i = self.step_idx
        t_hours = i * self.dt_hours
        demand = 1.0 + (self.growth_cap - 1.0) * (1.0 - np.exp(-self.growth_rate * t_hours))

        if intervention is not None:
            if intervention.o2_boost:
                self._o2_boost_remaining_steps = self.INTERVENTION_DURATION_STEPS
                self._o2_boost_value = intervention.o2_boost
            if intervention.refill_glucose:
                self.glucose = BASELINE_GLUCOSE * (1.0 + self._np_rng.normal(0, 0.04))

        o2_diffusion_rate = self.o2_diffusion_rate_base
        if self._o2_boost_remaining_steps > 0:
            o2_diffusion_rate += self._o2_boost_value
            self._o2_boost_remaining_steps -= 1

        glucose_frac = min(1.0, self.glucose / BASELINE_GLUCOSE)
        oxygen_frac = min(1.0, self.oxygen / AMBIENT_O2)

        self.decline.step(i, glucose_frac, oxygen_frac, self.rng, allow_events=not self.control)

        fluxes = compute_fluxes(
            glucose=self.glucose,
            oxygen=self.oxygen,
            enzyme_activity=self.decline.enzyme_activity * self.drug_ceiling,
            temperature_c=self.decline.temperature_c,
        )

        self.glucose = max(0.0, self.glucose - fluxes.glucose_consumption * self.dt_hours * GLUCOSE_CONSUMPTION_SCALE * demand)
        self.oxygen = self.oxygen + (AMBIENT_O2 - self.oxygen) * o2_diffusion_rate - fluxes.o2_consumption * self.dt_hours * O2_CONSUMPTION_SCALE * demand
        self.oxygen = max(0.0, min(AMBIENT_O2, self.oxygen))
        self.lactate = max(0.0, self.lactate + fluxes.lactate_production * self.dt_hours * LACTATE_ACCUMULATION_SCALE * demand)

        if self.decline.decline_active and self.limiting_factor is None:
            if self.decline.active_event is not None:
                self.limiting_factor = "adverse_event"
            elif oxygen_frac <= glucose_frac:
                self.limiting_factor = "oxygen"
            else:
                self.limiting_factor = "glucose"

        if (i + 1) % FEED_INTERVAL_STEPS == 0:
            self.glucose = BASELINE_GLUCOSE * (1.0 + self._np_rng.normal(0, 0.04))
            self.lactate *= 0.15
            self.oxygen = min(AMBIENT_O2, self.oxygen + 0.5 * (AMBIENT_O2 - self.oxygen))

        # Cached so callers that only have a handle on the simulator (not its
        # last returned dict -- e.g. the live dashboard's EWC fine-tune step)
        # can still get the exact ground truth from the most recent step.
        self.last_warburg_index = fluxes.warburg_index
        self.last_o2_consumption = fluxes.o2_consumption
        self.last_lactate_production = fluxes.lactate_production

        return {
            "t": t_hours, "glucose": self.glucose, "oxygen": self.oxygen, "lactate": self.lactate,
            "enzyme_activity": self.decline.enzyme_activity, "temperature_c": self.decline.temperature_c,
            "stress_level": self.decline.stress_level, "warburg_index": fluxes.warburg_index,
            "o2_consumption": fluxes.o2_consumption, "lactate_production": fluxes.lactate_production,
            "glucose_consumption": fluxes.glucose_consumption, "demand": demand,
            "decline_active": self.decline.decline_active,
        }


def simulate_well(
    well_id: str,
    seed: int,
    n_steps: int = N_STEPS,
    dt_hours: float = DT_HOURS,
    control: bool = False,
    toxin_dose: float = 0.0,
    ec50: float = 10.0,
    hill_slope: float = 2.0,
) -> WellTrajectory:
    sim = WellSimulator(well_id, seed, dt_hours=dt_hours, control=control, toxin_dose=toxin_dose, ec50=ec50, hill_slope=hill_slope)
    traj = WellTrajectory(well_id=well_id, seed=seed, control=control)
    keys = [
        "t", "glucose", "oxygen", "lactate", "enzyme_activity", "temperature_c",
        "stress_level", "warburg_index", "o2_consumption", "lactate_production",
        "glucose_consumption", "demand",
    ]
    arrays = {k: np.zeros(n_steps) for k in keys}
    decline_active_arr = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        state = sim.step(None)
        for k in keys:
            arrays[k][i] = state[k]
        decline_active_arr[i] = state["decline_active"]

    for k, v in arrays.items():
        setattr(traj, k, v)
    traj.decline_active = decline_active_arr
    traj.decline_onset_step = sim.decline.decline_onset_step
    traj.limiting_factor = sim.limiting_factor
    return traj


def simulate_plate(well_ids: list[str], base_seed: int, control_well_ids: set[str] | None = None) -> dict[str, WellTrajectory]:
    """Simulate every well on a plate; distinct seeds so wells diverge."""
    control_well_ids = control_well_ids or set()
    return {
        wid: simulate_well(wid, seed=base_seed * 1000 + idx, control=wid in control_well_ids)
        for idx, wid in enumerate(well_ids)
    }
