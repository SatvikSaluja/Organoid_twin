"""
The closed-loop control arena: runs a whole plate under one of three arms,
stepping every well's WellSimulator in lockstep (so the model-driven arm can
do real plate-graph inference each tick) and applying that arm's policy's
interventions as it goes.

  - no_control:    passive monitoring only, nothing is ever corrected
  - model_driven:  the actual system -- GNN health score + recommend/engine.py,
                    sensor readings only, exactly what a real deployment sees
  - oracle:        upper bound -- acts on ground-truth decline state directly

All three arms for a given base_seed use identical per-well RNG streams
(same seed -> same adverse-event timing/type, same sensor noise draws), so
the only thing that can differ between arms is the physical effect of an
intervention actually being applied. That's what makes the comparison fair.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from backend.biology.organoid_trajectory import N_STEPS, DT_HOURS, WellSimulator
from backend.config import SENSOR_TYPES
from backend.control.policies import ModelDrivenPolicy, NoInterventionPolicy, OraclePolicy
from backend.gnn.architecture import CAUSE_CLASSES, DEFAULT_WINDOW
from backend.gnn.plate_graph import PLATE_ADJACENCY
from backend.recommend.engine import compute_sensor_deltas, recommend_for_well
from backend.sensors.sensor_model import StreamingSensorState, true_signals_from_state

HEALTHY_THRESHOLD = 0.7


@dataclass
class WellArmResult:
    well_id: str
    arm: str
    health_series: np.ndarray
    stress_series: np.ndarray
    decline_onset_step: int | None
    limiting_factor: str | None
    n_interventions: int

    @property
    def mean_health(self) -> float:
        return float(self.health_series.mean())

    @property
    def healthy_hours(self) -> float:
        return float((self.health_series >= HEALTHY_THRESHOLD).sum() * DT_HOURS)


def _make_policy(arm: str):
    if arm == "no_control":
        return NoInterventionPolicy()
    if arm == "oracle":
        return OraclePolicy()
    if arm == "model_driven":
        return ModelDrivenPolicy()
    raise ValueError(f"unknown arm '{arm}'")


def run_arm(
    well_ids: list[str],
    base_seed: int,
    arm: str,
    n_steps: int = N_STEPS,
    model=None,
    on_tick=None,
) -> dict[str, WellArmResult]:
    """
    model is required (and only used) for arm='model_driven'.
    on_tick(t, sims, readings, interventions), if given, is called once per
    step -- the live dashboard's control-arena mode uses this to broadcast
    every tick instead of waiting for the whole run to finish.
    """
    policy = _make_policy(arm)
    sims = {wid: WellSimulator(wid, seed=base_seed * 1000 + idx) for idx, wid in enumerate(well_ids)}
    sensors = {wid: StreamingSensorState(seed=base_seed * 7919 + idx) for idx, wid in enumerate(well_ids)}
    history: dict[str, list[list[float]]] = {wid: [] for wid in well_ids}

    health_series = {wid: np.zeros(n_steps) for wid in well_ids}
    stress_series = {wid: np.zeros(n_steps) for wid in well_ids}
    intervention_counts = {wid: 0 for wid in well_ids}

    for t in range(n_steps):
        interventions = {wid: None for wid in well_ids}

        if arm == "oracle":
            for wid in well_ids:
                interventions[wid] = policy.decide(wid, sims[wid].decline, sims[wid].limiting_factor)

        elif arm == "model_driven" and t >= DEFAULT_WINDOW:
            x = torch.tensor(
                np.stack([history[wid][-DEFAULT_WINDOW:] for wid in well_ids]), dtype=torch.float32
            ).unsqueeze(0)  # (1, N, T, 4)
            with torch.no_grad():
                health_scores, _, cause_logits = model(x, PLATE_ADJACENCY, return_cause=True)
            causes = [CAUSE_CLASSES[c] for c in cause_logits.squeeze(0).argmax(dim=-1).tolist()]
            for i, wid in enumerate(well_ids):
                now, past = np.array(history[wid][-1]), np.array(history[wid][-DEFAULT_WINDOW])
                deltas = compute_sensor_deltas(now, past)
                rec = recommend_for_well(wid, deltas, float(health_scores[0, i]), cause=causes[i])
                interventions[wid] = policy.decide(wid, rec)

        readings = {}
        for wid in well_ids:
            state = sims[wid].step(interventions[wid])
            if interventions[wid] is not None:
                intervention_counts[wid] += 1
            health_series[wid][t] = 1.0 - (0.7 * state["stress_level"] + 0.3 * state["warburg_index"])
            stress_series[wid][t] = state["stress_level"]

            true_signals = true_signals_from_state(
                glucose=state["glucose"], oxygen=state["oxygen"], lactate=state["lactate"],
                stress_level=state["stress_level"], demand=state["demand"],
            )
            reading = sensors[wid].step(true_signals)
            readings[wid] = reading
            history[wid].append([reading[s] for s in SENSOR_TYPES])

        if on_tick is not None:
            on_tick(t, sims, readings, interventions)

    return {
        wid: WellArmResult(
            well_id=wid, arm=arm,
            health_series=health_series[wid], stress_series=stress_series[wid],
            decline_onset_step=sims[wid].decline.decline_onset_step,
            limiting_factor=sims[wid].limiting_factor,
            n_interventions=intervention_counts[wid],
        )
        for wid in well_ids
    }


@dataclass
class ThreeArmResult:
    base_seed: int
    arms: dict[str, dict[str, WellArmResult]] = field(default_factory=dict)

    def summary(self) -> dict:
        out = {}
        for arm, wells in self.arms.items():
            mean_healths = [w.mean_health for w in wells.values()]
            healthy_hours = [w.healthy_hours for w in wells.values()]
            onsets = [w.decline_onset_step for w in wells.values()]
            out[arm] = {
                "mean_health": float(np.mean(mean_healths)),
                "mean_healthy_hours": float(np.mean(healthy_hours)),
                "n_declined": sum(1 for o in onsets if o is not None),
                "total_interventions": sum(w.n_interventions for w in wells.values()),
            }
        return out


def run_three_arm_experiment(well_ids: list[str], base_seed: int, model, n_steps: int = N_STEPS) -> ThreeArmResult:
    result = ThreeArmResult(base_seed=base_seed)
    result.arms["no_control"] = run_arm(well_ids, base_seed, "no_control", n_steps=n_steps)
    result.arms["model_driven"] = run_arm(well_ids, base_seed, "model_driven", n_steps=n_steps, model=model)
    result.arms["oracle"] = run_arm(well_ids, base_seed, "oracle", n_steps=n_steps)
    return result
