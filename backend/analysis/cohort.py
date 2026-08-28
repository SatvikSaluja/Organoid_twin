"""
Runs the three-arm closed-loop experiment across several plates, persists
every well's outcome to the database (models/db.py's Experiment/ArmOutcome
tables), and aggregates it back into survival curves + a log-rank
significance test per arm pair -- the Cohort Analytics dashboard's data
source. Keeping this out of main.py keeps the route handlers thin.
"""
from dataclasses import dataclass

from backend.biology.organoid_trajectory import DT_HOURS, N_STEPS
from backend.control.closed_loop import run_three_arm_experiment
from backend.models.db import ArmOutcome, Experiment, get_session
from backend.analysis.survival import SurvivalCurve, kaplan_meier, log_rank_test

ARMS = ("no_control", "model_driven", "oracle")


def run_and_persist_experiment(well_ids: list[str], n_plates: int, base_seed: int, model, n_steps: int = N_STEPS) -> int:
    with get_session() as session:
        experiment = Experiment(n_plates=n_plates, n_steps=n_steps, base_seed=base_seed)
        session.add(experiment)
        session.flush()

        for p in range(n_plates):
            seed = base_seed + p
            result = run_three_arm_experiment(well_ids, seed, model, n_steps=n_steps)
            for arm_name, wells in result.arms.items():
                for wid, w in wells.items():
                    session.add(ArmOutcome(
                        experiment_id=experiment.id, arm=arm_name, well_id=wid, plate_seed=seed,
                        limiting_factor=w.limiting_factor, decline_onset_step=w.decline_onset_step,
                        declined=w.decline_onset_step is not None,
                        mean_health=w.mean_health, healthy_hours=w.healthy_hours,
                        n_interventions=w.n_interventions,
                    ))
        session.commit()
        return experiment.id


@dataclass
class ExperimentSummary:
    experiment_id: int
    created_at: str
    n_plates: int
    n_steps: int
    arm_stats: dict[str, dict]
    survival_curves: dict[str, SurvivalCurve]
    log_rank: dict[str, dict]  # "arm_a_vs_arm_b" -> {chi2, p_value}
    cause_breakdown: dict[str, dict[str, float]]  # cause -> arm -> mean_health


def _event_times_and_observed(rows: list[ArmOutcome], n_steps: int) -> tuple[list[float], list[bool]]:
    times, observed = [], []
    for r in rows:
        if r.declined:
            times.append(r.decline_onset_step * DT_HOURS)
            observed.append(True)
        else:
            times.append(n_steps * DT_HOURS)
            observed.append(False)
    return times, observed


def get_experiment_summary(experiment_id: int) -> ExperimentSummary:
    with get_session() as session:
        experiment = session.get(Experiment, experiment_id)
        rows = session.query(ArmOutcome).filter_by(experiment_id=experiment_id).all()

        by_arm = {arm: [r for r in rows if r.arm == arm] for arm in ARMS}

        arm_stats = {}
        survival_curves = {}
        for arm, arm_rows in by_arm.items():
            if not arm_rows:
                continue
            mean_healths = [r.mean_health for r in arm_rows]
            healthy_hours = [r.healthy_hours for r in arm_rows]
            arm_stats[arm] = {
                "n_wells": len(arm_rows),
                "mean_health": sum(mean_healths) / len(mean_healths),
                "mean_healthy_hours": sum(healthy_hours) / len(healthy_hours),
                "n_declined": sum(1 for r in arm_rows if r.declined),
                "total_interventions": sum(r.n_interventions for r in arm_rows),
            }
            times, observed = _event_times_and_observed(arm_rows, experiment.n_steps)
            curve = kaplan_meier(times, observed)
            survival_curves[arm] = curve

        log_rank = {}
        arm_list = list(by_arm.keys())
        for i in range(len(arm_list)):
            for j in range(i + 1, len(arm_list)):
                a, b = arm_list[i], arm_list[j]
                if not by_arm[a] or not by_arm[b]:
                    continue
                times_a, obs_a = _event_times_and_observed(by_arm[a], experiment.n_steps)
                times_b, obs_b = _event_times_and_observed(by_arm[b], experiment.n_steps)
                log_rank[f"{a}_vs_{b}"] = log_rank_test(times_a, obs_a, times_b, obs_b)

        cause_breakdown: dict[str, dict[str, float]] = {}
        no_control_causes = {r.well_id + str(r.plate_seed): (r.limiting_factor or "none") for r in by_arm.get("no_control", [])}
        for arm, arm_rows in by_arm.items():
            for r in arm_rows:
                cause = no_control_causes.get(r.well_id + str(r.plate_seed), r.limiting_factor or "none")
                cause_breakdown.setdefault(cause, {}).setdefault(arm, [])
                cause_breakdown[cause][arm].append(r.mean_health)
        for cause, arms in cause_breakdown.items():
            for arm, vals in arms.items():
                arms[arm] = sum(vals) / len(vals)

        return ExperimentSummary(
            experiment_id=experiment.id, created_at=experiment.created_at.isoformat(),
            n_plates=experiment.n_plates, n_steps=experiment.n_steps,
            arm_stats=arm_stats, survival_curves=survival_curves, log_rank=log_rank,
            cause_breakdown=cause_breakdown,
        )


def list_experiments() -> list[dict]:
    with get_session() as session:
        experiments = session.query(Experiment).order_by(Experiment.created_at.desc()).limit(50).all()
        return [
            {"id": e.id, "created_at": e.created_at.isoformat(), "n_plates": e.n_plates, "n_steps": e.n_steps}
            for e in experiments
        ]
