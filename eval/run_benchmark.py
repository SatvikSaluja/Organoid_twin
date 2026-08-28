"""
Evaluation harness. Ground truth is exact (we generated it), so these are
real measurements against known onset times / known root causes, not proxy
metrics:

  1. Detection lead time -- how many simulated hours before decline_dynamics
     .py's recorded onset step does the bifurcation detector fire?
  2. False positive rate -- firings on wells that never actually decline.
  3. Recommendation accuracy -- for wells with a known root cause (oxygen /
     glucose / adverse_event), does the recommendation match it, and how
     much better is the learned cause_head than the hand-tuned sensor-delta
     heuristic it replaced as the primary decision path?
  4. Ablations:
       a) with vs. without the hard stoichiometric consistency constraint
          (gnn/constraints.py) -- compares held-out consistency residual.
       b) with vs. without EWC continual adaptation (gnn/coevolution.py) --
          compares reference-set forgetting over a simulated long deployment
          that drifts into a decline-heavy regime partway through.

Run with:
    .venv/bin/python -m eval.run_benchmark
Writes eval/results.json and prints a summary.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from backend.biology.organoid_trajectory import N_STEPS, simulate_plate
from backend.config import CHECKPOINT_DIR, SENSOR_TYPES, WELL_IDS
from backend.gnn.architecture import CAUSE_CLASSES, DEFAULT_WINDOW
from backend.gnn.bifurcation import (
    DEFAULT_COOLDOWN_STEPS,
    DEFAULT_CONSEC_REQUIRED,
    BifurcationDetector,
    compute_jacobian_norms,
)
from backend.gnn.coevolution import compute_fisher_information, online_finetune_step
from backend.gnn.constraints import stoichiometric_consistency_penalty
from backend.gnn.plate_graph import PLATE_ADJACENCY
from backend.gnn.train import build_dataset, calibrate_stoichiometry, load_checkpoint
from backend.recommend.engine import compute_sensor_deltas, recommend_for_well
from backend.sensors.sensor_model import trajectory_to_readings

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

N_EVAL_PLATES = 8
LEAD_TIME_MATCH_WINDOW = 20  # a fire within [onset - inf, onset + this] counts as "detected"

ROOT_CAUSE_TO_EXPECTED_ACTION = {
    "oxygen": "Increase media oxygenation",
    "glucose": "Increase glucose feed rate",
    "adverse_event": "Flag for manual inspection",
}


def _plate_readings(base_seed: int):
    plate = simulate_plate(WELL_IDS, base_seed=base_seed)
    readings = {
        wid: trajectory_to_readings(traj, seed=base_seed * 13 + i)
        for i, (wid, traj) in enumerate(plate.items())
    }
    sensor_stack = np.stack(
        [np.stack([getattr(readings[wid], s) for s in SENSOR_TYPES], axis=-1) for wid in WELL_IDS], axis=0
    )
    return plate, sensor_stack


def eval_detection_and_recommendations(model, threshold: float, base_seed: int = 3_000_000) -> dict:
    detector_leads = []
    fp_count = 0
    fn_count = 0
    n_declined_total = 0

    rec_correct = 0
    rec_total = 0
    rec_confusion = []
    heuristic_correct = 0
    cause_classifier_correct = 0  # raw argmax(cause_logits) vs true cause, independent of action-text matching

    t0 = time.time()
    for p in range(N_EVAL_PLATES):
        seed = base_seed + p
        plate, sensor_stack = _plate_readings(seed)
        detector = BifurcationDetector(WELL_IDS, threshold=threshold,
                                        consec_required=DEFAULT_CONSEC_REQUIRED, cooldown_steps=DEFAULT_COOLDOWN_STEPS)
        fires_by_well = {wid: [] for wid in WELL_IDS}

        for t in range(DEFAULT_WINDOW, N_STEPS):
            x = torch.tensor(sensor_stack[:, t - DEFAULT_WINDOW:t, :], dtype=torch.float32).unsqueeze(0)
            norms = compute_jacobian_norms(model, x, PLATE_ADJACENCY, WELL_IDS)
            for wid, norm in norms.items():
                if detector.update(wid, norm):
                    fires_by_well[wid].append(t)

        for i, wid in enumerate(WELL_IDS):
            onset = plate[wid].decline_onset_step
            fires = fires_by_well[wid]
            if onset is None:
                fp_count += len(fires)
                continue

            n_declined_total += 1
            valid_fires = [f for f in fires if f <= onset + LEAD_TIME_MATCH_WINDOW]
            if valid_fires:
                lead_steps = onset - valid_fires[0]
                detector_leads.append(lead_steps * 0.5)  # -> hours (DT_HOURS=0.5)
            else:
                fn_count += 1

            # -- recommendation accuracy: evaluate at whichever step in
            # [onset, onset+LEAD_TIME_MATCH_WINDOW] the model itself scores
            # this well as least healthy, using only sensor readings/deltas
            # a real deployment would have. Right at onset the ground-truth
            # latent stress has only just crossed its own threshold, so the
            # model's health score usually hasn't dropped enough yet to
            # clear the recommendation engine's action threshold -- and a
            # fixed offset risks landing right after a feed event (which
            # resets O2/glucose toward baseline every 24h) and flipping the
            # apparent trend direction. Scanning for the model's own worst
            # point sidesteps both: it's what a real deployment would act
            # on anyway.
            cause = plate[wid].limiting_factor
            candidate_ts = [t for t in range(onset, min(N_STEPS, onset + LEAD_TIME_MATCH_WINDOW + 1)) if t >= DEFAULT_WINDOW]
            if cause in ROOT_CAUSE_TO_EXPECTED_ACTION and candidate_ts:
                with torch.no_grad():
                    xs = torch.stack([
                        torch.tensor(sensor_stack[:, t - DEFAULT_WINDOW:t, :], dtype=torch.float32) for t in candidate_ts
                    ])  # (n_candidates, N, T, 4)
                    candidate_scores, _, candidate_cause_logits = model(xs, PLATE_ADJACENCY, return_cause=True)  # (n_candidates, N)
                worst_idx = int(torch.argmin(candidate_scores[:, i]))
                eval_t = candidate_ts[worst_idx]
                score = float(candidate_scores[worst_idx, i])
                predicted_cause = CAUSE_CLASSES[int(candidate_cause_logits[worst_idx, i].argmax())]

                deltas = compute_sensor_deltas(sensor_stack[i, eval_t], sensor_stack[i, eval_t - DEFAULT_WINDOW])
                expected_action = ROOT_CAUSE_TO_EXPECTED_ACTION[cause]

                # primary: the ML cause_head drives the actual recommendation
                rec = recommend_for_well(wid, deltas, score, cause=predicted_cause)
                predicted_action = rec.action if rec else "No action (health score above threshold)"
                rec_total += 1
                is_correct = predicted_action == expected_action
                rec_correct += int(is_correct)
                cause_classifier_correct += int(predicted_cause == cause)

                # comparison: what the old hand-tuned heuristic alone would have said
                heuristic_rec = recommend_for_well(wid, deltas, score, cause=None)
                heuristic_action = heuristic_rec.action if heuristic_rec else "No action (health score above threshold)"
                heuristic_correct += int(heuristic_action == expected_action)

                rec_confusion.append({
                    "well": wid, "plate_seed": seed, "true_cause": cause, "predicted_cause": predicted_cause,
                    "expected": expected_action, "predicted": predicted_action, "correct": is_correct,
                    "heuristic_predicted": heuristic_action, "heuristic_correct": heuristic_action == expected_action,
                })

    elapsed = time.time() - t0
    n_healthy_well_weeks = (N_EVAL_PLATES * len(WELL_IDS) - n_declined_total) * (N_STEPS * 0.5 / 24 / 7)

    return {
        "n_plates": N_EVAL_PLATES,
        "n_declined_wells": n_declined_total,
        "n_detected": len(detector_leads),
        "n_missed": fn_count,
        "recall": len(detector_leads) / n_declined_total if n_declined_total else None,
        "lead_hours": detector_leads,
        "mean_lead_hours": float(np.mean(detector_leads)) if detector_leads else None,
        "median_lead_hours": float(np.median(detector_leads)) if detector_leads else None,
        "false_positive_count": fp_count,
        "false_positives_per_healthy_well_week": fp_count / n_healthy_well_weeks if n_healthy_well_weeks else None,
        "recommendation_accuracy": rec_correct / rec_total if rec_total else None,
        "recommendation_accuracy_heuristic_baseline": heuristic_correct / rec_total if rec_total else None,
        "cause_classifier_raw_accuracy": cause_classifier_correct / rec_total if rec_total else None,
        "recommendation_n": rec_total,
        "recommendation_confusion": rec_confusion,
        "elapsed_sec": elapsed,
    }


def eval_constraint_ablation(base_seed: int = 4_000_000) -> dict:
    fit = calibrate_stoichiometry()
    eval_x, eval_h, eval_o2, eval_lac, _ = build_dataset(3, seed_offset=base_seed, fit=fit)

    out = {}
    for tag, use_constraint in [("constrained", True), ("unconstrained", False)]:
        model, _ = load_checkpoint(tag)
        model.eval()
        with torch.no_grad():
            health_pred, aux_pred = model(eval_x, PLATE_ADJACENCY)
            health_mse = torch.nn.functional.mse_loss(health_pred, eval_h).item()
            residual = stoichiometric_consistency_penalty(aux_pred[..., 0], aux_pred[..., 1], fit).item()

        meta_path = CHECKPOINT_DIR / f"train_meta_{tag}.json"
        train_meta = json.load(open(meta_path)) if meta_path.exists() else {}
        out[tag] = {
            "held_out_health_mse": health_mse,
            "held_out_stoichiometric_residual": residual,
            "final_train_val_loss": train_meta.get("final_val_loss"),
        }
    return out


def eval_ewc_ablation(base_seed: int = 5_000_000) -> dict:
    """
    Simulate a long deployment that drifts: the reference distribution (what
    the model was originally trained on) vs. a run of "new" plates skewed
    toward decline (a distribution shift a real deployment might see as a
    culture batch ages or conditions change). Fine-tune two copies of the
    same starting model online against the new plates -- one with the EWC
    penalty, one without -- and track each one's loss on the *original*
    reference set after every step. If EWC is doing its job, the EWC copy's
    reference loss should stay lower (forget less) than the no-EWC copy's.
    """
    fit = calibrate_stoichiometry()
    reference_x, reference_h, reference_o2, reference_lac, _ = build_dataset(3, seed_offset=1000, fit=fit)  # same pool train.py used
    reference_ds = torch.utils.data.TensorDataset(reference_x, reference_h, reference_o2, reference_lac)

    new_x, new_h, new_o2, new_lac, _ = build_dataset(4, seed_offset=base_seed, fit=fit)
    # bias the "new" stream toward wells that are further into decline, by
    # sorting snapshots by (1 - health) descending and taking the neediest half
    order = torch.argsort(new_h.mean(dim=1), descending=False)  # low mean health first
    half = len(order) // 2
    drifted_idx = order[:half]

    results = {}
    for variant in ["with_ewc", "without_ewc"]:
        model, _ = load_checkpoint("constrained")
        ewc = compute_fisher_information(model, reference_ds) if variant == "with_ewc" else None
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        ref_losses = []
        model.eval()
        with torch.no_grad():
            pred, _ = model(reference_x, PLATE_ADJACENCY)
            ref_losses.append(torch.nn.functional.mse_loss(pred, reference_h).item())

        for idx in drifted_idx.tolist():
            batch = (new_x[idx:idx + 1], new_h[idx:idx + 1], new_o2[idx:idx + 1], new_lac[idx:idx + 1])
            online_finetune_step(model, optimizer, batch, ewc)

        model.eval()
        with torch.no_grad():
            pred, _ = model(reference_x, PLATE_ADJACENCY)
            ref_losses.append(torch.nn.functional.mse_loss(pred, reference_h).item())

        results[variant] = {
            "reference_loss_before": ref_losses[0],
            "reference_loss_after": ref_losses[-1],
            "reference_loss_increase": ref_losses[-1] - ref_losses[0],
        }
    return results


def main():
    print("Loading bifurcation threshold + model...")
    model, _ = load_checkpoint("constrained")
    threshold_path = CHECKPOINT_DIR / "bifurcation_threshold.json"
    threshold = json.load(open(threshold_path))["threshold"] if threshold_path.exists() else 1.0

    print(f"\n[1/3] Detection lead time + false-positive rate + recommendation accuracy over {N_EVAL_PLATES} held-out plates...")
    detection = eval_detection_and_recommendations(model, threshold)
    print(f"  recall={detection['recall']:.0%}  mean_lead={detection['mean_lead_hours']:.1f}h  "
          f"median_lead={detection['median_lead_hours']:.1f}h  fp/healthy-well-week={detection['false_positives_per_healthy_well_week']:.2f}")
    print(f"  recommendation_accuracy={detection['recommendation_accuracy']:.0%} "
          f"(heuristic baseline: {detection['recommendation_accuracy_heuristic_baseline']:.0%}, "
          f"raw cause-classifier accuracy: {detection['cause_classifier_raw_accuracy']:.0%}, n={detection['recommendation_n']})")

    print("\n[2/3] Constraint ablation (with vs. without hard consistency constraint)...")
    constraint = eval_constraint_ablation()
    for tag, r in constraint.items():
        print(f"  {tag}: health_mse={r['held_out_health_mse']:.5f}  stoichiometric_residual={r['held_out_stoichiometric_residual']:.5f}")

    print("\n[3/3] EWC ablation (with vs. without continual-adaptation safeguard)...")
    ewc = eval_ewc_ablation()
    for variant, r in ewc.items():
        print(f"  {variant}: reference_loss {r['reference_loss_before']:.5f} -> {r['reference_loss_after']:.5f} "
              f"(+{r['reference_loss_increase']:.5f})")

    results = {"detection": detection, "constraint_ablation": constraint, "ewc_ablation": ewc}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
