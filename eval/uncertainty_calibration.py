"""
Validates the MC-dropout uncertainty (gnn/uncertainty.py) actually means
something, rather than just reporting a plausible-looking number. Two checks,
both standard for regression uncertainty:

  1. Sharpness-vs-error correlation: samples the model reports higher
     uncertainty on should, on average, be the ones it's more wrong about.
     Reported as a Spearman correlation between predicted std and absolute
     error, plus a binned reliability table.
  2. Interval coverage: if predicted_std really behaved like a Gaussian
     standard deviation, ~68% of true values should fall within
     mean ± 1*std, and ~95% within mean ± 2*std. Reported coverage far from
     those targets means the uncertainty is under- or over-confident, even
     if it's directionally useful (check 1 can pass while this one fails --
     they measure different things).

Run with:
    .venv/bin/python -m eval.uncertainty_calibration
Writes eval/uncertainty_results.json and prints a summary.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from backend.biology.organoid_trajectory import N_STEPS, simulate_plate
from backend.config import SENSOR_TYPES, WELL_IDS
from backend.gnn.architecture import DEFAULT_WINDOW
from backend.gnn.plate_graph import PLATE_ADJACENCY
from backend.gnn.train import load_checkpoint
from backend.gnn.uncertainty import predict_with_uncertainty
from backend.sensors.sensor_model import trajectory_to_readings

RESULTS_PATH = Path(__file__).resolve().parent / "uncertainty_results.json"
N_EVAL_PLATES = 6
TIME_STRIDE = 4
N_BINS = 5


def _true_health(plate, wid, t) -> float:
    traj = plate[wid]
    return float(np.clip(1.0 - (0.7 * traj.stress_level[t] + 0.3 * traj.warburg_index[t]), 0.0, 1.0))


def collect_samples(model, base_seed: int = 6_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (predicted_mean, predicted_std, true_health), one entry per (well, sampled timestep)."""
    means, stds, truths = [], [], []

    for p in range(N_EVAL_PLATES):
        seed = base_seed + p
        plate = simulate_plate(WELL_IDS, base_seed=seed)
        readings = {
            wid: trajectory_to_readings(traj, seed=seed * 13 + i)
            for i, (wid, traj) in enumerate(plate.items())
        }
        sensor_stack = np.stack(
            [np.stack([getattr(readings[wid], s) for s in SENSOR_TYPES], axis=-1) for wid in WELL_IDS], axis=0
        )

        for t in range(DEFAULT_WINDOW, N_STEPS, TIME_STRIDE):
            x = torch.tensor(sensor_stack[:, t - DEFAULT_WINDOW:t, :], dtype=torch.float32).unsqueeze(0)
            mean, std = predict_with_uncertainty(model, x, PLATE_ADJACENCY, n_samples=20)
            mean, std = mean.squeeze(0).numpy(), std.squeeze(0).numpy()
            for i, wid in enumerate(WELL_IDS):
                means.append(mean[i])
                stds.append(std[i])
                truths.append(_true_health(plate, wid, t))

    return np.array(means), np.array(stds), np.array(truths)


def evaluate_calibration(means: np.ndarray, stds: np.ndarray, truths: np.ndarray) -> dict:
    errors = np.abs(means - truths)

    spearman_corr, spearman_p = stats.spearmanr(stds, errors)

    # Binned reliability table: sort by predicted std, split into N_BINS
    # equal-count bins, report mean std and mean error per bin.
    order = np.argsort(stds)
    bins = np.array_split(order, N_BINS)
    reliability = []
    for b in bins:
        reliability.append({
            "mean_predicted_std": float(stds[b].mean()),
            "mean_abs_error": float(errors[b].mean()),
            "n": len(b),
        })

    # Interval coverage: does the ±kσ interval contain the truth at the
    # nominal Gaussian rate?
    z1 = np.abs(means - truths) <= stds
    z2 = np.abs(means - truths) <= 2 * stds
    coverage_1sigma = float(z1.mean())
    coverage_2sigma = float(z2.mean())

    return {
        "n_samples": len(means),
        "spearman_std_vs_error": float(spearman_corr),
        "spearman_p_value": float(spearman_p),
        "reliability_bins": reliability,
        "coverage_1sigma": coverage_1sigma,
        "coverage_1sigma_target": 0.68,
        "coverage_2sigma": coverage_2sigma,
        "coverage_2sigma_target": 0.95,
        "mean_predicted_std": float(stds.mean()),
        "mean_abs_error": float(errors.mean()),
    }


def main():
    print("Loading model...")
    model, _ = load_checkpoint("constrained")

    print(f"Collecting samples across {N_EVAL_PLATES} held-out plates...")
    means, stds, truths = collect_samples(model)

    print("Evaluating calibration...")
    result = evaluate_calibration(means, stds, truths)

    print(f"\nn={result['n_samples']}")
    print(f"Spearman(predicted_std, |error|) = {result['spearman_std_vs_error']:.3f} (p={result['spearman_p_value']:.4f})")
    print("  -> positive & significant means higher-uncertainty predictions really are more often wrong")
    print("\nReliability bins (low to high predicted uncertainty):")
    for b in result["reliability_bins"]:
        print(f"  mean_std={b['mean_predicted_std']:.4f}  mean_abs_error={b['mean_abs_error']:.4f}  (n={b['n']})")
    print(f"\nCoverage: {result['coverage_1sigma']:.0%} within ±1σ (target 68%), {result['coverage_2sigma']:.0%} within ±2σ (target 95%)")
    if result["coverage_1sigma"] < 0.5:
        print("  -> under-confident/miscalibrated: the model is wrong more often than its own σ would predict.")
    elif result["coverage_1sigma"] > 0.85:
        print("  -> over-confident in the other direction: σ is larger than the errors actually warrant.")

    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
