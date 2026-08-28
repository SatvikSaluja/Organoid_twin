"""
Plotting for eval/run_benchmark.py output (eval/results.json).

Run after run_benchmark.py:
    .venv/bin/python -m eval.plots
Writes PNGs to eval/plots/.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

BG = "#0f1115"
FG = "#e7e9ee"
GRID = "#262b36"
ACCENT = "#60a5fa"
GOOD = "#4ade80"
BAD = "#f87171"


def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)


def plot_lead_time_distribution(detection: dict, out_path: Path):
    leads = detection.get("lead_hours", [])
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    if leads:
        ax.hist(leads, bins=min(12, max(4, len(leads) // 3)), color=ACCENT, edgecolor=BG)
        ax.axvline(detection["mean_lead_hours"], color=GOOD, linestyle="--", label=f"mean {detection['mean_lead_hours']:.1f}h")
        ax.axvline(detection["median_lead_hours"], color="#facc15", linestyle="--", label=f"median {detection['median_lead_hours']:.1f}h")
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID)
    ax.set_xlabel("Lead time (hours before ground-truth onset)")
    ax.set_ylabel("Wells detected")
    ax.set_title(f"Bifurcation detection lead time (recall {detection['recall']:.0%}, n={detection['n_declined_wells']})")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, dpi=150)
    plt.close(fig)


def plot_recommendation_accuracy(detection: dict, out_path: Path):
    conf = detection["recommendation_confusion"]
    by_cause, correct_by_cause = Counter(), Counter()
    for c in conf:
        by_cause[c["true_cause"]] += 1
        correct_by_cause[c["true_cause"]] += int(c["correct"])

    causes = sorted(by_cause.keys())
    accs = [correct_by_cause[c] / by_cause[c] for c in causes]
    ns = [by_cause[c] for c in causes]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    colors = [GOOD if a >= 0.7 else (ACCENT if a >= 0.4 else BAD) for a in accs]
    bars = ax.bar(causes, accs, color=colors)
    for b, a, n in zip(bars, accs, ns):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.0%} (n={n})", ha="center", color=FG, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recommendation accuracy")
    ax.set_title("Recommendation accuracy by true root cause")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, dpi=150)
    plt.close(fig)


def plot_constraint_ablation(constraint: dict, out_path: Path):
    tags = ["unconstrained", "constrained"]
    residuals = [constraint[t]["held_out_stoichiometric_residual"] for t in tags]
    mses = [constraint[t]["held_out_health_mse"] for t in tags]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor=BG)
    for ax, values, title in [(ax1, residuals, "Stoichiometric consistency residual\n(lower = more internally consistent)"),
                                (ax2, mses, "Held-out health-score MSE\n(lower = more accurate)")]:
        bars = ax.bar(tags, values, color=[BAD, GOOD])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", color=FG, fontsize=9)
        ax.set_title(title)
        _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, dpi=150)
    plt.close(fig)


def plot_ewc_ablation(ewc: dict, out_path: Path):
    variants = ["without_ewc", "with_ewc"]
    before = [ewc[v]["reference_loss_before"] for v in variants]
    after = [ewc[v]["reference_loss_after"] for v in variants]

    x = range(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    ax.bar([i - width / 2 for i in x], before, width, label="before online fine-tuning", color="#475569")
    ax.bar([i + width / 2 for i in x], after, width, label="after online fine-tuning", color=[BAD, GOOD])
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylabel("Reference-set (original task) loss")
    ax.set_title("EWC continual adaptation: forgetting on the original task")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, dpi=150)
    plt.close(fig)


def main():
    results = json.load(open(RESULTS_PATH))
    PLOTS_DIR.mkdir(exist_ok=True)

    plot_lead_time_distribution(results["detection"], PLOTS_DIR / "lead_time.png")
    plot_recommendation_accuracy(results["detection"], PLOTS_DIR / "recommendation_accuracy.png")
    plot_constraint_ablation(results["constraint_ablation"], PLOTS_DIR / "constraint_ablation.png")
    plot_ewc_ablation(results["ewc_ablation"], PLOTS_DIR / "ewc_ablation.png")

    print(f"Wrote plots to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
