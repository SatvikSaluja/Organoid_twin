"""
MC-dropout uncertainty (Gal & Ghahramani, 2016): run the model several times
with dropout deliberately left *on* (i.e. in .train() mode, at inference
time) and treat the spread across those stochastic forward passes as an
approximate posterior over the health score. A well the model is genuinely
unsure about should show high variance across samples; a well it's
confident about should look almost identical every time.

Used for two things downstream:
  - surfacing calibrated uncertainty bands in the dashboard, not just a
    point estimate
  - an active-sensing suggestion: wells with high uncertainty are the ones
    where closer/more frequent sampling would help most, the same logic a
    real high-throughput assay would use to allocate limited sensor time
    (see backend/analysis/active_sensing.py)
"""
import torch

from backend.gnn.architecture import PlateGNN

N_MC_SAMPLES = 20


def predict_with_uncertainty(
    model: PlateGNN, x_raw: torch.Tensor, adj: torch.Tensor, n_samples: int = N_MC_SAMPLES
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x_raw: (B, N, T, 4). Returns (mean_health, std_health), each (B, N).
    Deliberately puts the model in .train() mode to keep dropout stochastic,
    then restores whatever mode it was in before returning.
    """
    was_training = model.training
    model.train()
    try:
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                health_score, _ = model(x_raw, adj)
                samples.append(health_score)
        stacked = torch.stack(samples, dim=0)  # (n_samples, B, N)
        return stacked.mean(dim=0), stacked.std(dim=0)
    finally:
        model.train(was_training)
