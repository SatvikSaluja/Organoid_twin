"""
EWC (Elastic Weight Consolidation) continual adaptation.

As the model observes a live-streamed culture period, it periodically
fine-tunes on recently-seen conditions (e.g. a specific decline pattern
under-represented in the original training mix) via `online_finetune_step`.
An EWC penalty -- weighted by each parameter's Fisher information from the
*original* training distribution -- keeps that fine-tuning from overwriting
what the model already knows about the general healthy/stress/decline
distinction: parameters the original task was highly sensitive to (high
Fisher) are penalized hard for moving; parameters it barely used are left
free to adapt.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in
neural networks" (PNAS 2017).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backend.gnn.architecture import PlateGNN
from backend.gnn.plate_graph import PLATE_ADJACENCY


@dataclass
class EWCState:
    fisher: dict[str, torch.Tensor]
    theta_star: dict[str, torch.Tensor]
    # Calibrated empirically (see eval/run_benchmark.py's EWC ablation): this
    # model's Fisher values are tiny (~1e-6, see compute_fisher_information),
    # so a textbook-scale lambda (~100s) barely registers against the task
    # loss's gradient. 1e8 sits in the useful middle of the stability/
    # plasticity tradeoff -- meaningfully slows reference-set forgetting
    # without freezing the model against the new data entirely.
    lam: float = 1e8


def compute_fisher_information(model: PlateGNN, dataset: TensorDataset, n_batches: int = 60, batch_size: int = 4) -> EWCState:
    """
    Estimate the diagonal Fisher information of `model`'s parameters w.r.t.
    the health-score loss on `dataset` (the original/reference training
    distribution), by averaging squared per-batch gradients.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
    theta_star = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    model.eval()
    n_seen = 0
    for i, (xb, hb, o2b, lacb) in enumerate(loader):
        if i >= n_batches:
            break
        model.zero_grad()
        health_pred, aux_pred = model(xb, PLATE_ADJACENCY)
        loss = F.mse_loss(health_pred, hb) + 0.5 * (
            F.mse_loss(aux_pred[..., 0], o2b) + F.mse_loss(aux_pred[..., 1], lacb)
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
        n_seen += 1

    for name in fisher:
        fisher[name] /= max(1, n_seen)

    return EWCState(fisher=fisher, theta_star=theta_star)


def ewc_penalty(model: PlateGNN, ewc: EWCState) -> torch.Tensor:
    loss = torch.tensor(0.0)
    for name, p in model.named_parameters():
        if name in ewc.fisher:
            loss = loss + (ewc.fisher[name] * (p - ewc.theta_star[name]) ** 2).sum()
    return ewc.lam * loss


def online_finetune_step(
    model: PlateGNN,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ewc: EWCState | None,
) -> float:
    """One gradient step on a freshly-seen batch, optionally EWC-regularized.
    Pass ewc=None to get the "no continual-learning safeguard" ablation arm."""
    xb, hb, o2b, lacb = batch
    model.train()
    optimizer.zero_grad()
    health_pred, aux_pred = model(xb, PLATE_ADJACENCY)
    task_loss = F.mse_loss(health_pred, hb) + 0.5 * (
        F.mse_loss(aux_pred[..., 0], o2b) + F.mse_loss(aux_pred[..., 1], lacb)
    )
    loss = task_loss if ewc is None else task_loss + ewc_penalty(model, ewc)
    loss.backward()
    optimizer.step()
    return task_loss.item()
