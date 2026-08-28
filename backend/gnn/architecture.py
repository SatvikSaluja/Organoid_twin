"""
Multimodal GATv2 fusion model.

Per well, per timestep: the 4 sensor readings (normalized) go into a small
GRU that encodes the recent window into a per-well embedding (so the model
reasons over trends, not single readings). Those embeddings are then the
node features for 2-3 layers of GATv2-style attention over the plate graph
(plate_graph.py), letting each well's prediction be informed by its
neighbors. Two output heads:

  - health_score: continuous well-health estimate in [0, 1] (1 = healthy).
    Continuous rather than a fixed class label because the bifurcation
    detector (bifurcation.py) needs to differentiate it, and because a
    label is just a threshold on top of a score anyway.
  - aux (o2_consumption_norm, lactate_production_norm): normalized estimates
    of the two flux quantities constraints.py ties together via the same
    stoichiometric relationship metabolic_sim.py uses to generate the
    ground truth in the first place.
  - cause_logits (4-way: none / oxygen / glucose / adverse_event): a learned
    root-cause classifier, trained directly against decline_dynamics.py's
    ground-truth cause labels. This replaced an earlier hand-tuned,
    sensor-delta-threshold heuristic in recommend/engine.py that topped out
    around 24%/4% accuracy on oxygen/glucose attribution -- the underlying
    signal is genuinely present in the fused embedding (see eval results),
    it just wasn't recoverable from a few hand-picked thresholds.

GATv2 (Brody et al. 2021) reorders the original GAT attention so the
nonlinearity is applied after combining the two projected node vectors,
which fixes GAT's "static attention" limitation:
    e_ij = a^T LeakyReLU(W_l h_i + W_r h_j)
Implemented here as dense attention over a small (24-node) adjacency matrix
rather than via a sparse-graph library -- see plate_graph.py's docstring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SENSORS = 4
DEFAULT_WINDOW = 12       # timesteps of history fed to the temporal encoder
TEMPORAL_HIDDEN = 24
GNN_HIDDEN = 32
GNN_HEADS = 4
GNN_LAYERS = 3
LEAKY_SLOPE = 0.2
DROPOUT_P = 0.15  # kept active at inference time for MC-dropout uncertainty, see gnn/uncertainty.py

CAUSE_CLASSES = ["none", "oxygen", "glucose", "adverse_event"]


class GATv2Layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = GNN_HEADS):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin_l = nn.Linear(in_dim, out_dim * heads)
        self.lin_r = nn.Linear(in_dim, out_dim * heads)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        self.out_proj = nn.Linear(out_dim * heads, out_dim * heads)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, return_attention: bool = False):
        """
        h:   (B, N, in_dim)
        adj: (N, N) with 1s on edges (including self-loops), 0 elsewhere
        returns: (B, N, out_dim * heads), and optionally the attention
        weights (B, N, N, heads) -- which well influenced which, used by the
        interactive control panel's live attention-graph view.
        """
        B, N, _ = h.shape
        H = self.heads
        Dout = self.out_dim

        hl = self.lin_l(h).view(B, N, H, Dout)  # (B, N, H, D)
        hr = self.lin_r(h).view(B, N, H, Dout)

        # e_ij = a^T LeakyReLU(hl_i + hr_j), broadcast over all (i, j) pairs
        hl_i = hl.unsqueeze(2)  # (B, N, 1, H, D)
        hr_j = hr.unsqueeze(1)  # (B, 1, N, H, D)
        e = F.leaky_relu(hl_i + hr_j, LEAKY_SLOPE)          # (B, N, N, H, D)
        e = torch.einsum("bijhd,hd->bijh", e, self.attn)     # (B, N, N, H)

        mask = adj.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)  # softmax over neighbors j, per (b, i, h)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated node safety

        # weighted sum of neighbor value vectors (reuse hr as the value projection)
        out = torch.einsum("bijh,bjhd->bihd", alpha, hr)  # (B, N, H, D)
        out = out.reshape(B, N, H * Dout)
        out = self.out_proj(out)
        return (out, alpha) if return_attention else out


class TemporalEncoder(nn.Module):
    def __init__(self, n_sensors: int = N_SENSORS, hidden: int = TEMPORAL_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size=n_sensors, hidden_size=hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, n_sensors) -> (B, N, hidden)"""
        B, N, T, F_ = x.shape
        x = x.reshape(B * N, T, F_)
        _, h_n = self.gru(x)         # h_n: (1, B*N, hidden)
        h = h_n.squeeze(0).reshape(B, N, -1)
        return h


class PlateGNN(nn.Module):
    def __init__(
        self,
        n_sensors: int = N_SENSORS,
        temporal_hidden: int = TEMPORAL_HIDDEN,
        gnn_hidden: int = GNN_HIDDEN,
        heads: int = GNN_HEADS,
        n_layers: int = GNN_LAYERS,
    ):
        super().__init__()
        self.temporal = TemporalEncoder(n_sensors, temporal_hidden)

        dims = [temporal_hidden] + [gnn_hidden * heads] * n_layers
        self.gat_layers = nn.ModuleList([
            GATv2Layer(dims[i], gnn_hidden, heads=heads) for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(gnn_hidden * heads) for _ in range(n_layers)])
        self.residual_proj = nn.ModuleList([
            nn.Linear(dims[i], gnn_hidden * heads) if dims[i] != gnn_hidden * heads else nn.Identity()
            for i in range(n_layers)
        ])
        # Left on at inference time for MC-dropout uncertainty (gnn/uncertainty.py
        # runs the model in .train() mode deliberately) -- this is the whole
        # mechanism, not a training-only regularizer here.
        self.dropout = nn.Dropout(DROPOUT_P)

        final_dim = gnn_hidden * heads
        self.health_head = nn.Sequential(nn.Linear(final_dim, final_dim // 2), nn.ReLU(), nn.Linear(final_dim // 2, 1))
        self.aux_head = nn.Sequential(nn.Linear(final_dim, final_dim // 2), nn.ReLU(), nn.Linear(final_dim // 2, 2))
        self.cause_head = nn.Sequential(nn.Linear(final_dim, final_dim // 2), nn.ReLU(), nn.Linear(final_dim // 2, len(CAUSE_CLASSES)))

        # Per-sensor normalization stats, set by train.py before training and
        # persisted in the checkpoint (registered buffers -> saved/loaded by
        # state_dict automatically, unlike plain attributes).
        self.register_buffer("sensor_mean", torch.zeros(n_sensors))
        self.register_buffer("sensor_std", torch.ones(n_sensors))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.sensor_mean) / self.sensor_std

    def forward(self, x_raw: torch.Tensor, adj: torch.Tensor, return_attention: bool = False, return_cause: bool = False):
        """
        x_raw: (B, N, T, n_sensors) raw (unnormalized) sensor window
        adj:   (N, N) plate adjacency
        returns: health_score (B, N), aux (B, N, 2), then -- in this order,
        each only if requested -- cause_logits (B, N, len(CAUSE_CLASSES))
        and the last GAT layer's attention weights (B, N, N, heads).
        """
        x = self.normalize(x_raw)
        h = self.temporal(x)  # (B, N, temporal_hidden)

        last_attention = None
        n_layers = len(self.gat_layers)
        for idx, (gat, norm, res_proj) in enumerate(zip(self.gat_layers, self.norms, self.residual_proj)):
            want_attn = return_attention and idx == n_layers - 1
            gat_out = gat(h, adj, return_attention=want_attn)
            h_new, last_attention = gat_out if want_attn else (gat_out, last_attention)
            h = norm(h_new + res_proj(h))
            h = F.gelu(h)
            h = self.dropout(h)

        health_score = torch.sigmoid(self.health_head(h)).squeeze(-1)  # (B, N)
        aux = torch.sigmoid(self.aux_head(h))                           # (B, N, 2)

        outputs = [health_score, aux]
        if return_cause:
            outputs.append(self.cause_head(h))  # (B, N, 4) raw logits -- caller applies softmax/argmax
        if return_attention:
            outputs.append(last_attention)
        return tuple(outputs)
