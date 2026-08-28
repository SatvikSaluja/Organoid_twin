"""
Builds the well-to-well coupling graph: nodes are wells, edges connect wells
that share microenvironment (4-connected grid neighbors on the physical
plate -- adjacent wells share incubator airflow/thermal gradients). This is
what lets the GNN borrow signal across wells rather than acting as N
independent per-well classifiers.

Implemented as a small dense adjacency matrix rather than a sparse-graph
library (PyG/DGL): a 24-node plate is tiny, dense attention is simpler to
reason about and has no extra native-wheel dependencies to install for a
one-time build.
"""
import torch

from backend.config import NUM_WELLS, PLATE_COLS, PLATE_ROWS, WELL_IDS


def well_index_map() -> dict[str, int]:
    return {wid: i for i, wid in enumerate(WELL_IDS)}


def build_adjacency(rows: int = PLATE_ROWS, cols: int = PLATE_COLS, self_loops: bool = True) -> torch.Tensor:
    """
    Dense (N, N) adjacency matrix, N = rows*cols, in the same row-major well
    order as backend.config.WELL_IDS. Edge if wells are 4-connected neighbors
    on the plate grid.
    """
    n = rows * cols
    adj = torch.zeros((n, n), dtype=torch.float32)

    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[i, idx(nr, nc)] = 1.0

    if self_loops:
        adj += torch.eye(n)

    return adj


PLATE_ADJACENCY = build_adjacency()
WELL_INDEX = well_index_map()

assert PLATE_ADJACENCY.shape == (NUM_WELLS, NUM_WELLS)
