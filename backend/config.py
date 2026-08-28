"""
Central configuration for OrganoidTwin.

Everything that later modules (biology sim, sensors, GNN, WS stream) need to
agree on lives here, so the plate layout / noise profile / paths are defined
in exactly one place.
"""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.environ.get("ORGANOID_DB_URL", f"sqlite:///{DATA_DIR / 'organoid_twin.db'}")

CHECKPOINT_DIR = BASE_DIR / "gnn" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --- Plate geometry ------------------------------------------------------
# A standard multi-well plate. 4x6 = 24 wells is a common organoid culture
# format and keeps the plate graph small enough to eyeball while developing.
PLATE_ROWS = 4
PLATE_COLS = 6
ROW_LABELS = ["A", "B", "C", "D"][:PLATE_ROWS]


def well_ids():
    """All well IDs in plate order, e.g. A1, A2, ..., D6."""
    return [f"{r}{c + 1}" for r in ROW_LABELS for c in range(PLATE_COLS)]


WELL_IDS = well_ids()
NUM_WELLS = len(WELL_IDS)

# --- Sensor types ----------------------------------------------------------
SENSOR_TYPES = ["ph", "do2", "glucose_lactate", "impedance"]

# Stub-stage random-walk parameters (used until sensors/sensor_model.py
# replaces this with biology-grounded readings in a later step). Each sensor
# gets a plausible starting value, a step size, and soft bounds so wells
# stay in a physiologically-shaped range even during plumbing tests.
RANDOM_WALK_PARAMS = {
    "ph":              {"start": 7.4,  "step": 0.01,  "min": 6.5,  "max": 7.8},
    "do2":             {"start": 90.0, "step": 0.6,   "min": 0.0,  "max": 100.0},   # % air saturation
    "glucose_lactate": {"start": 5.0,  "step": 0.05,  "min": 0.0,  "max": 25.0},    # mM glucose proxy
    "impedance":       {"start": 500.0, "step": 3.0,  "min": 0.0,  "max": 2000.0},  # ohms, proxy for density
}

# --- Streaming -------------------------------------------------------------
WS_BROADCAST_INTERVAL_SEC = float(os.environ.get("ORGANOID_WS_INTERVAL", "1.0"))

# --- Sensor noise (placeholder magnitudes; superseded by sensors/noise_profiles.py) ---
SENSOR_NOISE_STD = {
    "ph": 0.02,
    "do2": 1.5,
    "glucose_lactate": 0.15,
    "impedance": 8.0,
}
