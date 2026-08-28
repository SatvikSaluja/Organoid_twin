"""
Powers the interactive control panel's "what if I intervene right now"
slider: clones a live well's simulator state (never mutates the real one),
rolls it forward under a hypothetical intervention, and returns the
predicted health trajectory -- so a user can preview an action before
deciding whether to actually apply it.
"""
from dataclasses import dataclass

import numpy as np
import torch

from backend.biology.organoid_trajectory import DT_HOURS, Intervention, WellSimulator
from backend.config import SENSOR_TYPES, WELL_IDS
from backend.gnn.architecture import DEFAULT_WINDOW, PlateGNN
from backend.gnn.plate_graph import PLATE_ADJACENCY
from backend.sensors.sensor_model import StreamingSensorState, true_signals_from_state

DEFAULT_HORIZON_STEPS = 24  # 12h preview


@dataclass
class WhatIfResult:
    well_id: str
    horizon_hours: list[float]
    health_baseline: list[float]     # no intervention
    health_intervened: list[float]   # with the hypothetical intervention


def run_whatif(
    well_id: str,
    live_sim: WellSimulator,
    live_sensor_state: StreamingSensorState,
    live_history: list[list[float]],
    other_wells_history: dict[str, list[list[float]]],
    model: PlateGNN,
    o2_boost: float,
    refill_glucose: bool,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
) -> WhatIfResult:
    """
    live_history / other_wells_history: each well's last >= DEFAULT_WINDOW
    sensor readings ([ph, do2, glucose_lactate, impedance] per step), so the
    model can be run with full plate graph context even though only one
    well's intervention is being explored. Other wells are held at their
    last known reading for the preview horizon (a reasonable assumption for
    a short lookahead).
    """
    intervention = Intervention(o2_boost=o2_boost, refill_glucose=refill_glucose) if (o2_boost or refill_glucose) else None

    def rollout(intervention_to_apply: Intervention | None) -> list[float]:
        sim = live_sim.clone()
        sensor_state = StreamingSensorState(seed=0)
        sensor_state._filtered_prev = dict(live_sensor_state._filtered_prev) if live_sensor_state._filtered_prev else None
        history = [row[:] for row in live_history]

        healths = []
        for step in range(horizon_steps):
            applied = intervention_to_apply if step == 0 else None
            state = sim.step(applied)
            true_signals = true_signals_from_state(
                glucose=state["glucose"], oxygen=state["oxygen"], lactate=state["lactate"],
                stress_level=state["stress_level"], demand=state["demand"],
            )
            reading = sensor_state.step(true_signals)
            history.append([reading[s] for s in SENSOR_TYPES])
            window = history[-DEFAULT_WINDOW:]

            plate_window = []
            for wid in WELL_IDS:
                if wid == well_id:
                    plate_window.append(window)
                else:
                    hist = other_wells_history.get(wid, window)
                    last = hist[-1]
                    padded = (hist[-DEFAULT_WINDOW:] + [last] * DEFAULT_WINDOW)[-DEFAULT_WINDOW:]
                    plate_window.append(padded)

            x = torch.tensor(np.array(plate_window), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                health_scores, _ = model(x, PLATE_ADJACENCY)
            healths.append(float(health_scores[0, WELL_IDS.index(well_id)]))
        return healths

    baseline = rollout(None)
    intervened = rollout(intervention)

    return WhatIfResult(
        well_id=well_id,
        horizon_hours=[(i + 1) * DT_HOURS for i in range(horizon_steps)],
        health_baseline=baseline,
        health_intervened=intervened,
    )
