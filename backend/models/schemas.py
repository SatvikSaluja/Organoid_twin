"""
Pydantic models shared across the API, the WebSocket stream, and (later) the
GNN / recommendation / explainer layers.

Design note: health is modeled as a continuous `health_score` in [0, 1]
(1 = healthy, 0 = fully declined) rather than a fixed class label. A
continuous score is what the bifurcation detector (section 6) needs to take
a derivative of, and a label is just a threshold on top of it — so we derive
`HealthLabel` from `health_score` instead of predicting it directly. This is
a placeholder derivation until the real GNN model exists.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class HealthLabel(str, Enum):
    healthy = "healthy"
    mild_stress = "mild_stress"
    declining = "declining"


def health_label_from_score(score: float) -> HealthLabel:
    if score >= 0.7:
        return HealthLabel.healthy
    if score >= 0.4:
        return HealthLabel.mild_stress
    return HealthLabel.declining


class SensorType(str, Enum):
    ph = "ph"
    do2 = "do2"
    glucose_lactate = "glucose_lactate"
    impedance = "impedance"


class SensorReading(BaseModel):
    """One multimodal sensor snapshot for a single well at a single time."""
    well_id: str
    timestamp: datetime
    ph: float
    do2: float = Field(..., description="Dissolved oxygen, % air saturation")
    glucose_lactate: float = Field(..., description="Glucose/lactate proxy, mM")
    impedance: float = Field(..., description="Impedance, ohms (proxy for cell density)")


class WellState(BaseModel):
    """Latest known state of a single well: raw reading + inferred health."""
    well_id: str
    timestamp: datetime
    reading: SensorReading
    health_score: float = Field(..., ge=0.0, le=1.0)
    health_label: HealthLabel
    driving_sensor: SensorType | None = Field(
        default=None,
        description="Which sensor most explains the current health score",
    )
    narration: str = Field(default="", description="Plain-language explanation of this well's current state")
    health_std: float = Field(default=0.0, ge=0.0, description="MC-dropout uncertainty on health_score (see gnn/uncertainty.py)")


class PlateStateMessage(BaseModel):
    """Full-plate payload broadcast over the WebSocket on every tick."""
    timestamp: datetime
    plate_rows: int
    plate_cols: int
    wells: list[WellState]


class EventType(str, Enum):
    regime_shift = "regime_shift"
    adverse_event = "adverse_event"
    manual_flag = "manual_flag"


class PlateEvent(BaseModel):
    """A discrete, loggable occurrence (bifurcation firing, injected event, etc.)."""
    well_id: str
    timestamp: datetime
    event_type: EventType
    description: str


class Recommendation(BaseModel):
    """A concrete, interpretable media-adjustment suggestion for one well."""
    well_id: str
    timestamp: datetime
    action: str
    reasoning: str


class HealthResponse(BaseModel):
    status: str = "ok"
    num_wells: int


class CalibrationPoint(BaseModel):
    """One point in the model's continual-adaptation history (EWC fine-tune loop)."""
    timestamp: datetime
    sim_hours: float
    reference_val_loss: float
    finetune_count: int


class LiveFrame(BaseModel):
    """Everything broadcast over the WebSocket on one tick."""
    plate_state: PlateStateMessage
    new_events: list[PlateEvent] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    calibration: CalibrationPoint | None = None
