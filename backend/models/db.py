"""
SQLAlchemy persistence layer: plates, wells, readings, events.

Kept intentionally simple (SQLite by default, see config.DATABASE_URL) since
this is a one-time local build, not a service that needs a production DB.
Readings/events are persisted so eval/run_benchmark.py can later replay a
full culture period and score detection lead-time against ground truth.
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from backend.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Plate(Base):
    __tablename__ = "plates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, default="default-plate")
    rows: Mapped[int] = mapped_column()
    cols: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    wells: Mapped[list["Well"]] = relationship(back_populates="plate", cascade="all, delete-orphan")


class Well(Base):
    __tablename__ = "wells"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate_id: Mapped[int] = mapped_column(ForeignKey("plates.id"))
    well_id: Mapped[str] = mapped_column(String, index=True)  # e.g. "B4"
    row: Mapped[int] = mapped_column()
    col: Mapped[int] = mapped_column()

    plate: Mapped["Plate"] = relationship(back_populates="wells")
    readings: Mapped[list["Reading"]] = relationship(back_populates="well", cascade="all, delete-orphan")
    events: Mapped[list["Event"]] = relationship(back_populates="well", cascade="all, delete-orphan")


class Reading(Base):
    __tablename__ = "readings"

    id: Mapped[int] = mapped_column(primary_key=True)
    well_id: Mapped[int] = mapped_column(ForeignKey("wells.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    ph: Mapped[float] = mapped_column(Float)
    do2: Mapped[float] = mapped_column(Float)
    glucose_lactate: Mapped[float] = mapped_column(Float)
    impedance: Mapped[float] = mapped_column(Float)

    health_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    well: Mapped["Well"] = relationship(back_populates="readings")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    well_id: Mapped[int] = mapped_column(ForeignKey("wells.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    event_type: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)

    well: Mapped["Well"] = relationship(back_populates="events")


class Experiment(Base):
    """One closed-loop control-arena run: N plates x 3 arms, persisted so the
    Cohort Analytics dashboard has real history to browse and compare
    instead of only ever showing the most recent run."""
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    n_plates: Mapped[int] = mapped_column(Integer)
    n_steps: Mapped[int] = mapped_column(Integer)
    base_seed: Mapped[int] = mapped_column(Integer)

    outcomes: Mapped[list["ArmOutcome"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class ArmOutcome(Base):
    """One well's outcome under one arm of one experiment -- the unit
    survival.py's Kaplan-Meier/log-rank functions and the cohort-analytics
    endpoint aggregate over."""
    __tablename__ = "arm_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    arm: Mapped[str] = mapped_column(String, index=True)  # "no_control" | "model_driven" | "oracle"
    well_id: Mapped[str] = mapped_column(String)
    plate_seed: Mapped[int] = mapped_column(Integer)

    limiting_factor: Mapped[str | None] = mapped_column(String, nullable=True)
    decline_onset_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    declined: Mapped[bool] = mapped_column(Boolean)  # False -> right-censored at n_steps for survival analysis
    mean_health: Mapped[float] = mapped_column(Float)
    healthy_hours: Mapped[float] = mapped_column(Float)
    n_interventions: Mapped[int] = mapped_column(Integer)

    experiment: Mapped["Experiment"] = relationship(back_populates="outcomes")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)


def get_or_create_plate(session: Session, well_ids: list[str], rows: int, cols: int) -> Plate:
    """Ensure a plate + its wells exist, returning the plate row-mapped for use."""
    plate = session.query(Plate).filter_by(name="default-plate").one_or_none()
    if plate is not None:
        return plate

    plate = Plate(name="default-plate", rows=rows, cols=cols)
    session.add(plate)
    session.flush()  # assign plate.id

    for idx, wid in enumerate(well_ids):
        row, col = divmod(idx, cols)
        session.add(Well(plate_id=plate.id, well_id=wid, row=row, col=col))

    session.commit()
    session.refresh(plate)
    return plate
