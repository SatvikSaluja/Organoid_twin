"""
Per-sensor-type calibrated noise (std dev) and response lag (as an
exponential-smoothing time constant, in simulation steps). This heterogeneity
-- pH read almost instantly, a dissolved-O2 probe visibly lagging, impedance
moving slower still -- is what makes fusing the four channels a real
problem rather than four independent thresholds.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseProfile:
    noise_std: float
    lag_tau_steps: float  # 1.0 = no lag (tracks instantly); higher = slower to respond


NOISE_PROFILES: dict[str, NoiseProfile] = {
    "ph": NoiseProfile(noise_std=0.015, lag_tau_steps=1.0),
    "do2": NoiseProfile(noise_std=1.2, lag_tau_steps=3.0),
    "glucose_lactate": NoiseProfile(noise_std=0.2, lag_tau_steps=1.0),
    "impedance": NoiseProfile(noise_std=6.0, lag_tau_steps=8.0),
}


def lag_filter_step(prev_filtered: float, raw: float, tau_steps: float) -> float:
    """One step of an exponential moving average with time-constant tau_steps."""
    alpha = 1.0 / max(1.0, tau_steps)
    return prev_filtered + alpha * (raw - prev_filtered)
