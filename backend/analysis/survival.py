"""
Kaplan-Meier survival estimation + log-rank test, applied to "time to
decline onset" rather than literal survival -- standard biostatistics
machinery, implemented from scratch (no extra dependency) since it's just a
step function and one significance test.

A well that never declines within the observed window is right-censored at
the end of that window, not treated as "never happens" -- the whole point
of KM is handling exactly that correctly.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SurvivalCurve:
    times: list[float]      # x-axis, hours
    survival: list[float]   # S(t), starts at 1.0
    n_at_risk_start: int
    n_events: int


def kaplan_meier(event_times_hours: list[float], observed: list[bool]) -> SurvivalCurve:
    """
    event_times_hours[i]: time of decline onset (if observed[i]) or time of
    censoring (last observed, still healthy) otherwise.
    """
    order = np.argsort(event_times_hours)
    times = np.array(event_times_hours)[order]
    obs = np.array(observed)[order]
    n = len(times)

    unique_times = sorted(set(times[obs]))  # only actual events define KM steps
    curve_times = [0.0]
    curve_surv = [1.0]
    survival = 1.0
    n_at_risk = n

    for t in unique_times:
        d = int(np.sum((times == t) & obs))  # events at this exact time
        at_risk_now = int(np.sum(times >= t))
        if at_risk_now > 0:
            survival *= (1.0 - d / at_risk_now)
        curve_times.append(float(t))
        curve_surv.append(survival)

    curve_times.append(float(times.max()) if n else 0.0)
    curve_surv.append(survival)

    return SurvivalCurve(
        times=curve_times, survival=curve_surv,
        n_at_risk_start=n, n_events=int(obs.sum()),
    )


def log_rank_test(
    times_a: list[float], observed_a: list[bool],
    times_b: list[float], observed_b: list[bool],
) -> dict:
    """Two-group log-rank test. Returns the chi-square statistic and p-value
    for the null hypothesis that both groups have the same survival function."""
    times_a, observed_a = np.array(times_a), np.array(observed_a)
    times_b, observed_b = np.array(times_b), np.array(observed_b)

    all_event_times = sorted(set(times_a[observed_a]) | set(times_b[observed_b]))
    o_minus_e = 0.0
    variance = 0.0

    for t in all_event_times:
        n_a = int(np.sum(times_a >= t))
        n_b = int(np.sum(times_b >= t))
        n = n_a + n_b
        if n == 0:
            continue
        d_a = int(np.sum((times_a == t) & observed_a))
        d_b = int(np.sum((times_b == t) & observed_b))
        d = d_a + d_b
        if n <= 1:
            continue

        expected_a = d * n_a / n
        var_t = d * (n_a / n) * (n_b / n) * (n - d) / (n - 1)

        o_minus_e += d_a - expected_a
        variance += var_t

    chi2_stat = (o_minus_e ** 2) / variance if variance > 0 else 0.0
    p_value = float(stats.chi2.sf(chi2_stat, df=1))
    return {"chi2": float(chi2_stat), "p_value": p_value}
