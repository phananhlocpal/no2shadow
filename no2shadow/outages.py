"""Simulated sensor outages overlaid on the measured series.

Every generator returns a boolean matrix (hours x stations), True while a station is offline. Outages start at
midnight and last `duration` hours; the expected daily start probability per station is about 0.10.
  make_events       : independent of the data (missing completely at random).
  make_events_mnar  : probability rises with the station's mean level inside the outage window (stress test).
  make_events_pre   : probability rises with the station's mean level in the 24 h before the start.
  make_events_corr  : shared-cause events that take down several stations on the same day.
"""
import warnings

import numpy as np


def make_events(n, stations, begin, duration, seed=31751, daily_prob=.10):
    if n < 0 or not 0 <= begin <= n or duration not in (0, 24, 72) or not 0 <= daily_prob <= 1:
        raise ValueError('invalid schedule configuration')
    offline = np.zeros((n, len(stations)), dtype=bool)
    if duration == 0:
        return offline
    rng = np.random.default_rng(seed)
    for t in range(begin, n, 24):
        chosen = np.flatnonzero(rng.random(len(stations)) < daily_prob)
        offline[t:min(n, t + duration), chosen] = True
    return offline


def make_events_mnar(raw, begin, duration, seed, beta, daily_prob=.10):
    """Like make_events, but the chance of a window rises with its (standardized) mean level.
    beta=0 is MCAR. Expected outage rate is kept near daily_prob. Simulator only: it reads the truth by design."""
    n, c = raw.shape
    starts = list(range(begin, n, 24))
    mu, sd = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = np.zeros((len(starts), c))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k, t in enumerate(starts):
            m = np.nanmean(raw[t:min(n, t + duration)], axis=0)
            z[k] = np.where(np.isfinite(m), (m - mu) / sd, 0.0)
    w = np.exp(beta * np.clip(z, -3, 3))
    p = np.clip(daily_prob * w / w.mean(), 0, 1)
    rng = np.random.default_rng(seed)
    offline = np.zeros((n, c), dtype=bool)
    for k, t in enumerate(starts):
        chosen = np.flatnonzero(rng.random(c) < p[k])
        offline[t:min(n, t + duration), chosen] = True
    return offline


def past_level(raw, starts, lookback, mu, sd):
    """Standardised mean level of each station in the `lookback` hours before each start (uses nothing at or after the start)."""
    z = np.zeros((len(starts), raw.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k, t in enumerate(starts):
            m = np.nanmean(raw[max(0, t - lookback):t], axis=0)
            z[k] = np.where(np.isfinite(m), (m - mu) / sd, 0.0)
    return z


def make_events_pre(raw, begin, duration, seed, beta, lookback=24, daily_prob=.10):
    """Outage chance rises with the station's own mean level in the `lookback` hours BEFORE the outage starts
    (observable at failure time). Expected rate stays near daily_prob. beta=0 is MCAR."""
    n, c = raw.shape
    starts = list(range(begin, n, 24))
    mu, sd = np.nanmean(raw, axis=0), np.nanstd(raw, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = past_level(raw, starts, lookback, mu, sd)
    w = np.exp(beta * np.clip(z, -3, 3))
    p = np.clip(daily_prob * w / w.mean(), 0, 1)
    rng = np.random.default_rng(seed)
    offline = np.zeros((n, c), dtype=bool)
    for k, t in enumerate(starts):
        chosen = np.flatnonzero(rng.random(c) < p[k])
        offline[t:min(n, t + duration), chosen] = True
    return offline


def make_events_corr(n, c, begin, duration, seed, reach, event_prob=.05, daily_prob=.10):
    """Shared-cause outages: with probability `event_prob` per day an event hits each station independently with
    probability `reach`; independent background outages top the per-station rate up to ~daily_prob."""
    rng = np.random.default_rng(seed)
    background = max(daily_prob - event_prob * reach, 0.0)
    offline = np.zeros((n, c), dtype=bool)
    for t in range(begin, n, 24):
        hit = rng.random(c) < background
        if rng.random() < event_prob:
            hit |= rng.random(c) < reach
        offline[t:min(n, t + duration), np.flatnonzero(hit)] = True
    return offline
