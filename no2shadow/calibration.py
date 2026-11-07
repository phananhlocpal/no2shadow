"""Calibrate 1-6 h prediction intervals of stations under outage with four residual (label) sources.

Every source feeds the same calibrator: two-sided empirical quantiles of signed residuals inside a calibration cell
(6 h age x issue hour, falling back to 6 h age, then to four age buckets when a cell has fewer than 100 residuals,
7 days or 3 stations). A residual is usable at origin t only if its target is observed and target + 72 h <= t.
  C1_archival    residuals of stations that were offline before the lock time, labels filled in retrospectively; frozen.
  C2_online      residuals of online stations (full context), rolling 30 days.
  C3_shadow      residuals of shadow forecasts (recent history of online stations hidden), rolling 30 days.
                 C3_shadow_d05 / _d10 calibrate at nominal level 0.85 / 0.90.
  C3x_crossmask  diagnostic only: offline-state forecasts of the same station taken from other outage schedules.
  C4_oracle      residuals of the outage targets themselves, 14-day block cross-fit; a reference, not deployable.
Outputs (results/calibration): summary by group, transport (target vs shadow-stream coverage), cell-level shares,
block-bootstrap contrasts and monthly stability, and a markdown report.

Usage:
    python -m no2shadow.calibration
Environment variables (optional): AQ_EVAL_SRC, AQ_EVAL_OUT, AQ_EVAL_PERIODS (default tune,evaluation; use holdout
for the EPA 2025 year), AQ_EVAL_TAG (pool all seeds into one bootstrap group).
"""
import csv, json, math, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / os.environ.get('AQ_EVAL_SRC', 'results/forecasts')
OUT = ROOT / os.environ.get('AQ_EVAL_OUT', 'results/calibration')
DELAY, WINDOW, ALPHA, HMAX, BLOCK = 72, 720, 0.2, 6, 14
EDGES = np.array([24, 48, 72])
AGE_STEP, N_AGE, NCELL = 6, 16, 64
POOLED = (100, 7, 3)
CALS = ('C1_archival', 'C2_online', 'C3_shadow', 'C3_shadow_d05', 'C3_shadow_d10', 'C3x_crossmask', 'C4_oracle')
COMMON = ('C1_archival', 'C2_online', 'C3_shadow', 'C4_oracle')
SPLIT_NAMES = ('fit', 'tune', 'evaluation', 'holdout')
PERIODS = tuple(os.environ.get('AQ_EVAL_PERIODS', 'tune,evaluation').split(','))
MODES = ('age_hour', 'age', 'bucket')   # 6 h age x issue hour; 6 h age; four age buckets
PRIMARY = 'hier'                # finest supported level per source: age_hour -> age -> bucket (same rule for every source)
DEV_SEEDS, CONFIRM_SEEDS = (2025,), (2026, 2027)   # seed used while building the method; seeds used for reporting
BOOT, BOOT_SEED = 2000, 7


def bucket(age):
    return np.searchsorted(EDGES, age, side='left')


def cell_of(age, off_idx, mode):
    """Calibration cell. Age is rounded to the 6h shadow grid; origin hour is the offset index (0-3)."""
    age, off_idx = np.asarray(age), np.asarray(off_idx).astype(int)
    a = np.clip(np.rint(age / AGE_STEP).astype(int), 1, N_AGE) - 1
    if mode == 'age_hour':
        return a + N_AGE * off_idx
    return a if mode == 'age' else bucket(age)


def online_key(cell, mode):
    cell = np.asarray(cell)
    out = (cell // N_AGE) * N_AGE if mode == 'age_hour' else np.zeros_like(cell)
    return int(out) if out.ndim == 0 else out


class Pool:
    """Residual rows grouped by key and sorted by target time (index)."""

    def __init__(self, key, time, day, site, res):
        order = np.lexsort((time, key))
        key, self.time, self.day, self.site, self.res = key[order], time[order], day[order], site[order], res[order]
        self.span = {}
        for k in np.unique(key):
            idx = np.flatnonzero(key == k)
            self.span[int(k)] = (idx[0], idx[-1] + 1)

    def query(self, key, hi, lo=None):
        if key not in self.span:
            return None
        a, b = self.span[key]
        t = self.time[a:b]
        j = a + np.searchsorted(t, hi, side='right')
        i = a if lo is None else a + np.searchsorted(t, lo, side='right')
        return self.day[i:j], self.site[i:j], self.res[i:j]


def interval(rows, rule=POOLED, alpha=ALPHA):
    if rows is None:
        return None
    day, site, res = rows
    n = len(res)
    if n < rule[0] or len(np.unique(day)) < rule[1] or len(np.unique(site)) < rule[2]:
        return None
    r = np.sort(res)
    k_lo, k_hi = math.floor(alpha / 2 * (n + 1)), math.ceil((1 - alpha / 2) * (n + 1))
    if k_lo < 1 or k_hi > n:
        return None
    return float(r[k_lo - 1]), float(r[k_hi - 1])


def load_mask(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def flat_base(M, which, mode):
    """Long residual rows from base forecasts. which: 'offline' (archival), 'online' (feedback state)."""
    P, truth, fb, off, age = M['pred'][..., 1], M['truth'], M['fb'], M['off'], M['age']
    O, C, H = truth.shape
    o, s, h = np.indices((O, C, H))
    res = truth - P
    ok = np.isfinite(res)
    ok &= (off[:, :, None] == 1) if which == 'offline' else ((off[:, :, None] == 0) & (fb == 1))
    pk = _pack(M, o[ok], s[ok], h[ok], res[ok], age[o[ok], s[ok]], mode)
    if which == 'online':
        pk = (online_key(pk[0], mode),) + pk[1:]             # online state has no age dimension
    return pk


def flat_shadow(M, mode):
    o, s, a = M['sh_o'], M['sh_s'].astype(int), M['sh_a']
    H = HMAX
    res = M['truth'][o, s, :] - M['sh_pred']                               # (N,6)
    ok = np.isfinite(res) & (M['fb'][o, s, :] == 1)
    n, h = np.nonzero(ok)
    return _pack(M, o[n], s[n], h, res[n, h], M['sh_age'][n], mode)


def flat_cross(M, others, mode):
    """Cross-schedule rows: offline-state forecasts from other outage schedules of a station that is online here."""
    parts = []
    for X in others:
        P, off = X['pred'][..., 1], X['off']
        res = M['truth'] - P
        ok = np.isfinite(res) & (off[:, :, None] == 1) & (M['off'][:, :, None] == 0) & (M['fb'] == 1)
        o, s, h = np.nonzero(ok)
        parts.append(_pack(M, o, s, h, res[o, s, h], X['age'][o, s], mode))
    return tuple(np.concatenate(z) for z in zip(*parts))


def _pack(M, o, s, h, res, age, mode):
    T = M['T']
    time = T[o] + h                                    # target time index (horizon h is 0-based here)
    origin = T[o] - 1
    cell = cell_of(age, M['offset'][o], mode)
    return cell, time, origin // 24, s, res


def make_pool(rows):
    key, time, day, site, res = rows
    return Pool(key, time, day, site, res)


def month_of(M, idx):
    t0 = np.datetime64(int(M['t0_ns']), 'ns')
    return (t0 + idx.astype('timedelta64[h]')).astype('datetime64[M]').astype(str), \
        ((t0.astype('datetime64[h]').astype(np.int64) + idx) % 24)


def targets_for(M, period, mode):
    code = SPLIT_NAMES.index(period)
    P, truth, off = M['pred'][..., 1], M['truth'], M['off']
    in_period = (M['split'] == code)[:, None, None]
    ok = in_period & (off[:, :, None] == 1) & np.isfinite(truth - P)
    o, s, h = np.nonzero(ok)
    return {'o': o, 's': s, 'h': h, 'res': (truth - P)[o, s, h], 'ab': bucket(M['age'][o, s]),
            'cell': cell_of(M['age'][o, s], M['offset'][o], mode), 'oh': M['offset'][o].astype(int),
            'origin': M['T'][o] - 1, 'time': M['T'][o] + h,
            'start': int(M['T'][M['split'] == code].min()) - 1, 'end': int(M['T'][M['split'] == code].max()) - 1}


def intervals_for(tg, fns):
    """fns: calibrator -> f(origin, ab, block) -> interval|None. Returns lo, hi arrays (nan unsupported)."""
    lo, hi = {}, {}
    keys = tg['origin'] * NCELL + tg['cell']
    uniq, inv = np.unique(keys, return_inverse=True)
    for name, fn in fns.items():
        l = np.full(len(uniq), np.nan); u = np.full(len(uniq), np.nan)
        for j, k in enumerate(uniq):
            iv = fn(int(k) // NCELL, int(k) % NCELL)
            if iv is not None:
                l[j], u[j] = iv
        lo[name], hi[name] = l[inv], u[inv]
    return lo, hi


def oracle_intervals(tg, period_days, mode):
    """Block cross-fit ceiling. In the primary mode a thin cell falls back to age bucket x hour, then age bucket
    (the oracle is not a deployable method; fallback only keeps the ceiling defined)."""
    block = (tg['origin'] // 24 - period_days) // BLOCK
    lo, hi = np.full(len(block), np.nan), np.full(len(block), np.nan)
    day, site = tg['origin'] // 24, tg['s']
    tiers = [tg['cell']] + ([tg['ab'] + 4 * tg['oh'], tg['ab']] if mode == 'age_hour' else [])
    for keyarr in tiers:
        todo = ~np.isfinite(lo)
        for c in np.unique(keyarr[todo]):
            in_c = keyarr == c
            for b in np.unique(block[in_c & todo]):
                iv = interval((day[in_c & (block != b)], site[in_c & (block != b)], tg['res'][in_c & (block != b)]))
                if iv is not None:
                    sel = in_c & (block == b) & todo
                    lo[sel], hi[sel] = iv
    return lo, hi, block


def combine_hier(results):
    """results: {mode: (tg, lo, hi, block, proxy)} for MODES. Finest supported level per target and calibrator."""
    base = results[MODES[0]]
    lo, hi, share = {}, {}, {}
    for c in base[1]:
        l, u, lev = np.full(len(base[0]['res']), np.nan), np.full(len(base[0]['res']), np.nan), np.full(len(base[0]['res']), len(MODES))
        for li, m in enumerate(MODES):
            ml, mu = results[m][1][c], results[m][2][c]
            take = ~np.isfinite(l) & np.isfinite(ml)
            l[take], u[take], lev[take] = ml[take], mu[take], li
        lo[c], hi[c] = l, u
        share[c] = [float((lev == li).mean()) for li in range(len(MODES) + 1)]
    return base[0], lo, hi, base[3], base[4], share


def metrics(res, lo, hi):
    ok = np.isfinite(lo)
    below, above = np.where(ok, np.maximum(0, lo - res), 0), np.where(ok, np.maximum(0, res - hi), 0)
    cov = ok & (res >= lo) & (res <= hi)
    return ok, cov, hi - lo, (hi - lo) + 2 / ALPHA * (below + above), ok & (res < lo), ok & (res > hi)


def run_mask(M, others, period, kind, mode=PRIMARY):
    tg = targets_for(M, period, mode)
    if not len(tg['res']):
        raise ValueError('no targets')
    start = tg['start']
    arch, online, shadow = make_pool(flat_base(M, 'offline', mode)), make_pool(flat_base(M, 'online', mode)), make_pool(flat_shadow(M, mode))
    cross = make_pool(flat_cross(M, others, mode)) if others else None
    win = lambda origin: (origin - DELAY, origin - DELAY - WINDOW)
    memo = {}

    def cached(tag, key, fn):
        k = (tag, key)
        if k not in memo:
            memo[k] = fn()
        return memo[k]
    fns = {
        'C1_archival': lambda o, ab: cached('c1', ab, lambda: interval(arch.query(ab, start - DELAY))),
        'C2_online': lambda o, ab: cached('c2', (o, online_key(ab, mode)), lambda: interval(online.query(online_key(ab, mode), *win(o)))),
        'C3_shadow': lambda o, ab: interval(shadow.query(ab, *win(o))),
        'C3_shadow_d05': lambda o, ab: interval(shadow.query(ab, *win(o)), alpha=ALPHA - .05),
        'C3_shadow_d10': lambda o, ab: interval(shadow.query(ab, *win(o)), alpha=ALPHA - .10),
    }
    if cross is not None:
        fns['C3x_crossmask'] = lambda o, ab: interval(cross.query(ab, *win(o)))
    lo, hi = intervals_for(tg, fns)
    lo['C4_oracle'], hi['C4_oracle'], block = oracle_intervals(tg, start // 24, mode)
    # proxy stream: prequential coverage of the C3 calibrator on the shadow rows themselves
    proxy = proxy_coverage(M, shadow, period, mode)
    return tg, lo, hi, block, proxy


def proxy_coverage(M, shadow, period, mode):
    """Prequential coverage of the C3 calibrator on the shadow rows themselves; returns {age bucket: (n, hits)}."""
    code = SPLIT_NAMES.index(period)
    rows = np.flatnonzero(M['split'][M['sh_o']] == code)
    o, s = M['sh_o'][rows], M['sh_s'][rows].astype(int)
    res = M['truth'][o, s, :] - M['sh_pred'][rows]
    ok = np.isfinite(res) & (M['fb'][o, s, :] == 1)
    cell = cell_of(M['sh_age'][rows], M['offset'][o], mode)
    keys = o.astype(np.int64) * NCELL + cell
    uniq, inv = np.unique(keys, return_inverse=True)
    lo, hi = np.full(len(uniq), np.nan), np.full(len(uniq), np.nan)
    for j, k in enumerate(uniq):
        origin = int(M['T'][int(k) // NCELL]) - 1
        iv = interval(shadow.query(int(k) % NCELL, origin - DELAY, origin - DELAY - WINDOW))
        if iv is not None:
            lo[j], hi[j] = iv
    lo, hi = lo[inv][:, None], hi[inv][:, None]
    use = ok & np.isfinite(lo)
    hit = use & (res >= lo) & (res <= hi)
    bk = bucket(M['sh_age'][rows])
    return {b: (int(use[bk == b].sum()), int(hit[bk == b].sum())) for b in range(4)}


def summarize(base, tg, lo, hi, M):
    rows, mon, hod = [], *month_of(M, tg['time'])
    groups = [('all', 'all', np.ones(len(tg['res']), bool))]
    for h in range(HMAX):
        groups.append(('horizon', str(h + 1), tg['h'] == h))
    for b in range(4):
        groups.append(('age_bucket', str(b), tg['ab'] == b))
    for v in range(24):
        groups.append(('hour_of_day', str(v), hod == v))
    for v in sorted(set(mon)):
        groups.append(('month', str(v), mon == v))
    ok = {c: np.isfinite(lo[c]) for c in lo}
    common = np.logical_and.reduce([ok[c] for c in COMMON])
    for gtype, g, sel in groups:
        for subset in ('own', 'common'):
            for c in lo:
                if subset == 'common' and c not in COMMON:
                    continue
                use = sel & (common if subset == 'common' else ok[c])
                n = int(use.sum())
                row = {**base, 'subset': subset, 'calibrator': c, 'group_type': gtype, 'group': g,
                       'targets': int(sel.sum()), 'n': n}
                if n:
                    _, cov, w, sc, bl, ab_ = metrics(tg['res'][use], lo[c][use], hi[c][use])
                    row.update(coverage=cov.mean(), width=w.mean(), interval_score=sc.mean(),
                               miss_below=bl.mean(), miss_above=ab_.mean())
                rows.append(row)
    return rows


def month_cells(M, idx, first):
    """Month index (from `first`, datetime64[M]) and 14-day block within month (0-2) for hour indices."""
    t0 = np.datetime64(int(M['t0_ns']), 'ns')
    d = (t0 + np.asarray(idx).astype('timedelta64[h]')).astype('datetime64[D]')
    mo = d.astype('datetime64[M]')
    dom = (d - mo.astype('datetime64[D]')).astype(int)
    return (mo - first).astype(int), np.minimum(dom // 14, 2)


def unit_blocks(tg, lo, hi, block, M=None):
    nb = (tg['end'] // 24 - tg['start'] // 24) // BLOCK + 1
    """Per-block sums for the bootstrap, common-supported targets, MCAR evaluation."""
    common = np.logical_and.reduce([np.isfinite(lo[c]) for c in COMMON])
    out = {'n': np.bincount(block[common], minlength=nb).astype(float)}
    for c in COMMON:
        _, cov, _, sc, _, _ = metrics(tg['res'], lo[c], hi[c])
        out['cov_' + c] = np.bincount(block[common], weights=cov[common], minlength=nb)
        out['score_' + c] = np.bincount(block[common], weights=sc[common], minlength=nb)
    if M is not None:                                   # per (month, 14-day block within month) sums for stability endpoints
        t0 = np.datetime64(int(M['t0_ns']), 'ns')
        first = (t0 + np.timedelta64(int(tg['start']) + 1, 'h')).astype('datetime64[M]')
        last = (t0 + np.timedelta64(int(tg['end']) + 1 + HMAX, 'h')).astype('datetime64[M]')
        nm = int((last - first).astype(int)) + 1
        mi, bi = month_cells(M, tg['time'], first)
        flat = mi * 3 + bi
        out['st_n'] = np.bincount(flat[common], minlength=nm * 3).reshape(nm, 3).astype(float)
        for c in COMMON:
            cov = metrics(tg['res'], lo[c], hi[c])[1]
            out['st_cov_' + c] = np.bincount(flat[common], weights=cov[common], minlength=nm * 3).reshape(nm, 3)
    return out


MIN_MONTH_N = 1000          # months with fewer mean targets per unit are excluded from stability endpoints


def stability_bootstrap(units):
    """E1 = mean |monthly coverage - 0.8|, E2 = worst monthly coverage; bootstrap resamples 14-day blocks within
    months (same draws for every unit and calibrator). Returns estimates and CIs including C1-C3 differences."""
    n = np.stack([u['st_n'] for u in units])                        # (U,M,3)
    cov = {c: np.stack([u['st_cov_' + c] for u in units]) for c in COMMON}
    months = [m for m in range(n.shape[1]) if n[:, m].sum() / len(units) >= MIN_MONTH_N]
    if not months:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    avail = {m: np.flatnonzero(n[:, m].sum(axis=0) > 0) for m in months}

    def stat(W):
        out = {}
        for c in COMMON:
            monthly = []
            for m in months:
                w = W[m]
                monthly.append(np.mean((cov[c][:, m] @ w) / np.maximum(n[:, m] @ w, 1)))
            monthly = np.array(monthly)
            out[c] = (float(np.mean(np.abs(monthly - .8))), float(monthly.min()))
        return out
    full = {m: np.ones(3) * (n[:, m].sum(axis=0) > 0) for m in months}
    est = stat(full)
    draws = []
    for _ in range(BOOT):
        W = {}
        for m in months:
            a = avail[m]
            w = np.zeros(3)
            np.add.at(w, rng.choice(a, size=len(a)), 1)
            W[m] = w
        draws.append(stat(W))
    ci = lambda f: [float(np.percentile([f(d) for d in draws], q)) for q in (2.5, 97.5)]
    out = {'months': len(months), 'E1': {c: est[c][0] for c in COMMON}, 'E2_worst_month': {c: est[c][1] for c in COMMON}}
    out['E1_C1_minus_C3'] = {'estimate': est['C1_archival'][0] - est['C3_shadow'][0], 'ci95': ci(lambda d: d['C1_archival'][0] - d['C3_shadow'][0])}
    out['E2_C3_minus_C1'] = {'estimate': est['C3_shadow'][1] - est['C1_archival'][1], 'ci95': ci(lambda d: d['C3_shadow'][1] - d['C1_archival'][1])}
    return out


def bootstrap(units, nb):
    """units: list of per-block dicts (same block structure). Returns percentile CIs of pooled statistics."""
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(nb, np.ones(nb) / nb, size=BOOT).astype(float)

    def stat(w):
        cov = {c: np.mean([(w @ u['cov_' + c]) / max(w @ u['n'], 1) for u in units]) for c in COMMON}
        sc = {c: np.mean([(w @ u['score_' + c]) / max(w @ u['n'], 1) for u in units]) for c in COMMON}
        return cov, sc
    est_cov, est_sc = stat(np.ones(nb))
    draws = [stat(w) for w in W]
    def ci(f):
        v = np.array([f(*d) for d in draws])
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    pairs = {
        'cov_C3_minus_C1': lambda c, s: c['C3_shadow'] - c['C1_archival'],
        'cov_C3_minus_C2': lambda c, s: c['C3_shadow'] - c['C2_online'],
        'cov_C4_minus_C3': lambda c, s: c['C4_oracle'] - c['C3_shadow'],
        'score_C3_minus_C1': lambda c, s: s['C3_shadow'] - s['C1_archival'],
        'score_C3_minus_C2': lambda c, s: s['C3_shadow'] - s['C2_online'],
    }
    out = {k: {'estimate': f(est_cov, est_sc), 'ci95': ci(f)} for k, f in pairs.items()}
    out['coverage'] = est_cov
    gap = est_cov['C4_oracle'] - est_cov['C1_archival']
    out['closure_C3'] = (est_cov['C3_shadow'] - est_cov['C1_archival']) / gap if gap >= .02 else None
    return out


def write_csv(path, rows):
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, transport, levels, units = [], [], [], {}
    for ds_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        ds = ds_dir.name
        for seed_dir in sorted(ds_dir.glob('seed_*')):
            seed = int(seed_dir.name.split('_')[1])
            masks = {p.stem: load_mask(p) for p in sorted(seed_dir.glob('*.npz'))}
            mcar = [k for k in masks if k.startswith('mcar')]
            for name, M in masks.items():
                kind = name.split('_')[0]
                others = [masks[k] for k in mcar if k != name] if kind == 'mcar' else []
                for X in others:
                    if not (np.array_equal(X['T'], M['T']) and np.allclose(X['truth'], M['truth'], equal_nan=True)):
                        raise ValueError('masks are not aligned')
                for period in PERIODS:
                    results = {m: run_mask(M, others, period, kind, m) for m in MODES}
                    for m in MODES[1:]:
                        if not np.array_equal(results[m][0]['res'], results[MODES[0]][0]['res']):
                            raise ValueError('target sets differ across modes')
                    tg, lo, hi, block, proxy, share = combine_hier(results)
                    for c, sh in share.items():
                        levels.append({'dataset': ds, 'seed': seed, 'mask': name, 'kind': kind, 'period': period, 'calibrator': c,
                                       **{f'share_{m}': sh[i] for i, m in enumerate(MODES)}, 'share_unsupported': sh[-1]})
                    for mode, (tg_m, lo_m, hi_m, block_m, proxy_m) in [*results.items(), ('hier', (tg, lo, hi, block, proxy))]:
                        base = {'dataset': ds, 'seed': seed, 'mask': name, 'kind': kind, 'period': period, 'cells': mode}
                        summary += summarize(base, tg_m, lo_m, hi_m, M)
                        if mode == 'hier':
                            if kind == 'mcar' and period == PERIODS[-1]:
                                units.setdefault((ds, seed), []).append(unit_blocks(tg_m, lo_m, hi_m, block_m, M))
                            continue
                        ok3 = np.isfinite(lo_m['C3_shadow'])
                        for b in range(4):
                            sel = ok3 & (tg_m['ab'] == b)
                            n_p, hit_p = proxy_m.get(b, (0, 0))
                            if sel.any() or n_p:
                                cov_t = float(metrics(tg_m['res'][sel], lo_m['C3_shadow'][sel], hi_m['C3_shadow'][sel])[1].mean()) if sel.any() else math.nan
                                cov_p = hit_p / n_p if n_p else math.nan
                                transport.append({**base, 'age_bucket': b, 'target_n': int(sel.sum()), 'target_coverage_c3': cov_t,
                                                  'proxy_n': n_p, 'proxy_coverage': cov_p, 'delta_target_minus_proxy': cov_t - cov_p})
                    print('completed', ds, seed, name, period, len(tg['res']), flush=True)
            write_csv(OUT / 'calibration_summary.csv', summary)
            write_csv(OUT / 'calibration_transport.csv', transport)
            write_csv(OUT / 'calibration_levels.csv', levels)
    report = {'per_dataset': {}, 'stability': {}}
    tag = os.environ.get('AQ_EVAL_TAG')
    groups = [(tag, None)] if tag else [('development_seed', DEV_SEEDS), ('reporting_seeds', CONFIRM_SEEDS)]
    for ds in sorted({k[0] for k in units}):
        entry, stab = {}, {}
        for label, seeds in groups:
            us = [u for (d, sd), v in units.items() if d == ds and (seeds is None or sd in seeds) for u in v]
            nb = max((len(u['n']) for u in us), default=0)
            if us and len({len(u['n']) for u in us}) == 1:
                entry[label] = bootstrap(us, nb)
                if len({u['st_n'].shape for u in us}) == 1:
                    stab[label] = stability_bootstrap(us)
        report['per_dataset'][ds] = entry
        report['stability'][ds] = stab
    (OUT / 'calibration_contrasts.json').write_text(json.dumps(report, indent=2))
    lines = ['# Calibration by label source', '',
             '| dataset | seeds | ' + ' | '.join(COMMON) + ' | closure C3 | C3-C1 [95% CI] |', '|---|---|' + '---:|' * (len(COMMON) + 2)]
    for ds, e in report['per_dataset'].items():
        for label, b in e.items():
            d = b['cov_C3_minus_C1']
            cl = b['closure_C3']
            lines.append(f'| {ds} | {label} | ' + ' | '.join(f'{b["coverage"][c]:.3f}' for c in COMMON) +
                         f' | {"n/a" if cl is None else format(cl, ".2f")} | {d["estimate"]:+.3f} [{d["ci95"][0]:+.3f}, {d["ci95"][1]:+.3f}] |')
    lines += ['', '## Monthly stability (E1 = mean |monthly coverage - 0.8|, E2 = worst month)', '',
              '| dataset | seeds | months | E1 C1 | E1 C3 | E1 C1-C3 [CI95] | E2 C1 | E2 C3 | E2 C3-C1 [CI95] |', '|---|---|---:|---:|---:|---|---:|---:|---|']
    for ds, e in report['stability'].items():
        for label, b in e.items():
            if not b:
                continue
            d1, d2 = b['E1_C1_minus_C3'], b['E2_C3_minus_C1']
            lines.append(f'| {ds} | {label} | {b["months"]} | {b["E1"]["C1_archival"]:.3f} | {b["E1"]["C3_shadow"]:.3f} | {d1["estimate"]:+.3f} [{d1["ci95"][0]:+.3f}, {d1["ci95"][1]:+.3f}] | '
                         f'{b["E2_worst_month"]["C1_archival"]:.3f} | {b["E2_worst_month"]["C3_shadow"]:.3f} | {d2["estimate"]:+.3f} [{d2["ci95"][0]:+.3f}, {d2["ci95"][1]:+.3f}] |')
    (OUT / 'CALIBRATION_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
