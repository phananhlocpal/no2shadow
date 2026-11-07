"""Additional analyses of the shadow label source (CPU only, no training).

R1  Coverage conditional on NO2 level (truth quantile and point-forecast quantile, thresholds from fit+tune periods),
    upper-bound recall / alert rate for exceedances of data-driven thresholds (P90, P95; no regulatory thresholds).
R2  Fairness of C2 vs C3: C3 collapsed to the same cell key as C2 (hour only); coverage by age, hour of day, horizon.
R3  Stronger mask-aware controls: recency-weighted C3 over a 90-day window; C1 updated with retrospective labels
    (30-day window and all matured history).
R4  Total-variation sensitivity (bin count, bin rule, block bootstrap) and direct quantile shifts.
R5  Per-station coverage with station level and missing rate.
R6  Cost / robustness: 1, 4, 8, 16 hide lengths; fraction of stations that provide shadow labels; forward-pass counts.
R7  Case-study series (one station, 14 days, deterministic selection) for a figure.

Reads forecasts written by forecast.py. Post-hoc analyses on the evaluation period.

Usage:
    python -m no2shadow.review   (AQ_REVIEW_SRC, AQ_REVIEW_OUT, AQ_REVIEW_SEEDS, AQ_REVIEW_PERIOD)
"""
import csv, json, math, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from . import calibration as ev
from . import audit as au

SRC = ROOT / os.environ.get('AQ_REVIEW_SRC', 'results/forecasts')
OUT = ROOT / os.environ.get('AQ_REVIEW_OUT', 'results/review')
SEEDS = [int(x) for x in os.environ.get('AQ_REVIEW_SEEDS', '2026,2027').split(',')]
PERIOD = os.environ.get('AQ_REVIEW_PERIOD', 'evaluation')
B14 = 14
WIN90 = 24 * 90
HALF_LIVES = {'C3_90d_hl14': 14.0, 'C3_90d_hl7': 7.0}
AGE_SETS = {'C3_ages8': tuple(range(12, 97, 12)), 'C3_ages4': (12, 36, 60, 84), 'C3_ages1': (48,)}
PROVIDERS = {'C3_prov50': 0.5, 'C3_prov25': 0.25}
CORE = ('C1_archival', 'C2_online', 'C3_shadow', 'C4_oracle')
VARIANTS = ('C3_hourcell', 'C1_upd30', 'C1_upd_all', 'C3_90d', *HALF_LIVES, *AGE_SETS, *PROVIDERS)
ALL = CORE + VARIANTS
CONTRASTS = {
    'C3-C2': ('C3_shadow', 'C2_online'), 'C3-C1': ('C3_shadow', 'C1_archival'), 'C4-C3': ('C4_oracle', 'C3_shadow'),
    'C3hour-C2': ('C3_hourcell', 'C2_online'), 'C3hour-C3': ('C3_hourcell', 'C3_shadow'),
    'C3_90d-C3': ('C3_90d', 'C3_shadow'), 'C3_90d_hl14-C3': ('C3_90d_hl14', 'C3_shadow'), 'C3_90d_hl7-C3': ('C3_90d_hl7', 'C3_shadow'),
    'C1upd30-C1': ('C1_upd30', 'C1_archival'), 'C1upd30-C3': ('C1_upd30', 'C3_shadow'), 'C1updall-C3': ('C1_upd_all', 'C3_shadow'),
    'C3ages8-C3': ('C3_ages8', 'C3_shadow'), 'C3ages4-C3': ('C3_ages4', 'C3_shadow'), 'C3ages1-C3': ('C3_ages1', 'C3_shadow'),
    'C3prov50-C3': ('C3_prov50', 'C3_shadow'), 'C3prov25-C3': ('C3_prov25', 'C3_shadow'),
}
LEVEL_LABELS = ('le_p50', 'p50_p90', 'p90_p95', 'gt_p95', 'gt_p90')
LEVEL_CONTRASTS = {'C3-C2': ('C3_shadow', 'C2_online'), 'C3-C1': ('C3_shadow', 'C1_archival')}
TV_BINS = (10, 20, 40, 80, 160)
TV_BOOT = 300


# ------------------------------------------------------------------ small helpers (unit tested)
def weighted_interval(rows, origin, half_life, rule=ev.POOLED, alpha=ev.ALPHA):
    """Recency-weighted two-sided empirical quantile interval of signed residuals (heuristic, no finite-sample correction).
    rows = (origin day, site, residual); weights halve every `half_life` days of origin-day lag."""
    if rows is None:
        return None
    day, site, res = rows
    n = len(res)
    if n < rule[0] or len(np.unique(day)) < rule[1] or len(np.unique(site)) < rule[2]:
        return None
    w = 0.5 ** (np.maximum(origin // 24 - day, 0) / half_life)
    idx = np.argsort(res, kind='stable')
    cw = np.cumsum(w[idx])
    cw = cw / cw[-1]
    r = res[idx]
    lo = r[min(int(np.searchsorted(cw, alpha / 2)), n - 1)]
    hi = r[min(int(np.searchsorted(cw, 1 - alpha / 2)), n - 1)]
    return float(lo), float(hi)


def level_groups(x, thr):
    """0: <=p50, 1: (p50,p90], 2: (p90,p95], 3: >p95 for thr=[p50,p90,p95]."""
    return np.searchsorted(np.asarray(thr), x, side='left')


def level_masks(g):
    return {'le_p50': g == 0, 'p50_p90': g == 1, 'p90_p95': g == 2, 'gt_p95': g == 3, 'gt_p90': g >= 2}


def tv_hist(a, b, edges):
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a)
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def unit_sums_sel(tg, lo, hi, names, sel):
    common = np.logical_and.reduce([np.isfinite(lo[c]) for c in names]) & sel
    blk = (tg['origin'] // 24 - tg['start'] // 24) // B14
    nb = (tg['end'] // 24 - tg['start'] // 24) // B14 + 1
    out = {'nb': nb, 'n': np.bincount(blk[common], minlength=nb).astype(float)}
    for c in names:
        out['cov_' + c] = np.bincount(blk[common], weights=au.covered(tg['res'], lo[c], hi[c])[common], minlength=nb)
    return out


# ------------------------------------------------------------------ forecasts -> intervals
def shadow_pack(M, mode, ages=None, sites=None):
    o, s, a = M['sh_o'], M['sh_s'].astype(int), M['sh_a']
    res = M['truth'][o, s, :] - M['sh_pred']
    ok = np.isfinite(res) & (M['fb'][o, s, :] == 1)
    if ages is not None:
        ok &= np.isin(M['ages'][a], ages)[:, None]
    if sites is not None:
        ok &= np.isin(s, sites)[:, None]
    n, h = np.nonzero(ok)
    return ev._pack(M, o[n], s[n], h, res[n, h], M['sh_age'][n], mode)


def provider_sites(M, frac, salt):
    C = M['truth'].shape[1]
    k = max(1, int(math.ceil(frac * C)))
    rng = np.random.default_rng(1000003 * salt + int(round(frac * 100)))
    return np.sort(rng.choice(C, k, replace=False))


def run_mode(M, period, mode, provs):
    tg = ev.targets_for(M, period, mode)
    start = tg['start']
    arch = ev.make_pool(ev.flat_base(M, 'offline', mode))
    online = ev.make_pool(ev.flat_base(M, 'online', mode))
    full = ev.flat_shadow(M, mode)
    pools = {'full': ev.make_pool(full), 'hour': ev.make_pool((ev.online_key(full[0], mode),) + full[1:])}
    for name, ages in AGE_SETS.items():
        pools[name] = ev.make_pool(shadow_pack(M, mode, ages=ages))
    for name, sites in provs.items():
        pools[name] = ev.make_pool(shadow_pack(M, mode, sites=sites))
    win = lambda o: (o - ev.DELAY, o - ev.DELAY - ev.WINDOW)
    win90 = lambda o: (o - ev.DELAY, o - ev.DELAY - WIN90)
    fns = {
        'C1_archival': lambda o, c: ev.interval(arch.query(c, start - ev.DELAY)),
        'C2_online': lambda o, c: ev.interval(online.query(ev.online_key(c, mode), *win(o))),
        'C3_shadow': lambda o, c: ev.interval(pools['full'].query(c, *win(o))),
        'C3_hourcell': lambda o, c: ev.interval(pools['hour'].query(ev.online_key(c, mode), *win(o))),
        'C1_upd30': lambda o, c: ev.interval(arch.query(c, *win(o))),
        'C1_upd_all': lambda o, c: ev.interval(arch.query(c, o - ev.DELAY)),
        'C3_90d': lambda o, c: ev.interval(pools['full'].query(c, *win90(o))),
    }
    for name, hl in HALF_LIVES.items():
        fns[name] = (lambda hl_: lambda o, c: weighted_interval(pools['full'].query(c, *win90(o)), o, hl_))(hl)
    for name in list(AGE_SETS) + list(provs):
        fns[name] = (lambda nm: lambda o, c: ev.interval(pools[nm].query(c, *win(o))))(name)
    lo, hi = ev.intervals_for(tg, fns)
    lo['C4_oracle'], hi['C4_oracle'], _ = ev.oracle_intervals(tg, start // 24, mode)
    return tg, lo, hi


def run_mask(M, period, salt):
    provs = {n: provider_sites(M, f, salt) for n, f in PROVIDERS.items()}
    res = {m: run_mode(M, period, m, provs) for m in ev.MODES}
    tg, lo, hi, _, _, share = ev.combine_hier({m: (r[0], r[1], r[2], None, {}) for m, r in res.items()})
    return tg, lo, hi


# ------------------------------------------------------------------ accumulation
def thresholds(M):
    sp = M['split'] <= 1
    t = M['truth'][sp]
    t = t[np.isfinite(t)]
    P = M['pred'][..., 1]
    sel = sp[:, None, None] & (M['off'][:, :, None] == 1) & np.isfinite(P)
    return np.quantile(t, [.5, .9, .95]), np.quantile(P[sel], [.5, .9, .95])


class Acc:
    def __init__(self):
        self.var = {c: np.zeros(5) for c in ALL}                       # total, supported, covered, width, score
        self.units = {k: [] for k in CONTRASTS}
        self.lvl_units = {(src, lab, k): [] for src in ('truth', 'point') for lab in ('gt_p90',) for k in LEVEL_CONTRASTS}
        self.lvl = {}                                                   # (src,label,variant) -> n, cov, miss_above, width
        self.exc = {}                                                   # (src,thr,variant) -> positives, hits, alerts, total
        self.grp = {}                                                   # (kind,key,variant) -> n, cov
        self.stn = {}
        self.fwd = []
        self.cols = None


def accumulate(acc, M, tg, lo, hi):
    res = tg['res']
    truth = M['truth'][tg['o'], tg['s'], tg['h']]
    point = truth - res
    for c in ALL:
        ok, cov, width, score, _, _ = ev.metrics(res, lo[c], hi[c])
        a = acc.var[c]
        a += (len(res), ok.sum(), cov[ok].sum(), width[ok].sum(), score[ok].sum())
    for k, (a, b) in CONTRASTS.items():
        acc.units[k].append(au.unit_sums(tg, lo, hi, [a, b], B14))
    common = np.logical_and.reduce([np.isfinite(lo[c]) for c in CORE])
    cov = {c: au.covered(res, lo[c], hi[c]) for c in CORE}
    above = {c: res > hi[c] for c in CORE}
    thr_t, thr_p = thresholds(M)
    for src, x, thr in (('truth', truth, thr_t), ('point', point, thr_p)):
        masks = level_masks(level_groups(x, thr))
        for lab, sel in masks.items():
            s_ = common & sel
            for c in CORE:
                v = acc.lvl.setdefault((src, lab, c), np.zeros(4))
                v += (s_.sum(), cov[c][s_].sum(), above[c][s_].sum(), (hi[c] - lo[c])[s_].sum())
        for (src2, lab, k), lst in acc.lvl_units.items():
            if src2 == src:
                a, b = LEVEL_CONTRASTS[k]
                lst.append(unit_sums_sel(tg, lo, hi, [a, b], masks[lab]))
    # exceedance of data-driven thresholds (truth above threshold; alert = upper bound reaches threshold)
    for j, tname in ((1, 'p90'), (2, 'p95')):
        thr = thr_t[j]
        pos = truth > thr
        for c in CORE:
            ub = point + hi[c]
            v = acc.exc.setdefault(('truth', tname, c), np.zeros(4))
            v += ((common & pos).sum(), (common & pos & (ub >= thr)).sum(), (common & (ub >= thr)).sum(), common.sum())
    # coverage by age bucket, hour of day, horizon
    for kind, key, n in (('age', tg['ab'], 4), ('hour', tg['oh'], 4), ('horizon', tg['h'], ev.HMAX)):
        for g in range(n):
            s_ = common & (key == g)
            for c in CORE:
                v = acc.grp.setdefault((kind, g, c), np.zeros(2))
                v += (s_.sum(), cov[c][s_].sum())
    for s in np.unique(tg['s'][common]):
        s_ = common & (tg['s'] == s)
        a = acc.stn.setdefault(int(s), {'n': 0, 'truth_sum': 0.0, **{c: 0.0 for c in CORE}})
        a['n'] += int(s_.sum())
        a['truth_sum'] += float(truth[s_].sum())
        for c in CORE:
            a[c] += float(cov[c][s_].sum())
    n_or = len(np.unique(M['sh_o']))
    acc.fwd.append({'shadow_forecasts': int(len(M['sh_o'])), 'origins_with_shadow': int(n_or),
                    'per_origin': len(M['sh_o']) / max(n_or, 1), 'n_ages': int(len(np.unique(M['ages']))), 'n_stations': int(M['truth'].shape[1])})
    miss = float(np.mean(~np.isfinite(M['truth'])))
    acc.miss = miss


def boot_pair(units, label, a, b):
    return au.boot_contrasts(units, {label: (a, b)}, B14)[label]


def mean_cov(units, c):
    return float(np.mean([np.sum(u['cov_' + c]) / max(np.sum(u['n']), 1) for u in units]))


# ------------------------------------------------------------------ distribution sensitivity
def tv_tables(ds, T, SH, ON):
    rows = []
    rng = np.random.default_rng(11)
    for b in range(4):
        tm = T['ab'] == b
        tb, sb = T['res'][tm], SH['res'][SH['ab'] == b]
        if len(tb) < 50 or len(sb) < 50:
            continue
        days = T['day'][tm]
        blk = (days // B14).astype(int)
        ub = np.unique(blk)
        iqr = float(np.subtract(*np.quantile(tb, [.75, .25])))
        pooled = np.concatenate([tb, sb, ON['res']])
        for kind in ('quantile', 'equal'):
            for nb in TV_BINS:
                edges = np.unique(np.quantile(pooled, np.linspace(0, 1, nb + 1))) if kind == 'quantile' else \
                    np.linspace(*np.quantile(pooled, [.01, .99]), nb + 1)
                par = days % 2
                floor = tv_hist(tb[par == 0], tb[par == 1], edges)
                est_s, est_o = tv_hist(tb, sb, edges), tv_hist(tb, ON['res'], edges)
                boots = []
                for _ in range(TV_BOOT):
                    pick = rng.choice(ub, len(ub))
                    x = np.concatenate([tb[blk == u] for u in pick])
                    boots.append(tv_hist(x, sb, edges))
                rows.append({'dataset': ds, 'age_group': b, 'bins': nb, 'rule': kind, 'tv_shadow': est_s, 'tv_shadow_lo': float(np.percentile(boots, 2.5)),
                             'tv_shadow_hi': float(np.percentile(boots, 97.5)), 'tv_online': est_o, 'noise_floor': floor, 'n_target': len(tb)})
        for q in (.1, .5, .9):
            rows.append({'dataset': ds, 'age_group': b, 'quantile_shift': q, 'shadow_minus_target_over_iqr': (float(np.quantile(sb, q)) - float(np.quantile(tb, q))) / max(iqr, 1e-9),
                         'online_minus_target_over_iqr': (float(np.quantile(ON['res'], q)) - float(np.quantile(tb, q))) / max(iqr, 1e-9)})
    return rows


# ------------------------------------------------------------------ case study
def case_study(ds, M, tg, lo, hi, stations, mask_name):
    common = np.logical_and.reduce([np.isfinite(lo[c]) for c in CORE]) & (tg['h'] == 0)
    if common.sum() == 0:
        return []
    best = None
    for s in np.unique(tg['s'][common]):
        sel = common & (tg['s'] == s)
        days = tg['origin'][sel] // 24
        for d0 in np.unique(days):
            n = int(((days >= d0) & (days < d0 + 14)).sum())
            if best is None or n > best[0]:
                best = (n, int(s), int(d0))
    n, s, d0 = best
    sel = common & (tg['s'] == s) & (tg['origin'] // 24 >= d0) & (tg['origin'] // 24 < d0 + 14)
    t0 = np.datetime64(int(M['t0_ns']), 'ns').astype('datetime64[h]')
    truth = M['truth'][tg['o'], tg['s'], tg['h']]
    point = truth - tg['res']
    out = []
    for i in np.flatnonzero(sel):
        r = {'dataset': ds, 'mask': mask_name, 'station': stations[s], 'time': str(t0 + int(tg['time'][i])), 'age_bucket': int(tg['ab'][i]),
             'truth': float(truth[i]), 'point': float(point[i])}
        for c in CORE:
            r[c + '_lo'], r[c + '_hi'] = float(point[i] + lo[c][i]), float(point[i] + hi[c][i])
        out.append(r)
    return out


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {k: [] for k in ('variants', 'contrasts', 'level', 'level_ci', 'exceed', 'groups', 'tv', 'stations', 'cost', 'case')}
    for ds_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        ds = ds_dir.name
        acc, parts_t, parts_s, parts_o, stations, first = Acc(), [], [], [], None, True
        unit_i = 0
        for seed in SEEDS:
            for path in sorted((ds_dir / f'seed_{seed}').glob('mcar_*.npz')):
                M = ev.load_mask(path)
                stations = [str(x) for x in M['stations']]
                t, sh, on = au.residual_rows(M, PERIOD)
                parts_t.append(t); parts_s.append(sh); parts_o.append(on)
                tg, lo, hi = run_mask(M, PERIOD, unit_i)
                accumulate(acc, M, tg, lo, hi)
                if first:
                    R['case'] += case_study(ds, M, tg, lo, hi, stations, path.stem)
                    first = False
                unit_i += 1
                print('done', ds, seed, path.stem, flush=True)
        for c in ALL:
            n, sup, cov, w, sc = acc.var[c]
            R['variants'].append({'dataset': ds, 'variant': c, 'support_share': sup / max(n, 1), 'coverage_on_supported': cov / max(sup, 1),
                                  'mean_width': w / max(sup, 1), 'mean_interval_score': sc / max(sup, 1), 'n_targets': int(n)})
        for k, (a, b) in CONTRASTS.items():
            e, l, u = boot_pair(acc.units[k], k, a, b)
            R['contrasts'].append({'dataset': ds, 'contrast': k, 'cov_a': mean_cov(acc.units[k], a), 'cov_b': mean_cov(acc.units[k], b),
                                   'diff': e, 'ci_lo': l, 'ci_hi': u})
        for (src, lab, c), v in sorted(acc.lvl.items()):
            R['level'].append({'dataset': ds, 'grouping': src, 'level_group': lab, 'source': c, 'n': int(v[0]), 'coverage': v[1] / max(v[0], 1),
                               'miss_above': v[2] / max(v[0], 1), 'mean_width': v[3] / max(v[0], 1)})
        for (src, lab, k), lst in acc.lvl_units.items():
            a, b = LEVEL_CONTRASTS[k]
            e, l, u = boot_pair(lst, k, a, b)
            R['level_ci'].append({'dataset': ds, 'grouping': src, 'level_group': lab, 'contrast': k, 'diff': e, 'ci_lo': l, 'ci_hi': u})
        for (src, tname, c), v in sorted(acc.exc.items()):
            R['exceed'].append({'dataset': ds, 'threshold': tname, 'source': c, 'positives': int(v[0]), 'recall_upper_bound': v[1] / max(v[0], 1),
                                'alert_rate': v[2] / max(v[3], 1), 'n': int(v[3])})
        for (kind, g, c), v in sorted(acc.grp.items()):
            R['groups'].append({'dataset': ds, 'by': kind, 'key': g, 'source': c, 'n': int(v[0]), 'coverage': v[1] / max(v[0], 1)})
        T, SH, ON = au.cat(parts_t), au.cat(parts_s), au.cat(parts_o)
        R['tv'] += tv_tables(ds, T, SH, ON)
        for s, a in sorted(acc.stn.items()):
            R['stations'].append({'dataset': ds, 'station': stations[s], 'n_targets': a['n'], 'mean_truth': a['truth_sum'] / max(a['n'], 1),
                                  'missing_share_of_truth': acc.miss, **{c: a[c] / max(a['n'], 1) for c in CORE}})
        f = acc.fwd[0]
        R['cost'].append({'dataset': ds, 'n_stations': f['n_stations'], 'hide_lengths': f['n_ages'], 'mean_shadow_forwards_per_origin': float(np.mean([x['per_origin'] for x in acc.fwd])),
                          'upper_bound_forwards_per_origin_all_online': f['n_stations'] * f['n_ages']})
    for k, rows in R.items():
        au.write_csv(OUT / f'review_{k}.csv', rows)
    write_report(R)
    (OUT / 'REVIEW_DONE.json').write_text(json.dumps({'complete': True, 'seeds': SEEDS, 'period': PERIOD, 'epa_2025_loaded': PERIOD == 'holdout'}))


def write_report(R):
    L = ['# Shadow review analyses', '']
    L += ['## R3/R6 variants: support, coverage on own supported targets, width, interval score', '',
          '| dataset | variant | support | coverage | width | interval score |', '|---|---|---:|---:|---:|---:|']
    for r in R['variants']:
        L.append(f"| {r['dataset']} | {r['variant']} | {r['support_share']:.3f} | {r['coverage_on_supported']:.3f} | {r['mean_width']:.2f} | {r['mean_interval_score']:.2f} |")
    L += ['', '## Paired contrasts on jointly supported targets (14-day block bootstrap, 95% CI)', '',
          '| dataset | contrast | cov a | cov b | diff [CI] |', '|---|---|---:|---:|---|']
    for r in R['contrasts']:
        L.append(f"| {r['dataset']} | {r['contrast']} | {r['cov_a']:.3f} | {r['cov_b']:.3f} | {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    L += ['', '## R1 coverage by NO2 level (common-supported targets)', '',
          '| dataset | grouping | level group | source | n | coverage | miss above | width |', '|---|---|---|---|---:|---:|---:|---:|']
    for r in R['level']:
        L.append(f"| {r['dataset']} | {r['grouping']} | {r['level_group']} | {r['source']} | {r['n']} | {r['coverage']:.3f} | {r['miss_above']:.3f} | {r['mean_width']:.1f} |")
    L += ['', '### Top-decile contrasts', '', '| dataset | grouping | contrast | diff [CI] |', '|---|---|---|---|']
    for r in R['level_ci']:
        L.append(f"| {r['dataset']} | {r['grouping']} | {r['contrast']} | {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    L += ['', '### Exceedance of data-driven thresholds (truth above fit+tune P90/P95)', '',
          '| dataset | threshold | source | positives | recall of upper bound | alert rate |', '|---|---|---|---:|---:|---:|']
    for r in R['exceed']:
        L.append(f"| {r['dataset']} | {r['threshold']} | {r['source']} | {r['positives']} | {r['recall_upper_bound']:.3f} | {r['alert_rate']:.3f} |")
    L += ['', '## R2 coverage by age bucket / hour / horizon', '', '| dataset | by | key | C1 | C2 | C3 | C4 |', '|---|---|---:|---:|---:|---:|---:|']
    keys = sorted({(r['dataset'], r['by'], r['key']) for r in R['groups']})
    for ds, by, key in keys:
        v = {r['source']: r['coverage'] for r in R['groups'] if (r['dataset'], r['by'], r['key']) == (ds, by, key)}
        L.append(f"| {ds} | {by} | {key} | {v['C1_archival']:.3f} | {v['C2_online']:.3f} | {v['C3_shadow']:.3f} | {v['C4_oracle']:.3f} |")
    L += ['', '## R4 TV sensitivity (40 quantile bins and range of bin counts)', '', '| dataset | age group | rule | bins | TV shadow [CI] | TV online | noise floor |', '|---|---:|---|---:|---|---:|---:|']
    for r in R['tv']:
        if 'tv_shadow' in r:
            L.append(f"| {r['dataset']} | {r['age_group']} | {r['rule']} | {r['bins']} | {r['tv_shadow']:.3f} [{r['tv_shadow_lo']:.3f}, {r['tv_shadow_hi']:.3f}] | {r['tv_online']:.3f} | {r['noise_floor']:.3f} |")
    L += ['', '## R6 cost: shadow forecasts per origin', '', '| dataset | stations | hide lengths | mean shadow forwards/origin | upper bound |', '|---|---:|---:|---:|---:|']
    for r in R['cost']:
        L.append(f"| {r['dataset']} | {r['n_stations']} | {r['hide_lengths']} | {r['mean_shadow_forwards_per_origin']:.1f} | {r['upper_bound_forwards_per_origin_all_online']} |")
    L += ['', 'Wall-clock time and memory are not measured here.']
    (OUT / 'SHADOW_REVIEW_REPORT.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
