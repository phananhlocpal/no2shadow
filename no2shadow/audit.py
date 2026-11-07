"""Post-hoc audit of the shadow label source on stored forecasts (CPU only).

B1  Do shadow residuals resemble the residuals of real outage targets? Quantiles, KS, TV by age group; per-station checks.
B2  Is the C2 -> C3 gain explained by sample size or by dependence among rows? Matched-size, one-horizon-per-row and
    four-hide-length variants of C3, evaluated on the same target set.
B3  Coverage by horizon (including an oracle calibrated per horizon) and bootstrap block-length sensitivity.

Reads forecasts written by forecast.py and reuses the calibrator, pools and targets of calibration.py.
Post-hoc analyses on the evaluation period.

Usage:
    python -m no2shadow.audit   (AQ_AUDIT_SRC, AQ_AUDIT_OUT, AQ_AUDIT_SEEDS, AQ_AUDIT_PERIOD)
"""
import csv, json, math, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from . import calibration as ev

SRC = ROOT / os.environ.get('AQ_AUDIT_SRC', 'results/forecasts')
OUT = ROOT / os.environ.get('AQ_AUDIT_OUT', 'results/audit')
SEEDS = [int(x) for x in os.environ.get('AQ_AUDIT_SEEDS', '2026,2027').split(',')]
PERIOD = os.environ.get('AQ_AUDIT_PERIOD', 'evaluation')
BLOCKS = (7, 14, 28)
BOOT, BOOT_SEED = 2000, 7
CORE = ('C1_archival', 'C2_online', 'C3_shadow', 'C4_oracle')
VARIANTS = ('C3_cap', 'C3_oneh', 'C3_4ages')
SUB_AGES = (12, 36, 60, 84)
QSTAT = []       # (n shadow rows, n online rows) per finest-level query, for the size diagnostic


def write_csv(path, rows):
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ shadow-row variants
def shadow_rows(M, mode, variant):
    o, s, a = M['sh_o'], M['sh_s'].astype(int), M['sh_a']
    res = M['truth'][o, s, :] - M['sh_pred']
    ok = np.isfinite(res) & (M['fb'][o, s, :] == 1)
    if variant == '4ages':
        ok &= np.isin(M['ages'][a], SUB_AGES)[:, None]
    n, h = np.nonzero(ok)
    if variant == 'oneh':                                   # one random horizon per shadow forecast row
        keep = h == ((n.astype(np.int64) * 2654435761 + 12345) % ev.HMAX)
        n, h = n[keep], h[keep]
    return ev._pack(M, o[n], s[n], h, res[n, h], M['sh_age'][n], mode)


def oracle_variant(tg, period_days, tiers):
    """Block cross-fit oracle with explicit tier keys (same logic as ev.oracle_intervals, block length ev.BLOCK)."""
    block = (tg['origin'] // 24 - period_days) // ev.BLOCK
    lo, hi = np.full(len(block), np.nan), np.full(len(block), np.nan)
    day, site = tg['origin'] // 24, tg['s']
    for keyarr in tiers:
        todo = ~np.isfinite(lo)
        for c in np.unique(keyarr[todo]):
            in_c = keyarr == c
            for b in np.unique(block[in_c & todo]):
                tr = in_c & (block != b)
                iv = ev.interval((day[tr], site[tr], tg['res'][tr]))
                if iv is not None:
                    sel = in_c & (block == b) & todo
                    lo[sel], hi[sel] = iv
    return lo, hi


def run_mask_audit(M, period, mode):
    tg = ev.targets_for(M, period, mode)
    start = tg['start']
    arch = ev.make_pool(ev.flat_base(M, 'offline', mode))
    online = ev.make_pool(ev.flat_base(M, 'online', mode))
    pools = {'full': ev.make_pool(shadow_rows(M, mode, 'full')), 'oneh': ev.make_pool(shadow_rows(M, mode, 'oneh')),
             '4ages': ev.make_pool(shadow_rows(M, mode, '4ages'))}
    win = lambda origin: (origin - ev.DELAY, origin - ev.DELAY - ev.WINDOW)

    def cap(o, c):
        rows = pools['full'].query(c, *win(o))
        ro = online.query(ev.online_key(c, mode), *win(o))
        if rows is None or ro is None:
            return None
        m = len(ro[2])
        if mode == 'age_hour':
            QSTAT.append((len(rows[2]), m))
        if len(rows[2]) > m:
            idx = np.random.default_rng(o * 1000 + c).choice(len(rows[2]), m, replace=False)
            rows = tuple(a[idx] for a in rows)
        return ev.interval(rows)
    fns = {
        'C1_archival': lambda o, c: ev.interval(arch.query(c, start - ev.DELAY)),
        'C2_online': lambda o, c: ev.interval(online.query(ev.online_key(c, mode), *win(o))),
        'C3_shadow': lambda o, c: ev.interval(pools['full'].query(c, *win(o))),
        'C3_cap': cap,
        'C3_oneh': lambda o, c: ev.interval(pools['oneh'].query(c, *win(o))),
        'C3_4ages': lambda o, c: ev.interval(pools['4ages'].query(c, *win(o))),
    }
    lo, hi = ev.intervals_for(tg, fns)
    lo['C4_oracle'], hi['C4_oracle'], _ = ev.oracle_intervals(tg, start // 24, mode)
    if mode == 'age_hour':                                  # horizon-specific oracle vs the matching horizon-free oracle
        d0 = start // 24
        base = tg['ab'] + 4 * tg['oh']
        lo['C4_nohz'], hi['C4_nohz'] = oracle_variant(tg, d0, [base, tg['ab']])
        lo['C4_hz'], hi['C4_hz'] = oracle_variant(tg, d0, [base * 8 + tg['h'], tg['ab'] * 8 + tg['h']])
    return tg, lo, hi, None, None


def combined(M, period):
    res = {m: run_mask_audit(M, period, m) for m in ev.MODES}
    # the two horizon oracles exist only for the finest mode; copy them through unchanged
    extra = {k: (res['age_hour'][1][k], res['age_hour'][2][k]) for k in ('C4_nohz', 'C4_hz')}
    for m in ev.MODES:
        for k in extra:
            res[m][1].pop(k, None); res[m][2].pop(k, None)
    tg, lo, hi, _, _, share = ev.combine_hier({m: (r[0], r[1], r[2], None, {}) for m, r in res.items()})
    for k, (l, u) in extra.items():
        lo[k], hi[k] = l, u
    return tg, lo, hi


# ------------------------------------------------------------------ statistics
def covered(res, lo, hi):
    return np.isfinite(lo) & (res >= lo) & (res <= hi)


def unit_sums(tg, lo, hi, names, B):
    common = np.logical_and.reduce([np.isfinite(lo[c]) for c in names])
    blk = (tg['origin'] // 24 - tg['start'] // 24) // B
    nb = (tg['end'] // 24 - tg['start'] // 24) // B + 1
    out = {'nb': nb, 'n': np.bincount(blk[common], minlength=nb).astype(float)}
    for c in names:
        out['cov_' + c] = np.bincount(blk[common], weights=covered(tg['res'], lo[c], hi[c])[common], minlength=nb)
    return out


def boot_contrasts(units, contrasts, B):
    """units: list of unit_sums dicts. contrasts: {label: (a, b)} meaning cov_a - cov_b. Mean over units of block-pooled coverage."""
    nb = units[0]['nb']
    if any(u['nb'] != nb for u in units):
        raise ValueError('unit block structure differs')
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(nb, np.ones(nb) / nb, size=BOOT).astype(float)

    def cov(w, c):
        return np.mean([(w @ u['cov_' + c]) / max(w @ u['n'], 1) for u in units])
    out = {}
    one = np.ones(nb)
    for label, (a, b) in contrasts.items():
        est = cov(one, a) - cov(one, b)
        d = np.array([cov(w, a) - cov(w, b) for w in W])
        out[label] = (est, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out


def quantiles(x, qs=(.1, .5, .9)):
    return [float(np.quantile(x, q)) if len(x) else math.nan for q in qs]


def ks_stat(a, b):
    if not len(a) or not len(b):
        return math.nan
    z = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), z, side='right') / len(a)
    fb = np.searchsorted(np.sort(b), z, side='right') / len(b)
    return float(np.max(np.abs(fa - fb)))


def tv_stat(a, b, edges):
    if not len(a) or not len(b):
        return math.nan
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a)
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


# ------------------------------------------------------------------ B1: residual distributions
def residual_rows(M, period):
    code = ev.SPLIT_NAMES.index(period)
    # real outage targets
    tg = ev.targets_for(M, period, 'bucket')
    # shadow rows inside the period
    rows = np.flatnonzero(M['split'][M['sh_o']] == code)
    o, s = M['sh_o'][rows], M['sh_s'][rows].astype(int)
    res = M['truth'][o, s, :] - M['sh_pred'][rows]
    ok = np.isfinite(res) & (M['fb'][o, s, :] == 1)
    n, h = np.nonzero(ok)
    sh = {'res': res[n, h], 'ab': ev.bucket(M['sh_age'][rows][n]), 's': s[n], 'day': (M['T'][o[n]] - 1) // 24}
    # online-state rows inside the period
    P, truth, fb, off = M['pred'][..., 1], M['truth'], M['fb'], M['off']
    inp = (M['split'] == code)[:, None, None]
    oko = inp & np.isfinite(truth - P) & (off[:, :, None] == 0) & (fb == 1)
    oo, so, ho = np.nonzero(oko)
    on = {'res': (truth - P)[oo, so, ho], 's': so, 'day': (M['T'][oo] - 1) // 24}
    t = {'res': tg['res'], 'ab': tg['ab'], 's': tg['s'], 'day': tg['origin'] // 24}
    return t, sh, on


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k in parts[0]}


def dist_tables(ds, T, SH, ON, stations):
    rows, st_rows = [], []
    for b in range(4):
        tb = T['res'][T['ab'] == b]
        sb = SH['res'][SH['ab'] == b]
        if len(tb) < 50 or len(sb) < 50:
            continue
        edges = np.unique(np.quantile(np.concatenate([tb, sb, ON['res']]), np.linspace(0, 1, 41)))
        par = T['day'][T['ab'] == b] % 2
        null = tv_stat(tb[par == 0], tb[par == 1], edges)
        for src, x in (('shadow', sb), ('online', ON['res'])):
            q10, q50, q90 = quantiles(x)
            cov_static = float(np.mean((tb >= q10) & (tb <= q90)))
            rows.append({'dataset': ds, 'age_group': b, 'source': src, 'n_target': len(tb), 'n_source': len(x),
                         **{f'q{int(q*100)}_target': v for q, v in zip((.1, .5, .9), quantiles(tb))},
                         'q10_source': q10, 'q50_source': q50, 'q90_source': q90,
                         'ks': ks_stat(tb, x), 'tv': tv_stat(tb, x, edges), 'tv_noise_floor': null, 'target_cov_of_source_q10_q90': cov_static})
    # per-station median shift, shadow vs target, matched by age group, normalised by target IQR of the group
    for s in np.unique(T['s']):
        num, den = 0.0, 0.0
        for b in range(4):
            tm = (T['s'] == s) & (T['ab'] == b)
            sm = (SH['s'] == s) & (SH['ab'] == b)
            tb = T['res'][T['ab'] == b]
            if tm.sum() < 30 or sm.sum() < 30 or len(tb) < 50:
                continue
            iqr = np.subtract(*np.quantile(tb, [.75, .25]))
            num += tm.sum() * (np.median(T['res'][tm]) - np.median(SH['res'][sm])) / max(iqr, 1e-9)
            den += tm.sum()
        if den > 0:
            st_rows.append({'dataset': ds, 'station': stations[s], 'n_target': int((T['s'] == s).sum()), 'median_shift_over_iqr': num / den})
    return rows, st_rows


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dist_rows, shift_rows, stat_rows, abl_rows, hz_rows, blk_rows, pool_rows = [], [], [], [], [], [], []
    for ds_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        ds = ds_dir.name
        parts_t, parts_s, parts_o, units, stations = [], [], [], {B: [] for B in BLOCKS}, None
        hz = {c: np.zeros((ev.HMAX, 2)) for c in list(CORE) + ['C4_nohz', 'C4_hz']}
        stn = {}
        for seed in SEEDS:
            for path in sorted((ds_dir / f'seed_{seed}').glob('mcar_*.npz')):
                M = ev.load_mask(path)
                stations = [str(x) for x in M['stations']]
                t, sh, on = residual_rows(M, PERIOD)
                parts_t.append(t); parts_s.append(sh); parts_o.append(on)
                tg, lo, hi = combined(M, PERIOD)
                names_main = list(CORE)
                names_abl = ['C2_online', 'C3_shadow', *VARIANTS]
                for B in BLOCKS:
                    units[B].append((unit_sums(tg, lo, hi, names_main, B), unit_sums(tg, lo, hi, names_abl, B)))
                # horizon and station tables (common set of the four core sources, pooled over units)
                common = np.logical_and.reduce([np.isfinite(lo[c]) for c in CORE])
                cov = {c: covered(tg['res'], lo[c], hi[c]) for c in CORE}
                for h in range(ev.HMAX):
                    sel = common & (tg['h'] == h)
                    for c in CORE:
                        hz[c][h] += (cov[c][sel].sum(), sel.sum())
                hcommon = np.logical_and.reduce([np.isfinite(lo[c]) for c in ('C4_nohz', 'C4_hz', 'C3_shadow')])
                for h in range(ev.HMAX):
                    sel = hcommon & (tg['h'] == h)
                    for c in ('C4_nohz', 'C4_hz'):
                        hz[c][h] += (covered(tg['res'], lo[c], hi[c])[sel].sum(), sel.sum())
                for s in np.unique(tg['s'][common]):
                    sel = common & (tg['s'] == s)
                    a = stn.setdefault(s, {c: np.zeros(2) for c in CORE})
                    for c in CORE:
                        a[c] += (cov[c][sel].sum(), sel.sum())
                print('done', ds, seed, path.stem, flush=True)
        T, SH, ON = cat(parts_t), cat(parts_s), cat(parts_o)
        d, s_ = dist_tables(ds, T, SH, ON, stations)
        dist_rows += d; shift_rows += s_
        for s, a in sorted(stn.items()):
            stat_rows.append({'dataset': ds, 'station': stations[s], 'n_targets': int(a['C1_archival'][1]),
                              **{c: a[c][0] / max(a[c][1], 1) for c in CORE}})
        for h in range(ev.HMAX):
            hz_rows.append({'dataset': ds, 'horizon': h + 1, **{c: hz[c][h, 0] / max(hz[c][h, 1], 1) for c in hz},
                            'n_core': int(hz['C1_archival'][h, 1]), 'n_oracle_pair': int(hz['C4_nohz'][h, 1])})
        for B in BLOCKS:
            core = [u[0] for u in units[B]]
            res = boot_contrasts(core, {'C3-C1': ('C3_shadow', 'C1_archival'), 'C3-C2': ('C3_shadow', 'C2_online'),
                                        'C4-C3': ('C4_oracle', 'C3_shadow')}, B)
            for k, (e, l, u) in res.items():
                blk_rows.append({'dataset': ds, 'block_days': B, 'contrast': k, 'estimate': e, 'ci_lo': l, 'ci_hi': u})
            if B == 14:
                abl = [u[1] for u in units[B]]
                contr = {}
                for v in ('C3_shadow',) + VARIANTS:
                    contr[f'{v}-C2_online'] = (v, 'C2_online')
                    if v != 'C3_shadow':
                        contr[f'{v}-C3_shadow'] = (v, 'C3_shadow')
                res = boot_contrasts(abl, contr, B)
                cov_all = {c: np.mean([(np.sum(u['cov_' + c]) / max(np.sum(u['n']), 1)) for u in abl]) for c in ['C2_online', 'C3_shadow', *VARIANTS]}
                n_all = float(np.mean([np.sum(u['n']) for u in abl]))
                for c, v in cov_all.items():
                    row = {'dataset': ds, 'variant': c, 'coverage': v, 'mean_common_targets_per_unit': n_all}
                    if f'{c}-C2_online' in res:
                        row['minus_C2'] = res[f'{c}-C2_online'][0]; row['minus_C2_lo'] = res[f'{c}-C2_online'][1]; row['minus_C2_hi'] = res[f'{c}-C2_online'][2]
                    if f'{c}-C3_shadow' in res:
                        row['minus_C3'] = res[f'{c}-C3_shadow'][0]; row['minus_C3_lo'] = res[f'{c}-C3_shadow'][1]; row['minus_C3_hi'] = res[f'{c}-C3_shadow'][2]
                    abl_rows.append(row)
        # shadow-row volume vs online-row volume (per unit): descriptive size of the calibration pools
        n_sh, n_on = len(SH['res']), len(ON['res'])
        q = np.array(QSTAT) if QSTAT else np.zeros((1, 2))
        pool_rows.append({'dataset': ds, 'real_outage_targets': int(len(T['res'])), 'shadow_rows': n_sh, 'online_rows': n_on,
                          'median_rows_per_query_shadow': float(np.median(q[:, 0])), 'median_rows_per_query_online': float(np.median(q[:, 1])),
                          'share_queries_where_shadow_exceeds_online': float(np.mean(q[:, 0] > q[:, 1]))})
        QSTAT.clear()
    write_csv(OUT / 'audit_distribution.csv', dist_rows)
    write_csv(OUT / 'audit_station_shift.csv', shift_rows)
    write_csv(OUT / 'audit_station_coverage.csv', stat_rows)
    write_csv(OUT / 'audit_ablation.csv', abl_rows)
    write_csv(OUT / 'audit_horizon.csv', hz_rows)
    write_csv(OUT / 'audit_block_sensitivity.csv', blk_rows)
    write_csv(OUT / 'audit_pool_sizes.csv', pool_rows)
    lines = ['# Shadow audit', '']
    lines += ['## B1. Residual distributions, shadow / online vs real outage targets', '',
              '| dataset | age group | source | q10 t/s | q50 t/s | q90 t/s | KS | TV | TV noise floor | target coverage of source q10-q90 |', '|---|---|---|---|---|---|---:|---:|---:|---:|']
    for r in dist_rows:
        lines.append(f"| {r['dataset']} | {r['age_group']} | {r['source']} | {r['q10_target']:.1f}/{r['q10_source']:.1f} | {r['q50_target']:.1f}/{r['q50_source']:.1f} | "
                     f"{r['q90_target']:.1f}/{r['q90_source']:.1f} | {r['ks']:.3f} | {r['tv']:.3f} | {r['tv_noise_floor']:.3f} | {r['target_cov_of_source_q10_q90']:.3f} |")
    lines += ['', '## B1b. Per-station median shift (shadow vs target) over target IQR', '', '| dataset | stations | mean abs | p90 abs | max abs |', '|---|---:|---:|---:|---:|']
    for ds in sorted({r['dataset'] for r in shift_rows}):
        v = np.abs([r['median_shift_over_iqr'] for r in shift_rows if r['dataset'] == ds])
        lines.append(f'| {ds} | {len(v)} | {v.mean():.3f} | {np.percentile(v, 90):.3f} | {v.max():.3f} |')
    lines += ['', '## B1c. Per-station coverage spread (common-supported targets)', '', '| dataset | source | min | max | sd | share within +-0.10 of 0.8 |', '|---|---|---:|---:|---:|---:|']
    for ds in sorted({r['dataset'] for r in stat_rows}):
        rs = [r for r in stat_rows if r['dataset'] == ds and r['n_targets'] >= 100]
        for c in CORE:
            v = np.array([r[c] for r in rs])
            if len(v):
                lines.append(f'| {ds} | {c} | {v.min():.3f} | {v.max():.3f} | {v.std():.3f} | {np.mean(np.abs(v - .8) <= .1):.2f} |')
    lines += ['', '## B2. Sample-size and dependence ablations of C3 (common-supported targets)', '',
              '| dataset | variant | coverage | minus C2 [CI] | minus C3 [CI] |', '|---|---|---:|---|---|']
    for r in abl_rows:
        f = lambda k: f"{r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]" if k in r else ''
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['coverage']:.3f} | {f('minus_C2')} | {f('minus_C3')} |")
    lines += ['', '## B3a. Coverage by horizon', '', '| dataset | h | C1 | C2 | C3 | C4 | C4 no-horizon | C4 per-horizon |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in hz_rows:
        lines.append(f"| {r['dataset']} | {r['horizon']} | {r['C1_archival']:.3f} | {r['C2_online']:.3f} | {r['C3_shadow']:.3f} | {r['C4_oracle']:.3f} | {r['C4_nohz']:.3f} | {r['C4_hz']:.3f} |")
    lines += ['', '## B3b. Bootstrap block-length sensitivity (95% CI)', '', '| dataset | block days | contrast | estimate | CI |', '|---|---:|---|---:|---|']
    for r in blk_rows:
        lines.append(f"| {r['dataset']} | {r['block_days']} | {r['contrast']} | {r['estimate']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |")
    lines += ['', '## B2b. Calibration rows per finest-level query', '', '| dataset | median shadow | median online | share queries with shadow > online |', '|---|---:|---:|---:|']
    for r in pool_rows:
        lines.append(f"| {r['dataset']} | {r['median_rows_per_query_shadow']:.0f} | {r['median_rows_per_query_online']:.0f} | {r['share_queries_where_shadow_exceeds_online']:.2f} |")
    (OUT / 'SHADOW_AUDIT_REPORT.md').write_text('\n'.join(lines))
    (OUT / 'AUDIT_DONE.json').write_text(json.dumps({'complete': True, 'seeds': SEEDS, 'period': PERIOD, 'epa_2025_loaded': PERIOD == 'holdout'}))


if __name__ == '__main__':
    main()
