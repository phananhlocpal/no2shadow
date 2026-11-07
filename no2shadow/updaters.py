"""Online conformal updaters fed with the online label stream or the shadow label stream.

Each updater receives absolute residuals and returns a symmetric radius around the point forecast.
  online : residuals of online stations, cell = issue hour.
  shadow : shadow residuals (four hiding lengths 12/36/60/84 h, one per age bucket), cell = age bucket x issue hour.
A residual reaches the updater 72 h after its target time; the interval at an origin uses the updater state after all
arrivals up to that origin. Methods: SAOCP, FACI and NExConformal from salesforce/online_conformal (commit 95790a9,
unmodified) and ROLL_ABS, a 30-day rolling quantile of |residual| used as a simple reference.

Usage:
    python -m no2shadow.updaters
Environment variables (optional): AQ_UPD_SRC (default results/forecasts_fresh, schedules 31756-31760),
AQ_UPD_OUT, AQ_UPD_PERIOD (evaluation or holdout), AQ_UPD_METHODS, AQ_UPD_UNITS (seed:schedule list), AQ_UPD_PROCS.
"""
import csv, json, math, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from . import calibration as ev

SRC = ROOT / os.environ.get('AQ_UPD_SRC', 'results/forecasts_fresh')
OUT = ROOT / os.environ.get('AQ_UPD_OUT', 'results/updaters')
PERIOD = os.environ.get('AQ_UPD_PERIOD', 'evaluation')
ALPHA, DELAY, WINDOW, MIN_N, HMAX = 0.2, 72, 720, 100, 6
METHODS = tuple(os.environ.get('AQ_UPD_METHODS', 'SAOCP,FACI,NEx,ROLL_ABS').split(','))
SOURCES = ('online', 'shadow')
SUB_AGES = (12, 36, 60, 84)


def streams(M, source):
    """Return cell, target time, arrival time, residual, origin day, station arrays for one source."""
    T = M['T']
    if source == 'online':
        P, truth, fb, off = M['pred'][..., 1], M['truth'], M['fb'], M['off']
        res = truth - P
        ok = np.isfinite(res) & (off[:, :, None] == 0) & (fb == 1)
        o, s, h = np.nonzero(ok)
        cell = 4 * M['offset'][o].astype(int)
        r = res[o, s, h]
    else:
        keep = np.isin(M['ages'][M['sh_a']], SUB_AGES)
        o, s = M['sh_o'][keep], M['sh_s'][keep].astype(int)
        res2 = M['truth'][o, s, :] - M['sh_pred'][keep]
        ok = np.isfinite(res2) & (M['fb'][o, s, :] == 1)
        n, h = np.nonzero(ok)
        cell = ev.bucket(M['sh_age'][keep][n]) + 4 * M['offset'][o[n]].astype(int)
        r = res2[n, h]
        o, s = o[n], s[n]
    time_idx = T[o] + h
    return cell, time_idx, time_idx + DELAY, r, (T[o] - 1) // 24, s


def make_updater(method, scale):
    from online_conformal.saocp import SAOCP
    from online_conformal.faci import FACI
    from online_conformal.nex_conformal import NExConformal
    kw = dict(model=None, train_data=None, coverage=1 - ALPHA, horizon=1)
    if method == 'SAOCP':
        return SAOCP(max_scale=scale, **kw)
    return FACI(**kw) if method == 'FACI' else NExConformal(**kw)


def series_radii(method, arrival, res, queries, lock):
    """Radius at each query origin using only scores with arrival <= origin (stable arrival order)."""
    order = np.argsort(arrival, kind='stable')
    arrival, res = arrival[order], res[order]
    out = {q: math.nan for q in queries}
    if not len(res):
        return out
    pre = np.abs(res[arrival <= lock])
    if method == 'SAOCP' and not len(pre):
        return out
    scale = max(float(pre.max()) * math.sqrt(3), 1e-6) if len(pre) else 1.0
    model = make_updater(method, scale)
    i = 0
    for q in queries:
        j = int(np.searchsorted(arrival, q, side='right'))
        if j > i:
            model.update(pd.Series(res[i:j]), pd.Series(np.zeros(j - i)), horizon=1)
            i = j
        if i >= MIN_N:
            out[q] = float(model.predict(horizon=1)[1])
    return out


def rolling_radii(cell, time_idx, day, site, res, queries_by_cell):
    pool = ev.make_pool((cell, time_idx, day, site, np.abs(res)))
    out = {}
    for c, qs in queries_by_cell.items():
        for q in qs:
            rows = pool.query(c, q - DELAY, q - DELAY - WINDOW)
            r = math.nan
            if rows is not None:
                d, st, a = rows
                n = len(a)
                if n >= 100 and len(np.unique(d)) >= 7 and len(np.unique(st)) >= 3:
                    k = math.ceil((1 - ALPHA) * (n + 1))
                    if k <= n:
                        r = float(np.sort(a)[k - 1])
            out[(c, q)] = r
    return out


def evaluate(tg, radius, M):
    ok = np.isfinite(radius)
    a = np.abs(tg['res'])
    cov = ok & (a <= radius)
    width = 2 * radius
    score = width + 2 / ALPHA * np.where(ok, np.maximum(0, a - radius), 0)
    mon, _ = ev.month_of(M, tg['time'])
    rows = []
    groups = [('all', 'all', np.ones(len(a), bool))] + [('month', str(m), mon == m) for m in sorted(set(mon))]
    for gt, g, sel in groups:
        use = sel & ok
        n = int(use.sum())
        row = {'group_type': gt, 'group': g, 'targets': int(sel.sum()), 'n': n}
        if n:
            row.update(coverage=float(cov[use].mean()), width=float(width[use].mean()), interval_score=float(score[use].mean()))
        rows.append(row)
    return rows


def run_task(args):
    path, ds, seed, mask, method, source = args
    t0 = time.time()
    M = ev.load_mask(path)
    tg = ev.targets_for(M, PERIOD, 'bucket')
    cell, tidx, arrival, res, day, site = streams(M, source)
    tcell = (tg['ab'] if source == 'shadow' else 0 * tg['ab']) + 4 * tg['oh']
    lock = tg['start']
    radius = np.full(len(tcell), math.nan)
    if method == 'ROLL_ABS':
        qb = {int(c): np.unique(tg['origin'][tcell == c]) for c in np.unique(tcell)}
        rr = rolling_radii(cell, tidx, day, site, res, qb)
        radius = np.array([rr.get((int(c), int(o)), math.nan) for c, o in zip(tcell, tg['origin'])])
    else:
        for c in np.unique(tcell):
            sel = tcell == c
            qs = np.unique(tg['origin'][sel])
            m = cell == c
            rad = series_radii(method, arrival[m], res[m], qs, lock)
            radius[sel] = [rad[o] for o in tg['origin'][sel]]
    rows = evaluate(tg, radius, M)
    base = {'dataset': ds, 'seed': seed, 'mask': mask, 'method': method, 'source': source,
            'stream_rows': int(len(res)), 'seconds': round(time.time() - t0, 1)}
    return [{**base, **r} for r in rows]


def main():
    units = [tuple(x.split(':')) for x in os.environ.get('AQ_UPD_UNITS', '2026:31756,2027:31756,2026:31757,2027:31757').split(',')]
    tasks = []
    for ds in sorted(p.name for p in SRC.iterdir() if p.is_dir()):
        for seed, mask in units:
            path = SRC / ds / f'seed_{seed}' / f'mcar_{mask}.npz'
            if not path.exists():
                raise FileNotFoundError(path)
            for method in METHODS:
                for source in SOURCES:
                    tasks.append((str(path), ds, int(seed), f'mcar_{mask}', method, source))
    # slowest first: shadow streams, SAOCP/FACI
    weight = lambda t: (t[5] == 'shadow') * 4 + (t[4] in ('SAOCP', 'FACI')) * 2
    tasks.sort(key=weight, reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)
    nproc = max(1, min(os.cpu_count() or 1, int(os.environ.get('AQ_UPD_PROCS', '8'))))
    print('tasks', len(tasks), 'procs', nproc, flush=True)
    rows = []
    with get_context('fork').Pool(nproc) as pool:
        for k, part in enumerate(pool.imap_unordered(run_task, tasks), 1):
            rows += part
            first = part[0]
            print('completed', k, '/', len(tasks), first['dataset'], first['seed'], first['method'], first['source'], 'stream', first['stream_rows'], 'sec', first['seconds'], flush=True)
            ev.write_csv(OUT / 'updaters_summary.csv', rows)
    lines = ['# Online updaters x label sources', '',
             '| dataset | method | source | coverage | width | interval score | supported |', '|---|---|---|---:|---:|---:|---:|']
    for ds in sorted({r['dataset'] for r in rows}):
        for m in METHODS:
            for src in SOURCES:
                rr = [r for r in rows if r['dataset'] == ds and r['method'] == m and r['source'] == src and r['group_type'] == 'all']
                if rr:
                    mean = lambda k: np.mean([r[k] for r in rr if k in r])
                    lines.append(f'| {ds} | {m} | {src} | {mean("coverage"):.3f} | {mean("width"):.1f} | {mean("interval_score"):.1f} | '
                                 f'{np.mean([r["n"] / r["targets"] for r in rr]):.2f} |')
    (OUT / 'SHADOW_UPDATERS_REPORT.md').write_text('\n'.join(lines))
    (OUT / 'UPDATERS_DONE.json').write_text(json.dumps({'complete': True, 'units': units, 'methods': METHODS, 'epa_2025_loaded': PERIOD == 'holdout', 'period': PERIOD}))


if __name__ == '__main__':
    main()
