"""Shadow forecasts: base forecasts under a simulated outage schedule plus causal shadow forecasts.

For each forecast origin t (four per day, at 0/6/12/18 h) the frozen model forecasts horizons 1-6 for every station
from the 96 h context visible under the outage schedule. For every station that is online at t-1 and every hiding
length a in {6, 12, ..., 96} h, the last a hours of that station are hidden on top of the real mask (other stations keep
their real context) and the same model forecasts the station again. Nothing after t-1 is used. A shadow residual becomes
a calibration label only once its target is observed and 72 h have passed; that rule is applied in calibration.py.

Output: one .npz per (dataset, model seed, outage schedule) in results/forecasts/<dataset>/seed_<seed>/.

Usage:
    python -m no2shadow.forecast
Environment variables (optional):
    AQ_SHADOW_OUT        output folder (default results/forecasts)
    AQ_SHADOW_DATASETS   comma list (default beijing_multisite,kdd2018_london,epa_la)
    AQ_SHADOW_SEEDS      model seeds (default 2025,2026,2027)
    AQ_SHADOW_MCAR_SEEDS outage schedule seeds (default 31751..31755)
    AQ_SHADOW_MNAR_SEEDS model seeds that also get level-dependent schedules (default 2025)
    AQ_SHADOW_SCENARIOS  extra mechanisms, e.g. pre:1,pre:2,corr:0.6,corr:0.9
    AQ_SHADOW_HOLDOUT    lock date for the EPA 2025 held-out year, e.g. 2025-01-01 (EPA only)
    AQ_DEVICE            cuda or cpu (default: cuda when available)
"""
import json, os, time
from pathlib import Path
import numpy as np

from .data import load_dataset as _load
from .outages import make_events, make_events_corr, make_events_mnar, make_events_pre

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get('AQ_SHADOW_OUT', 'results/forecasts')
MODELS = ROOT / 'models'
DATASETS = tuple(os.environ.get('AQ_SHADOW_DATASETS', 'beijing_multisite,kdd2018_london,epa_la').split(','))
HOLDOUT = os.environ.get('AQ_SHADOW_HOLDOUT', '')
MCAR_SEEDS = tuple(int(x) for x in os.environ.get('AQ_SHADOW_MCAR_SEEDS', '31751,31752,31753,31754,31755').split(','))
MNAR_SEEDS = (41751, 41752, 41753)
MNAR_BETAS = (1.0, 2.0)
OFFSETS = (0, 6, 12, 18)
AGES = tuple(range(6, 97, 6))        # 16 hiding lengths; calibration cells round ages to 6 h
HMAX, WIN, DURATION = 6, 96, 72
SPLITS = ('fit', 'tune', 'evaluation', 'holdout')
SCENARIOS = [x for x in os.environ.get('AQ_SHADOW_SCENARIOS', '').split(',') if x]
SCEN_SEEDS = tuple(int(x) for x in os.environ.get('AQ_SHADOW_SCEN_SEEDS', '51751,51752,51753').split(','))


def mask_list(seed, with_mnar):
    """(name, mask seed, kind, parameter). kind: mcar | mnar (future-window level) | pre (past-level) | corr (shared events)."""
    masks = [(f'mcar_{s}', s, 'mcar', 0.0) for s in MCAR_SEEDS]
    if with_mnar:
        masks += [(f'mnar{int(b)}_{s}', s, 'mnar', b) for b in MNAR_BETAS for s in MNAR_SEEDS]
    for sc in SCENARIOS:
        kind, val = sc.split(':')
        val = float(val)
        tag = f'{kind}{int(val)}' if kind == 'pre' else f'{kind}{int(round(val * 100))}'
        masks += [(f'{tag}_{s}', s, kind, val) for s in SCEN_SEEDS]
    return masks


def origins_for(splits):
    T, S, O = [], [], []
    for code, (name, lo, hi) in enumerate(splits):
        for oi, off in enumerate(OFFSETS):
            for t in range(max(lo + off, WIN), hi - 24 + 1, 24):
                T.append(t); S.append(code); O.append(oi)
    order = np.argsort(T, kind='stable')
    T, S, O = np.array(T)[order], np.array(S)[order], np.array(O)[order]
    if len(set(T.tolist())) != len(T):
        raise ValueError('duplicate origins')
    return T.astype(np.int64), S.astype(np.int8), O.astype(np.int8)


def state_arrays(visible, T):
    """Number of visible hours in the context window and age (hours since the last visible value) per origin and station."""
    n, c = visible.shape
    cum = np.vstack([np.zeros((1, c), np.int32), np.cumsum(visible, axis=0)])
    last = np.maximum.accumulate(np.where(visible, np.arange(n)[:, None] + 1, 0), axis=0)
    return (cum[T] - cum[T - WIN]).astype(np.int16), np.minimum(WIN, T[:, None] - last[T - 1]).astype(np.int16)


def shadow_state(vis):
    """vis: (N,WIN) visibility of one station inside the window after hiding. Same definition as state_arrays."""
    last = np.where(vis, np.arange(WIN)[None, :] + 1, 0).max(axis=1)
    return np.where(last > 0, WIN - last, WIN).astype(np.int16), vis.sum(axis=1).astype(np.int16)


def load_dataset(name):
    return _load(name, holdout=bool(HOLDOUT))


def forecast(model, x, m, mean, std, device, batch=256):
    """x: masked standardized inputs (N,WIN,C); m: mask (N,WIN,C). Returns (N,HMAX,C,3) in original units."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            q = model(torch.from_numpy(x[i:i + batch]).to(device), torch.from_numpy(m[i:i + batch]).to(device))
            q = q.cpu().numpy()[:, :HMAX] * std[None, None, :, None] + mean[None, None, :, None]
            out.append(q.astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, HMAX, len(mean), 3), np.float32)


def process_mask(model, raw, mean, std, T, offline, device, ages=AGES, fc=None):
    fc = fc or (lambda x, m: forecast(model, x, m, mean, std, device))
    n, c = raw.shape
    visible = np.isfinite(raw) & ~offline
    xz = np.nan_to_num((raw - mean) / std, nan=0).astype(np.float32)
    idx = T[:, None] + np.arange(-WIN, 0)[None, :]
    vwin, xwin = visible[idx], xz[idx]
    base = fc(xwin * vwin, vwin.astype(np.float32))                      # (O,6,C,3)
    hidx = T[:, None] + np.arange(HMAX)[None, :]
    ctx, age = state_arrays(visible, T)
    res = {'pred': base.transpose(0, 2, 1, 3), 'truth': raw[hidx].transpose(0, 2, 1).astype(np.float32),
           'fb': visible[hidx].transpose(0, 2, 1).astype(np.uint8), 'off': offline[T - 1].astype(np.uint8),
           'age': age, 'ctx': ctx}
    so, ss, sa, sage, sctx, spred = [], [], [], [], [], []
    for ai, a in enumerate(ages):
        for s in range(c):
            elig = np.flatnonzero(visible[T - 1, s])
            if not len(elig):
                continue
            v = vwin[elig].copy()
            v[:, WIN - a:, s] = False
            q = fc(xwin[elig] * v, v.astype(np.float32))[:, :, s, 1]    # median only
            ag, cx = shadow_state(v[:, :, s])
            so.append(elig.astype(np.int32)); ss.append(np.full(len(elig), s, np.int16))
            sa.append(np.full(len(elig), ai, np.int8)); sage.append(ag); sctx.append(cx); spred.append(q)
    cat = lambda xs, dt, shape: np.concatenate(xs) if xs else np.zeros(shape, dt)
    res.update(sh_o=cat(so, np.int32, (0,)), sh_s=cat(ss, np.int16, (0,)), sh_a=cat(sa, np.int8, (0,)),
               sh_age=cat(sage, np.int16, (0,)), sh_ctx=cat(sctx, np.int16, (0,)),
               sh_pred=cat(spred, np.float32, (0, HMAX)))
    return res


def main():
    import torch
    import pandas as pd
    from .model import RGQF
    device = torch.device(os.environ.get('AQ_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu'))
    seeds = [int(x) for x in os.environ.get('AQ_SHADOW_SEEDS', '2025,2026,2027').split(',')]
    mnar_seeds = [int(x) for x in os.environ.get('AQ_SHADOW_MNAR_SEEDS', '2025').split(',') if x]
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for ds in DATASETS:
        dates, stations, raw, train_end, val_end, fit_end, tune_end = load_dataset(ds)
        splits = [('fit', val_end, fit_end), ('tune', fit_end, tune_end), ('evaluation', tune_end, len(raw))]
        if HOLDOUT:
            hold = int(dates[dates == pd.Timestamp(HOLDOUT)].index[0])
            if ds != 'epa_la' or len(DATASETS) != 1:
                raise ValueError('holdout mode is EPA-only')
            splits = [('fit', val_end, fit_end), ('tune', fit_end, tune_end), ('evaluation', tune_end, hold), ('holdout', hold, len(raw))]
        T, S, O = origins_for(splits)
        for seed in seeds:
            cp = torch.load(MODELS / ds / f'seed_{seed}/checkpoint.pt', map_location=device, weights_only=False)
            if list(cp['stations']) != list(stations) or [cp['train_end'], cp['val_end']] != [train_end, val_end]:
                raise ValueError('checkpoint does not match the dataset splits')
            model = RGQF(64, 24, True).to(device)
            model.load_state_dict(cp['model'])
            model.eval()
            mean, std = np.asarray(cp['mean'], np.float32), np.asarray(cp['std'], np.float32)
            folder = OUT / ds / f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            for name, ms, kind, beta in mask_list(seed, seed in mnar_seeds):
                path = folder / f'{name}.npz'
                if path.exists():
                    continue
                if kind == 'mnar':
                    offline = make_events_mnar(raw, val_end, DURATION, ms, beta)
                elif kind == 'pre':
                    offline = make_events_pre(raw, val_end, DURATION, ms, beta)
                elif kind == 'corr':
                    offline = make_events_corr(len(raw), len(stations), val_end, DURATION, ms, beta)
                else:
                    offline = make_events(len(raw), stations, val_end, DURATION, seed=ms)
                res = process_mask(model, raw, mean, std, T, offline, device)
                tmp = path.with_name(path.stem + '.tmp.npz')
                np.savez_compressed(tmp, T=T, split=S, offset=O, ages=np.array(AGES), stations=np.array(stations),
                                    t0_ns=np.int64(dates.iloc[0].value), **res)
                tmp.replace(path)
                print('completed', ds, seed, name, 'shadow_rows', len(res['sh_o']), 'elapsed_min', round((time.time() - started) / 60, 1), flush=True)
    (OUT / 'FORECAST_DONE.json').write_text(json.dumps({'complete': True, 'datasets': DATASETS, 'seeds': seeds,
        'mnar_seeds': mnar_seeds, 'offsets': OFFSETS, 'ages': AGES, 'holdout_loaded': bool(HOLDOUT)}))


if __name__ == '__main__':
    main()
