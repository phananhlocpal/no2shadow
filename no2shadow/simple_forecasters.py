"""Two forecasters without training, used to check that the shadow source is not specific to the neural model.

persistence : last visible value repeated over horizons 1-6.
seasonal    : mean of the visible values at the same clock hour in the previous 1-4 days (persistence if none).
Both see the same masked context and the same shadow hiding. Scalers use the training period only. Outputs follow the
schema of forecast.py (q10/q90 are placeholders), so calibration.py runs unchanged on results/forecasts_<kind>.

Usage:
    python -m no2shadow.simple_forecasters   (AQ_SIMPLE_KINDS=persistence,seasonal)
"""
import os, sys, time
from pathlib import Path
import numpy as np

from . import forecast as sf

ROOT = sf.ROOT
HMAX, WIN = sf.HMAX, sf.WIN


def make_fc(kind, mean, std):
    def fc(x, m):
        N, L, C = x.shape
        pos = np.arange(L)[None, :, None]
        last = np.where(m > 0, pos, -1).max(axis=1)
        val = np.take_along_axis(x, np.clip(last, 0, None)[:, None, :], axis=1)[:, 0, :]
        val = np.where(last >= 0, val, 0.0)
        p = np.repeat(val[:, None, :], HMAX, axis=1)
        if kind == 'seasonal':
            hs = np.arange(HMAX)
            num = sum(x[:, WIN - 24 * k + hs, :] * m[:, WIN - 24 * k + hs, :] for k in (1, 2, 3, 4))
            cnt = sum(m[:, WIN - 24 * k + hs, :] for k in (1, 2, 3, 4))
            p = np.where(cnt > 0, num / np.maximum(cnt, 1), p)
        p = p * std[None, None, :] + mean[None, None, :]
        return np.stack((p - 1, p, p + 1), axis=-1).astype(np.float32)
    return fc


def main():
    from .outages import make_events
    kinds = os.environ.get('AQ_SIMPLE_KINDS', 'persistence,seasonal').split(',')
    started = time.time()
    for kind in kinds:
        out = ROOT / f'results/forecasts_{kind}'
        for ds in sf.DATASETS:
            dates, stations, raw, train_end, val_end, fit_end, tune_end = sf.load_dataset(ds)
            splits = [('fit', val_end, fit_end), ('tune', fit_end, tune_end), ('evaluation', tune_end, len(raw))]
            T, S, O = sf.origins_for(splits)
            mean = np.nanmean(raw[:train_end], axis=0).astype(np.float32)
            std = np.nanstd(raw[:train_end], axis=0).astype(np.float32)
            std = np.where(std > 0, std, 1.0).astype(np.float32)
            folder = out / ds / 'seed_0'
            folder.mkdir(parents=True, exist_ok=True)
            fc_std = make_fc(kind, mean, std)
            for name, ms, kind, beta in sf.mask_list(0, False):
                path = folder / f'{name}.npz'
                if path.exists():
                    continue
                offline = make_events(len(raw), stations, val_end, sf.DURATION, seed=ms)
                res = sf.process_mask(None, raw, mean, std, T, offline, None, fc=fc_std)
                tmp = path.with_name(path.stem + '.tmp.npz')
                np.savez_compressed(tmp, T=T, split=S, offset=O, ages=np.array(sf.AGES), stations=np.array(stations),
                                    t0_ns=np.int64(dates.iloc[0].value), **res)
                tmp.replace(path)
                print('completed', kind, ds, name, 'shadow_rows', len(res['sh_o']), 'elapsed_min', round((time.time() - started) / 60, 1), flush=True)


if __name__ == '__main__':
    main()
