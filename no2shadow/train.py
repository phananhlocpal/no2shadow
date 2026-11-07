"""Train the forecaster for each dataset and model seed (pinball loss, early stopping on validation).

Scalers are fitted on the training period only. During training one random 12 h block per series is hidden with
probability 0.25. Checkpoints are written to models/<dataset>/seed_<seed>/checkpoint.pt.

Usage:
    python -m no2shadow.train [--datasets beijing_multisite,kdd2018_london,epa_la] [--seeds 2025,2026,2027]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DATASETS, load_dataset
from .model import RGQF, Windows, augment_blocks, pinball

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs' / 'train.json').read_text())
MODELS = ROOT / 'models'


def train_one(name, seed, device):
    dates, stations, raw, train_end, val_end, fit_end, tune_end = load_dataset(name)
    mean = np.nanmean(raw[:train_end], axis=0).astype('float32')
    std = np.nanstd(raw[:train_end], axis=0).astype('float32')
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(f'{name}: a station has no training data')
    std = np.where(std > 1e-6, std, 1).astype('float32')
    mask = np.isfinite(raw).astype('float32')
    x = np.nan_to_num((raw - mean) / std, nan=0).astype('float32')
    c, h = CONFIG['context'], CONFIG['horizon']
    train = DataLoader(Windows(x, mask, c, train_end, c, h, 1), batch_size=CONFIG['batch_size'], shuffle=True)
    valid = DataLoader(Windows(x, mask, train_end, val_end, c, h, h), batch_size=CONFIG['batch_size'], shuffle=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = RGQF(CONFIG['hidden_size'], h, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    out = MODELS / name / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    best, patience, history = float('inf'), 0, []
    for epoch in range(CONFIG['epochs']):
        model.train()
        losses = []
        for a, b, y, ym, _ in train:
            a, b, y, ym = (v.to(device) for v in (a, b, y, ym))
            a, b = augment_blocks(a, b, CONFIG['augmentation_probability'], CONFIG['block_hours'])
            loss = pinball(model(a, b), y, ym)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        total = count = 0.
        with torch.no_grad():
            for a, b, y, ym, _ in valid:
                a, b, y, ym = (v.to(device) for v in (a, b, y, ym))
                m = float(ym.sum())
                total += float(pinball(model(a, b), y, ym)) * m
                count += m
        val = total / count
        history.append({'epoch': epoch + 1, 'train_pinball': float(np.mean(losses)), 'val_pinball': val})
        print('train', name, seed, epoch + 1, round(val, 5), flush=True)
        if val < best - 1e-6:
            best, patience = val, 0
            torch.save({'model': model.state_dict(), 'mean': mean, 'std': std, 'stations': stations, 'seed': seed,
                        'train_end': train_end, 'val_end': val_end, 'fit_end': fit_end, 'tune_end': tune_end}, out / 'checkpoint.pt')
        else:
            patience += 1
            if patience >= CONFIG['patience']:
                break
    (out / 'history.json').write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', default=','.join(DATASETS))
    parser.add_argument('--seeds', default='2025,2026,2027')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    for name in args.datasets.split(','):
        for seed in (int(s) for s in args.seeds.split(',')):
            train_one(name, seed, torch.device(args.device))


if __name__ == '__main__':
    main()
