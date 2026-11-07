"""Download and prepare the three hourly NO2 panels, and load them with fixed chronological splits.

Beijing (12 stations, ug/m3) and London (KDD Cup 2018, ug/m3) come from public archives listed in
configs/data_sources.json. Los Angeles (EPA AQS hourly NO2, ppb, GMT) is fetched per year; monitors are selected
from 2023 only (at least 80% observed, lowest POC per site) and kept fixed for 2024 and 2025.
Missing values are kept as NaN; nothing is imputed.

Usage:
    python -m no2shadow.data            # Beijing, London, EPA 2023 and 2024
    python -m no2shadow.data --holdout  # additionally EPA 2025 (held-out year)
"""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
SOURCES = json.loads((ROOT / 'configs' / 'data_sources.json').read_text())
DATASETS = ('beijing_multisite', 'kdd2018_london', 'epa_la')
EPA_URL = 'https://aqs.epa.gov/aqsweb/airdata/hourly_42602_{year}.zip'
EPA_COLUMNS = ['State Code', 'County Code', 'Site Num', 'POC', 'Date GMT', 'Time GMT', 'Sample Measurement', 'Units of Measure']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expected_sha=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print('downloading', url, flush=True)
        tmp = dest.with_suffix(dest.suffix + '.part')
        with urllib.request.urlopen(url, timeout=300) as response, tmp.open('wb') as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.replace(dest)
    if expected_sha and sha256(dest) != expected_sha:
        raise ValueError(f'SHA-256 mismatch for {dest}')
    return dest


# ------------------------------------------------------------------ Beijing and London
def beijing_panel():
    outer = zipfile.ZipFile(download(SOURCES['beijing']['url'], RAW / 'beijing.zip', SOURCES['beijing']['sha256']))
    nested = next(n for n in outer.namelist() if n.endswith('.zip'))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(nested)))
    panels = []
    for name in sorted(n for n in inner.namelist() if n.endswith('.csv')):
        x = pd.read_csv(inner.open(name))
        index = pd.to_datetime(x[['year', 'month', 'day', 'hour']])
        panels.append(pd.Series(pd.to_numeric(x['NO2'], errors='coerce').values, index=index, name=str(x.station.iloc[0])))
    if len(panels) != 12:
        raise ValueError(f'expected 12 Beijing stations, got {len(panels)}')
    return pd.concat(panels, axis=1)


def london_panel():
    z = zipfile.ZipFile(download(SOURCES['kdd2018']['url'], RAW / 'kdd.zip', SOURCES['kdd2018']['sha256']))
    name = next(n for n in z.namelist() if n.endswith(('.tsf', '.ts')))
    attributes, series, in_data = [], [], False
    for line in z.open(name):
        line = line.decode('cp1252').strip()
        if not line or line.startswith('#'):
            continue
        if line.lower().startswith('@attribute '):
            attributes.append(line.split()[1])
        elif line.lower() == '@data':
            in_data = True
        elif in_data:
            fields = line.split(':')
            if len(fields) != len(attributes) + 1:
                raise ValueError('unsupported TSF row schema')
            meta = dict(zip(attributes, fields[:-1]))
            desc = ' '.join(str(v) for v in meta.values()).lower()
            if 'london' not in desc or 'no2' not in desc.replace('_', '').replace('.', ''):
                continue
            dates = [v for v in fields[:-1] if len(v) >= 10 and v[:4].isdigit() and v[4] == '-']
            start = pd.to_datetime(dates[0], format='%Y-%m-%d %H-%M-%S') if dates else pd.Timestamp('2017-01-01')
            values = pd.to_numeric(pd.Series(fields[-1].split(',')).replace('?', np.nan), errors='coerce').to_numpy()
            key = '_'.join(f'{k}={v}' for k, v in meta.items() if k != 'start_timestamp')
            series.append(pd.Series(values, index=pd.date_range(start, periods=len(values), freq='h'), name=key))
    if not series:
        raise ValueError('no London NO2 series found')
    return pd.concat(series, axis=1)


def write_panel(name, panel):
    if panel.index.has_duplicates or panel.columns.has_duplicates:
        raise ValueError(f'{name}: duplicate timestamps or stations')
    panel = panel.sort_index().asfreq('h').astype(float)
    panel.index.name = 'date'
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / f'{name}_observed.csv')
    print(name, 'rows', len(panel), 'stations', panel.shape[1], 'missing', round(float(panel.isna().mean().mean()), 3), flush=True)


# ------------------------------------------------------------------ EPA Los Angeles
def epa_year(year):
    archive = download(EPA_URL.format(year=year), RAW / f'hourly_42602_{year}.zip')
    parts = []
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
        if len(names) != 1:
            raise ValueError('unexpected EPA archive')
        with z.open(names[0]) as source:
            for chunk in pd.read_csv(source, usecols=EPA_COLUMNS, chunksize=150000,
                                     dtype={'State Code': str, 'County Code': str, 'Site Num': str, 'POC': str}):
                x = chunk[(chunk['State Code'] == '06') & (chunk['County Code'] == '037')].copy()
                if not x.empty:
                    parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    if set(x['Units of Measure'].dropna().unique()) != {'Parts per billion'}:
        raise ValueError('unexpected units')
    x['date'] = pd.to_datetime(x['Date GMT'] + ' ' + x['Time GMT'], errors='raise')
    x['monitor'] = x['State Code'] + '-' + x['County Code'] + '-' + x['Site Num'] + '-' + x['POC']
    x['value'] = pd.to_numeric(x['Sample Measurement'], errors='coerce')
    if x.duplicated(['date', 'monitor']).any():
        raise ValueError('duplicate monitor-hour rows')
    index = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
    return x.pivot(index='date', columns='monitor', values='value').reindex(index)


def select_monitors(panel_2023):
    """At least 80% observed in 2023; one monitor (lowest POC) per site."""
    counts = panel_2023.notna().sum()
    eligible = sorted(counts[counts >= int(.8 * len(panel_2023))].index)
    chosen = {}
    for monitor in eligible:
        site, poc = monitor.rsplit('-', 1)
        if site not in chosen or int(poc) < int(chosen[site].rsplit('-', 1)[1]):
            chosen[site] = monitor
    return sorted(chosen.values())


def prepare_epa(years):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    monitors_file = PROCESSED / 'epa_la_monitors.json'
    if not monitors_file.exists():
        monitors = select_monitors(epa_year(2023))
        monitors_file.write_text(json.dumps({'rule': 'at least 80% observed in 2023, lowest POC per site', 'monitors': monitors}, indent=2))
    monitors = json.loads(monitors_file.read_text())['monitors']
    for year in years:
        panel = epa_year(year).reindex(columns=monitors)
        panel.rename_axis('date').to_csv(PROCESSED / f'epa_la_{year}_ppb.csv.gz', compression='gzip')
        print('epa_la', year, 'monitors', len(monitors), 'missing', round(float(panel.isna().mean().mean()), 3), flush=True)


# ------------------------------------------------------------------ loading with chronological splits
def load_dataset(name, holdout=False):
    """Returns dates, stations, values (hours x stations, NaN = missing) and split boundaries
    train_end, val_end (model training/validation), fit_end, tune_end (calibration history, tuning, then evaluation).
    Beijing/London: 40% train, 10% validation, then fit to 65%, tune to 80%, last 20% evaluation.
    Los Angeles: train 2023, validation 2024-01..06, fit 2024-07..08, tune 2024-09, evaluation 2024-10..12;
    with holdout=True the 2025 year is appended after the evaluation period."""
    if name == 'epa_la':
        monitors = json.loads((PROCESSED / 'epa_la_monitors.json').read_text())['monitors']
        years = (2023, 2024, 2025) if holdout else (2023, 2024)
        df = pd.concat([pd.read_csv(PROCESSED / f'epa_la_{y}_ppb.csv.gz', parse_dates=['date']).set_index('date')[monitors] for y in years])
        if not df.index.equals(pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31 23:00', freq='h')):
            raise ValueError('invalid EPA timeline')
        stations = list(monitors)
        dates = pd.Series(df.index)
        raw = df.to_numpy(dtype=np.float32)
        train_end = int(df.index.get_loc(pd.Timestamp('2024-01-01')))
        val_end = int(df.index.get_loc(pd.Timestamp('2024-07-01')))
        fit_end = int(df.index.get_loc(pd.Timestamp('2024-09-01')))
        tune_end = int(df.index.get_loc(pd.Timestamp('2024-10-01')))
    else:
        if holdout:
            raise ValueError('holdout data exist only for epa_la')
        df = pd.read_csv(PROCESSED / f'{name}_observed.csv', parse_dates=['date'])
        dates, stations = df.date, list(df.columns[1:])
        raw = df[stations].to_numpy(dtype=np.float32)
        n = len(raw)
        train_end, val_end = int(n * .4), int(n * .5)
        fit_end = val_end + 24 * (int(n * .65 - val_end) // 24)
        tune_end = n - int(n * .2)
    if not (dates.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError('timeline is not strictly hourly')
    return dates, stations, raw, train_end, val_end, fit_end, tune_end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', action='store_true', help='also prepare EPA Los Angeles 2025')
    args = parser.parse_args()
    write_panel('beijing_multisite', beijing_panel())
    write_panel('kdd2018_london', london_panel())
    prepare_epa((2023, 2024, 2025) if args.holdout else (2023, 2024))


if __name__ == '__main__':
    main()
