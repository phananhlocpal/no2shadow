"""Summarise the outage-mechanism sensitivity run (MCAR vs past-level selection vs shared-cause outages).

Reads calibration.py outputs and the stored forecasts of the same run; writes scen_summary.csv and SCEN_REPORT.md.

Usage:
    python -m no2shadow.scenarios   (AQ_SCEN_EVAL, AQ_SCEN_SRC, AQ_SCEN_PERIOD)
"""
import csv, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / os.environ.get('AQ_SCEN_EVAL', 'results/calibration_scenarios')
SRC = ROOT / os.environ.get('AQ_SCEN_SRC', 'results/forecasts_scenarios')
PERIOD = os.environ.get('AQ_SCEN_PERIOD', 'evaluation')
CORE = ('C1_archival', 'C2_online', 'C3_shadow', 'C4_oracle')
ORDER = ('mcar', 'pre1', 'pre2', 'corr60', 'corr90')


def rows(name):
    return list(csv.DictReader((EVAL / name).open()))


def wavg(sel, key):
    n = sum(float(x['n']) for x in sel)
    return (sum(float(x[key]) * float(x['n']) for x in sel) / n) if n else float('nan'), n


def network_stats(ds):
    """Mean share of stations online at the origins of real outage targets, and shadow rows per origin, per kind."""
    out = {}
    for kind in ORDER:
        fr, w, sh, offs, cells = 0.0, 0.0, 0, 0.0, 0
        for p in sorted((SRC / ds).glob(f'seed_*/{kind}_*.npz')):
            with np.load(p) as z:
                code = {'tune': 1, 'evaluation': 2}[PERIOD]
                sel = z['split'] == code
                off = z['off'][sel]                                   # (origins, stations) offline at origin
                n_off = off.sum(axis=1)                               # offline stations per origin
                S = off.shape[1]
                offs += float(off.sum()); cells += off.size
                fr += float((n_off * (1 - n_off / S)).sum()); w += float(n_off.sum())
                sh += int(np.isin(z['sh_o'], np.flatnonzero(sel)).sum())
        if w:
            out[kind] = (fr / w, sh, offs / cells)
    return out


def main():
    S = [x for x in rows('calibration_summary.csv') if x['period'] == PERIOD and x['cells'] == 'hier']
    L = [x for x in rows('calibration_levels.csv') if x['period'] == PERIOD]
    T = [x for x in rows('calibration_transport.csv') if x['period'] == PERIOD and x['cells'] == 'bucket']
    out = []
    for ds in sorted({x['dataset'] for x in S}):
        net = network_stats(ds)
        for kind in ORDER:
            sel = lambda c, sub: [x for x in S if x['dataset'] == ds and x['kind'] == kind and x['group_type'] == 'all' and x['subset'] == sub and x['calibrator'] == c]
            if not sel('C3_shadow', 'common'):
                continue
            r = {'dataset': ds, 'scenario': kind}
            for c in CORE:
                r[c], n = wavg(sel(c, 'common'), 'coverage')
            r['common_targets'] = int(n)
            for c, tag in (('C3_shadow', 'C3_own'), ('C3_shadow_d05', 'C3_d05_own'), ('C3_shadow_d10', 'C3_d10_own')):
                r[tag], _ = wavg(sel(c, 'own'), 'coverage')
            r['C3_miss_below'], _ = wavg(sel('C3_shadow', 'common'), 'miss_below')
            r['C3_miss_above'], _ = wavg(sel('C3_shadow', 'common'), 'miss_above')
            lv = [x for x in L if x['dataset'] == ds and x['kind'] == kind and x['calibrator'] == 'C3_shadow']
            r['C3_unsupported_share'] = float(np.mean([float(x['share_unsupported']) for x in lv])) if lv else float('nan')
            tr = [x for x in T if x['dataset'] == ds and x['kind'] == kind and x['delta_target_minus_proxy'] not in ('', 'nan')]
            w = sum(float(x['target_n']) for x in tr)
            r['transport_delta'] = sum(float(x['delta_target_minus_proxy']) * float(x['target_n']) for x in tr) / w if w else float('nan')
            r['online_share_at_outage_origins'], r['shadow_rows'], r['offline_rate'] = net.get(kind, (float('nan'), 0, float('nan')))
            out.append(r)
    cols = list(dict.fromkeys(k for r in out for k in r))
    with (EVAL / 'scen_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(out)
    lines = ['# Outage-mechanism sensitivity', '',
             '| dataset | scenario | C1 | C2 | C3 | C4 | C3 d=0.05 | C3 d=0.10 | transport delta | offline rate | online share at outage origins | C3 unsupported |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in out:
        lines.append(f"| {r['dataset']} | {r['scenario']} | {r['C1_archival']:.3f} | {r['C2_online']:.3f} | {r['C3_shadow']:.3f} | {r['C4_oracle']:.3f} | "
                     f"{r['C3_d05_own']:.3f} | {r['C3_d10_own']:.3f} | {r['transport_delta']:+.3f} | {r['offline_rate']:.3f} | {r['online_share_at_outage_origins']:.2f} | {r['C3_unsupported_share']:.3f} |")
    (EVAL / 'SCEN_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
