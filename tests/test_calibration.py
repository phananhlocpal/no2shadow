import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from no2shadow import forecast as sf
from no2shadow import calibration as ev

N, C = 24 * 160, 6
SPLITS = [('fit', 480, 1920), ('tune', 1920, 2880), ('evaluation', 2880, N)]


def persistence(x, m):
    """Fake forecaster: repeat the last visible standardized value (error grows with staleness)."""
    pos = np.arange(x.shape[1])[None, :, None]
    last = np.where(m > 0, pos, -1).max(axis=1)                       # (N,C)
    val = np.take_along_axis(x, np.clip(last, 0, None)[:, None, :], axis=1)[:, 0, :]
    val = np.where(last >= 0, val, 0.0)
    p = np.repeat(val[:, None, :], sf.HMAX, axis=1)
    return np.stack((p - 1, p, p + 1), axis=-1).astype(np.float32)


def make_raw(seed=0, shift_after=None):
    rng = np.random.default_rng(seed)
    raw = (np.cumsum(rng.normal(0, 1, (N, C)), axis=0) * 0.3).astype(np.float32)
    if shift_after is not None:
        raw[shift_after:] += 1000.0
    return raw


def build(raw, mask_seeds=(31751, 31752), beta=0.0):
    T, S, O = sf.origins_for(SPLITS)
    out = {}
    for ms in mask_seeds:
        offline = sf.make_events_mnar(raw, 480, 72, ms, beta)
        res = sf.process_mask(None, raw, np.zeros(C, np.float32), np.ones(C, np.float32), T, offline, None, fc=persistence)
        out[f'mcar_{ms}'] = {'T': T, 'split': S, 'offset': O, 't0_ns': np.int64(0), **res}
    return out


class ShadowForecastTests(unittest.TestCase):
    def test_shadow_state_matches_global_state(self):
        rng = np.random.default_rng(1)
        vis = rng.random((400, 3)) > 0.3
        t = np.array([300])
        a, s = 36, 1
        hid = vis.copy(); hid[t[0] - a:t[0], s] = False
        ctx, age = sf.state_arrays(hid, t)
        win = hid[t[0] - sf.WIN:t[0], s][None, :]
        sage, sctx = sf.shadow_state(win)
        self.assertEqual(int(age[0, s]), int(sage[0]))
        self.assertEqual(int(ctx[0, s]), int(sctx[0]))
        self.assertGreaterEqual(int(sage[0]), a)

    def test_mnar_rate_and_level_selection(self):
        raw = make_raw(3)
        base = sf.make_events_mnar(raw, 480, 24, 5, 0.0)
        mnar = sf.make_events_mnar(raw, 480, 24, 5, 2.0)
        self.assertAlmostEqual(base[480:].mean(), 0.10, delta=0.03)
        self.assertAlmostEqual(mnar[480:].mean(), 0.10, delta=0.04)
        z = (raw - raw.mean(0)) / raw.std(0)
        self.assertGreater(z[480:][mnar[480:]].mean(), z[480:][base[480:]].mean() + 0.3)

    def test_origins_cover_four_offsets_without_cross_split_targets(self):
        T, S, O = sf.origins_for(SPLITS)
        self.assertEqual(set(O.tolist()), {0, 1, 2, 3})
        for code, (name, lo, hi) in enumerate(SPLITS):
            t = T[S == code]
            self.assertTrue((t >= lo).all() and (t + 24 <= hi).all())

    def test_shadow_rows_only_for_online_stations_and_age_at_least_hidden(self):
        masks = build(make_raw(0), (31751,))
        M = masks['mcar_31751']
        self.assertTrue((M['off'][M['sh_o'], M['sh_s'].astype(int)] == 0).all())
        hidden = np.array(sf.AGES)[M['sh_a']]
        self.assertTrue((M['sh_age'] >= hidden).all())


class ShadowEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.masks = build(make_raw(0), (31751, 31752, 31753))

    def run_one(self, masks, name='mcar_31751', period='evaluation', mode='age_hour'):
        others = [v for k, v in masks.items() if k != name]
        return ev.run_mask(masks[name], others, period, 'mcar', mode)

    def test_sources_rank_as_expected_on_staleness_driven_errors(self):
        tg, lo, hi, block, proxy = self.run_one(self.masks)
        common = np.logical_and.reduce([np.isfinite(lo[c]) for c in ev.COMMON])
        self.assertGreater(common.sum(), 300)
        cov = {c: ev.metrics(tg['res'], lo[c], hi[c])[1][common].mean() for c in ev.COMMON}
        self.assertLess(cov['C2_online'], 0.6)                       # fresh-context residuals miss stale-context targets
        self.assertGreater(cov['C3_shadow'], 0.7, cov)
        self.assertLess(cov['C3_shadow'], 0.9, cov)
        self.assertGreater(cov['C4_oracle'], 0.5, cov)               # ~40 synthetic days: cross-fit oracle is noisy
        self.assertGreater(cov['C4_oracle'], cov['C2_online'] + 0.2, cov)
        self.assertGreater(cov['C3_shadow'], cov['C2_online'] + 0.15)
        n_p, hit_p = zip(*proxy.values())
        self.assertAlmostEqual(sum(hit_p) / sum(n_p), cov['C3_shadow'], delta=0.08)   # transport gap small when exchangeable

    def test_delta_levels_widen_intervals(self):
        tg, lo, hi, _, _ = self.run_one(self.masks)
        ok = np.isfinite(lo['C3_shadow']) & np.isfinite(lo['C3_shadow_d10'])
        w0, w10 = (hi['C3_shadow'] - lo['C3_shadow'])[ok], (hi['C3_shadow_d10'] - lo['C3_shadow_d10'])[ok]
        self.assertTrue((w10 >= w0 - 1e-9).all() and w10.mean() > w0.mean())

    def test_no_future_information_in_shadow_intervals(self):
        cut = 3300
        clean = build(make_raw(0), (31751, 31752, 31753))
        dirty = build(make_raw(0, shift_after=cut), (31751, 31752, 31753))
        a = self.run_one(clean)
        b = self.run_one(dirty)
        ta, tb = a[0], b[0]
        early = (ta['time'] + 1 < cut) & (ta['origin'] + 6 < cut)
        self.assertGreater(early.sum(), 100)
        same = np.array_equal(ta['origin'], tb['origin']) and np.array_equal(ta['ab'], tb['ab'])
        self.assertTrue(same)
        for c in ('C2_online', 'C3_shadow', 'C3x_crossmask'):
            x, y = a[1][c][early], b[1][c][early]
            self.assertTrue(np.allclose(x, y, equal_nan=True), c)

    def test_bootstrap_ci_orders_c3_above_c1_when_gap_large(self):
        tg, lo, hi, block, _ = self.run_one(self.masks)
        unit = ev.unit_blocks(tg, lo, hi, block)
        out = ev.bootstrap([unit], len(unit['n']))
        self.assertIn('cov_C3_minus_C1', out)
        self.assertLessEqual(out['cov_C3_minus_C1']['ci95'][0], out['cov_C3_minus_C1']['estimate'])

    def test_hour_conditioned_cells_equalize_coverage_across_origin_hours(self):
        n = 24 * 400
        splits = [('fit', 480, 4800), ('tune', 4800, 7200), ('evaluation', 7200, n)]
        rng = np.random.default_rng(5)
        sd = np.where(np.arange(n) % 24 < 12, 0.2, 1.6)[:, None]          # volatility depends on hour of day
        raw = np.cumsum(rng.normal(0, 1, (n, C)) * sd, axis=0).astype(np.float32)
        T, S, O = sf.origins_for(splits)
        masks = {}
        for ms in (31751, 31752, 31753):
            off = sf.make_events_mnar(raw, 480, 72, ms, 0.0)
            res = sf.process_mask(None, raw, np.zeros(C, np.float32), np.ones(C, np.float32), T, off, None, fc=persistence)
            masks[f'mcar_{ms}'] = {'T': T, 'split': S, 'offset': O, 't0_ns': np.int64(0), **res}
        spread = {}
        for mode in ('bucket', 'age_hour'):
            tg, lo, hi, _, _ = self.run_one(masks, mode=mode)
            ok = np.isfinite(lo['C3_shadow'])
            cov = ev.metrics(tg['res'], lo['C3_shadow'], hi['C3_shadow'])[1]
            offs = masks['mcar_31751']['offset'][tg['o']]
            per = [cov[ok & (offs == k)].mean() for k in range(4)]
            spread[mode] = max(per) - min(per)
        self.assertLess(spread['age_hour'], spread['bucket'] - 0.03, spread)

    def test_hier_uses_finest_supported_level_and_reports_shares(self):
        res = {m: self.run_one(self.masks, mode=m) for m in ev.MODES}
        tg, lo, hi, block, proxy, share = ev.combine_hier(res)
        for c in ev.COMMON:
            fine = res['age_hour'][1][c]
            ok = np.isfinite(fine)
            self.assertTrue(np.array_equal(lo[c][ok], fine[ok]))             # finest level wins when supported
            self.assertAlmostEqual(sum(share[c]), 1.0)
            self.assertAlmostEqual(share[c][0], ok.mean())


class HoldoutTests(unittest.TestCase):
    SPL = [('fit', 480, 1920), ('tune', 1920, 2880), ('evaluation', 2880, 3300), ('holdout', 3300, N)]

    def make(self, shift_after=None):
        raw = make_raw(0, shift_after)
        T, S, O = sf.origins_for(self.SPL)
        masks = {}
        for ms in (31756, 31757, 31758):
            off = sf.make_events_mnar(raw, 480, 72, ms, 0.0)
            res = sf.process_mask(None, raw, np.zeros(C, np.float32), np.ones(C, np.float32), T, off, None, fc=persistence)
            masks[f'mcar_{ms}'] = {'T': T, 'split': S, 'offset': O, 't0_ns': np.int64(0), **res}
        return masks

    def test_holdout_targets_stay_inside_period_and_no_future_leak(self):
        cut = 3600
        a, b = self.make(), self.make(shift_after=cut)
        T, S, _ = sf.origins_for(self.SPL)
        t = T[S == 3]
        self.assertTrue((t >= 3300).all() and (t + 24 <= N).all())
        outs = []
        for masks in (a, b):
            others = [v for k, v in masks.items() if k != 'mcar_31756']
            outs.append(ev.run_mask(masks['mcar_31756'], others, 'holdout', 'mcar', 'age_hour'))
        ta, tb = outs[0][0], outs[1][0]
        self.assertTrue((ta['origin'] >= 3299).all())
        early = (ta['time'] + 1 < cut) & (ta['origin'] + 6 < cut)
        self.assertGreater(early.sum(), 50)
        for c in ('C1_archival', 'C2_online', 'C3_shadow'):
            self.assertTrue(np.allclose(outs[0][1][c][early], outs[1][1][c][early], equal_nan=True), c)


if __name__ == '__main__':
    unittest.main()
