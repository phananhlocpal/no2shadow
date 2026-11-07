import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from no2shadow import forecast as sf
from no2shadow import outages as oc


def level_series(n=24 * 900, c=8, seed=0):
    """Persistent AR(1) level per station so that past level predicts future level."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, c))
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + rng.normal(0, 0.3, c)
    return x.astype(np.float32)


class ScenarioTests(unittest.TestCase):
    def test_rates_stay_near_ten_percent(self):
        raw = level_series()
        for off in (oc.make_events_pre(raw, 480, 72, 5, 0.0), oc.make_events_pre(raw, 480, 72, 5, 2.0),
                    oc.make_events_corr(len(raw), raw.shape[1], 480, 72, 5, 0.6), oc.make_events_corr(len(raw), raw.shape[1], 480, 72, 5, 0.9)):
            self.assertTrue(0.18 < off[480:].mean() < 0.42, off[480:].mean())   # 72h windows at 10% daily start rate

    def test_pre_selection_follows_past_level_and_not_future_only(self):
        raw = level_series()
        mu, sd = raw.mean(0), raw.std(0)
        z = (raw - mu) / sd
        def start_levels(off):
            starts = np.argwhere(off[1:] & ~off[:-1]) + [1, 0]
            starts = starts[starts[:, 0] >= 480]
            past = np.mean([z[max(0, t - 24):t, s].mean() for t, s in starts])
            return past
        base = start_levels(oc.make_events_pre(raw, 480, 72, 9, 0.0))
        sel = start_levels(oc.make_events_pre(raw, 480, 72, 9, 2.0))
        self.assertGreater(sel, base + 0.2, (sel, base))

    def test_past_level_ignores_everything_from_the_start_on(self):
        raw = level_series()
        mu, sd = raw.mean(0), raw.std(0)
        starts = list(range(480, 480 + 24 * 20, 24))
        z1 = oc.past_level(raw, starts, 24, mu, sd)
        mod = raw.copy()
        mod[480 + 24 * 10:] += 100.0
        z2 = oc.past_level(mod, starts, 24, mu, sd)
        self.assertTrue(np.allclose(z1[:11], z2[:11]))          # starts up to the change see identical past windows
        self.assertFalse(np.allclose(z1[11:], z2[11:]))

    def test_corr_events_make_simultaneous_outages_more_likely(self):
        raw = level_series()
        c = raw.shape[1]
        ind = oc.make_events_corr(len(raw), c, 480, 72, 3, 0.0, event_prob=0.0)
        cor = oc.make_events_corr(len(raw), c, 480, 72, 3, 0.9)
        v_ind = ind[480:].sum(axis=1).var(); v_cor = cor[480:].sum(axis=1).var()
        self.assertGreater(v_cor, v_ind * 1.2, (v_cor, v_ind))

    def test_mask_list_names(self):
        old = (sf.SCENARIOS, sf.SCEN_SEEDS)
        sf.SCENARIOS, sf.SCEN_SEEDS = ['pre:1', 'pre:2', 'corr:0.6', 'corr:0.9'], (7, 8)
        try:
            names = [m[0] for m in sf.mask_list(2025, False)]
        finally:
            sf.SCENARIOS, sf.SCEN_SEEDS = old
        for n in ('pre1_7', 'pre2_8', 'corr60_7', 'corr90_8'):
            self.assertIn(n, names)


if __name__ == '__main__':
    unittest.main()
