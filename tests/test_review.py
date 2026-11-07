import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from no2shadow import calibration as ev
from no2shadow import review as rv


def rows(n=600, seed=0):
    r = np.random.default_rng(seed)
    day = r.integers(0, 30, n)
    return day, r.integers(0, 5, n), r.normal(0, 1, n)


class ReviewTests(unittest.TestCase):
    def test_uniform_weights_match_plain_interval(self):
        day, site, res = rows()
        w = rv.weighted_interval((day, site, res), 31 * 24, 1e9)
        p = ev.interval((day, site, res))
        self.assertAlmostEqual(w[0], p[0], delta=0.1)
        self.assertAlmostEqual(w[1], p[1], delta=0.1)

    def test_recent_rows_dominate(self):
        day = np.concatenate([np.arange(300) % 8, 22 + np.arange(300) % 8])
        site = np.tile(np.arange(5), 120)
        res = np.concatenate([np.full(300, 10.0) + np.linspace(0, 1, 300), np.linspace(-1, 1, 300)])
        lo, hi = rv.weighted_interval((day, site, res), 30 * 24, 1.0)
        self.assertLess(hi, 2.0)                     # old rows with residual ~10 are down-weighted away
        plain = ev.interval((day, site, res))
        self.assertGreater(plain[1], 5.0)

    def test_support_rule_returns_none(self):
        day, site, res = rows(50)
        self.assertIsNone(rv.weighted_interval((day, site, res), 31 * 24, 7.0))
        self.assertIsNone(rv.weighted_interval(None, 0, 7.0))

    def test_level_groups(self):
        g = rv.level_groups(np.array([0., 5., 9., 10., 11., 20.]), [5., 10., 12.])
        self.assertEqual(list(g), [0, 0, 1, 1, 2, 3])
        m = rv.level_masks(g)
        self.assertEqual(int(m['gt_p90'].sum()), 2)

    def test_tv_hist_bounds(self):
        a = np.random.default_rng(1).normal(0, 1, 2000)
        edges = np.linspace(-4, 4, 21)
        self.assertAlmostEqual(rv.tv_hist(a, a, edges), 0.0)
        self.assertGreater(rv.tv_hist(a, a + 6, edges), 0.9)


if __name__ == '__main__':
    unittest.main()
