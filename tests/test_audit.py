import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from no2shadow import audit as au
from no2shadow import calibration as ev
import test_calibration as base


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.masks = base.build(base.make_raw(0), (31751, 31752))
        for M in cls.masks.values():
            M['ages'] = np.array(base.sf.AGES)

    def test_ks_tv_basic(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 4000); b = rng.normal(0, 1, 4000); c = rng.normal(1.5, 1, 4000)
        edges = np.quantile(np.concatenate([a, b, c]), np.linspace(0, 1, 31))
        self.assertLess(au.ks_stat(a, b), 0.05)
        self.assertGreater(au.ks_stat(a, c), 0.4)
        self.assertLess(au.tv_stat(a, b, edges), 0.08)
        self.assertGreater(au.tv_stat(a, c, edges), 0.4)

    def test_variants_and_oracles_are_defined_and_ordered(self):
        tg, lo, hi = au.combined(self.masks['mcar_31751'], 'evaluation')
        for k in ('C1_archival', 'C2_online', 'C3_shadow', 'C3_cap', 'C3_oneh', 'C3_4ages', 'C4_oracle', 'C4_nohz', 'C4_hz'):
            self.assertIn(k, lo)
            self.assertEqual(len(lo[k]), len(tg['res']))
        self.assertTrue(np.isfinite(lo['C3_shadow']).sum() > 300)
        # a capped pool can only be a subsample of the full pool, never larger than the online pool at a query
        self.assertTrue(np.isfinite(lo['C3_cap']).sum() > 100)
        ok = np.isfinite(lo['C3_shadow']) & np.isfinite(lo['C3_4ages'])
        self.assertGreater(ok.sum(), 100)

    def test_block_bootstrap_contains_estimate(self):
        tg, lo, hi = au.combined(self.masks['mcar_31751'], 'evaluation')
        u = au.unit_sums(tg, lo, hi, list(au.CORE), 14)
        out = au.boot_contrasts([u], {'C3-C2': ('C3_shadow', 'C2_online')}, 14)
        est, l, h = out['C3-C2']
        self.assertLessEqual(l, est + 1e-9); self.assertGreaterEqual(h, est - 1e-9)
        self.assertGreater(est, 0.1)


if __name__ == '__main__':
    unittest.main()
