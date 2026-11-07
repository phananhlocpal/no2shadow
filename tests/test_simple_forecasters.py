import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from no2shadow import forecast as sf
from no2shadow import simple_forecasters as ss


class SimpleForecasterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(3, 96, 4)).astype(np.float32)
        self.m = (rng.random((3, 96, 4)) > 0.3).astype(np.float32)
        self.x *= self.m
        self.mean, self.std = np.full(4, 10, np.float32), np.full(4, 2, np.float32)

    def test_persistence_repeats_last_visible_value(self):
        q = ss.make_fc('persistence', self.mean, self.std)(self.x, self.m)
        self.assertEqual(q.shape, (3, 6, 4, 3))
        for b in range(3):
            for c in range(4):
                idx = np.flatnonzero(self.m[b, :, c])
                expect = self.x[b, idx[-1], c] * 2 + 10 if len(idx) else 10.0
                self.assertTrue(np.allclose(q[b, :, c, 1], expect, atol=1e-5))

    def test_seasonal_uses_same_clock_hour_of_previous_days(self):
        q = ss.make_fc('seasonal', self.mean, self.std)(self.x, self.m)
        b, c, h = 1, 2, 3
        pos = [96 - 24 * k + h for k in (1, 2, 3, 4)]
        vals = [self.x[b, p, c] for p in pos if self.m[b, p, c] > 0]
        if vals:
            self.assertAlmostEqual(float(q[b, h, c, 1]), float(np.mean(vals) * 2 + 10), places=4)

    def test_runs_through_shadow_pipeline_and_hides_history(self):
        n, c = 24 * 60, 4
        rng = np.random.default_rng(1)
        raw = (20 + np.cumsum(rng.normal(0, 1, (n, c)), axis=0)).astype(np.float32)
        T, S, O = sf.origins_for([('fit', 480, 960), ('tune', 960, 1200), ('evaluation', 1200, n)])
        off = sf.make_events_mnar(raw, 480, 72, 5, 0.0)
        fc = ss.make_fc('persistence', self.mean[:c], self.std[:c])
        res = sf.process_mask(None, raw, self.mean[:c], self.std[:c], T, off, None, fc=fc, ages=(6, 48))
        self.assertGreater(len(res['sh_o']), 100)
        self.assertTrue((res['sh_age'] >= np.array((6, 48))[res['sh_a']]).all())
        self.assertTrue(np.isfinite(res['sh_pred']).all())


if __name__ == '__main__':
    unittest.main()
