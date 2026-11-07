import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from no2shadow import forecast as sf
from no2shadow import calibration as ev
from no2shadow import updaters as up
import test_calibration as base


def mask_dict(raw, seed=31751):
    M = base.build(raw, (seed,))[f'mcar_{seed}']
    M['ages'] = np.array(sf.AGES)
    return M


class UpdaterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = mask_dict(base.make_raw(0))

    def test_streams_use_delay_and_correct_sources(self):
        M = self.M
        for source in up.SOURCES:
            cell, t, arrival, res, day, site = up.streams(M, source)
            self.assertTrue((arrival - t == up.DELAY).all())
            self.assertTrue(len(res) > 1000 and np.isfinite(res).all())
        cell_o = up.streams(M, 'online')[0]
        cell_s = up.streams(M, 'shadow')[0]
        self.assertTrue((cell_o % 4 == 0).all() and cell_o.max() <= 12)       # origin hour only
        self.assertTrue(cell_s.max() <= 15 and len(np.unique(cell_s)) > 4)    # age bucket x hour
        keep = np.isin(M['ages'][M['sh_a']], up.SUB_AGES)
        self.assertTrue(keep.sum() < len(keep))                                # subsampled ages

    def test_radii_ignore_unarrived_scores_for_all_methods(self):
        n = 24 * 160
        clean, dirty = mask_dict(base.make_raw(0)), mask_dict(base.make_raw(0, shift_after=3300))
        tg = ev.targets_for(clean, 'evaluation', 'bucket')
        queries = np.unique(tg['origin'])
        queries = queries[queries < 3300]
        lock = tg['start']
        for method in ('NEx', 'SAOCP', 'FACI'):
            outs = []
            for M in (clean, dirty):
                cell, t, arrival, res, day, site = up.streams(M, 'shadow')
                m = cell == 5
                outs.append(up.series_radii(method, arrival[m], res[m], queries, lock))
            same = [np.isclose(outs[0][q], outs[1][q], equal_nan=True) for q in queries]
            self.assertTrue(all(same), method)
            self.assertTrue(np.isfinite(list(outs[0].values())).any(), method)

    def test_end_to_end_tasks_cover_roughly_at_nominal_for_shadow(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'm.npz'
            M = self.M
            np.savez_compressed(path, **M)
            for method in ('ROLL_ABS', 'NEx', 'SAOCP', 'FACI'):
                rows = up.run_task((str(path), 'synthetic', 1, 'mcar_31751', method, 'shadow'))
                allrow = [r for r in rows if r['group_type'] == 'all'][0]
                self.assertGreater(allrow['n'] / allrow['targets'], 0.3, (method, allrow))
                self.assertGreater(allrow['coverage'], 0.5, (method, allrow))
                self.assertLess(allrow['coverage'], 0.97, (method, allrow))
            on = [r for r in up.run_task((str(path), 'synthetic', 1, 'mcar_31751', 'ROLL_ABS', 'online')) if r['group_type'] == 'all'][0]
            sh = [r for r in up.run_task((str(path), 'synthetic', 1, 'mcar_31751', 'ROLL_ABS', 'shadow')) if r['group_type'] == 'all'][0]
            self.assertGreater(sh['coverage'], on['coverage'] + 0.1)           # stale-context errors need the shadow stream


if __name__ == '__main__':
    unittest.main()
