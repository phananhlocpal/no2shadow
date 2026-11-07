import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from no2shadow import data
from no2shadow.outages import make_events


class DataTests(unittest.TestCase):
    def test_monitor_selection_uses_completeness_and_lowest_poc(self):
        idx = pd.date_range('2023-01-01', periods=100, freq='h')
        panel = pd.DataFrame({'06-037-0001-1': 1.0, '06-037-0001-3': 1.0, '06-037-0002-2': 1.0, '06-037-0003-1': np.nan}, index=idx)
        panel.iloc[:30, 2] = np.nan                                  # 70% observed: not eligible
        self.assertEqual(data.select_monitors(panel), ['06-037-0001-1'])

    def test_outage_schedule_is_deterministic_and_starts_at_begin(self):
        a = make_events(500, list('abcd'), 100, 72, seed=31751)
        b = make_events(500, list('abcd'), 100, 72, seed=31751)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(a[:100].any())
        self.assertTrue(a.any())


if __name__ == '__main__':
    unittest.main()
