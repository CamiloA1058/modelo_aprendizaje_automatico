"""
The run form must start with the thresholds used in the thesis model
(growth > 15 %, drop > 10 %), so a default run reproduces the reported results.
"""

import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "dashboard" / "dashboard.html").read_text(encoding="utf-8")


def _default_value(input_id):
    match = re.search(r'id="%s"\s+value="([^"]*)"' % input_id, HTML)
    return match.group(1) if match else None


class RunFormDefaultsTest(unittest.TestCase):

    def test_growth_threshold_defaults_to_15_percent(self):
        self.assertEqual(_default_value("param-growth"), "15")

    def test_reduce_threshold_defaults_to_10_percent(self):
        self.assertEqual(_default_value("param-reduce"), "10")


if __name__ == "__main__":
    unittest.main()
