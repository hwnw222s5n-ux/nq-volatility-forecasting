"""Regression checks for timing, model definitions, source data and local price updates."""

import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean, stdev

from research import (BENCHMARK_WINDOW, DECAY, SHORT_WINDOW, Observation, SAMPLE_NAMES,
                      VolatilityModels, latest_from_csv, load_returns, qlike_gap, score, walk_forward)


class ForecastTests(unittest.TestCase):
    def test_estimates_match_definitions(self):
        series = [(i % 7 - 3) / 200 for i in range(20)]
        models = VolatilityModels()
        for r in series:
            models.observe(r)
        estimates = models.predict()
        if estimates[2] is None:
            self.fail("All three forecasts should be available after 20 returns.")
        short, benchmark, weighted = estimates
        self.assertAlmostEqual(short, stdev(series[-5:]))
        self.assertAlmostEqual(benchmark, stdev(series))
        self.assertAlmostEqual(weighted**2, fmean(r*r for r in series))
        models.observe(-.08)
        updated = models.predict()[2]
        if updated is None:
            self.fail("EWMA should remain available after another return.")
        self.assertAlmostEqual(updated**2, DECAY*weighted**2 + (1-DECAY)*.08**2)

    def test_warmup_distinguishes_flat_prices_from_missing_history(self):
        model = VolatilityModels()
        for _ in range(SHORT_WINDOW - 1):
            model.observe(0.0)
        self.assertEqual(model.predict(), (None, None, None))
        model.observe(0.0)
        self.assertEqual(model.predict(), (0.0, None, None))
        for _ in range(BENCHMARK_WINDOW - SHORT_WINDOW - 1):
            model.observe(0.0)
        self.assertEqual(model.predict(), (0.0, None, None))
        model.observe(0.0)
        self.assertEqual(model.predict(), (0.0, 0.0, 0.0))

    def test_revealing_a_new_return_only_changes_later_predictions(self):
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(25)]
        observations = [Observation(left, right, .01) for left, right in zip(dates, dates[1:])]
        original = walk_forward(observations)
        changed_index = 21
        observations[changed_index] = observations[changed_index]._replace(actual=-.25)
        changed = walk_forward(observations)
        self.assertEqual(original[0], changed[0])
        self.assertEqual(original[1].sigmas, changed[1].sigmas)
        self.assertEqual(changed[1].actual, -.25)
        self.assertNotEqual(original[2].sigmas, changed[2].sigmas)

    def test_calendar_filter_restarts_model_history(self):
        observations, _ = load_returns()
        result = walk_forward(observations, outside_expiry=True)
        eligible = [r for r in observations if date(2024, 4, 1) <= r.cutoff < date(2024, 6, 1)]
        first_after_march = next(r for r in result if r.target >= date(2024, 4, 1))
        self.assertEqual(first_after_march.cutoff, eligible[19].target)
        self.assertAlmostEqual(first_after_march.sigmas[1], stdev(r.actual for r in eligible[:20]))
        self.assertAlmostEqual(first_after_march.sigmas[2]**2, fmean(r.actual**2 for r in eligible[:20]))

    def test_frozen_snapshot_and_scores(self):
        observations, source = load_returns()
        self.assertEqual(len(observations), 2492)
        self.assertTrue(source["matches_reference_closes"])
        forecasts = walk_forward(observations)
        self.assertEqual((forecasts[0].cutoff, forecasts[0].target),
                         (date(2023, 12, 29), date(2024, 1, 2)))
        self.assertEqual(forecasts[-1].target, date(2025, 11, 28))
        scores = score(forecasts, SAMPLE_NAMES[0])
        self.assertEqual([r["sessions"] for r in scores], [482, 482, 482])
        self.assertEqual([r["inside"] for r in scores], [309, 339, 350])
        self.assertAlmostEqual(scores[2]["qlike_gap_vs_benchmark"], -.06459, places=5)
        self.assertEqual(qlike_gap(-.03, .01, .01), 0)
        self.assertTrue(math.isfinite(qlike_gap(0, 0, .01)))

    def test_local_csv_produces_a_short_forecast_and_catches_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "closes.csv"
            dates = ["2025-12-01", "2025-12-02", "2025-12-03",
                     "2025-12-04", "2025-12-05", "2025-12-08"]
            closes = [100, 101, 98, 98, 101, 102]
            text = "date,close\n" + "".join(f"{d},{c}\n" for d, c in zip(dates, closes))
            path.write_text(text, encoding="utf-8-sig")
            cutoff, close, sigmas = latest_from_csv(path)
            self.assertEqual((cutoff, close), (date(2025, 12, 8), 102))
            self.assertAlmostEqual(sigmas[0], stdev((b-a)/a for a, b in zip(closes, closes[1:])))
            self.assertEqual(sigmas[1:], (None, None))
            path.write_text(text + "2025-12-08,103\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                latest_from_csv(path)

    def test_local_csv_needs_enough_closes_before_forecasting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "closes.csv"
            for body in ("", "2025-12-01,100\n"):
                with self.subTest(body=body):
                    path.write_text("date,close\n" + body, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "consecutive session closes"):
                        latest_from_csv(path)


if __name__ == "__main__":
    unittest.main()
