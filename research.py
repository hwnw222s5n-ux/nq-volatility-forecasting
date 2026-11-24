"""Run the frozen NQ volatility comparison and rebuild the offline dashboard."""

import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import deque
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from statistics import fmean, quantiles, stdev
from typing import NamedTuple

from get_history import atomic_text, csv_text


ROOT = Path(__file__).resolve().parent
TEST_START = date(2024, 1, 1)
SHORT_WINDOW = 5
BENCHMARK_WINDOW = 20
DECAY = 0.94
MODEL_NAMES = (f"{SHORT_WINDOW}-day rolling", f"{BENCHMARK_WINDOW}-day rolling", "EWMA")
EXPIRY_MONTHS = (3, 6, 9, 12)
SAMPLE_NAMES = ("Full holdout", "Outside expiry months")
VARIANCE_FLOOR = 1e-12

FullSigmas = tuple[float, float, float]
ShortOnlySigmas = tuple[float, None, None]


class Observation(NamedTuple):
    cutoff: date
    target: date
    actual: float


class Forecast(NamedTuple):
    cutoff: date
    target: date
    actual: float
    sigmas: FullSigmas


def load_returns():
    directory = ROOT / "data"
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    content = (directory / "nq_returns.csv").read_text(encoding="utf-8")
    if hashlib.sha256(content.encode()).hexdigest() != source["return_csv_sha256"]:
        raise ValueError("The input changed after import. Rebuild it with python get_history.py.")
    rows: list[Observation] = []
    for row in csv.DictReader(io.StringIO(content)):
        cutoff = date.fromisoformat(row["cutoff_date"])
        target = date.fromisoformat(row["target_date"])
        actual = float(row["actual_return"])
        if not 1 <= (target - cutoff).days <= 4 or (rows and cutoff != rows[-1].target):
            raise ValueError(f"Broken date sequence ending on {target}")
        if not math.isfinite(actual) or actual <= -1:
            raise ValueError(f"Invalid NQ return on {target}")
        rows.append(Observation(cutoff, target, actual))
    if len(rows) != source["returns"]:
        raise ValueError("Return count differs from the source record.")
    return rows, source


class VolatilityModels:
    """State at one close, shared by the historical test and new forecasts."""

    def __init__(self) -> None:
        if not 2 <= SHORT_WINDOW < BENCHMARK_WINDOW or not 0 < DECAY < 1:
            raise ValueError("Choose 2 <= short window < benchmark window and 0 < decay < 1.")
        self.recent: deque[float] = deque(maxlen=BENCHMARK_WINDOW)
        self.ewma_variance: float | None = None

    def observe(self, actual: float) -> None:
        self.recent.append(actual)
        if self.ewma_variance is not None:
            self.ewma_variance = DECAY * self.ewma_variance + (1 - DECAY) * actual**2
        elif len(self.recent) == BENCHMARK_WINDOW:
            self.ewma_variance = fmean(r**2 for r in self.recent)

    def predict(self) -> tuple[None, None, None] | ShortOnlySigmas | FullSigmas:
        if len(self.recent) < SHORT_WINDOW:
            return None, None, None
        short = stdev(list(self.recent)[-SHORT_WINDOW:])
        if self.ewma_variance is None:
            return short, None, None
        return short, stdev(self.recent), math.sqrt(self.ewma_variance)


def walk_forward(observations: Iterable[Observation], *, start: date = TEST_START,
                 outside_expiry: bool = False) -> list[Forecast]:
    """Each estimate uses only returns that ended on or before its cutoff."""
    models = VolatilityModels()
    forecasts: list[Forecast] = []
    for cutoff, target, actual in observations:
        if outside_expiry and (cutoff.month in EXPIRY_MONTHS or target.month in EXPIRY_MONTHS):
            models = VolatilityModels()
            continue

        if target >= start:
            sigmas = models.predict()
            # EWMA becomes available after the same warm-up as the benchmark.
            if sigmas[2] is not None:
                forecasts.append(Forecast(cutoff, target, actual, sigmas))
        # Reveal the target return to the models only after recording the forecasts.
        models.observe(actual)
    return forecasts


def latest_from_csv(path: Path) -> tuple[date, float, ShortOnlySigmas | FullSigmas]:
    models = VolatilityModels()
    previous: tuple[date, float] | None = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"date", "close"}.issubset(rows.fieldnames or []):
            raise ValueError("Use the column names date,close.")
        for line, row in enumerate(rows, start=2):
            try:
                day = date.fromisoformat(row["date"].strip())
                close = float(row["close"])
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid date or close on CSV line {line}.") from error
            if not math.isfinite(close) or close <= 0:
                raise ValueError(f"Expected a positive finite close on {day}.")
            if previous is not None:
                previous_day, previous_close = previous
                if day <= previous_day:
                    raise ValueError(f"Duplicate or out-of-order date on CSV line {line}.")
                if (day - previous_day).days > 4:
                    raise ValueError(f"Check the missing trading sessions between {previous_day} and {day}.")
                models.observe((close - previous_close) / previous_close)
            previous = day, close
    sigmas = models.predict()
    if previous is None or sigmas[0] is None:
        raise ValueError(f"Use at least {SHORT_WINDOW + 1} consecutive session closes for the short model.")
    last_day, last_close = previous
    return last_day, last_close, sigmas


def qlike_gap(actual, sigma, benchmark_sigma):
    """Gaussian quasi-likelihood loss, model minus benchmark, on one target."""
    h = max(sigma**2, VARIANCE_FLOOR)
    benchmark = max(benchmark_sigma**2, VARIANCE_FLOOR)
    return math.log(h / benchmark) + actual**2 * (1 / h - 1 / benchmark)


def block_interval(gaps):
    """Paired circular moving-block bootstrap of the mean daily loss difference."""
    n = len(gaps)
    length = min(20, n)
    doubled = gaps + gaps
    rng = random.Random(17)
    means = []
    for _ in range(2000):
        draw = []
        while len(draw) < n:
            first = rng.randrange(n)
            draw.extend(doubled[first:first + length])
        means.append(fmean(draw[:n]))
    limits = quantiles(means, n=40, method="inclusive")
    return limits[0], limits[-1]


def score(forecasts, sample):
    if not forecasts:
        raise ValueError(f"No eligible forecasts for {sample}. Check dates and window lengths.")
    summary = []
    for index, name in enumerate(MODEL_NAMES):
        gaps = [qlike_gap(f.actual, f.sigmas[index], f.sigmas[1]) for f in forecasts]
        inside = sum(abs(f.actual) <= f.sigmas[index] for f in forecasts)
        lower, upper = None, None
        if sample == SAMPLE_NAMES[0]:
            lower, upper = (0.0, 0.0) if index == 1 else block_interval(gaps)
        summary.append({
            "sample": sample, "model": name, "sessions": len(forecasts),
            "inside": inside, "coverage_pct": 100 * inside / len(forecasts),
            "mean_half_width_pct": 100 * fmean(f.sigmas[index] for f in forecasts),
            "qlike_gap_vs_benchmark": fmean(gaps), "ci_low": lower, "ci_high": upper,
        })
    return summary


def build_report():
    observations, source = load_returns()
    samples = {
        SAMPLE_NAMES[0]: walk_forward(observations),
        SAMPLE_NAMES[1]: walk_forward(observations, outside_expiry=True),
    }
    scores = [row for sample, forecasts in samples.items() for row in score(forecasts, sample)]
    output = ROOT / "results"
    atomic_text(output / "model_scores.csv", csv_text(scores[0], (r.values() for r in scores)))
    fields = ["sample", "cutoff_date", "target_date", "actual_return", "sigma_short",
              "sigma_benchmark", "sigma_ewma"]
    daily = ((sample, f.cutoff, f.target, f.actual, *f.sigmas)
             for sample, forecasts in samples.items() for f in forecasts)
    atomic_text(output / "daily_forecasts.csv", csv_text(fields, daily))

    # The viewer contains derived percentages and forecasts; it opens without a server.
    latest = VolatilityModels()
    for observation in observations:
        latest.observe(observation.actual)
    study = {
        "models": MODEL_NAMES, "decay": DECAY, "source": source, "scores": scores,
        "latest": latest.predict(),
        "samples": {sample: [[str(f.cutoff), str(f.target), f.actual, *f.sigmas] for f in forecasts]
                    for sample, forecasts in samples.items()},
    }
    template = (ROOT / "dashboard.template.html").read_text(encoding="utf-8")
    atomic_text(ROOT / "dashboard.html", template.replace("__STUDY_JSON__", json.dumps(study)))
    return samples, scores, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closes", type=Path, metavar="CSV",
                        help="Forecast the next session from your own date,close CSV.")
    args = parser.parse_args()
    if args.closes:
        try:
            day, close, sigmas = latest_from_csv(args.closes)
        except (OSError, ValueError) as exc:
            parser.exit(2, f"{exc}\n")
        print(f"Next-session forecast using closes through {day} | Last close {close:,.2f}")
        for model, sigma in zip(MODEL_NAMES, sigmas):
            if sigma is None:
                print(f"{model:<17} available after {BENCHMARK_WINDOW + 1} closes")
            else:
                points = close * sigma
                print(f"{model:<17} {sigma:.2%} | {points:,.1f} points | "
                      f"reference band {close - points:,.1f} to {close + points:,.1f}")
        return
    samples, scores, source = build_report()
    print(f"NQ=F | {source['priced_dates']:,} closes | {source['first_close']} to {source['last_close']}")
    if not source["matches_reference_closes"]:
        print("The price history differs from the published snapshot. Results reflect this copy.")
    for sample, forecasts in samples.items():
        print(f"\n{sample} | {len(forecasts)} targets | {forecasts[0].target} to {forecasts[-1].target}")
        print(f"{'Model':<17} {'Coverage':>9} {'Half-width':>11} {'QLIKE gap':>12}")
        for row in scores:
            if row["sample"] == sample:
                print(f"{row['model']:<17} {row['coverage_pct']:>8.1f}%"
                      f" {row['mean_half_width_pct']:>10.2f}% {row['qlike_gap_vs_benchmark']:>+12.3f}")
    print("\nOpen dashboard.html to explore the forecasts.")


if __name__ == "__main__":
    main()
