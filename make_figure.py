"""Recreate the LinkedIn image. Only this optional script uses Matplotlib."""
from collections import defaultdict
from datetime import date
from math import ceil, sqrt
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from research import DECAY, MODEL_NAMES, ROOT, SAMPLE_NAMES, build_report

PAPER = "#fbfaf6"
INK = "#192f36"
MUTED = "#4c6167"
RULE = "#d6ddda"
GREEN = "#00776c"
BLUE = "#45678b"
RUST = "#a44428"
COLORS = (RUST, BLUE, GREEN)


def plus(value):
    return f"{value:+.3f}".replace("-", "−") if value else "0.000"


def main():
    samples, scores, source = build_report()
    forecasts = samples[SAMPLE_NAMES[0]]
    full = [s for s in scores if s["sample"] == SAMPLE_NAMES[0]]
    months = defaultdict(list)
    for f in forecasts:
        months[date(f.target.year, f.target.month, 1)].append(f)
    dates = list(months)
    actual = [100 * sqrt(fmean(f.actual**2 for f in rows)) for rows in months.values()]
    predicted = [[100 * sqrt(fmean(f.sigmas[i]**2 for f in rows)) for rows in months.values()]
                 for i in range(len(MODEL_NAMES))]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.unicode_minus": True})
    fig = plt.figure(figsize=(10, 10.75), dpi=160, facecolor=PAPER)

    def note(x, y, content, **kw):
        fig.text(x, y, content, transform=fig.transFigure, **kw)

    def line(y):
        fig.add_artist(Line2D([.063, .95], [y, y], transform=fig.transFigure, lw=.8, color=RULE))

    note(.063, .955, "RJ HILL   /   QUANTITATIVE RESEARCH", fontsize=10.5, fontweight="bold", color=GREEN)
    note(.95, .955, "NQ=F   /   DAILY", fontsize=10, ha="right", color=MUTED)
    line(.936)
    note(.06, .882, "Forecasting daily NQ volatility", fontsize=30, fontfamily="DejaVu Serif")
    period = f"{forecasts[0].target:%B %Y} – {forecasts[-1].target:%B %Y}"
    note(.064, .844, f"{len(forecasts)} one-session-ahead forecasts   ·   {period}",
         fontsize=13.3, color=MUTED)
    note(.064, .785, "How the forecasts tracked daily moves", fontsize=16.4, fontweight="bold")
    note(.064, .759, "Monthly root-mean-square of daily returns and forecast volatilities (%)",
         fontsize=10.8, color=MUTED)
    ax = fig.add_axes((.102, .453, .836, .244), facecolor=PAPER)
    ax.plot(dates, actual, color=INK, lw=2.7, marker="o", ms=3.5, label="Observed move", zorder=4)
    for i, style in enumerate([":", "--", "-"]):
        label = MODEL_NAMES[i] if i < 2 else f"EWMA ({DECAY:.2f})"
        ax.plot(dates, predicted[i], color=COLORS[i], ls=style, lw=1.9, label=label)
    ax.set_ylim(0, max(2, ceil(max(actual + sum(predicted, [])))))
    ax.set_xlim(dates[0], dates[-1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ticks = dates[::4]
    if dates[-1] != ticks[-1]:
        ticks.append(dates[-1])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="y", color=RULE, lw=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=7, labelsize=10)
    ax.legend(loc="lower left", bbox_to_anchor=(-.052, 1.045), frameon=False, ncol=4,
              fontsize=10.1, columnspacing=1.7, handlelength=2.1, borderaxespad=0)

    line(.394)
    note(.064, .355, "Which model scored better?", fontsize=16.4, fontweight="bold")
    note(.064, .329, f"Mean QLIKE loss difference vs. {MODEL_NAMES[1]}. Lower is better.",
         fontsize=10.8, color=MUTED)
    lo = min(s["ci_low"] for s in full)
    hi = max(s["ci_high"] for s in full)
    forest = fig.add_axes((.275, .186, .361, .116), facecolor=PAPER)
    forest.set_xlim(min(-.24, lo-.05), max(.2, hi+.06))
    forest.set_ylim(-.6, 2.6)
    forest.axvline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
    for i, s in enumerate(full):
        y = 2-i
        gap = s["qlike_gap_vs_benchmark"]
        forest.hlines(y, s["ci_low"], s["ci_high"], color=COLORS[i], lw=2.1)
        if s["ci_low"] != s["ci_high"]:
            forest.vlines([s["ci_low"], s["ci_high"]], y-.10, y+.10, color=COLORS[i], lw=1.2)
        forest.plot(gap, y, "o", color=COLORS[i], ms=6, zorder=3)
        yf = .186 + .116 * (y+.6)/3.2
        name = MODEL_NAMES[i] if i != 2 else f"EWMA ({DECAY:.2f})"
        note(.064, yf, name, fontsize=12, va="center")
        note(.746, yf, plus(gap), fontsize=12, ha="right", va="center")
        note(.939, yf, f"{s['coverage_pct']:.1f}%", fontsize=12, ha="right", va="center")
    forest.yaxis.set_visible(False)
    forest.xaxis.set_major_locator(MaxNLocator(nbins=4))
    forest.tick_params(length=0, pad=3, labelsize=9)
    for s in forest.spines.values():
        s.set_visible(False)
    note(.746, .303, "Mean gap", fontsize=9.8, fontweight="bold", ha="right", color=MUTED)
    note(.939, .303, "Inside ±1σ", fontsize=9.8, fontweight="bold", ha="right", color=MUTED)
    note(.064, .147, "Whiskers show paired 95% block-bootstrap intervals for the average loss gap.",
         fontsize=10, color=MUTED)
    line(.123)
    # Descriptive conclusions for this frozen comparison, kept conditional on its actual results.
    short, _, weighted = full
    if (MODEL_NAMES[:2] == ("5-day rolling", "20-day rolling") and short["ci_low"] > 0
            and weighted["qlike_gap_vs_benchmark"] < 0 < weighted["ci_high"]):
        takeaway = "The five-day window had higher loss. EWMA edged the benchmark,\nbut its interval includes a tie."
    else:
        best = min(full, key=lambda s: s["qlike_gap_vs_benchmark"])
        takeaway = f"{best['model']} had the lowest average loss in this test.\nThe intervals show uncertainty in these differences."
    note(.064, .095, takeaway, va="top", fontsize=12.1, linespacing=1.35, fontweight="bold")
    years = f"{source['first_close'][:4]}–{source['last_close'][:4]}"
    note(.064, .025, f"Yahoo Finance NQ=F daily close   ·   {source['priced_dates']:,} closes, {years}",
         fontsize=9.7, color=MUTED)
    fig.savefig(ROOT / "NQ_volatility_LinkedIn.png", dpi=160, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
