"""Rebuild the study's price and return files from Yahoo Finance daily NQ=F closes."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
START = date(2016, 1, 1)
END_EXCLUSIVE = date(2025, 12, 1)
URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/NQ%3DF"
    "?interval=1d&period1=1451606400&period2=1764547200"
)
REFERENCE_HASH = "9bc2807ee6436612583e288c5bd8057b0a39e4c75c624087311b1cc33aebebdc"


def atomic_text(path, content):
    """Commit a complete UTF-8 file, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w",
                                         encoding="utf-8", newline="") as temp:
            temp_path = Path(temp.name)
            temp.write(content)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def csv_text(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def read_quotes(payload):
    chart = payload["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("Yahoo returned no price series.")
    result = chart["result"][0]
    meta = result["meta"]
    if (meta["symbol"], meta["instrumentType"], meta["dataGranularity"]) != ("NQ=F", "FUTURE", "1d"):
        raise ValueError("The source must contain daily prices for NQ=F.")
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    if len(timestamps) != len(closes):
        raise ValueError("The source returned different numbers of timestamps and closes.")

    prices, missing = [], []
    prior = None
    for timestamp, close in zip(timestamps, closes):
        # For this fixed series, the provider's daily stamps have the same
        # calendar date in UTC and New York. Never use the PC's local timezone.
        day = datetime.fromtimestamp(timestamp, timezone.utc).date()
        if not START <= day < END_EXCLUSIVE:
            raise ValueError(f"Unexpected date outside the requested range: {day}")
        if prior is not None and day <= prior:
            raise ValueError(f"Duplicate or out-of-order source date: {day}")
        prior = day
        if close is None:
            missing.append(day.isoformat())
            continue
        if day.weekday() > 4 or not math.isfinite(close) or close <= 0:
            raise ValueError(f"Unexpected daily observation on {day}: {close}")
        prices.append((day, round(close, 2)))
    if not prices or (prices[0][0], prices[-1][0]) != (date(2016, 1, 4), date(2025, 11, 28)):
        raise ValueError("The source did not cover the full frozen study period.")
    if any((b[0] - a[0]).days > 4 for a, b in zip(prices, prices[1:])):
        raise ValueError("Unexpected gap of more than four calendar days. Inspect the source.")
    return prices, missing


def build_files(payload, directory=DATA):
    prices, missing = read_quotes(payload)
    close_csv = csv_text(["date", "close"], ((d, f"{p:.2f}") for d, p in prices))
    return_csv = csv_text(
        ["cutoff_date", "target_date", "actual_return"],
        ((d0, d1, (p1 - p0) / p0) for (d0, p0), (d1, p1) in zip(prices, prices[1:])),
    )
    digest = hashlib.sha256(close_csv.encode()).hexdigest()
    metadata = {
        "source": "Yahoo Finance", "symbol": "NQ=F", "field": "daily close",
        "url": URL, "request_start": str(START), "request_end_exclusive": str(END_EXCLUSIVE),
        "first_close": str(prices[0][0]), "last_close": str(prices[-1][0]),
        "last_close_price": prices[-1][1],
        "priced_dates": len(prices), "returns": len(prices) - 1,
        "null_close_dates_omitted": missing, "close_csv_sha256": digest,
        "return_csv_sha256": hashlib.sha256(return_csv.encode()).hexdigest(),
        "matches_reference_closes": digest == REFERENCE_HASH,
    }
    atomic_text(directory / "nq_closes.csv", close_csv)
    atomic_text(directory / "nq_returns.csv", return_csv)
    atomic_text(directory / "source.json", json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json", type=Path, help="Use a saved response from the source URL.")
    args = parser.parse_args()
    if args.from_json:
        payload = json.loads(args.from_json.read_text(encoding="utf-8"))
    else:
        request = Request(URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except (URLError, TimeoutError, ValueError) as exc:
            raise SystemExit(f"Price download failed: {exc}\nThe bundled returns work offline with python research.py.") from exc
    meta = build_files(payload)
    print(f"Saved {meta['priced_dates']:,} closes and {meta['returns']:,} returns in {DATA}")
    print("Snapshot matches the reference." if meta["matches_reference_closes"] else
          "The vendor snapshot differs from the reference. Review data/source.json before comparison.")


if __name__ == "__main__":
    main()
