"""Download forex OHLC from Yahoo Finance into the CSV format backtest.py expects.

Setup:
    pip install yfinance

Usage:
    python fetch_data.py                          # EURUSD, 1h, last 730 days -> EURUSD_H1.csv
    python fetch_data.py GBPUSD USDJPY            # several pairs
    python fetch_data.py EURUSD --interval 1d --period 10y

Then:
    python backtest.py --csv EURUSD_H1.csv --walk-forward

Yahoo limits intraday history: 1h ~730 days, 15m/5m ~60 days. For longer
intraday history, export from MT5 or Dukascopy instead.
For JPY pairs run backtest.py with --pip 0.01.
"""
import argparse
import sys

import yfinance as yf

SUFFIX = {"1h": "H1", "30m": "M30", "15m": "M15", "5m": "M5", "1d": "D1", "1wk": "W1"}


def fetch(pair: str, interval: str, period: str):
    df = yf.download(f"{pair}=X", interval=interval, period=period,
                     auto_adjust=False, progress=False)
    if df.empty:
        return None
    if hasattr(df.columns, "levels"):  # newer yfinance returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["open", "high", "low", "close"]
    df.index.name = "time"
    return df[~df.index.duplicated()].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=["EURUSD"], help="e.g. EURUSD GBPUSD USDJPY")
    ap.add_argument("--interval", default="1h", choices=list(SUFFIX))
    ap.add_argument("--period", default="730d", help="e.g. 60d, 730d, 5y, max")
    a = ap.parse_args()

    for pair in a.pairs:
        pair = pair.upper().replace("/", "")
        df = fetch(pair, a.interval, a.period)
        if df is None:
            print(f"{pair}: no data returned (rate-limited or bad period/interval?)", file=sys.stderr)
            continue
        path = f"{pair}_{SUFFIX[a.interval]}.csv"
        df.to_csv(path)
        print(f"{pair}: {len(df)} bars {df.index[0]} -> {df.index[-1]} saved to {path}")


if __name__ == "__main__":
    main()
