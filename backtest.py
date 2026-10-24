"""EMA crossover backtest for forex.

Usage:
    python backtest.py                       # synthetic random-walk data (sanity check)
    python backtest.py --csv EURUSD_H1.csv   # CSV with columns: time,open,high,low,close

Signal: long when fast EMA > slow EMA, short otherwise. The signal computed on
bar t is traded at the close of bar t and earns the return of bar t+1.
"""
import argparse

import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df.sort_index()


def synthetic(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Driftless random walk around 1.10. Any strategy should NOT profit here."""
    rng = np.random.default_rng(seed)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"close": close}, index=idx)


def backtest(close: pd.Series, fast: int, slow: int, spread_pips: float,
             pip: float, leverage: float) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    pos = np.sign(ema_f - ema_s).shift(1).fillna(0)  # trade next bar, no look-ahead

    ret = close.pct_change().fillna(0)
    trades = pos.diff().abs().fillna(0)  # 2 when flipping long<->short
    cost = trades * (spread_pips * pip / close) / 2  # half-spread per side
    strat = pos * ret * leverage - cost * leverage

    out = pd.DataFrame({"pos": pos, "ret": strat})
    out["equity"] = (1 + out["ret"]).cumprod()
    return out


def report(res: pd.DataFrame, bars_per_year: int) -> dict:
    r = res["ret"]
    eq = res["equity"]
    dd = eq / eq.cummax() - 1
    sharpe = r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else 0.0
    return {
        "total_return_%": (eq.iloc[-1] - 1) * 100,
        "max_drawdown_%": dd.min() * 100,
        "sharpe": sharpe,
        "num_trades": int((res["pos"].diff().abs() > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--spread-pips", type=float, default=1.0)
    ap.add_argument("--pip", type=float, default=0.0001)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--bars-per-year", type=int, default=24 * 252)
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="fraction of data used as in-sample; rest is out-of-sample")
    a = ap.parse_args()

    df = load_csv(a.csv) if a.csv else synthetic()
    cut = int(len(df) * a.train_frac)
    for name, part in (("in-sample", df.iloc[:cut]), ("out-of-sample", df.iloc[cut:])):
        res = backtest(part["close"], a.fast, a.slow, a.spread_pips, a.pip, a.leverage)
        stats = report(res, a.bars_per_year)
        print(f"[{name}] " + "  ".join(f"{k}={v:.2f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
