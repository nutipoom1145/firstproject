"""EMA crossover backtest for forex, with ATR stop loss and walk-forward testing.

Usage:
    python backtest.py                         # simple in/out-of-sample split, synthetic data
    python backtest.py --walk-forward          # walk-forward optimisation
    python backtest.py --csv EURUSD_H1.csv     # CSV columns: time,open,high,low,close

Rules:
  * Signal: long when fast EMA > slow EMA, short otherwise, computed on the
    close of bar t-1 and executed at the open of bar t (no look-ahead).
  * Stop loss: entry -/+ atr_mult * ATR(atr_n). Checked intrabar using high/low;
    if the bar gaps through the stop we exit at the open. After a stop-out we
    stay flat until the signal flips (no instant re-entry in the same direction).
  * Position size: risk `risk` of current equity on each trade
    (notional = equity * risk / stop distance %), capped at `max_leverage`.
  * Costs: half the spread on each entry and each exit.
"""
import argparse
import itertools

import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    return df.sort_index()[["open", "high", "low", "close"]]


def synthetic(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Driftless random-walk OHLC around 1.10. No strategy should profit here."""
    rng = np.random.default_rng(seed)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.0003, (2, n))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def backtest(df: pd.DataFrame, fast: int, slow: int, atr_n: int, atr_mult: float,
             spread_pips: float, pip: float, risk: float, max_leverage: float) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sig = np.sign(c_ema(df, fast) - c_ema(df, slow)).to_numpy()
    a = atr(df, atr_n).to_numpy()
    half = spread_pips * pip / 2  # price units, per side

    n = len(df)
    rets = np.zeros(n)
    pos_out = np.zeros(n)
    entries = np.zeros(n, dtype=bool)
    equity = 1.0
    d, notional, stop, blocked = 0, 0.0, 0.0, 0

    for i in range(1, n):
        want = int(sig[i - 1])
        if want != blocked:
            blocked = 0
        pnl = 0.0
        prev_c = c[i - 1]
        ref = prev_c  # price the open position is currently marked at

        if d != 0:
            hit_px = None
            if d == 1 and o[i] <= stop or d == -1 and o[i] >= stop:
                hit_px = o[i]
            elif d == 1 and l[i] <= stop or d == -1 and h[i] >= stop:
                hit_px = stop
            if hit_px is not None:
                pnl += d * notional * (hit_px / ref - 1) - notional * half / ref
                blocked, d = d, 0
            elif want != d:
                pnl += d * notional * (o[i] / ref - 1) - notional * half / ref
                d = 0
            else:
                pnl += d * notional * (c[i] / ref - 1)

        if d == 0 and want != 0 and want != blocked and a[i - 1] > 0:
            dist = atr_mult * a[i - 1]
            notional = min(equity * risk / (dist / o[i]), equity * max_leverage)
            d, stop = want, o[i] - want * dist
            entries[i] = True
            pnl -= notional * half / o[i]
            if d == 1 and l[i] <= stop or d == -1 and h[i] >= stop:  # stopped on entry bar
                pnl += d * notional * (stop / o[i] - 1) - notional * half / o[i]
                blocked, d = d, 0
            else:
                pnl += d * notional * (c[i] / o[i] - 1)

        rets[i] = pnl / equity
        equity += pnl
        pos_out[i] = d

    out = pd.DataFrame({"pos": pos_out, "ret": rets, "entry": entries}, index=df.index)
    out["equity"] = (1 + out["ret"]).cumprod()
    return out


def c_ema(df: pd.DataFrame, span: int) -> pd.Series:
    return df["close"].ewm(span=span, adjust=False).mean()


def stats(ret: pd.Series, entries: pd.Series, bars_per_year: int) -> dict:
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(bars_per_year) if ret.std() > 0 else 0.0
    return {
        "return_%": (eq.iloc[-1] - 1) * 100,
        "max_dd_%": dd.min() * 100,
        "sharpe": sharpe,
        "trades": int(entries.sum()),
    }


def fmt(s: dict) -> str:
    return "  ".join(f"{k}={v:.2f}" for k, v in s.items())


def walk_forward(df, grid, train, test, bpy, **cost):
    """Optimise on a rolling train window, trade the next `test` bars with the winner."""
    oos_ret, oos_entry, rows = [], [], []
    start = 0
    while start + train + test <= len(df):
        tr_end, te_end = start + train, start + train + test
        best, best_sh = None, -np.inf
        for fast, slow, am in grid:
            if fast >= slow:
                continue
            r = backtest(df.iloc[start:tr_end], fast, slow, 14, am, **cost)
            sh = stats(r["ret"], r["entry"], bpy)["sharpe"]
            if sh > best_sh:
                best, best_sh = (fast, slow, am), sh
        # run on train+test so EMAs/ATR are warm, keep only the test segment
        r = backtest(df.iloc[start:te_end], best[0], best[1], 14, best[2], **cost).iloc[train:]
        oos_ret.append(r["ret"])
        oos_entry.append(r["entry"])
        s = stats(r["ret"], r["entry"], bpy)
        rows.append((df.index[tr_end], best, best_sh, s))
        start += test
    return pd.concat(oos_ret), pd.concat(oos_entry), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--atr-n", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=2.0, help="stop distance in ATRs")
    ap.add_argument("--risk", type=float, default=0.01, help="fraction of equity risked per trade")
    ap.add_argument("--max-leverage", type=float, default=10.0)
    ap.add_argument("--spread-pips", type=float, default=1.0)
    ap.add_argument("--pip", type=float, default=0.0001)
    ap.add_argument("--bars-per-year", type=int, default=24 * 252)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--wf-train", type=int, default=6000, help="bars in each optimisation window")
    ap.add_argument("--wf-test", type=int, default=2000, help="bars traded after each optimisation")
    a = ap.parse_args()

    df = load_csv(a.csv) if a.csv else synthetic()
    cost = dict(spread_pips=a.spread_pips, pip=a.pip, risk=a.risk, max_leverage=a.max_leverage)

    if a.walk_forward:
        grid = list(itertools.product([5, 10, 20, 30], [40, 50, 100, 200], [1.5, 2.0, 3.0]))
        ret, ent, rows = walk_forward(df, grid, a.wf_train, a.wf_test, a.bars_per_year, **cost)
        for t, params, train_sh, s in rows:
            print(f"from {t:%Y-%m-%d} params(fast,slow,atr_mult)={params} train_sharpe={train_sh:.2f} -> test: {fmt(s)}")
        print(f"[walk-forward OOS total] {fmt(stats(ret, ent, a.bars_per_year))}")
        return

    cut = int(len(df) * a.train_frac)
    for name, part in (("in-sample", df.iloc[:cut]), ("out-of-sample", df.iloc[cut:])):
        r = backtest(part, a.fast, a.slow, a.atr_n, a.atr_mult, **cost)
        print(f"[{name}] {fmt(stats(r['ret'], r['entry'], a.bars_per_year))}")


if __name__ == "__main__":
    main()
