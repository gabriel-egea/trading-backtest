import numpy as np
import pandas as pd


def sharpe(returns, ann=252):
    r = returns.dropna()
    if r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(ann)


def max_drawdown(equity):
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return dd.min()


def calmar(returns, ann=252):
    mdd = max_drawdown((1 + returns.dropna()).cumprod())
    if mdd == 0:
        return np.nan
    return returns.dropna().mean() * ann / abs(mdd)


def information_ratio(returns, bench_returns, ann=252):
    active = returns.dropna() - bench_returns.reindex(returns.dropna().index)
    if active.std() == 0:
        return np.nan
    return active.mean() / active.std() * np.sqrt(ann)


def hit_ratio(returns):
    r = returns.dropna()
    return (r > 0).sum() / len(r)


def summary(returns, bench_returns=None, label=""):
    r   = returns.dropna()
    ann = 252
    res = {
        "ann return (%)":  round(r.mean() * ann * 100, 2),
        "ann vol (%)":     round(r.std()  * np.sqrt(ann) * 100, 2),
        "Sharpe":          round(sharpe(r), 3),
        "Max Drawdown (%)":round(max_drawdown((1 + r).cumprod()) * 100, 2),
        "Calmar":          round(calmar(r), 3),
        "Hit ratio (%)":   round(hit_ratio(r) * 100, 2),
    }
    if bench_returns is not None:
        res["Info Ratio"] = round(information_ratio(r, bench_returns), 3)

    print(f"\n--- {label} ---")
    for k, v in res.items():
        print(f"  {k:<22}: {v}")
    return res
