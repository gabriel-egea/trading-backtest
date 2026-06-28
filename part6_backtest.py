import numpy as np
import pandas as pd


def run_backtest(returns, weights, transaction_cost=0.0005):
    w      = weights.reindex(returns.index).fillna(0.0)
    w_prev = w.shift(1).fillna(0.0)

    port_returns = (w_prev * returns).sum(axis=1)
    turnover     = w.diff().abs().sum(axis=1).fillna(0.0)
    port_returns -= transaction_cost * turnover

    equity = (1 + port_returns).cumprod()
    return pd.DataFrame({
        "returns":  port_returns,
        "equity":   equity,
        "turnover": turnover,
    })


def benchmark_returns(returns):
    r_bench  = returns.mean(axis=1)
    equity   = (1 + r_bench).cumprod()
    return pd.DataFrame({
        "returns": r_bench,
        "equity":  equity,
    })
