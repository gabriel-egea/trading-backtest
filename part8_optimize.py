import numpy as np
import pandas as pd
from itertools import product
from part2_momentum import momentum_signal, equal_weight
from backtest import run_backtest
from metrics import sharpe


def grid_search_momentum(prices, returns, short_range, long_range,
                         transaction_cost=0.0005):
    results = []
    for sw, lw in product(short_range, long_range):
        if sw >= lw:
            continue
        sig = momentum_signal(prices, sw, lw)
        w   = equal_weight(sig)
        res = run_backtest(returns, w, transaction_cost)
        sh  = sharpe(res["returns"])
        results.append({"short": sw, "long": lw, "sharpe": sh})
    return pd.DataFrame(results).sort_values("sharpe", ascending=False)


def train_test_split(prices, returns, train_end="2019-12-31"):
    mask_tr = prices.index <= train_end
    mask_te = prices.index >  train_end
    return (prices[mask_tr], returns[mask_tr],
            prices[mask_te], returns[mask_te])


def walk_forward(prices, returns, short_range, long_range,
                 train_window=504, test_window=126,
                 transaction_cost=0.0005):
    n     = len(prices)
    start = train_window
    all_returns = []

    while start + test_window <= n:
        tr_prices  = prices.iloc[start - train_window:start]
        tr_returns = returns.iloc[start - train_window:start]
        te_prices  = prices.iloc[start:start + test_window]
        te_returns = returns.iloc[start:start + test_window]

        best_sh, best_sw, best_lw = -np.inf, 20, 50
        for sw, lw in product(short_range, long_range):
            if sw >= lw:
                continue
            sig = momentum_signal(tr_prices, sw, lw)
            w   = equal_weight(sig)
            res = run_backtest(tr_returns, w, transaction_cost)
            sh  = sharpe(res["returns"])
            if sh > best_sh:
                best_sh, best_sw, best_lw = sh, sw, lw

        # apply best params on test window
        sig = momentum_signal(
            pd.concat([tr_prices.iloc[-best_lw:], te_prices]),
            best_sw, best_lw
        ).iloc[best_lw:]
        w   = equal_weight(sig)
        res = run_backtest(te_returns, w, transaction_cost)
        res["best_short"] = best_sw
        res["best_long"]  = best_lw
        all_returns.append(res)
        start += test_window

    return pd.concat(all_returns).sort_index()
