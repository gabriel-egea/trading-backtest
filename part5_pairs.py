import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def find_cointegrated_pairs(prices, formation_prices, pvalue_threshold=0.05):
    tickers = list(prices.columns)
    pairs   = []

    for a, b in combinations(tickers, 2):
        pa = formation_prices[a].values
        pb = formation_prices[b].values

        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue

        res   = OLS(pa, add_constant(pb)).fit()
        beta  = res.params[1]
        alpha = res.params[0]
        spread = pa - beta * pb - alpha

        adf_pval = adfuller(spread, maxlag=1, autolag=None)[1]
        if adf_pval < pvalue_threshold:
            pairs.append((a, b, beta, alpha, adf_pval))

    return pairs


def pairs_signals(prices, formation_window=252, trading_window=63,
                  entry_z=2.0, exit_z=0.5, pvalue_threshold=0.05):
    n_rows = len(prices)
    start  = formation_window

    all_signals = {}

    for t in range(start, n_rows - trading_window, trading_window):
        form_prices  = prices.iloc[t - formation_window:t]
        trade_prices = prices.iloc[t:t + trading_window]

        pairs = find_cointegrated_pairs(prices, form_prices, pvalue_threshold)

        for a, b, beta, alpha, _ in pairs:
            key = (a, b)
            spread = trade_prices[a] - beta * trade_prices[b] - alpha
            mu_s   = spread.mean()
            sig_s  = spread.std()

            if sig_s < 1e-8:
                continue

            z = (spread - mu_s) / sig_s
            n = len(z)
            sig_arr = np.zeros(n)
            prev    = 0.0

            for i in range(n):
                zi = z.iloc[i]
                if zi > entry_z:
                    sig_arr[i] = -1.0
                elif zi < -entry_z:
                    sig_arr[i] = 1.0
                elif abs(zi) < exit_z:
                    sig_arr[i] = 0.0
                else:
                    sig_arr[i] = prev
                prev = sig_arr[i]

            dates = trade_prices.index
            for i, date in enumerate(dates):
                if key not in all_signals:
                    all_signals[key] = {}
                all_signals[key][date] = sig_arr[i]

    return all_signals


def pairs_portfolio_returns(prices, returns, all_signals):
    port_returns = pd.Series(0.0, index=returns.index)
    n_active     = pd.Series(0,   index=returns.index)

    for (a, b), sig_dict in all_signals.items():
        dates = sorted(sig_dict.keys())
        for i, date in enumerate(dates[1:], 1):
            prev_date = dates[i-1]
            sig = sig_dict[prev_date]
            if sig == 0 or date not in returns.index:
                continue
            if a not in returns.columns or b not in returns.columns:
                continue
            pnl = sig * returns.loc[date, a] - sig * returns.loc[date, b]
            port_returns.loc[date] += pnl
            n_active.loc[date]     += 1

    active = n_active > 0
    port_returns[active] /= n_active[active]
    return port_returns
