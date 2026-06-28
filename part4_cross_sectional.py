import numpy as np
import pandas as pd


def cross_sectional_momentum(prices, lookback=252, skip=21,
                              top_pct=0.20, rebalance_freq=21):
    n_rows, n_cols = prices.shape
    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for t in range(lookback, n_rows):
        if t % rebalance_freq != 0:
            sig.iloc[t] = sig.iloc[t-1]
            continue

        score = np.log(prices.iloc[t-skip] / prices.iloc[t-lookback])
        if score.isna().all():
            continue

        ranked     = score.rank(pct=True, na_option="keep")
        long_mask  = ranked >= (1 - top_pct)
        short_mask = ranked <= top_pct

        n_long  = long_mask.sum()
        n_short = short_mask.sum()

        row = pd.Series(0.0, index=prices.columns)
        if n_long  > 0: row[long_mask]  =  1.0 / n_long
        if n_short > 0: row[short_mask] = -1.0 / n_short
        sig.iloc[t] = row

    sig.iloc[:lookback] = np.nan
    return sig
