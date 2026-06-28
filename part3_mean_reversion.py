import numpy as np
import pandas as pd


def mean_reversion_signal(prices, window=20, entry_z=2.0, exit_z=0.5):
    mu    = prices.rolling(window, min_periods=window).mean()
    sigma = prices.rolling(window, min_periods=window).std()
    z     = (prices - mu) / sigma

    n_rows, n_cols = prices.shape
    sig = np.zeros((n_rows, n_cols))
    sig[:] = np.nan

    z_arr = z.values
    for t in range(1, n_rows):
        if np.all(np.isnan(z_arr[t])):
            continue
        for i in range(n_cols):
            zi = z_arr[t, i]
            if np.isnan(zi):
                sig[t, i] = np.nan
                continue
            prev = sig[t-1, i] if not np.isnan(sig[t-1, i]) else 0.0
            if zi > entry_z:
                sig[t, i] = -1.0
            elif zi < -entry_z:
                sig[t, i] = 1.0
            elif abs(zi) < exit_z:
                sig[t, i] = 0.0
            else:
                sig[t, i] = prev

    return pd.DataFrame(sig, index=prices.index, columns=prices.columns)
