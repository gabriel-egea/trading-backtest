import numpy as np
import pandas as pd


def momentum_signal(prices, short_window=20, long_window=50):
    if short_window >= long_window:
        raise ValueError("short_window must be < long_window.")
    ma_short = prices.rolling(short_window, min_periods=short_window).mean()
    ma_long  = prices.rolling(long_window,  min_periods=long_window).mean()
    signal   = (ma_short > ma_long).astype(float)
    signal[ma_long.isna()] = np.nan
    return signal


def equal_weight(signals):
    filled = signals.fillna(0.0)
    denom  = filled.abs().sum(axis=1).replace(0, np.nan)
    return filled.div(denom, axis=0).fillna(0.0)


def signal_diagnostics(signals, label=""):
    filled = signals.fillna(0)
    long_pct   = (filled  > 0).values.mean() * 100
    short_pct  = (filled  < 0).values.mean() * 100
    trade_freq = (filled.diff().abs() > 0).values.mean() * 100
    print(f"[{label}]  long {long_pct:.1f}%  short {short_pct:.1f}%"
          f"  trade freq {trade_freq:.1f}%"
          f"  valid from {signals.first_valid_index().date()}")
