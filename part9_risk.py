import numpy as np
import pandas as pd
from scipy.stats import norm


def var_historical(returns, alpha=0.05):
    return -np.quantile(returns.dropna(), alpha)


def var_parametric(returns, alpha=0.05):
    r = returns.dropna()
    return -(r.mean() + norm.ppf(alpha) * r.std())


def cvar(returns, alpha=0.05):
    r   = returns.dropna()
    var = var_historical(r, alpha)
    return -r[r < -var].mean()


def rolling_var(returns, window=252, alpha=0.05):
    return returns.rolling(window).quantile(alpha).abs()


def risk_contribution(weights_df, returns_df):
    w     = weights_df.fillna(0.0)
    r     = returns_df.reindex(w.index).fillna(0.0)
    w_avg = w.mean(axis=0).values.reshape(-1)
    cov   = np.cov(r.values.T)
    if cov.ndim == 0:
        return pd.Series(0.0, index=returns_df.columns)
    port_vol = float(np.sqrt(w_avg @ cov @ w_avg))
    if port_vol == 0:
        return pd.Series(0.0, index=returns_df.columns)
    rc = w_avg * (cov @ w_avg) / port_vol
    return pd.Series(rc, index=returns_df.columns)


def risk_report(returns, weights_df, returns_df, alpha=0.05, label=""):
    r = returns.dropna()
    print(f"\n--- Risk report [{label}] ---")
    print(f"  VaR {int((1-alpha)*100)}% historical : {var_historical(r, alpha)*100:.3f}%")
    print(f"  VaR {int((1-alpha)*100)}% parametric : {var_parametric(r, alpha)*100:.3f}%")
    print(f"  CVaR {int((1-alpha)*100)}%           : {cvar(r, alpha)*100:.3f}%")
    rc  = risk_contribution(weights_df, returns_df)
    top = rc.abs().nlargest(3)
    print(f"  Top 3 risk contributors:")
    for ticker, val in top.items():
        print(f"    {ticker:<12} RC = {val*100:.4f}%")
