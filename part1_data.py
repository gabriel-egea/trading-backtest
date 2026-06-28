import numpy as np
import pandas as pd

CAC40_TICKERS = [
    "AI.PA",   "AIR.PA",  "ALO.PA",  "MT.AS",   "ATO.PA",
    "CS.PA",   "BNP.PA",  "EN.PA",   "CAP.PA",  "CA.PA",
    "ACA.PA",  "BN.PA",   "DSY.PA",  "EDEN.PA", "EL.PA",
    "ERF.PA",  "RMS.PA",  "KER.PA",  "LR.PA",   "OR.PA",
    "MC.PA",   "ML.PA",   "ORA.PA",  "RI.PA",   "PUB.PA",
    "RNO.PA",  "SAF.PA",  "SGO.PA",  "SAN.PA",  "SU.PA",
    "GLE.PA",  "STLAM.MI","STM.PA",  "TEP.PA",  "HO.PA",
    "TTE.PA",  "URW.AS",  "VIE.PA",  "DG.PA",   "WLN.PA",
]


def download_cac40(start="2010-01-01", end="2024-12-31",
                   tickers=None, min_history=0.9):
    import yfinance as yf
    if tickers is None:
        tickers = CAC40_TICKERS
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    threshold = int(len(raw) * min_history)
    prices = raw.dropna(axis=1, thresh=threshold).ffill().dropna()
    prices.index = pd.to_datetime(prices.index)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return {"prices": prices, "returns": log_returns, "tickers": list(prices.columns)}


def generate_cac40(start="2010-01-01", end="2024-12-31", n_stocks=40, seed=42):
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    T, N  = len(dates), min(n_stocks, len(CAC40_TICKERS))

    # one-factor model: r_{i,t} = beta_i * r_market_t + epsilon_{i,t}
    r_market  = rng.normal(0.00025, 0.010, T)
    betas     = rng.uniform(0.5,  1.5,  N)
    idio_vols = rng.uniform(0.008, 0.020, N)
    mus       = rng.uniform(-0.0001, 0.0003, N)

    idio        = rng.normal(0, 1, (T, N)) * idio_vols
    log_returns = mus + betas * r_market[:, None] + idio
    prices_arr  = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    tickers = CAC40_TICKERS[:N]
    return {
        "prices":  pd.DataFrame(prices_arr, index=dates, columns=tickers),
        "returns": pd.DataFrame(log_returns, index=dates, columns=tickers),
        "tickers": tickers,
    }


def summary_stats(data):
    r, ann = data["returns"], 252
    stats = pd.DataFrame({
        "mean ann (%)": (r.mean() * ann * 100).round(1),
        "vol ann (%)":  (r.std() * np.sqrt(ann) * 100).round(1),
        "skew":          r.skew().round(2),
        "kurt":          r.kurt().round(2),
        "min (%)":      (r.min() * 100).round(2),
        "max (%)":      (r.max() * 100).round(2),
    })
    corr = r.corr().values.copy()
    np.fill_diagonal(corr, np.nan)
    print(f"Period : {r.index[0].date()} -> {r.index[-1].date()}")
    print(f"Stocks : {r.shape[1]}   Days : {r.shape[0]}")
    print(f"\n{stats.head(8).to_string()}")
    print(f"\nAvg pairwise correlation : {np.nanmean(corr):.3f}")
