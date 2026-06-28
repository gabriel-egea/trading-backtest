import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product


def plot_cumulative_returns(results_dict, benchmark, save="cumulative_returns.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, res in results_dict.items():
        ax.plot(res["equity"], label=label)
    ax.plot(benchmark["equity"], label="Benchmark (buy & hold)",
            color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Cumulative returns — CAC 40 strategies vs benchmark")
    ax.set_ylabel("Portfolio value (start = 1)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"Saved {save}")


def plot_drawdowns(results_dict, benchmark, save="drawdowns.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, res in results_dict.items():
        eq   = res["equity"]
        peak = eq.cummax()
        dd   = (eq - peak) / peak * 100
        ax.plot(dd, label=label, alpha=0.8)
    eq_b   = benchmark["equity"]
    dd_b   = (eq_b - eq_b.cummax()) / eq_b.cummax() * 100
    ax.plot(dd_b, label="Benchmark", color="black", linestyle="--", linewidth=1.2)
    ax.fill_between(dd_b.index, dd_b, 0, alpha=0.06, color="black")
    ax.set_title("Drawdowns (%)")
    ax.set_ylabel("Drawdown (%)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"Saved {save}")


def plot_rolling_sharpe(results_dict, window=252, save="rolling_sharpe.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, res in results_dict.items():
        r  = res["returns"]
        rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(window)
        ax.plot(rs, label=label, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling Sharpe ratio ({window}-day window)")
    ax.set_ylabel("Sharpe"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"Saved {save}")


def plot_return_distribution(results_dict, benchmark, save="return_distribution.png"):
    n   = len(results_dict) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    items = list(results_dict.items()) + [("Benchmark", benchmark)]
    for ax, (label, res) in zip(axes, items):
        r = res["returns"].dropna() * 100
        ax.hist(r, bins=60, density=True, alpha=0.7, color="steelblue")
        mu, sigma = r.mean(), r.std()
        x  = np.linspace(r.min(), r.max(), 200)
        ax.plot(x, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2),
                color="red", linewidth=1.5, label="Normal fit")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Daily return (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density")
    fig.suptitle("Return distributions vs Gaussian fit")
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"Saved {save}")


def plot_momentum_signal(prices, signal, ticker="BNP.PA",
                         save="momentum_signal.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    p   = prices[ticker]
    ma20 = p.rolling(20).mean()
    ma50 = p.rolling(50).mean()
    sig  = signal[ticker]

    ax1.plot(p,    label="Price",   alpha=0.8, linewidth=0.9)
    ax1.plot(ma20, label="MA 20",   linewidth=1.2)
    ax1.plot(ma50, label="MA 50",   linewidth=1.2)
    ax1.fill_between(p.index, p.min(), p.max(),
                     where=(sig > 0), alpha=0.1, color="green", label="Long")
    ax1.set_title(f"Momentum signal — {ticker}")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(sig, color="green", linewidth=0.8)
    ax2.set_ylabel("Signal"); ax2.set_ylim(-0.1, 1.1)
    ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"Saved {save}")


def plot_sharpe_heatmap(prices, returns, short_range, long_range,
                        transaction_cost=0.0005, save="sharpe_heatmap.png"):
    from part2_momentum import momentum_signal, equal_weight
    from backtest import run_backtest
    from metrics import sharpe as sharpe_fn

    matrix = pd.DataFrame(index=short_range, columns=long_range, dtype=float)
    for sw, lw in product(short_range, long_range):
        if sw >= lw:
            matrix.loc[sw, lw] = np.nan
            continue
        sig = momentum_signal(prices, sw, lw)
        w   = equal_weight(sig)
        res = run_backtest(returns, w, transaction_cost)
        matrix.loc[sw, lw] = sharpe_fn(res["returns"])

    fig, ax = plt.subplots(figsize=(8, 5))
    import matplotlib.colors as mcolors
    vmin = matrix.min().min()
    vmax = matrix.max().max()
    vcenter = 0.0
    if not (vmin < vcenter < vmax):
        vcenter = (vmin + vmax) / 2
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn",
                   norm=norm, aspect="auto")
    ax.set_xticks(range(len(long_range)));  ax.set_xticklabels(long_range)
    ax.set_yticks(range(len(short_range))); ax.set_yticklabels(short_range)
    ax.set_xlabel("Long window"); ax.set_ylabel("Short window")
    ax.set_title("Sharpe ratio heatmap — momentum parameters")
    for i in range(len(short_range)):
        for j in range(len(long_range)):
            val = matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(save, dpi=120)
    print(f"Saved {save}")
