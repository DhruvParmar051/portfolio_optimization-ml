"""
reporting.py

Creates visualization and a simple HTML report from portfolio backtest outputs.

Inputs (expected files):
 - results/portfolio/portfolio_cumulative.csv
 - results/portfolio/portfolio_weights.csv
 - models/arima_expanding/arima_expanding_summary.csv   (optional)

Outputs:
 - results/report/cumulative.png
 - results/report/drawdown.png
 - results/report/rolling_sharpe.png
 - results/report/allocation_heatmap.png
 - results/report/report.html
 - results/report/metrics_summary.csv

Author: Dhruv
Date: 2025-11-02
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Tuple

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
PORTFOLIO_DIR = os.path.join(os.getcwd(), "results")
REPORT_DIR = os.path.join(os.getcwd(), "results", "report")
MODEL_SUMMARY = os.path.join(os.getcwd(), "models", "arima_expanding_summary.csv")

os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------
# Helper math functions
# -----------------------
def compute_drawdown(series: pd.Series) -> pd.DataFrame:
    """Compute drawdown series and max drawdown."""
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return pd.DataFrame({"Cumulative": cumulative, "Drawdown": drawdown})


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling Sharpe ratio (window in trading days; default ~3 months)."""
    roll_mean = returns.rolling(window).mean() * 252
    roll_std = returns.rolling(window).std() * np.sqrt(252)
    sharpe = roll_mean / roll_std
    return sharpe


# -----------------------
# Plotting functions
# -----------------------
def plot_cumulative(cumulative: pd.Series, outpath: str):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    cumulative.plot(ax=ax)
    ax.set_title("Portfolio Cumulative Return")
    ax.set_ylabel("Cumulative Value (growth of 1)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Saved cumulative plot → {outpath}")


def plot_drawdown(drawdown_df: pd.DataFrame, outpath: str):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    drawdown_df["Drawdown"].plot(ax=ax)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Saved drawdown plot → {outpath}")


def plot_rolling_sharpe(sharpe: pd.Series, outpath: str):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    sharpe.plot(ax=ax)
    ax.set_title("Rolling Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Saved rolling sharpe plot → {outpath}")


def plot_allocation_heatmap(weights: pd.DataFrame, outpath: str, top_n: int = 25):
    """
    Plot allocation heatmap across rebalances.
    weights: 2D array (rows = rebalance times, cols = stocks) or DataFrame
    Shows top_n stocks by average weight.
    """
    if isinstance(weights, np.ndarray):
        weights = pd.DataFrame(weights)

    # convert to DataFrame if necessary and add index
    if weights.shape[1] == 0:
        logging.warning("Weight matrix empty — skipping allocation heatmap.")
        return

    avg_weights = weights.mean(axis=0).sort_values(ascending=False)
    top_stocks = avg_weights.head(top_n).index
    heat_df = weights[top_stocks] if isinstance(weights, pd.DataFrame) else pd.DataFrame(weights[:, :len(top_stocks)], columns=top_stocks)

    plt.figure(figsize=(12, min(8, 0.15 * heat_df.shape[0] + 2)))
    ax = sns.heatmap(heat_df.T, cmap="viridis", cbar_kws={"label": "Weight"})
    ax.set_ylabel("Stock")
    ax.set_xlabel("Rebalance Step")
    plt.title("Allocation Heatmap (Top {} Stocks)".format(top_n))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logging.info(f"Saved allocation heatmap → {outpath}")


# -----------------------
# Report assembly
# -----------------------
def load_inputs() -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load portfolio cumulative, weights, and optional model summary.
    Returns (portfolio_returns, weights_df, model_summary_df)
    """
    cum_path = os.path.join(PORTFOLIO_DIR, "portfolio_summary.csv")
    weights_path = os.path.join(PORTFOLIO_DIR, "optimized_weights.csv")

    if not os.path.exists(cum_path):
        raise FileNotFoundError(f"Missing {cum_path}. Run optimize_portfolio first.")

    cumulative = pd.read_csv(cum_path, index_col=0, parse_dates=True).iloc[:, 0]
    cumulative.index = pd.to_datetime(cumulative.index)

    weights_df = None
    if os.path.exists(weights_path):
        # weights file may be wide or long; try to load sensibly
        try:
            w = pd.read_csv(weights_path)
            # if rows equal to rebalances and columns to stocks, keep as-is
            weights_df = w
        except Exception:
            weights_df = None

    model_summary_df = pd.read_csv(MODEL_SUMMARY) if os.path.exists(MODEL_SUMMARY) else None

    return cumulative, weights_df, model_summary_df


def compute_metrics_from_returns(portfolio_cum: pd.Series) -> pd.Series:
    """Given cumulative series (growth of 1), compute standard metrics using returns."""
    portfolio_cum = portfolio_cum.sort_index()
    returns = portfolio_cum.pct_change().dropna()
    total_return = portfolio_cum.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
    drawdown_df = compute_drawdown(returns)
    max_dd = drawdown_df["Drawdown"].min() if not drawdown_df.empty else np.nan

    metrics = {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }
    return pd.Series(metrics)


def generate_html_report(image_paths: dict, metrics: pd.Series, model_summary: pd.DataFrame = None, outpath: str = None):
    """Create a self-contained simple HTML report referencing the generated PNGs and metrics table."""
    if outpath is None:
        outpath = os.path.join(REPORT_DIR, "report.html")

    html_parts = []
    html_parts.append("<html><head><title>Portfolio Backtest Report</title></head><body>")
    html_parts.append("<h1>Portfolio Backtest Report</h1>")

    # Metrics table
    html_parts.append("<h2>Key Metrics</h2>")
    html_parts.append(metrics.to_frame("Value").to_html(border=0))

    # Model summary (optional)
    if model_summary is not None:
        html_parts.append("<h2>Model Summary (per-stock)</h2>")
        html_parts.append(model_summary.head(50).to_html(index=False, border=0))

    # Images
    html_parts.append("<h2>Charts</h2>")
    for title, path in image_paths.items():
        if os.path.exists(path):
            html_parts.append(f"<h3>{title}</h3>")
            html_parts.append(f'<img src="{os.path.basename(path)}" style="max-width:100%;height:auto;"><br>')

    html_parts.append("</body></html>")

    # write html and copy images to same folder
    with open(outpath, "w") as f:
        f.write("\n".join(html_parts))

    # copy images into report dir (so HTML can open them locally)
    for _, p in image_paths.items():
        if os.path.exists(p):
            dst = os.path.join(REPORT_DIR, os.path.basename(p))
            if os.path.abspath(p) != os.path.abspath(dst):
                try:
                    from shutil import copyfile
                    copyfile(p, dst)
                except Exception:
                    logging.warning(f"Could not copy {p} to {dst}")

    logging.info(f"HTML report generated → {outpath}")


# -----------------------
# Main entrypoint
# -----------------------
def generate_report():
    try:
        cumulative, weights_df, model_summary = load_inputs()

        # compute returns series from cumulative growth-of-1
        portfolio_returns = cumulative.pct_change().dropna()

        # compute drawdown dataframe
        drawdown_df = compute_drawdown(portfolio_returns)

        # rolling sharpe
        r_sharpe = rolling_sharpe(portfolio_returns, window=63)

        # save plots
        cumulative_path = os.path.join(REPORT_DIR, "cumulative.png")
        drawdown_path = os.path.join(REPORT_DIR, "drawdown.png")
        sharpe_path = os.path.join(REPORT_DIR, "rolling_sharpe.png")
        heatmap_path = os.path.join(REPORT_DIR, "allocation_heatmap.png")

        plot_cumulative(cumulative, cumulative_path)
        plot_drawdown(drawdown_df, drawdown_path)
        plot_rolling_sharpe(r_sharpe, sharpe_path)

        if weights_df is not None:
            # For the heatmap, ensure weights_df is numeric matrix-like
            try:
                # If weights file contains many columns, interpret as rebalance rows
                wmat = weights_df.select_dtypes(include=[np.number])
                if not wmat.empty:
                    plot_allocation_heatmap(wmat, heatmap_path, top_n=25)
                else:
                    logging.warning("No numeric columns in weights dataframe for heatmap.")
            except Exception as e:
                logging.warning(f"Allocation heatmap skipped due to error: {e}")
        else:
            logging.warning("Weights CSV missing - skipping allocation heatmap.")

        # compute metrics and save
        metrics = compute_metrics_from_returns(cumulative)
        metrics_path = os.path.join(REPORT_DIR, "metrics_summary.csv")
        metrics.to_csv(metrics_path, header=True)
        logging.info(f"Metrics saved → {metrics_path}")

        # generate html report with images
        image_paths = {
            "Cumulative Returns": cumulative_path,
            "Drawdown": drawdown_path,
            "Rolling Sharpe": sharpe_path,
            "Allocation Heatmap": heatmap_path
        }
        generate_html_report(image_paths, metrics, model_summary, outpath=os.path.join(REPORT_DIR, "report.html"))

        logging.info("Reporting complete. All artifacts saved under results/report/")

    except Exception as e:
        logging.exception(f"Reporting pipeline failed: {e}")


