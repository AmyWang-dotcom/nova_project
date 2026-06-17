"""Contract audit and quantitative backtest engine."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_PAIR, PAIR_META, SUPPORTED_PAIRS, YFINANCE_PAIR_TICKERS
from .indicators import _multi_factor_series
from .market_data import (
    _fetch_alpha_vantage_spot_data,
    _fetch_yfinance_spot_data,
    fetch_real_market_data,
    generate_fx_kline,
)
from .utils import _calendar_date_index, _sanitize_spot_dataframe, _sanitize_spot_series

TRADING_DAYS = 365
RISK_FREE_ANNUAL = 0.02
def _annualized_volatility(daily_returns: pd.Series) -> float:
    std = daily_returns.std(ddof=1)
    if std is None or np.isnan(std) or std == 0:
        return 0.0
    return float(std * np.sqrt(TRADING_DAYS))
def _historical_var_95(daily_cash_flows: pd.Series) -> float:
    """95% Historical VaR: 5th percentile of daily absolute CNH cash flows."""
    clean = daily_cash_flows.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(np.percentile(clean.values, 5))
def _maximum_drawdown(cumulative_equity: pd.Series) -> float:
    """Peak-to-trough decline on cumulative equity curve (positive magnitude)."""
    clean = cumulative_equity.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return 0.0
    running_peak = clean.cummax()
    drawdown = (clean - running_peak) / running_peak.replace(0, np.nan)
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(abs(drawdown.min()))
def _sharpe_ratio(daily_returns: pd.Series, rf_daily: float) -> float:
    std = daily_returns.std(ddof=1)
    if std is None or np.isnan(std) or std == 0:
        return 0.0
    excess = daily_returns.mean() - rf_daily
    return float(excess / std * np.sqrt(TRADING_DAYS))
def _compute_strategy_metrics(
    daily_cnh: pd.Series,
    daily_returns: pd.Series,
    rf_daily: float,
) -> dict[str, Any]:
    cumulative = daily_cnh.cumsum()
    wealth_index = (1.0 + daily_returns).cumprod()
    return {
        "daily_cnh": daily_cnh,
        "daily_returns": daily_returns,
        "cumulative_equity": cumulative,
        "wealth_index": wealth_index,
        "total_value": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
        "annualized_volatility": _annualized_volatility(daily_returns),
        "var_95": _historical_var_95(daily_cnh),
        "max_drawdown": _maximum_drawdown(cumulative),
        "sharpe_ratio": _sharpe_ratio(daily_returns, rf_daily),
    }
def _compute_multi_factor_hedge_ratios(
    spot: pd.Series,
    ma5: pd.Series | None = None,
) -> np.ndarray:
    """
    Three-factor integrated hedge model (weights 40% / 30% / 30%):
    F1 trend (Spot vs MA5), F2 vol (Spot vs 20d Bollinger upper), F3 macro (Spot vs MA30).
    Score → hedge: ≥0.7 → 80%, [0.4, 0.7) → 55%, <0.4 → 30%.
    """
    return _multi_factor_series(spot, ma5)["hedge_ratio"]
def advanced_quantitative_analysis(
    df_history: pd.DataFrame,
    monthly_revenue: float = 100_000,
) -> dict[str, Any]:
    """
    Compare Unhedged Spot vs NOVA Multi-Factor Dynamic Overlay.
    Factors: MA5 trend, 20d Bollinger upper-band stretch, 30d macro deviation.
    Returns metrics, equity curves, and timelines for dashboard rendering.
    """
    df = _sanitize_spot_dataframe(df_history.copy())
    spot = _sanitize_spot_series(df["Spot"].astype(float))
    ma5 = (
        _sanitize_spot_series(df["MA5"].astype(float))
        if "MA5" in df.columns
        else spot.rolling(window=5, min_periods=1).mean().ffill().bfill()
    )
    daily_usd = monthly_revenue / 30.0
    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS

    # Strategy A — unhedged spot conversion
    benchmark_cnh = daily_usd * spot
    benchmark_returns = benchmark_cnh.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Strategy B — NOVA multi-factor dynamic overlay
    hedge_ratio = _compute_multi_factor_hedge_ratios(spot, ma5)
    effective_rate = (hedge_ratio * ma5) + ((1.0 - hedge_ratio) * spot)
    nova_cnh = daily_usd * effective_rate
    nova_returns = nova_cnh.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    benchmark = _compute_strategy_metrics(benchmark_cnh, benchmark_returns, rf_daily)
    nova = _compute_strategy_metrics(nova_cnh, nova_returns, rf_daily)

    timeline = df["date"].tolist() if "date" in df.columns else list(range(len(df)))
    data_source = df.attrs.get("data_source", "unknown")

    return {
        "pair": "USD/CNH",
        "data_source": data_source,
        "ticker": df.attrs.get("ticker", YFINANCE_PAIR_TICKERS.get("USD/CNH", "USDCNH=X")),
        "observation_days": len(df),
        "monthly_revenue_usd": monthly_revenue,
        "daily_usd_exposure": daily_usd,
        "timeline": timeline,
        "hedge_ratio_series": hedge_ratio.tolist(),
        "benchmark": {
            **benchmark,
            "label": "Unhedged Spot Benchmark",
            "daily_cnh": benchmark["daily_cnh"].tolist(),
            "daily_returns": benchmark["daily_returns"].tolist(),
            "cumulative_equity": benchmark["cumulative_equity"].tolist(),
        },
        "nova": {
            **nova,
            "label": "NOVA Dynamic Overlay",
            "daily_cnh": nova["daily_cnh"].tolist(),
            "daily_returns": nova["daily_returns"].tolist(),
            "cumulative_equity": nova["cumulative_equity"].tolist(),
        },
        "metrics_comparison": {
            "total_value": {
                "benchmark": benchmark["total_value"],
                "nova": nova["total_value"],
            },
            "annualized_volatility": {
                "benchmark": benchmark["annualized_volatility"],
                "nova": nova["annualized_volatility"],
            },
            "var_95": {
                "benchmark": benchmark["var_95"],
                "nova": nova["var_95"],
            },
            "max_drawdown": {
                "benchmark": benchmark["max_drawdown"],
                "nova": nova["max_drawdown"],
            },
            "sharpe_ratio": {
                "benchmark": benchmark["sharpe_ratio"],
                "nova": nova["sharpe_ratio"],
            },
        },
    }
