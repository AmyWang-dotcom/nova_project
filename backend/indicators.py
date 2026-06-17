"""Technical indicators and multi-factor desk metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_PAIR, NOVA_RISK_WEIGHTS, OFFSHORE_QUOTE_CCY, PAIR_META, VAT_MAP
from .knowledge_base import get_imf_macro_knowledge_entry
from .market_data import (
    _extract_close_column,
    fetch_china_money_central_parity,
    fetch_fx_kline,
    fetch_tnx_fedwatch_proxy,
    get_fx_metrics,
)
from .risk_engine import compute_environment_risk_assessment
from .utils import _sanitize_spot_series

def resolve_vat_rate(pair: str, custom_vat_rate: float | None = None) -> float:
    """Pair VAT baseline with EU slider override for EUR/CNH desks."""
    base_rate = float(VAT_MAP.get(pair, 0.0))
    if pair == "EUR/CNH" and custom_vat_rate is not None:
        return float(custom_vat_rate)
    return base_rate
def calculate_vat_net_payout_series(
    historical_df: pd.DataFrame,
    gross_sales_base: float,
    pair: str,
    custom_vat_rate: float | None = None,
) -> pd.Series:
    """
    Synthesize VAT net payout: gross_sales_base × (1 − VAT) × historical close.
    EUR/CNH accepts a dynamic member-state VAT override from the UI slider.
    """
    close = _extract_close_column(historical_df)
    if close.empty:
        return pd.Series(dtype=float)

    vat_rate = resolve_vat_rate(pair, custom_vat_rate)
    gross_base = max(0.0, float(gross_sales_base))
    payout = gross_base * (1.0 - vat_rate) * close
    return _sanitize_spot_series(payout)
def compute_rsi_and_macd(
    payout_series: pd.Series,
    rsi_period: int = 14,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, Any]:
    """
    Wilder RSI + MACD on VAT net payout (pandas `.diff()` / `.ewm()` only).
    Returns latest scalar points plus the full indicator frame.
    """
    empty: dict[str, Any] = {
        "rsi_val": 50.0,
        "macd_hist": 0.0,
        "macd_line_val": 0.0,
        "macd_signal_val": 0.0,
        "frame": pd.DataFrame(),
    }
    series = _sanitize_spot_series(payout_series.astype(float))
    if series.empty:
        return empty

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / max(1, int(rsi_period))
    avg_gain = gain.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    ema_fast = series.ewm(span=max(1, int(fast_period)), adjust=False).mean()
    ema_slow = series.ewm(span=max(1, int(slow_period)), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=max(1, int(signal_period)), adjust=False).mean()
    macd_histogram = macd_line - macd_signal

    frame = pd.DataFrame(
        {
            "net_payout": series,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
        }
    )
    idx = -1

    def _safe_scalar(col: str, default: float = 0.0) -> float:
        raw = frame[col].iloc[idx]
        if pd.isna(raw):
            return default
        return float(raw)

    return {
        "rsi_val": round(_safe_scalar("rsi", 50.0), 2),
        "macd_hist": round(_safe_scalar("macd_histogram", 0.0), 6),
        "macd_line_val": round(_safe_scalar("macd_line", 0.0), 6),
        "macd_signal_val": round(_safe_scalar("macd_signal", 0.0), 6),
        "frame": frame,
    }
def _macd_histogram_cross_flags(histogram: pd.Series) -> tuple[bool, bool]:
    """Detect MACD histogram zero-line crosses on the latest bar."""
    if histogram is None or len(histogram) < 2:
        return False, False
    prev = float(histogram.iloc[-2])
    curr = float(histogram.iloc[-1])
    if np.isnan(prev) or np.isnan(curr):
        return False, False
    crossed_below = prev >= 0.0 and curr < 0.0
    crossed_above = prev <= 0.0 and curr > 0.0
    return crossed_below, crossed_above
def compute_net_payout_technical_metrics(
    df: pd.DataFrame,
    pair: str | None = None,
    gross_sales_base: float = 10_000.0,
    custom_vat_rate: float | None = None,
) -> dict[str, Any]:
    """RSI/MACD snapshot on tax-adjusted VAT net payout for the active desk pair."""
    active_pair = pair or df.attrs.get("pair", DEFAULT_PAIR)
    if df.empty or _extract_close_column(df).empty:
        return {}

    vat_rate = resolve_vat_rate(active_pair, custom_vat_rate)
    payout_series = calculate_vat_net_payout_series(
        df,
        gross_sales_base=gross_sales_base,
        pair=active_pair,
        custom_vat_rate=custom_vat_rate,
    )
    ta = compute_rsi_and_macd(payout_series)
    frame = ta.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}

    rsi_val = float(ta.get("rsi_val", 50.0))
    macd_hist = float(ta.get("macd_hist", 0.0))
    macd_line = float(ta.get("macd_line_val", 0.0))
    macd_signal = float(ta.get("macd_signal_val", 0.0))
    crossed_below, crossed_above = _macd_histogram_cross_flags(frame["macd_histogram"])

    return {
        "net_payout_rsi": rsi_val,
        "net_payout_macd_line": macd_line,
        "net_payout_macd_signal": macd_signal,
        "net_payout_macd_histogram": macd_hist,
        "macd_histogram_crossed_below": crossed_below,
        "macd_histogram_crossed_above": crossed_above,
        "vat_deduction_rate": vat_rate,
        "gross_sales_base": float(gross_sales_base),
        "net_payout_latest": float(frame["net_payout"].iloc[-1]),
    }
def build_nova_risk_advisory(
    selected_pair: str,
    rsi_val: float,
    macd_hist: float,
    *,
    vat_rate: float | None = None,
    gross_sales_base: float | None = None,
    macd_line: float = 0.0,
    macd_signal: float = 0.0,
) -> dict[str, str]:
    """
    Executive NOVA Risk Advisory — VAT net payout RSI/MACD saturation filter.
    """
    pair = selected_pair or DEFAULT_PAIR
    quote = OFFSHORE_QUOTE_CCY
    vat_pct = int(round(float(vat_rate or VAT_MAP.get(pair, 0.0)) * 100))
    gross_label = (
        f"{float(gross_sales_base):,.0f}" if gross_sales_base is not None else "N/A"
    )

    base_ccy = PAIR_META.get(pair, {}).get("base_ccy", "FX")
    if rsi_val > 70.0 and macd_hist < 0.0:
        return {
            "signal_class": "risk_alert",
            "text": (
                f"📊 **NOVA Risk Advisory ({pair})**: Your custom tax-adjusted VAT Net Payout RSI "
                f"is highly saturated at {rsi_val:.2f} (Overbought), with a bearish MACD crossover "
                f"({macd_hist:.4f}). Momentum is reversing. **Action Recommended**: Lock in your net "
                f"{base_ccy} proceeds immediately using Window Forwards to immunize against {quote} appreciation."
            ),
        }

    if rsi_val < 30.0 and macd_hist > 0.0:
        return {
            "signal_class": "opportunity",
            "text": (
                f"📊 **NOVA Risk Advisory ({pair})**: Net Payout RSI shows oversold exhaustion at "
                f"{rsi_val:.2f} with bullish MACD histogram divergence ({macd_hist:.4f}). "
                f"**Action Recommended**: Selling pressure is fading; delay immediate spot conversion "
                f"to capture upcoming rebounds."
            ),
        }

    return {
        "signal_class": "normal",
        "text": (
            f"📊 **NOVA Risk Advisory ({pair}) — Corporate Audit Log**: VAT Net Payout momentum is "
            f"within neutral stability bounds. Gross revenue base **{gross_label}** · applied VAT "
            f"**{vat_pct}%** · RSI **{rsi_val:.2f}** · MACD Line **{macd_line:.6f}** · Signal "
            f"**{macd_signal:.6f}** · Histogram **{macd_hist:.6f}** ({quote} desk). "
            f"Maintain baseline hedge ladder; no saturation reversal trigger active."
        ),
    }
def build_net_payout_anti_saturation_advisory(metrics: dict[str, Any]) -> dict[str, str]:
    """Backward-compatible wrapper delegating to `build_nova_risk_advisory`."""
    return build_nova_risk_advisory(
        metrics.get("pair", DEFAULT_PAIR),
        float(metrics.get("net_payout_rsi", 50.0)),
        float(metrics.get("net_payout_macd_histogram", 0.0)),
        vat_rate=metrics.get("vat_deduction_rate"),
        gross_sales_base=metrics.get("gross_sales_base"),
        macd_line=float(metrics.get("net_payout_macd_line", 0.0)),
        macd_signal=float(metrics.get("net_payout_macd_signal", 0.0)),
    )
def _multi_factor_series(
    spot: pd.Series,
    ma5: pd.Series | None = None,
) -> dict[str, Any]:
    """Vectorized multi-factor components aligned with the backtest hedge engine."""
    spot = _sanitize_spot_series(spot.astype(float))
    ma5_series = (
        _sanitize_spot_series(ma5.astype(float))
        if ma5 is not None
        else spot.rolling(window=5, min_periods=1).mean().ffill().bfill()
    )
    ma20 = spot.rolling(window=20, min_periods=1).mean().ffill().bfill()
    std20 = spot.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
    upper_band = ma20 + 2.0 * std20
    ma30 = spot.rolling(window=30, min_periods=1).mean()

    factor_1 = np.where(spot >= ma5_series, 1.0, 0.0)
    factor_2 = np.where(spot >= upper_band, 1.0, 0.0)
    factor_3 = np.where(spot >= ma30, 1.0, 0.0)
    total_score = (factor_1 * 0.4) + (factor_2 * 0.3) + (factor_3 * 0.3)
    hedge_ratio = np.select(
        [total_score >= 0.7, total_score >= 0.4],
        [0.80, 0.55],
        default=0.30,
    )

    return {
        "ma5_series": ma5_series,
        "ma30": ma30,
        "upper_band": upper_band,
        "factor_1": factor_1,
        "factor_2": factor_2,
        "factor_3": factor_3,
        "total_score": total_score,
        "hedge_ratio": hedge_ratio,
    }
def get_live_desk_multi_factor_metrics(
    df: pd.DataFrame,
    pair: str | None = None,
    gross_sales_base: float = 10_000.0,
    custom_vat_rate: float | None = None,
) -> dict[str, Any]:
    """Live desk metrics + MA30, Bollinger upper, multi-factor score, and hedge ratio."""
    metrics = get_fx_metrics(df, pair=pair)
    if not metrics or df.empty:
        return metrics

    spot_series = df["close"].astype(float)
    ma5_input = df["ma5"].astype(float) if "ma5" in df.columns else None
    mf = _multi_factor_series(spot_series, ma5_input)
    idx = -1

    spot = float(spot_series.iloc[idx])
    ma5_val = float(mf["ma5_series"].iloc[idx])
    ma30_val = float(mf["ma30"].iloc[idx])
    bollinger_upper = float(mf["upper_band"].iloc[idx])
    total_score = float(mf["total_score"][idx])
    legacy_hedge_ratio = float(mf["hedge_ratio"][idx])

    metrics.update(
        {
            "ma30": ma30_val,
            "bollinger_upper": bollinger_upper,
            "multi_factor_score": round(total_score, 3),
            "legacy_market_hedge_ratio": legacy_hedge_ratio,
            "factor_trend": float(mf["factor_1"][idx]),
            "factor_bollinger": float(mf["factor_2"][idx]),
            "factor_macro": float(mf["factor_3"][idx]),
        }
    )
    metrics.update(
        compute_net_payout_technical_metrics(
            df,
            pair=metrics.get("pair"),
            gross_sales_base=gross_sales_base,
            custom_vat_rate=custom_vat_rate,
        )
    )

    imf_entry = get_imf_macro_knowledge_entry()
    imf_indicators = (imf_entry or {}).get("macro_indicators")
    china_money = fetch_china_money_central_parity()
    tnx_proxy = fetch_tnx_fedwatch_proxy()

    risk_assessment = compute_environment_risk_assessment(
        metrics,
        market_factor_score=total_score,
        imf_indicators=imf_indicators if isinstance(imf_indicators, dict) else None,
        china_money=china_money,
        tnx_proxy=tnx_proxy,
    )

    metrics.update(risk_assessment)
    nova_advisory = build_nova_risk_advisory(
        metrics.get("pair", DEFAULT_PAIR),
        float(metrics.get("net_payout_rsi", 50.0)),
        float(metrics.get("net_payout_macd_histogram", 0.0)),
        vat_rate=metrics.get("vat_deduction_rate"),
        gross_sales_base=metrics.get("gross_sales_base", gross_sales_base),
        macd_line=float(metrics.get("net_payout_macd_line", 0.0)),
        macd_signal=float(metrics.get("net_payout_macd_signal", 0.0)),
    )
    metrics["net_payout_signal_class"] = nova_advisory.get("signal_class", "normal")
    metrics["net_payout_advisory"] = nova_advisory.get("text", "")
    metrics["nova_risk_advisory"] = metrics["net_payout_advisory"]
    metrics["nova_risk_signal_class"] = metrics["net_payout_signal_class"]
    return metrics
