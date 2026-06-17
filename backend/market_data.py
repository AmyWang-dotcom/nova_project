"""Market data ingestion and FX feed helpers."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

try:
    import streamlit as st
except ImportError:  # pragma: no cover — optional when backend imported outside Streamlit
    st = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

from .constants import (
    ALPHA_VANTAGE_PAIR_MAP,
    CHINA_MACRO_ANCHOR,
    DEFAULT_PAIR,
    HTTP_TIMEOUT_SEC,
    PAIR_META,
    SUPPORTED_PAIRS,
    YFINANCE_PAIR_TICKERS,
)
from .utils import (
    _calendar_date_index,
    _coerce_calendar_dates,
    _month_end_pulse,
    _sanitize_spot_dataframe,
    _sanitize_spot_series,
)

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False
try:
    import streamlit as st
except ImportError:  # pragma: no cover — optional when backend imported outside Streamlit
    st = None  # type: ignore[assignment]
def get_secure_secret(key: str) -> str:
    """Resolve a credential from Streamlit secrets, then OS environment."""
    if st is not None:
        try:
            secrets_obj = getattr(st, "secrets", None)
            if secrets_obj is not None and key in secrets_obj:
                value = secrets_obj[key]
                if value is not None and str(value).strip():
                    return str(value).strip()
        except (FileNotFoundError, KeyError, TypeError, RuntimeError):
            pass
    return os.environ.get(key, "").strip()
def get_secure_openai_key() -> str:
    return get_secure_secret("OPENAI_API_KEY")
def get_secure_alpha_vantage_key() -> str:
    return get_secure_secret("ALPHA_VANTAGE_KEY")
def _network_error_types() -> tuple[type[BaseException], ...]:
    """Exception types captured for outbound HTTP/HTTPS fetches."""
    errs: tuple[type[BaseException], ...] = (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    )
    if REQUESTS_AVAILABLE:
        import requests.exceptions as req_exc

        errs = errs + (req_exc.RequestException,)
    return errs
def _fetch_json_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """GET JSON with strict timeout; returns None on any network/parse failure."""
    hdrs = headers or {}
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, timeout=HTTP_TIMEOUT_SEC, headers=hdrs)
            response.raise_for_status()
            payload = response.json()
        else:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as response:
                payload = json.load(response)
        return payload if isinstance(payload, dict) else None
    except _network_error_types():
        return None
def generate_fx_kline(pair: str = DEFAULT_PAIR, days: int = 30) -> pd.DataFrame:
    """Simulate daily FX bars for supported pairs with pair-specific dynamics."""
    if pair not in PAIR_META:
        raise ValueError(f"Unsupported pair: {pair}. Choose from {SUPPORTED_PAIRS}")

    meta = PAIR_META[pair]
    rng = np.random.default_rng(meta["seed"])
    dates = _calendar_date_index(days)
    n = len(dates)
    t = np.linspace(0, 1, n)

    floor = meta["floor"]
    ceiling = meta["ceiling"]
    center = meta["center"] + meta.get("cycle_amp", 0.04) * np.sin(2 * np.pi * t)
    month_pulse = _month_end_pulse(dates, scale=0.025 if pair != "HKD/CNH" else 0.004)

    close = np.empty(n)
    close[0] = meta["init"]

    if pair == "HKD/CNH":
        peg_mid = meta["peg_mid"]
        weakening = np.linspace(0, meta["weakening_total"], n)
        for i in range(1, n):
            shock = rng.normal(0, meta["shock"])
            target = peg_mid - weakening[i] + 0.003 * np.sin(4 * np.pi * t[i])
            reversion = (target - close[i - 1]) * meta["reversion"]
            close[i] = close[i - 1] + shock + reversion + month_pulse[i]
            if close[i] < floor + 0.002:
                close[i] = floor + rng.uniform(0.002, 0.006)
            elif close[i] > ceiling - 0.002:
                close[i] = ceiling - rng.uniform(0.002, 0.006)
    else:
        for i in range(1, n):
            shock = rng.normal(0, meta["shock"])
            reversion = (center[i] - close[i - 1]) * meta["reversion"]
            close[i] = close[i - 1] + shock + reversion + month_pulse[i]
            if close[i] > ceiling - 0.01:
                close[i] = ceiling - rng.uniform(0.01, 0.03)
            elif close[i] < floor + 0.01:
                close[i] = floor + rng.uniform(0.01, 0.03)

    close = np.clip(close, floor, ceiling)
    decimals = 4

    open_ = np.roll(close, 1)
    open_[0] = close[0] + rng.normal(0, meta["shock"] * 0.6)
    wick = 0.018 if pair != "HKD/CNH" else 0.004
    high = np.maximum(open_, close) + rng.uniform(wick * 0.2, wick, n)
    low = np.minimum(open_, close) - rng.uniform(wick * 0.2, wick, n)
    high = np.clip(high, floor, ceiling)
    low = np.clip(low, floor, ceiling)

    vol_scale = 1e6 if pair != "HKD/CNH" else 3e5
    vol_proxy = np.abs(np.diff(close, prepend=close[0])) * vol_scale + rng.integers(
        int(vol_scale * 2), int(vol_scale * 6), n
    )
    volatility = pd.Series(close).pct_change().abs().fillna(0) * 100

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_, decimals),
            "high": np.round(high, decimals),
            "low": np.round(low, decimals),
            "close": np.round(close, decimals),
            "volume": vol_proxy.astype(int),
            "volatility_pct": np.round(volatility.values * 100, 3),
        }
    )
    df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df.attrs["pair"] = pair
    df.attrs["data_source"] = "simulated"
    return df
def _spot_range_from_series(spot: pd.Series) -> tuple[float, float]:
    """Min/max of a spot close series (full precision, ignores NaN)."""
    clean = pd.to_numeric(spot, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 0.0
    return float(clean.min()), float(clean.max())
def resolve_30d_range_from_kline(df: pd.DataFrame) -> tuple[float, float]:
    """30-day min/max spot range from kline close column (ignores stale attrs / synthetic OHLC)."""
    if df.empty or "close" not in df.columns:
        return 0.0, 0.0
    return _spot_range_from_series(df["close"])
def _attach_spot_range_attrs(spot_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Persist 30-day spot range on the kline frame (derived from close column)."""
    low_30d, high_30d = resolve_30d_range_from_kline(df)
    df.attrs["low_30d"] = low_30d
    df.attrs["high_30d"] = high_30d
    return df
def _spot_frame_to_kline(spot_df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Convert real Spot/MA5 frame into Plotly-ready OHLCV bars.
    Open/High/Low are proportionally derived from Close (~0.5% bands).
    Synthetic wicks are not clipped to PAIR_META floor/ceiling — that bound
    collapsed 30-day range when real spot traded below the simulator floor.
    """
    meta = PAIR_META[pair]
    rng = np.random.default_rng(meta["seed"])

    close = spot_df["Spot"].astype(float).values
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    if n > 1:
        open_[1:] = close[:-1]

    band = close * 0.005
    wick = rng.uniform(0.15, 1.0, n) * band
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    low = np.maximum(low, close * 0.95)

    vol_scale = 1e6 if pair != "HKD/CNH" else 3e5
    vol_proxy = np.abs(np.diff(close, prepend=close[0])) * vol_scale + rng.integers(
        int(vol_scale * 2), int(vol_scale * 6), n
    )
    volatility = pd.Series(close).pct_change().abs().fillna(0) * 100

    ma5 = (
        spot_df["MA5"].astype(float).values
        if "MA5" in spot_df.columns
        else pd.Series(close).rolling(5, min_periods=1).mean().values
    )

    chart_dates = _coerce_calendar_dates(spot_df["date"], n)
    df = pd.DataFrame(
        {
            "date": chart_dates,
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": vol_proxy.astype(int),
            "volatility_pct": np.round(volatility.values * 100, 3),
            "ma5": np.round(ma5, 4),
        }
    )
    df.attrs["pair"] = pair
    df.attrs["data_source"] = spot_df.attrs.get("data_source", "unknown")
    df.attrs["ticker"] = spot_df.attrs.get(
        "ticker", YFINANCE_PAIR_TICKERS.get(pair, "USDCNH=X")
    )
    return df
def fetch_fx_kline(pair: str = DEFAULT_PAIR, days: int = 30) -> pd.DataFrame:
    """Live desk K-line: yfinance Spot ingestion → OHLCV candlestick frame."""
    if pair not in PAIR_META:
        raise ValueError(f"Unsupported pair: {pair}. Choose from {SUPPORTED_PAIRS}")
    spot_df = fetch_real_market_data(days=days, pair=pair)
    df = _spot_frame_to_kline(spot_df, pair)
    return _attach_spot_range_attrs(spot_df, df)
def fetch_usdcnh_kline(days: int = 30) -> pd.DataFrame:
    """Backward-compatible alias."""
    return fetch_fx_kline("USD/CNH", days=days)
def get_fx_metrics(df: pd.DataFrame, pair: str | None = None) -> dict[str, Any]:
    if df.empty:
        return {}

    pair = pair or df.attrs.get("pair", DEFAULT_PAIR)
    meta = PAIR_META[pair]
    base = meta["base_ccy"]

    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    ma5_col = (
        pd.to_numeric(df["ma5"], errors="coerce").ffill().bfill()
        if "ma5" in df.columns
        else close.rolling(window=5, min_periods=1).mean().ffill().bfill()
    )

    latest = df.iloc[-1]
    prev_raw = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
    prev = prev_raw if prev_raw and not np.isnan(prev_raw) else float(close.iloc[-1]) or 1.0
    spot_raw = float(close.iloc[-1])
    ma5_raw = float(ma5_col.iloc[-1])
    pct = (spot_raw - prev) / prev * 100 if prev else 0.0

    p75 = float(close.quantile(0.75))
    p25 = float(close.quantile(0.25))
    spot = spot_raw if spot_raw and not np.isnan(spot_raw) else prev
    ma5 = ma5_raw if ma5_raw and not np.isnan(ma5_raw) else spot

    latest_date_raw = latest["date"]
    if hasattr(latest_date_raw, "strftime"):
        latest_date_str = latest_date_raw.strftime("%Y-%m-%d")
    else:
        latest_date_str = str(latest_date_raw)[:10]

    if spot >= p75:
        risk_level = "HIGH"
        risk_color = "#ef4444"
        if pair == "EUR/CNH":
            risk_advice = (
                "EUR/CNH is elevated: lock 60–75% of net EUR receivables (post-VAT) "
                "with a 3-month forward before remittance week."
            )
        elif pair == "HKD/CNH":
            risk_advice = (
                "HKD is weak vs CNH (low HKD/CNH): accelerate CNH conversion for "
                "approved onshore payables; tranche-convert HKD receipts rather than hoarding."
            )
        elif pair == "GBP/CNH":
            risk_advice = (
                "GBP/CNH is elevated: lock 60–70% of net GBP receivables (post-UK VAT) "
                "with a 3-month forward before Amazon UK remittance week."
            )
        else:
            risk_advice = (
                "USD/CNH is elevated: lock 60–80% of confirmed USD receivables "
                "with a 3-month forward to protect conversion value."
            )
    elif spot <= p25:
        risk_level = "LOW"
        risk_color = "#22c55e"
        if pair == "HKD/CNH":
            risk_advice = (
                "HKD/CNH is firm (HKD strong): favorable for spot CNH conversion; "
                "convert ~40% now, keep buffer for peg-volatility headlines."
            )
        elif pair == "EUR/CNH":
            risk_advice = (
                "EUR/CNH is attractive: spot-convert ~40% of net EUR; "
                "keep 30% unhedged for potential ECB-driven rebound."
            )
        elif pair == "GBP/CNH":
            risk_advice = (
                "GBP/CNH is attractive: spot-convert ~40% of net GBP; "
                "keep 30% unhedged for potential BoE-driven rebound on UK payouts."
            )
        else:
            risk_advice = (
                "USD/CNH is favorable: spot-convert ~40% now; "
                "keep residual exposure for potential USD rebound."
            )
    else:
        risk_level = "MEDIUM"
        risk_color = "#f59e0b"
        risk_advice = (
            f"Neutral {pair} zone: ladder forwards—add 10% hedge for each +50 bps move."
        )

    ccy_word = base
    ma_signal = (
        f"{ccy_word} strengthening (spot > MA5)"
        if spot > ma5
        else f"CNH firming vs {ccy_word} (spot < MA5)"
    )
    if spot > ma5 and risk_level != "LOW":
        risk_advice += f" | MA5 trigger: short-term {ccy_word} strength—prioritize locking."

    low_30d, high_30d = resolve_30d_range_from_kline(df)

    return {
        "pair": pair,
        "pair_label": meta["label"],
        "base_ccy": base,
        "symbol": meta["symbol"],
        "latest_date": latest_date_str,
        "spot": spot,
        "ma5": ma5,
        "pct_change": round(float(pct), 3),
        "high_30d": high_30d,
        "low_30d": low_30d,
        "p75": float(p75),
        "p25": float(p25),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_advice": risk_advice,
        "ma_signal": ma_signal,
        "data_source": df.attrs.get("data_source", "unknown"),
        "ticker": df.attrs.get("ticker", YFINANCE_PAIR_TICKERS.get(pair, "")),
    }
def _extract_close_column(historical_df: pd.DataFrame) -> pd.Series:
    """Resolve Yahoo/Alpha Vantage close column (`Close` or `close` or `Spot`)."""
    for col in ("Close", "close", "Spot"):
        if col in historical_df.columns:
            return pd.to_numeric(historical_df[col], errors="coerce")
    return pd.Series(dtype=float)
CHINA_MONEY_CCPR_URL = (
    "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew"
)
TNX_FEDWATCH_PROXY_TICKER = "^TNX"
_MACRO_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CHINAMONEY_CACHE_FILE = _MACRO_CACHE_DIR / "chinamoney_ccpr.json"
_chinamoney_memory_cache: dict[str, Any] | None = None
def _beijing_trading_date() -> str:
    """Calendar date in Asia/Shanghai for intraday China Money cache keys."""
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
def _load_chinamoney_disk_cache() -> dict[str, Any] | None:
    if not _CHINAMONEY_CACHE_FILE.exists():
        return None
    try:
        with _CHINAMONEY_CACHE_FILE.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("cache_date") == _beijing_trading_date():
            return payload.get("result")
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return None
def _save_chinamoney_disk_cache(result: dict[str, Any]) -> None:
    try:
        _MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _CHINAMONEY_CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(
                {"cache_date": _beijing_trading_date(), "result": result},
                handle,
                ensure_ascii=False,
            )
    except OSError:
        pass
def _parse_chinamoney_ccpr_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract latest official USD/CNY central parity from China Money JSON."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return None

    heads = payload.get("data", {}).get("head", [])
    if not isinstance(heads, list) or "USD/CNY" not in heads:
        return None

    usd_idx = heads.index("USD/CNY")
    latest = records[0]
    values = latest.get("values")
    if not isinstance(values, list) or len(values) <= usd_idx:
        return None

    try:
        usd_cny = float(values[usd_idx])
    except (TypeError, ValueError):
        return None

    return {
        "status": "ok",
        "usd_cny": usd_cny,
        "parity_date": str(latest.get("date", "")),
        "source": "chinamoney",
        "policy_note": (
            "Official onshore CNY (RMB) central parity (USD/CNY) published by CFETS/China Money; "
            "use as regulatory policy boundary vs offshore CNH spot."
        ),
    }
def fetch_china_money_central_parity() -> dict[str, Any]:
    """
    China Money dynamic policy boundary layer.
    Intraday cache (memory + local file). On any failure → {"status": "failed"} only.
    """
    global _chinamoney_memory_cache

    today = _beijing_trading_date()
    if (
        _chinamoney_memory_cache is not None
        and _chinamoney_memory_cache.get("cache_date") == today
    ):
        cached = _chinamoney_memory_cache.get("result")
        if isinstance(cached, dict):
            return cached

    disk_cached = _load_chinamoney_disk_cache()
    if isinstance(disk_cached, dict) and disk_cached.get("status") == "ok":
        _chinamoney_memory_cache = {"cache_date": today, "result": disk_cached}
        return disk_cached

    payload = _fetch_json_url(
        CHINA_MONEY_CCPR_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NOVA-FX-Copilot/1.0; +quant-desk)",
            "Referer": "https://www.chinamoney.com.cn/",
        },
    )
    if payload is None:
        return {"status": "failed"}

    try:
        parsed = _parse_chinamoney_ccpr_payload(payload)
        if parsed is None:
            return {"status": "failed"}

        _chinamoney_memory_cache = {"cache_date": today, "result": parsed}
        _save_chinamoney_disk_cache(parsed)
        return parsed
    except (TypeError, ValueError, KeyError):
        return {"status": "failed"}
def _analyze_tnx_yield_series(
    close: pd.Series,
    *,
    fedwatch_data_source: str = "yfinance",
    ticker_label: str | None = None,
) -> dict[str, Any]:
    """Compute 10Y yield MA5/MA30, Bollinger middle band, and dovish death-cross signal."""
    yields = close.astype(float).dropna().sort_index().tail(30)
    if len(yields) < 10:
        raise ValueError("insufficient TNX history")

    ma5 = yields.rolling(window=5, min_periods=1).mean()
    ma30 = yields.rolling(window=30, min_periods=1).mean()
    bb_mid = yields.rolling(window=20, min_periods=1).mean()

    current = float(yields.iloc[-1])
    ma5_val = float(ma5.iloc[-1])
    ma30_val = float(ma30.iloc[-1])
    bb_mid_val = float(bb_mid.iloc[-1])

    death_cross = ma5_val < ma30_val and current < bb_mid_val
    dovish_signal = death_cross
    rate_cut_probability = 0.80 if dovish_signal else max(0.15, min(0.45, 0.35))

    return {
        "status": "ok",
        "simulated": False,
        "fedwatch_data_source": fedwatch_data_source,
        "ticker": ticker_label or TNX_FEDWATCH_PROXY_TICKER,
        "current_yield": round(current, 4),
        "ma5_yield": round(ma5_val, 4),
        "ma30_yield": round(ma30_val, 4),
        "bollinger_middle": round(bb_mid_val, 4),
        "death_cross": death_cross,
        "dovish_signal": dovish_signal,
        "rate_cut_probability": rate_cut_probability,
        "signal_label": (
            "DOVISH_FED_CUT_BET (Death Cross + below Bollinger mid)"
            if dovish_signal
            else "NEUTRAL_OR_HAWKISH_BIAS"
        ),
        "usd_downside_risk": "ELEVATED" if dovish_signal else "MODERATE",
    }
def _extract_yfinance_close_series(raw: pd.DataFrame) -> pd.Series:
    """Normalize yfinance OHLCV/MultiIndex frames to a single Close price series."""
    if raw is None or raw.empty:
        raise ValueError("empty yfinance dataframe")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        close = raw.iloc[:, 0]

    series = pd.to_numeric(close, errors="coerce").dropna()
    if series.empty:
        raise ValueError("no valid close prices in yfinance frame")
    return series.sort_index()
def _fetch_tnx_yield_series_yfinance() -> pd.Series:
    """Tier 1 — ^TNX daily closes via yfinance."""
    import yfinance as yf

    raw = yf.download(
        TNX_FEDWATCH_PROXY_TICKER,
        period="3mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    return _extract_yfinance_close_series(raw)
def _alphavantage_treasury_payload_error(payload: dict[str, Any]) -> bool:
    """True when Alpha Vantage returns rate-limit or error envelopes instead of yields."""
    if not isinstance(payload, dict):
        return True
    data = payload.get("data")
    if isinstance(data, list) and len(data) >= 10:
        return False
    return any(key in payload for key in ("Note", "Information", "Error Message"))
def _parse_alphavantage_treasury_yield_series(payload: dict[str, Any]) -> pd.Series:
    """Parse TREASURY_YIELD daily 10-year rows into a chronologically sorted yield series."""
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing Alpha Vantage treasury data rows")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_value = row.get("value")
        raw_date = row.get("date")
        if raw_value in (None, ".", ""):
            continue
        try:
            value = float(raw_value)
            dates.append(pd.to_datetime(raw_date))
            values.append(value)
        except (TypeError, ValueError):
            continue

    if len(values) < 10:
        raise ValueError("insufficient Alpha Vantage treasury yield history")

    series = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series
def _fetch_tnx_yield_series_alphavantage() -> pd.Series:
    """Tier 2 — 10Y Treasury daily yields via Alpha Vantage TREASURY_YIELD."""
    api_key = get_secure_alpha_vantage_key()
    if not api_key:
        raise ValueError("Alpha Vantage API key not configured (st.secrets / ALPHA_VANTAGE_KEY)")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TREASURY_YIELD&interval=daily&maturity=10year"
        f"&apikey={api_key}"
    )

    payload = _fetch_json_url(url)
    if payload is None:
        raise ValueError("Alpha Vantage TREASURY_YIELD network timeout or transport error")

    if _alphavantage_treasury_payload_error(payload):
        raise ValueError("Alpha Vantage TREASURY_YIELD payload error or rate limited")

    return _parse_alphavantage_treasury_yield_series(payload)
def _simulate_tnx_fedwatch_proxy() -> dict[str, Any]:
    """Structural simulation for demo continuity when yfinance is unavailable."""
    rng = np.random.default_rng(2026)
    base_yield = 4.15
    trend = np.linspace(0.08, -0.12, 30)
    noise = rng.normal(0, 0.02, 30)
    simulated = pd.Series(base_yield + trend + noise)

    result = _analyze_tnx_yield_series(
        simulated,
        fedwatch_data_source="simulated",
        ticker_label="SIMULATED_10Y_YIELD",
    )
    result["simulated"] = True
    result["rate_cut_probability"] = 0.80
    result["dovish_signal"] = True
    result["death_cross"] = True
    result["signal_label"] = "SIMULATED_DOVISH_FED_CUT_BET (Demo continuity mode)"
    result["usd_downside_risk"] = "ELEVATED"
    return result
def fetch_tnx_fedwatch_proxy() -> dict[str, Any]:
    """
    CME FedWatch cross-asset proxy — 3-tier escalation for US 10Y Treasury yields:
      Tier 1: yfinance ^TNX
      Tier 2: Alpha Vantage TREASURY_YIELD (10year, daily)
      Tier 3: structural simulation (demo continuity)
    Death cross: MA5 < MA30 and spot yield below 20d Bollinger middle → dovish / USD risk-off.
    """
    # Tier 1 — Primary: yfinance ^TNX
    try:
        close = _fetch_tnx_yield_series_yfinance()
        return _analyze_tnx_yield_series(
            close,
            fedwatch_data_source="yfinance",
            ticker_label=TNX_FEDWATCH_PROXY_TICKER,
        )
    except Exception:
        pass

    # Tier 2 — Secondary: Alpha Vantage TREASURY_YIELD
    try:
        close = _fetch_tnx_yield_series_alphavantage()
        return _analyze_tnx_yield_series(
            close,
            fedwatch_data_source="alphavantage",
            ticker_label="TREASURY_YIELD_10Y",
        )
    except Exception:
        pass

    # Tier 3 — Ultimate fallback: structural simulation
    print(
        "Warning: Both yfinance and Alpha Vantage links interrupted. "
        "Entering structural simulation mode.",
        flush=True,
    )
    return _simulate_tnx_fedwatch_proxy()
def _generate_fallback_spot_data(days: int, pair: str = DEFAULT_PAIR) -> pd.DataFrame:
    """Pair-aware Spot/MA5 fallback via the legacy simulator."""
    kline = generate_fx_kline(pair=pair, days=days)
    calendar = _calendar_date_index(days)
    df = pd.DataFrame(
        {
            "date": calendar,
            "Spot": kline["close"].values,
            "MA5": kline["ma5"].values,
        }
    )
    df.attrs["data_source"] = "simulated_fallback"
    df.attrs["ticker"] = YFINANCE_PAIR_TICKERS.get(pair, "USDCNH=X")
    df.attrs["pair"] = pair
    return df
def _generate_fallback_backtest_data(days: int = 180, pair: str = DEFAULT_PAIR) -> pd.DataFrame:
    """Deterministic random-walk fallback when live market feeds are unavailable."""
    return _generate_fallback_spot_data(days=days, pair=pair)
def _alpha_vantage_payload_error(payload: dict[str, Any]) -> bool:
    """True when Alpha Vantage returns rate-limit or error envelopes instead of OHLC."""
    if not isinstance(payload, dict):
        return True
    if payload.get("Time Series FX (Daily)"):
        return False
    return any(key in payload for key in ("Note", "Information", "Error Message"))
def _fetch_alpha_vantage_spot_data(
    days: int,
    pair: str,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    """Ingest Alpha Vantage FX_DAILY close series for a supported pair."""
    api_key = get_secure_alpha_vantage_key()
    if not api_key:
        return None

    from_ccy, to_ccy = ALPHA_VANTAGE_PAIR_MAP[pair]
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={from_ccy}&to_symbol={to_ccy}"
        f"&apikey={api_key}"
    )

    payload = _fetch_json_url(url)
    if payload is None:
        return None

    if _alpha_vantage_payload_error(payload):
        return None

    series = payload.get("Time Series FX (Daily)")
    if not isinstance(series, dict) or not series:
        return None

    rows: list[dict[str, Any]] = []
    for date_str, bar in series.items():
        if not isinstance(bar, dict):
            continue
        close_raw = bar.get("4. close")
        if close_raw is None:
            continue
        spot = pd.to_numeric(close_raw, errors="coerce")
        if pd.isna(spot):
            continue
        rows.append({"date": pd.to_datetime(date_str), "Spot": float(spot)})

    if not rows:
        return None

    out = pd.DataFrame(rows).dropna(subset=["Spot"]).sort_values("date")
    if len(out) < min_rows:
        return None

    out["Spot"] = out["Spot"].round(4)
    out["MA5"] = out["Spot"].rolling(window=5, min_periods=1).mean().ffill().bfill()
    out = _sanitize_spot_dataframe(out.tail(days).reset_index(drop=True))
    out.attrs["data_source"] = "Alpha Vantage"
    out.attrs["ticker"] = f"{from_ccy}/{to_ccy}"
    out.attrs["pair"] = pair
    return out
def _fetch_yfinance_spot_data(
    days: int,
    pair: str,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    """Backup live ingestion via Yahoo Finance when Alpha Vantage is unavailable."""
    ticker = YFINANCE_PAIR_TICKERS.get(pair, "USDCNH=X")

    try:
        import yfinance as yf

        yf_ticker = yf.Ticker(ticker)
        for period in ("1y", "6mo", "3mo", f"{max(days, 30)}d"):
            hist = yf_ticker.history(period=period, interval="1d", auto_adjust=True)
            parsed = _normalize_yfinance_frame(
                hist, days, ticker=ticker, pair=pair, min_rows=min_rows
            )
            if parsed is not None:
                return parsed

        raw = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        parsed = _normalize_yfinance_frame(
            raw, days, ticker=ticker, pair=pair, min_rows=min_rows
        )
        if parsed is not None:
            return parsed

        end = datetime.now()
        start = end - pd.Timedelta(days=int(days * 1.5))
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        return _normalize_yfinance_frame(
            raw, days, ticker=ticker, pair=pair, min_rows=min_rows
        )
    except Exception:
        return None
def _extract_close_series(raw: pd.DataFrame) -> pd.Series | None:
    """Extract Close prices from flat or MultiIndex yfinance frames."""
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        for col in raw.columns:
            if str(col[0]).lower() == "close":
                series = raw[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series
        return None

    for name in ("Close", "close", "Adj Close"):
        if name in raw.columns:
            series = raw[name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series
    return None
def _normalize_yfinance_frame(
    raw: pd.DataFrame,
    days: int,
    ticker: str,
    pair: str,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    """Parse yfinance OHLCV into date / Spot / MA5."""
    close = _extract_close_series(raw)
    if close is None:
        return None

    if isinstance(raw.index, pd.DatetimeIndex):
        dates = pd.to_datetime(raw.index)
    elif "Date" in raw.columns:
        dates = pd.to_datetime(raw["Date"])
    elif "date" in raw.columns:
        dates = pd.to_datetime(raw["date"])
    else:
        return None

    if isinstance(dates, pd.DatetimeIndex) and dates.tz is not None:
        dates = dates.tz_localize(None)

    spot_series = _sanitize_spot_series(pd.Series(close.values, dtype=float))
    out = pd.DataFrame({"date": dates, "Spot": spot_series.values})
    out = out.dropna(subset=["Spot"]).sort_values("date")
    if len(out) < min_rows:
        return None

    out["Spot"] = out["Spot"].round(4)
    out["MA5"] = out["Spot"].rolling(window=5, min_periods=1).mean().ffill().bfill()
    out = _sanitize_spot_dataframe(out.tail(days).reset_index(drop=True))
    if out is None or len(out) < min_rows:
        return None
    out.attrs["data_source"] = "yfinance"
    out.attrs["ticker"] = ticker
    out.attrs["pair"] = pair
    return out
def fetch_real_market_data(
    days: int = 180,
    pair: str = DEFAULT_PAIR,
) -> pd.DataFrame:
    """
    Market data pipeline: Alpha Vantage FX_DAILY → yfinance backup → simulator fallback.
    Returns date / Spot / MA5 for all SUPPORTED_PAIRS (incl. GBP/CNH).
    """
    if pair not in PAIR_META:
        raise ValueError(f"Unsupported pair: {pair}. Choose from {SUPPORTED_PAIRS}")

    min_rows = max(5, min(20, days // 2))

    alpha_df = _fetch_alpha_vantage_spot_data(days=days, pair=pair, min_rows=min_rows)
    if alpha_df is not None:
        return alpha_df

    yfinance_df = _fetch_yfinance_spot_data(days=days, pair=pair, min_rows=min_rows)
    if yfinance_df is not None:
        return yfinance_df

    return _generate_fallback_backtest_data(days=days, pair=pair)
def generate_backtest_data(days: int = 180) -> pd.DataFrame:
    """Backward-compatible alias for USD/CNH backtest ingestion."""
    return fetch_real_market_data(days=days, pair="USD/CNH")
