"""Shared utility helpers."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    ONSHORE_QUOTE_CCY,
    _COLLOQUIAL_ROUTER,
    _MAX_USER_QUERY_LEN,
    _PROMPT_INJECTION_PATTERNS,
)

def _cofer_reserve_pct(cofer: dict[str, Any], currency: str = ONSHORE_QUOTE_CCY) -> Any:
    """COFER sovereign reserve share; tolerates legacy ``RMB`` keys in cached KB snapshots."""
    if currency == ONSHORE_QUOTE_CCY:
        return cofer.get(ONSHORE_QUOTE_CCY, cofer.get("RMB"))
    return cofer.get(currency)
def _fmt_cofer_reserve_label(cofer: dict[str, Any]) -> str:
    """Display-safe COFER CNY/RMB share for macro UI blocks."""
    raw = _cofer_reserve_pct(cofer)
    if raw is None:
        return "N/A"
    try:
        return f"{float(raw)}"
    except (TypeError, ValueError):
        return str(raw)
def sanitize_user_query(query: str) -> str:
    """
    Prompt-injection guard: strip override phrases and cap length before RAG/LLM routing.
    """
    if not isinstance(query, str):
        return ""
    text = query.strip()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_USER_QUERY_LEN]
def _serialize_structural_doc_payload(label: str, payload: Any) -> str:
    """Serialize dense microstructure / macro reasoning blocks for LLM context."""
    if payload is None:
        return ""
    if isinstance(payload, (dict, list)):
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"**{label}:**\n```json\n{serialized}\n```"
    text = str(payload).strip()
    return f"**{label}:** {text}" if text else ""
def _month_end_pulse(dates: pd.DatetimeIndex, scale: float) -> np.ndarray:
    pulse = np.zeros(len(dates))
    for i, d in enumerate(dates):
        if d.day >= 25:
            pulse[i] = -scale * (d.day - 24) / 7
    return pulse
def _calendar_date_index(periods: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Real business-day calendar ending today — professional desk X-axis."""
    end_ts = (end or pd.Timestamp.now()).normalize()
    return pd.date_range(end=end_ts, periods=periods, freq="B")
def _coerce_calendar_dates(series: pd.Series, periods: int) -> pd.DatetimeIndex:
    """Ensure OHLC frames always carry parseable calendar dates, never ordinal labels."""
    parsed = pd.to_datetime(series, errors="coerce")
    if len(parsed) == periods and parsed.notna().all():
        idx = pd.DatetimeIndex(parsed)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx.normalize()
    return _calendar_date_index(periods)
def _sanitize_spot_series(series: pd.Series) -> pd.Series:
    """Forward/backward-fill yfinance spot paths to block NaN propagation in indicators."""
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if cleaned.isna().all():
        return cleaned.fillna(0.0)
    return cleaned
def _sanitize_spot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Harden date/Spot frames (GBPCNH=X and peers) before MA/volatility math."""
    if df is None or df.empty or "Spot" not in df.columns:
        return df
    out = df.copy()
    out["Spot"] = _sanitize_spot_series(out["Spot"])
    if out["Spot"].isna().any():
        out["Spot"] = out["Spot"].ffill().bfill().fillna(0.0)
    if "MA5" in out.columns:
        out["MA5"] = (
            out["Spot"].rolling(window=5, min_periods=1).mean().ffill().bfill()
        )
    return out.dropna(subset=["Spot"]).reset_index(drop=True)
_CROSS_LANG_MAPPING: dict[str, list[str]] = {
    "锁汇": ["hedge", "forward", "lock", "locking"],
    "远期": ["forward", "lock", "locking"],
    "对冲": ["hedge", "exposure"],
    "亏损": ["bleeding", "loss"],
    "亚马逊": ["amazon", "seller"],
    "买": ["buy", "payout"],
    "结汇": ["convert", "settlement"],
    "风险": ["risk", "exposure"],
    "合规": ["audit", "compliance", "safe"],
    "宏观": ["macro", "trend", "q2"],
    "汇率": ["macro", "spot", "trend"],
    "港币": ["hkd", "peg", "depreciation"],
    "港元": ["hkd", "peg", "depreciation"],
    "欧元": ["eur", "euro", "vat"],
    "英镑": ["gbp", "uk", "british", "pound", "vat"],
    "美元": ["usd", "forward", "exposure"],
    "结售汇": ["forward", "settlement", "convert"],
    "人民币": ["cnh", "cny", "forward", "settlement", "convert"],
    "离岸人民币": ["cnh", "offshore", "hibor", "squeeze"],
}
_CURRENCY_ATTACH_CODES = ("usd", "eur", "hkd", "gbp", "cnh", "cny")
def _split_attached_currency_codes(text: str) -> str:
    """Fix typos like 'USDnext' / 'EURpayout' → 'usd next' / 'eur payout'."""
    for code in _CURRENCY_ATTACH_CODES:
        text = re.sub(
            rf"\b({code})(?=[a-z])",
            r"\1 ",
            text,
            flags=re.IGNORECASE,
        )
    return text
def _tokenize(text: str) -> set[str]:
    """
    Tokenize with colloquial English expansion, currency typo splitting,
    and Chinese→English semantic bridge for KB alignment.
    """
    normalized = _split_attached_currency_codes(text.lower())
    tokens: set[str] = set()

    for phrase, mapped in sorted(_COLLOQUIAL_ROUTER.items(), key=lambda x: -len(x[0])):
        if phrase in normalized:
            tokens.update(mapped)
            tokens.add(phrase.replace(" ", "_"))

    for zh_term, en_tags in _CROSS_LANG_MAPPING.items():
        if zh_term in text:
            tokens.update(en_tags)

    tokens.update(re.findall(r"[a-z0-9]{2,}", normalized))
    return tokens
def _trim_to_word_budget(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"
