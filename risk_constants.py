"""
NOVA hedge-ratio tier mapping — single source of truth for UI and backend.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypedDict


class RiskPreferenceMapping(TypedDict):
    high_threshold: int
    low_threshold: int
    high_ratio: float
    medium_ratio: float
    low_ratio: float


DEFAULT_MAPPING: RiskPreferenceMapping = {
    "high_threshold": 67,
    "low_threshold": 33,
    "high_ratio": 0.80,
    "medium_ratio": 0.55,
    "low_ratio": 0.30,
}

HEDGE_RATIO_HELP_TEXT = (
    "Hedge ratio is determined by the final composite risk score: "
    "≥67 → 80%, 34–66 → 55%, ≤33 → 30%."
)

SEGMENT_COLORS = {
    "low": "#22c55e",
    "medium": "#eab308",
    "high": "#ef4444",
}


def normalize_mapping(raw: dict[str, Any] | None) -> RiskPreferenceMapping:
    """Coerce user/session input into a valid mapping."""
    base = deepcopy(DEFAULT_MAPPING)
    if not raw:
        return base
    for key in base:
        if key in raw and raw[key] is not None:
            base[key] = type(base[key])(raw[key])  # type: ignore[call-overload]
    low_th = int(base["low_threshold"])
    high_th = int(base["high_threshold"])
    if low_th >= high_th:
        base["low_threshold"] = DEFAULT_MAPPING["low_threshold"]
        base["high_threshold"] = DEFAULT_MAPPING["high_threshold"]
    base["high_threshold"] = max(1, min(99, int(base["high_threshold"])))
    base["low_threshold"] = max(0, min(base["high_threshold"] - 1, int(base["low_threshold"])))
    for ratio_key in ("high_ratio", "medium_ratio", "low_ratio"):
        base[ratio_key] = max(0.0, min(1.0, float(base[ratio_key])))
    return base


def hedge_ratio_from_score(
    score: float,
    mapping: dict[str, Any] | None = None,
) -> float:
    """Map final composite risk score (0–100) to hedge ratio."""
    m = normalize_mapping(mapping)
    if score >= m["high_threshold"]:
        return float(m["high_ratio"])
    if score > m["low_threshold"]:
        return float(m["medium_ratio"])
    return float(m["low_ratio"])


def format_mapping_rules(mapping: dict[str, Any] | None = None) -> list[str]:
    """Human-readable tier rules for tooltips and mapping preview."""
    m = normalize_mapping(mapping)
    mid_lo = m["low_threshold"] + 1
    mid_hi = m["high_threshold"] - 1
    return [
        f"Score ≥ {m['high_threshold']} → {m['high_ratio']:.0%}",
        f"Score {mid_lo}–{mid_hi} → {m['medium_ratio']:.0%}",
        f"Score ≤ {m['low_threshold']} → {m['low_ratio']:.0%}",
    ]


def resolve_display_hedge_ratio(
    metrics: dict[str, Any],
    mapping: dict[str, Any] | None = None,
) -> float | None:
    """Recompute hedge ratio from final score when a custom mapping is active."""
    final_score = metrics.get("final_risk_score")
    if final_score is not None:
        return hedge_ratio_from_score(float(final_score), mapping)
    raw = metrics.get("recommended_hedge_ratio")
    if raw is None:
        return None
    return float(raw)
