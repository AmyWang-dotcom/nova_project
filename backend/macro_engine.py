"""Macro and policy risk intelligence layer."""
from __future__ import annotations

from typing import Any

from .constants import CHINA_MACRO_ANCHOR, DEFAULT_PAIR, IMF_MACRO_KB_ID, _PAIR_RESERVE_CCY
from .knowledge_base import (
    compute_imf_ppp_spreads,
    format_imf_macro_indicators_block,
    get_imf_macro_knowledge_entry,
)
from .market_data import fetch_china_money_central_parity, fetch_tnx_fedwatch_proxy, TNX_FEDWATCH_PROXY_TICKER

def compute_macro_risk_score(
    pair: str,
    imf_indicators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """IMF CPI / GDP / COFER structural risk score on a 0–100 scale."""
    from .risk_engine import _normalize_component_score, _risk_rating_from_score

    entry = get_imf_macro_knowledge_entry()
    indicators = imf_indicators or (entry or {}).get("macro_indicators", {})
    if not isinstance(indicators, dict):
        return {
            "macro_score": 0.0,
            "components": {},
            "macro_risk_level": "LOW",
        }

    cpi = indicators.get("inflation_cpi_yoy", {})
    gdp = indicators.get("gdp_growth_forecast", {})
    cofer = indicators.get("cofer_sovereign_reserves_share", {})
    spreads = compute_imf_ppp_spreads(pair, indicators)

    dest_label = spreads.get("destination_label", "US")
    china_cpi = float(spreads.get("china_cpi", 0.0))
    dest_cpi = float(spreads.get("destination_cpi", china_cpi))
    inflation_spread = float(spreads.get("cpi_spread_vs_china", dest_cpi - china_cpi))

    china_gdp = float(gdp.get(CHINA_MACRO_ANCHOR, gdp.get("China", 0.0)))
    dest_gdp = float(gdp.get(dest_label, 0.0))
    gdp_spread = china_gdp - dest_gdp

    reserve_ccy = _PAIR_RESERVE_CCY.get(pair, "USD")
    reserve_share = float(cofer.get(reserve_ccy, 0.0))

    if inflation_spread > 2.0:
        inflation_points = 25
    elif inflation_spread > 1.0:
        inflation_points = 15
    elif inflation_spread > 0.0:
        inflation_points = 5
    else:
        inflation_points = 0

    if gdp_spread > 3.0:
        gdp_points = 25
    elif gdp_spread > 1.0:
        gdp_points = 15
    else:
        gdp_points = 5

    if reserve_share > 50.0:
        cofer_points = 25
    elif reserve_share > 40.0:
        cofer_points = 15
    else:
        cofer_points = 5

    raw_points = inflation_points + gdp_points + cofer_points
    macro_score = _normalize_component_score(raw_points, 75.0)

    return {
        "macro_score": macro_score,
        "macro_risk_level": _risk_rating_from_score(macro_score),
        "components": {
            "destination_label": dest_label,
            "destination_cpi": round(dest_cpi, 3),
            "china_cpi": round(china_cpi, 3),
            "inflation_spread_pp": round(inflation_spread, 2),
            "inflation_points": inflation_points,
            "destination_gdp": round(dest_gdp, 3),
            "china_gdp": round(china_gdp, 3),
            "gdp_spread_pp": round(gdp_spread, 2),
            "gdp_points": gdp_points,
            "reserve_currency": reserve_ccy,
            "reserve_share_pct": round(reserve_share, 2),
            "cofer_points": cofer_points,
            "raw_points": raw_points,
        },
    }
def _fedwatch_policy_points(tnx_proxy: dict[str, Any] | None) -> tuple[int, str]:
    if not tnx_proxy or tnx_proxy.get("status") != "ok":
        return 15, "neutral"

    if tnx_proxy.get("dovish_signal"):
        return 5, "dovish"

    ma5 = float(tnx_proxy.get("ma5_yield", 0.0))
    ma30 = float(tnx_proxy.get("ma30_yield", 0.0))
    current = float(tnx_proxy.get("current_yield", 0.0))
    bb_mid = float(tnx_proxy.get("bollinger_middle", current))
    if ma5 > ma30 and current > bb_mid:
        return 25, "hawkish"
    return 15, "neutral"
def compute_policy_risk_score(
    spot: float,
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """FedWatch proxy + CFETS parity deviation policy risk on a 0–100 scale."""
    from .risk_engine import _normalize_component_score, _risk_rating_from_score

    china_money = china_money if china_money is not None else fetch_china_money_central_parity()
    tnx_proxy = tnx_proxy if tnx_proxy is not None else fetch_tnx_fedwatch_proxy()

    fedwatch_points, fedwatch_state = _fedwatch_policy_points(tnx_proxy)

    parity = None
    deviation_bps = 0.0
    if china_money.get("status") == "ok" and china_money.get("usd_cny") is not None:
        parity = float(china_money["usd_cny"])
        if parity:
            deviation_bps = abs(float(spot) - parity) / parity * 10_000.0

    if deviation_bps > 300:
        cfets_points = 25
    elif deviation_bps > 150:
        cfets_points = 15
    else:
        cfets_points = 5

    raw_points = fedwatch_points + cfets_points
    policy_score = _normalize_component_score(raw_points, 50.0)

    return {
        "policy_score": policy_score,
        "policy_risk_level": _risk_rating_from_score(policy_score),
        "components": {
            "fedwatch_state": fedwatch_state,
            "fedwatch_points": fedwatch_points,
            "fedwatch_signal_label": tnx_proxy.get("signal_label", "N/A"),
            "cfets_parity": round(parity, 4) if parity is not None else None,
            "cfets_deviation_bps": round(deviation_bps, 1),
            "cfets_points": cfets_points,
            "raw_points": raw_points,
        },
    }
def build_macro_intelligence_layer(
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
    imf_indicators: dict[str, Any] | None = None,
    *,
    include_imf: bool = True,
) -> str:
    """
    Assemble [REAL-TIME MACRO INTELLIGENCE LAYER].
    China Money lines appear ONLY on successful fetch — zero placeholders on failure.
    IMF structured baseline is always injected when available in the knowledge base.
    """
    china_money = china_money if china_money is not None else fetch_china_money_central_parity()
    tnx_proxy = tnx_proxy if tnx_proxy is not None else fetch_tnx_fedwatch_proxy()

    lines: list[str] = ["[REAL-TIME MACRO INTELLIGENCE LAYER]"]

    if include_imf:
        imf_block = format_imf_macro_indicators_block(imf_indicators)
        if imf_block:
            lines.append(imf_block)

    if china_money.get("status") == "ok" and china_money.get("usd_cny") is not None:
        lines.extend(
            [
                "- **China Money Policy Boundary (CFETS Central Parity)**",
                f"  - USD/CNY official parity: **{float(china_money['usd_cny']):.4f}** "
                f"(fixing date: {china_money.get('parity_date', 'N/A')})",
                f"  - Interpretation: {china_money.get('policy_note', '')}",
            ]
        )

    if tnx_proxy.get("status") == "ok":
        feed = tnx_proxy.get("fedwatch_data_source", "unknown")
        if tnx_proxy.get("simulated"):
            feed_label = " [SIMULATED]"
        elif feed == "alphavantage":
            feed_label = " [Alpha Vantage TREASURY_YIELD 10Y]"
        elif feed == "yfinance":
            feed_label = " [yfinance ^TNX]"
        else:
            feed_label = ""
        ticker = tnx_proxy.get("ticker", TNX_FEDWATCH_PROXY_TICKER)
        lines.extend(
            [
                f"- **CME FedWatch Proxy — US 10Y Treasury ({ticker}){feed_label}**",
                f"  - Current yield: **{tnx_proxy.get('current_yield', 0):.4f}%** | "
                f"MA5: **{tnx_proxy.get('ma5_yield', 0):.4f}%** | "
                f"MA30: **{tnx_proxy.get('ma30_yield', 0):.4f}%** | "
                f"Bollinger mid (20d): **{tnx_proxy.get('bollinger_middle', 0):.4f}%**",
                f"  - Death cross (MA5 < MA30) + below Bollinger mid: **{tnx_proxy.get('death_cross')}**",
                f"  - Signal: **{tnx_proxy.get('signal_label')}** | "
                f"Implied Fed cut probability: **{float(tnx_proxy.get('rate_cut_probability', 0)):.0%}** | "
                f"USD downside risk: **{tnx_proxy.get('usd_downside_risk')}**",
            ]
        )

    if len(lines) == 1:
        return ""

    return "\n".join(lines)
def _rag_retrieved_imf_entry(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return IMF KB entry when RAG retrieval surfaces the macro inflation anchor."""
    for doc in docs:
        if doc.get("id") == IMF_MACRO_KB_ID:
            return doc
    return get_imf_macro_knowledge_entry()
def _build_imf_macro_application_advisory(
    pair: str,
    metrics: dict[str, Any] | None = None,
    macro_indicators: dict[str, Any] | None = None,
) -> str:
    """Desk-facing IMF PPP / GDP / COFER synthesis for long-horizon hedge questions."""
    indicators = macro_indicators
    if indicators is None:
        entry = get_imf_macro_knowledge_entry()
        indicators = (entry or {}).get("macro_indicators")
    if not isinstance(indicators, dict):
        return ""

    spreads = compute_imf_ppp_spreads(pair, indicators)
    cpi = indicators.get("inflation_cpi_yoy", {})
    gdp = indicators.get("gdp_growth_forecast", {})
    cofer = indicators.get("cofer_sovereign_reserves_share", {})

    dest = spreads["destination_label"]
    spread = spreads["cpi_spread_vs_china"]
    ma30 = metrics.get("ma30") if metrics else None
    hedge = metrics.get("recommended_hedge_ratio") if metrics else None

    lines = [
        "**📊 IMF Macro Application (PPP · GDP · COFER)**",
        f"- CPI spread ({dest} vs China): **{spread:+.1f} pp** "
        f"({spreads['destination_cpi']}% vs {spreads['china_cpi']}%) → "
        "PPP implies destination-currency purchasing power erosion on 90–180d receivables; "
        "bias toward strategic forward hedge.",
        f"- GDP velocity: US **{gdp.get('US')}%** vs Eurozone **{gdp.get('Eurozone')}%** "
        f"vs China **{gdp.get('China')}%** → relative capital-flow support for USD over EUR.",
        f"- COFER anchor: USD **{cofer.get('USD')}%** global reserves share → "
        "structural liquidity backstop in tail-risk panics.",
    ]
    if ma30 is not None and hedge is not None:
        lines.append(
            f"- Desk integration ({pair}): MA30 **{float(ma30):.4f}** × "
            f"multi-factor hedge **{float(hedge):.1%}** — align IMF 90–180d budget "
            "with technical overlay."
        )
    return "\n".join(lines)
def _build_macro_synthesis_advisory(
    metrics: dict[str, Any],
    china_money: dict[str, Any] | None,
    tnx_proxy: dict[str, Any] | None,
) -> str:
    """Desk-facing synthesis tying macro layers to hedge ratio guidance."""
    lines: list[str] = ["**🌐 Macro × Multi-Factor Synthesis**"]
    spot = metrics.get("spot")
    hedge = metrics.get("recommended_hedge_ratio")
    pair = metrics.get("pair", DEFAULT_PAIR)

    if china_money and china_money.get("usd_cny") is not None and spot is not None:
        parity = float(china_money["usd_cny"])
        spot_f = float(spot)
        deviation_bps = (spot_f - parity) / parity * 10_000
        lines.append(
            f"- China Money USD/CNY parity **{parity:.4f}** vs desk spot **{spot_f:.4f}** "
            f"→ offshore deviation **{deviation_bps:+.0f} bps** "
            f"({'CNH rich vs policy mid' if deviation_bps > 0 else 'CNH cheap vs policy mid'})."
        )

    if tnx_proxy and tnx_proxy.get("status") == "ok":
        if tnx_proxy.get("dovish_signal"):
            lines.append(
                f"- ^TNX dovish structure (death cross + below Bollinger mid) → "
                f"**{float(tnx_proxy.get('rate_cut_probability', 0.8)):.0%}** implied Fed-cut pricing; "
                "favor defensive USD/CNH locks on receivables."
            )
        else:
            lines.append(
                "- ^TNX structure neutral/hawkish → USD supported; keep baseline hedge, "
                "avoid over-locking sub-MA5 spot dips."
            )

    if hedge is not None:
        lines.append(
            f"- NOVA multi-factor overlay on **{pair}**: recommended hedge "
            f"**{float(hedge):.1%}** (integrate with IMF 90d+ playbook when tail risk)."
        )

    return "\n".join(lines) if len(lines) > 1 else ""
