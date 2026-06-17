"""Agent prompt assembly and RAG response builders."""
from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any

import numpy as np

from .constants import CHINA_MACRO_ANCHOR, DEFAULT_PAIR, IMF_MACRO_KB_ID, PAIR_META
from .knowledge_base import (
    compute_imf_ppp_spreads,
    format_imf_macro_indicators_block,
    get_imf_macro_knowledge_entry,
)
from .macro_engine import (
    _build_imf_macro_application_advisory,
    _build_macro_synthesis_advisory,
    _rag_retrieved_imf_entry,
    build_macro_intelligence_layer,
)
from .market_data import fetch_china_money_central_parity, fetch_tnx_fedwatch_proxy
from .retrieval import (
    _simple_semantic_search,
    assemble_retrieved_playbooks_context,
    retrieve_knowledge,
)
from .risk_engine import (
    build_risk_score_explanation,
    compute_nova_multidimensional_risk_assessment,
)
from .utils import _tokenize, _trim_to_word_budget, sanitize_user_query

def build_live_market_context_prompt(metrics: dict[str, Any]) -> str:
    """High-priority live desk snapshot injected ahead of each copilot user turn."""
    pair = metrics.get("pair", DEFAULT_PAIR)
    spot = float(metrics.get("spot", 0.0))
    ma5_val = float(metrics.get("ma5", 0.0))
    ma30_val = float(metrics.get("ma30", spot))
    bollinger_upper = float(metrics.get("bollinger_upper", spot))
    risk_level = metrics.get("multi_factor_risk_rating", metrics.get("risk_level", "MEDIUM"))
    hedge_ratio = metrics.get("recommended_hedge_ratio")
    enterprise_ready = bool(metrics.get("enterprise_assessment_ready"))
    data_source = metrics.get("data_source", "unknown")
    latest_date = metrics.get("latest_date", "")

    enterprise_lines = ""
    if enterprise_ready and hedge_ratio is not None and risk_level is not None:
        enterprise_lines = (
            f"- Knowledge Risk Score (0–100): {metrics.get('knowledge_score', 0):.1f}\n"
            f"- Final Composite Risk Score: {metrics.get('final_risk_score', 0):.1f} / 100\n"
            f"- Enterprise Risk Rating: {risk_level}\n"
            f"- Recommended Enterprise Hedge Ratio: {float(hedge_ratio):.1%}\n"
        )
    else:
        enterprise_lines = (
            "- Enterprise Risk Assessment: **PENDING** — awaiting FX exposure scenario from user.\n"
            "- Environment Monitoring only: Market / Macro / Policy layers active.\n"
        )

    return (
        "[SYSTEM LIVE MARKET CONTEXT - HIGH PRIORITY]\n"
        f"- Trading Desk: {pair}\n"
        f"- Data Feed: {data_source} (as of {latest_date})\n"
        f"- Real-time Spot: {spot:.4f}\n"
        f"- MA5 Dynamics: {ma5_val:.4f}\n"
        f"- MA30 Macro Trend: {ma30_val:.4f}\n"
        f"- Bollinger Upper Barrier (20d, 2σ): {bollinger_upper:.4f}\n"
        f"- Legacy Market Factor Score (0–1): {metrics.get('multi_factor_score', 0):.3f}\n"
        f"- Market Risk Score (0–100): {metrics.get('market_score', 0):.1f}\n"
        f"- Macro Risk Score (0–100): {metrics.get('macro_score', 0):.1f}\n"
        f"- Policy Risk Score (0–100): {metrics.get('policy_score', 0):.1f}\n"
        f"{enterprise_lines}"
        f"- Factor Flags — Trend(MA5): {int(metrics.get('factor_trend', 0))} | "
        f"Bollinger: {int(metrics.get('factor_bollinger', 0))} | "
        f"Macro(MA30): {int(metrics.get('factor_macro', 0))}\n"
        f"- VAT Net Payout RSI (applied VAT {int(float(metrics.get('vat_deduction_rate', 0)) * 100)}%): "
        f"{float(metrics.get('net_payout_rsi', 50.0)):.2f}\n"
        f"- VAT Net Payout MACD Histogram: "
        f"{float(metrics.get('net_payout_macd_histogram', 0.0)):.6f}\n"
        f"- Anti-Saturation Copilot Signal: "
        f"{metrics.get('net_payout_signal_class', 'normal').upper()}\n"
        f"- Advisory: {metrics.get('net_payout_advisory', 'N/A')}\n"
        "\n"
        'Cross-reference the "Corporate FX Risk Management Manual" (retrieved via RAG) '
        "with the LIVE snapshot above to deliver an explicit, mathematical execution advisory. "
        "If the user states any numeric payout or receivable exposure, you MUST compute the "
        "precise base-currency notional to forward-lock using the Calculated Multi-Factor Hedge "
        "Ratio and Real-time Spot, and show implied CNH settlement."
    )
NOVA_AGENT_SYSTEM_INSTRUCTIONS = """
You are NOVA, an institutional-grade cross-border FX Risk Copilot built for global enterprise treasuries. You process multi-factor market data, macro IMF policy boundaries, and tech indicators into robust hedging strategies.

For every single query, you must structure your response using the exact three-stage analytical pipeline below. Your text must be professional, compact, and completely in English (Reuters standard), weaving dynamic data points naturally into the reasoning layer:

### ⚡ NOVA Multi-Factor Risk Advisory

**1. Quantitative Ingestion & Market Signals [DATA]**
- State the relative positioning of the current Spot Rate against the MA5/MA30 factor corridor and the 30-Day Range boundaries.
- Seamlessly weave in active numbers from the macro engine: the specific IMF 2026 growth anchor status, current sovereign yield spreads (like Alpha Vantage 10Y Treasury), and the CFETS Central Parity boundary drift.

**2. Transmission Channel & Treasury Risk Impact [ANALYSIS]**
- Explain how the observed macro rate/policy spreads interact with short-term quantitative momentum (RSI/MACD saturation or exhaustion traps).
- Detail the impact on cross-border merchant micro-revenues (e.g., how platform payout cycles or VAT withholding adjustments are affected by this technical/macro convergence).

**3. Optimized Execution Framework [CONCLUSION]**
- Deliver a crystal-clear, non-ambiguous risk management directive (e.g., Layered Window Forward Execution, Delayed Discretionary Spot Clearance, or Meta/Google Ad Spend Asset Matching).
- Specify exact execution metrics: hedge ratios (e.g., 35% Layered Hedge) and protective price brackets to immunize cash flows from immediate volatility shocks.

Negative Constraint: Do not print raw JSON dictionary dumps, and never exceed 200 words in total. Maintain a cohesive, bulleted, and executive-ready typography.
"""
NOVA_IMF_MACRO_REASONING_RULES = (
    "IMF MACRO APPLICATION RULES (mandatory when IMF quantitative baseline is present):\n"
    "1) Analyze Inflation Spreads (PPP): Explicitly calculate the structural CPI gap between the "
    "destination market (US at 3.4% or Eurozone at 2.4%) and the home jurisdiction (China at 0.3%). "
    "State the spread in percentage points and explain that, under Purchasing Power Parity, a higher "
    "inflation differential erodes the destination currency's purchasing power over a 90–180-day tail, "
    "justifying a strategic forward hedge on cross-border receivables.\n"
    "2) Synthesize GDP Velocity: Reference GDP growth forecasts (US 2.7% vs Eurozone 0.8% vs China 4.6%) "
    "to explain relative capital-flow and yield support—e.g., stronger US GDP velocity vs Eurozone stagnation "
    "supports USD over EUR on a fundamental basis when budgeting multi-currency locks.\n"
    "3) Sovereign Trust Check: Cite COFER sovereign reserve shares (USD 58.4%, EUR 19.8%, CNY 2.3%) to "
    "reassure the merchant that USD liquidity remains the ultimate structural anchor during extreme market "
    "panics; use this when users ask about long-term lock-in, treasury safety, or 90–180-day budgeting.\n"
    "4) For long-term lock-in or financial budgeting questions, integrate IMF metrics with the live desk "
    "MA30/Bollinger/multi-factor hedge ratio—never invent IMF figures beyond those injected."
)
def build_nova_system_instructions(include_imf_rules: bool = True) -> str:
    """Assemble NOVA system prompt with optional IMF reasoning enforcement."""
    if include_imf_rules and get_imf_macro_knowledge_entry():
        return f"{NOVA_AGENT_SYSTEM_INSTRUCTIONS}\n\n{NOVA_IMF_MACRO_REASONING_RULES}"
    return NOVA_AGENT_SYSTEM_INSTRUCTIONS
def generate_dynamic_agent_prompt(
    user_query: str,
    metrics: dict[str, Any] | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Final brain: aggregate IMF baseline, China Money (success-only), ^TNX Fed proxy, live desk.
    Returns full agent prompt string plus structured macro state for the UI/RAG pipeline.
    """
    user_query = sanitize_user_query(user_query)
    imf_entry = _rag_retrieved_imf_entry(retrieved_docs or [])
    imf_indicators = (imf_entry or {}).get("macro_indicators")
    if not isinstance(imf_indicators, dict):
        imf_entry = get_imf_macro_knowledge_entry()
        imf_indicators = (imf_entry or {}).get("macro_indicators")

    china_money = fetch_china_money_central_parity()
    tnx_proxy = fetch_tnx_fedwatch_proxy()
    macro_layer = build_macro_intelligence_layer(
        china_money, tnx_proxy, imf_indicators=imf_indicators
    )
    live_desk_layer = build_live_market_context_prompt(metrics or {}) if metrics else ""
    system_instructions = build_nova_system_instructions(include_imf_rules=True)

    playbooks_layer = assemble_retrieved_playbooks_context(retrieved_docs)

    sections = [system_instructions]
    if macro_layer:
        sections.append(macro_layer)
    if live_desk_layer:
        sections.append(live_desk_layer)
    if playbooks_layer:
        sections.append(playbooks_layer)
    sections.append(f"User question:\n{user_query}")

    agent_prompt = "\n\n".join(sections)
    active_pair = (metrics or {}).get("pair", DEFAULT_PAIR)

    return {
        "agent_prompt": agent_prompt,
        "system_instructions": system_instructions,
        "macro_layer": macro_layer,
        "imf_macro_block": format_imf_macro_indicators_block(imf_indicators),
        "imf_macro_indicators": imf_indicators if isinstance(imf_indicators, dict) else None,
        "imf_ppp_spreads": compute_imf_ppp_spreads(active_pair, imf_indicators)
        if isinstance(imf_indicators, dict)
        else None,
        "live_desk_layer": live_desk_layer,
        "user_query": user_query,
        "china_money": china_money if china_money.get("status") == "ok" else None,
        "tnx_proxy": tnx_proxy if tnx_proxy.get("status") == "ok" else None,
        "macro_available": bool(macro_layer),
        "imf_retrieved": bool(imf_entry),
        "playbooks_layer": playbooks_layer,
        "retrieved_docs": list(retrieved_docs or []),
    }
_EXPOSURE_MULTIPLIERS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:billion|bn)\b", re.IGNORECASE), 1_000_000_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:million|mil|mio|mn)\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*百万", re.IGNORECASE), 1_000_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*万", re.IGNORECASE), 10_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*[kK]\b"), 1_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*m\b(?![a-z])", re.IGNORECASE), 1_000_000),
]
def _parse_notional_exposure(query: str, base_ccy: str) -> float | None:
    """Extract a numeric receivable/payout exposure from natural language."""
    normalized = query.replace(",", "")
    for pattern, multiplier in _EXPOSURE_MULTIPLIERS:
        match = pattern.search(normalized)
        if match:
            return float(match.group(1)) * multiplier

    ccy_pattern = (
        r"(?:usd|eur|hkd|dollar|dollars|euro|euros|"
        + re.escape(base_ccy.lower())
        + r"|\$|€|£)\s*([\d,]+(?:\.\d+)?)"
    )
    match = re.search(ccy_pattern, normalized, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))

    match = re.search(
        r"([\d,]+(?:\.\d+)?)\s*(?:usd|eur|hkd|gbp|dollar|dollars|euro|euros|pound|pounds)\b",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1).replace(",", ""))

    plain = re.search(r"\b([\d,]{4,}(?:\.\d+)?)\b", normalized)
    if plain:
        value = float(plain.group(1).replace(",", ""))
        if value >= 1_000:
            return value
    return None
def _build_execution_advisory(query: str, metrics: dict[str, Any]) -> str:
    """Quantitative execution ticket when user states a numeric exposure."""
    notional = _parse_notional_exposure(query, metrics.get("base_ccy", "USD"))
    if notional is None or notional <= 0:
        return ""

    hedge_ratio = _resolved_hedge_ratio(metrics)
    spot = float(metrics.get("spot", 0.0))
    ma5_val = float(metrics.get("ma5", spot))
    base = metrics.get("base_ccy", "USD")
    symbol = metrics.get("symbol", "$")
    pair = metrics.get("pair", DEFAULT_PAIR)
    mf_risk = metrics.get("multi_factor_risk_rating", metrics.get("risk_level", "MEDIUM"))

    lock_notional = notional * hedge_ratio
    residual_notional = notional - lock_notional
    cnh_at_spot_locked = lock_notional * spot
    cnh_at_spot_residual = residual_notional * spot
    cnh_at_spot_total = notional * spot
    effective_rate = (hedge_ratio * ma5_val) + ((1.0 - hedge_ratio) * spot)
    blended_cnh = notional * effective_rate

    return (
        f"**⚡ Real-Time Execution Ticket — {pair}**\n"
        f"- Stated exposure: **{symbol}{notional:,.2f} {base}**\n"
        f"- Multi-factor risk rating: **{mf_risk}** → hedge ratio **{hedge_ratio:.1%}**\n"
        f"- **Forward-lock now:** **{symbol}{lock_notional:,.2f} {base}** "
        f"({hedge_ratio:.1%} of gross exposure)\n"
        f"- **Keep unhedged / spot-flex:** **{symbol}{residual_notional:,.2f} {base}** "
        f"({1.0 - hedge_ratio:.1%} residual)\n"
        f"- Implied CNH at live spot **{spot:.4f}**: locked leg ≈ **¥{cnh_at_spot_locked:,.2f}** | "
        f"residual ≈ **¥{cnh_at_spot_residual:,.2f}** | total ≈ **¥{cnh_at_spot_total:,.2f}**\n"
        f"- NOVA blended effective rate (hedge×MA5 + residual×Spot): **{effective_rate:.4f}** → "
        f"modeled CNH receipt ≈ **¥{blended_cnh:,.2f}**\n"
        f"- **Desk instruction:** Execute a **{hedge_ratio:.0%}** forward/swap on "
        f"**{symbol}{lock_notional:,.2f} {base}** at or better than **{spot:.4f}**; "
        f"retain **{symbol}{residual_notional:,.2f}** for upside if spot mean-reverts below MA5."
    )
def _detect_intent(query: str, tokens: set[str]) -> str:
    q = query.lower()
    if tokens & {"hedge", "forward", "locking", "bleeding", "bleed", "loss"} or any(
        p in q for p in ("losing money", "stop bleeding", "bleed", "lost money")
    ):
        return "hedge"
    if tokens & {"vat", "euro", "eur", "remittance", "uk", "gbp", "british", "pound"}:
        return "eur_vat"
    if tokens & {"hkd", "peg", "depreciation"} or "hong kong" in q:
        return "hkd_peg"
    if tokens & {"macro", "trend", "q2", "seasonal", "june", "cheap"} or "trap" in q:
        return "macro"
    if tokens & {"amazon", "tiktok", "seller", "payout"}:
        return "seller_case"
    if tokens & {"safe", "worry", "worried", "audit", "compliance"}:
        return "compliance"
    return "general"
def _pair_action(intent: str, pair: str) -> str:
    if intent == "eur_vat" or (pair == "EUR/CNH" and intent in ("hedge", "general")):
        return (
            "**EUR / VAT playbook**\n"
            "1. Model net EUR after VAT—not platform gross GMV.\n"
            "2. Forward-hedge 65% of net EUR due in 60–90 days when EUR/CNH is rich.\n"
            "3. Align conversions with OSS/VAT filing dates to avoid double FX hits.\n"
            "4. Keep invoices that satisfy both tax authority and bank FX desks."
        )
    if intent == "hkd_peg" or (pair == "HKD/CNH" and intent in ("hedge", "general")):
        return (
            "**HKD peg-stress playbook**\n"
            "1. Watch HKD/CNH vs 30-day band; weakness → tranche conversions (40/30/30).\n"
            "2. Pre-fund 2 weeks of onshore CNY payroll during HKD weakening phases.\n"
            "3. Use 2–4 week forwards on net CNH shortfall—avoid speculative HKD hoarding.\n"
            "4. Consolidate USD+HKD exposure before adding new hedges."
        )
    if intent == "hedge":
        if pair == "EUR/CNH":
            return (
                "**Executable EUR hedge plan**\n"
                "1. Export 90-day net EUR receivable calendar (post-VAT).\n"
                "2. Forward-sell **65%** of EUR due in 60+ days (3M tenor).\n"
                "3. Keep **35%** spot; add **+10%** if EUR/CNH hits a new 30-day high.\n"
                "4. Log notionals in your exposure ledger for audit."
            )
        if pair == "HKD/CNH":
            return (
                "**Executable HKD hedge plan**\n"
                "1. Map weekly HKD inflows vs CNH/onshore CNY outflows.\n"
                "2. Forward/conversion hedge **50–60%** of 30-day net CNH need.\n"
                "3. Keep **40%** flexible for peg-stabilization bounces.\n"
                "4. Pause discretionary HKD cash piles during weak-HKD headlines."
            )
        if pair == "GBP/CNH":
            return (
                "**Executable GBP hedge plan**\n"
                "1. Export 90-day net GBP receivable calendar (post-UK VAT).\n"
                "2. Forward-sell **65%** of GBP due in 60+ days (3M tenor).\n"
                "3. Keep **35%** spot; add **+10%** if GBP/CNH hits a new 30-day high.\n"
                "4. Log notionals for HMRC + bank FX audit readiness."
            )
        return (
            "**Executable USD hedge plan**\n"
            "1. Export 90-day USD receivable calendar from marketplace reports.\n"
            "2. Forward-sell **70%** of USD due in 60+ days (3M tenor).\n"
            "3. Keep **30%** unhedged; add **+10%** on new 30-day highs.\n"
            "4. Log every hedge line for bank audit readiness."
        )
    if intent == "macro":
        ranges = {
            "USD/CNH": "7.15–7.32",
            "EUR/CNH": "7.70–8.00",
            "HKD/CNH": "0.910–0.940 (peg-guided)",
            "GBP/CNH": "8.75–9.35",
        }
        band = ranges.get(pair, "pair-specific simulation band")
        return (
            f"**Macro playbook ({pair})**\n"
            f"1. Base simulation range **{band}**—ladder forwards, no all-in spot bets.\n"
            "2. Front-load conversions before month-end corporate FX rush.\n"
            "3. Stress-test a shock scenario; add +10% hedge two weeks before large CNH conversion windows."
        )
    if intent == "seller_case":
        return (
            "**Marketplace seller checklist**\n"
            "1. Bucket cash flows by currency and 30/60/90-day payout windows.\n"
            "2. Never run naked float through a promo without a hedge plan.\n"
            "3. Post-campaign: convert 40% spot from day 7; roll the rest in 1M forwards.\n"
            "4. Pause adding forwards if spot < MA5 for 3 sessions."
        )
    if intent == "compliance":
        return (
            "**Safety & compliance steps**\n"
            "1. Hedge only documented open receivables—no speculative notional.\n"
            "2. Bundle invoice + logistics + settlement proof for the bank.\n"
            "3. Refresh KYC files quarterly to avoid payout-day hedge delays."
        )
    return (
        f"**General FX policy ({pair})**\n"
        "1. Quantify exposure and payout dates weekly.\n"
        "2. Map hedge ratio to risk light: HIGH 70% / MEDIUM 40% / LOW 20%.\n"
        "3. Reconcile actual payouts vs hedged notionals every month."
    )
def _resolved_hedge_ratio(metrics: dict[str, Any]) -> float:
    """Enterprise hedge ratio when available; otherwise legacy market sleeve."""
    raw = metrics.get("recommended_hedge_ratio")
    if raw is not None:
        return float(raw)
    legacy = metrics.get("legacy_market_hedge_ratio")
    if legacy is not None:
        return float(legacy)
    return 0.55
def build_executive_treasury_action(
    metrics: dict[str, Any],
    pair: str,
    *,
    imf_indicators: dict[str, Any] | None = None,
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
) -> str:
    """Tripartite Data → Analysis → Conclusion advisory aligned with NOVA_AGENT_SYSTEM_INSTRUCTIONS."""
    active_pair = metrics.get("pair", pair)
    spot = float(metrics.get("spot", 0.0))
    ma5 = float(metrics.get("ma5", spot))
    ma30 = float(metrics.get("ma30", spot))
    boll_upper = float(metrics.get("bollinger_upper", spot))
    low_30d = float(metrics.get("low_30d", spot * 0.99))
    high_30d = float(metrics.get("high_30d", spot * 1.01))
    hedge_ratio = _resolved_hedge_ratio(metrics)
    rsi = float(metrics.get("net_payout_rsi", 50.0))
    macd_hist = float(metrics.get("net_payout_macd_histogram", 0.0))
    vat_rate = float(metrics.get("vat_deduction_rate", 0.0))
    signal = str(metrics.get("net_payout_signal_class", "normal"))

    if spot > ma5 and spot > ma30:
        corridor = (
            f"Spot **{spot:.4f}** trades **above** the MA5/MA30 corridor "
            f"(MA5 {ma5:.4f}, MA30 {ma30:.4f})"
        )
    elif spot < ma30:
        corridor = (
            f"Spot **{spot:.4f}** sits **below** MA30 support "
            f"(MA5 {ma5:.4f}, MA30 {ma30:.4f})"
        )
    else:
        corridor = (
            f"Spot **{spot:.4f}** is **inside** the MA5/MA30 corridor "
            f"(MA5 {ma5:.4f}, MA30 {ma30:.4f})"
        )

    spreads = (
        compute_imf_ppp_spreads(active_pair, imf_indicators)
        if isinstance(imf_indicators, dict)
        else compute_imf_ppp_spreads(active_pair)
    )
    dest_label = spreads.get("destination_label", "US")
    macro_bits: list[str] = [f"30-day range **{low_30d:.4f}–{high_30d:.4f}**"]

    if isinstance(imf_indicators, dict):
        gdp = imf_indicators.get("gdp_growth_forecast", {})
        china_gdp = float(gdp.get("China", gdp.get(CHINA_MACRO_ANCHOR, 4.6)))
        dest_gdp = float(gdp.get(dest_label, 2.0))
        cpi_spread = spreads.get("cpi_spread_vs_china")
        macro_bits.append(
            f"IMF 2026 GDP anchor **{dest_label} {dest_gdp:.1f}%** vs China **{china_gdp:.1f}%**"
        )
        if cpi_spread is not None:
            macro_bits.append(f"CPI spread **{cpi_spread:+.1f}pp** vs China")

    if tnx_proxy and tnx_proxy.get("status") == "ok":
        macro_bits.append(
            f"US 10Y **{float(tnx_proxy.get('current_yield', 0.0)):.2f}%** "
            f"({tnx_proxy.get('signal_label', 'yield proxy')})"
        )

    if china_money and china_money.get("status") == "ok" and china_money.get("usd_cny"):
        parity = float(china_money["usd_cny"])
        drift_bps = (spot - parity) / parity * 10_000 if parity else 0.0
        macro_bits.append(
            f"CFETS parity **{parity:.4f}** ({drift_bps:+.0f} bps offshore drift)"
        )

    data_line = f"- {corridor}; {', '.join(macro_bits)}."

    if rsi >= 70:
        momentum = "RSI saturation"
    elif rsi <= 30:
        momentum = "RSI exhaustion"
    else:
        momentum = "neutral RSI momentum"
    macd_tone = "bearish MACD histogram" if macd_hist < 0 else "bullish MACD histogram"
    vat_pct = vat_rate * 100.0

    if signal == "risk_alert":
        payout_impact = (
            f"Platform payout cycles face compressed net CNH conversion as VAT-adjusted "
            f"receivables ({vat_pct:.0f}% withholding overlay) meet fading momentum."
        )
    elif signal == "opportunity":
        payout_impact = (
            f"Merchant micro-revenues retain upside optionality—VAT net payouts ({vat_pct:.0f}%) "
            f"can defer spot clearance without breaching remittance SLAs."
        )
    else:
        payout_impact = (
            f"Cross-border payout cadence remains stable; VAT-adjusted net flows ({vat_pct:.0f}%) "
            f"absorb macro noise without forced spot conversion."
        )

    analysis_line = (
        f"- Sovereign yield/CPI spreads interact with **{momentum}** and **{macd_tone}** "
        f"on net-payout technicals. {payout_impact}"
    )

    if signal == "risk_alert":
        directive = "Layered Window Forward Execution"
        bracket = f"protective floor **{low_30d:.4f}**, Bollinger cap **{boll_upper:.4f}**"
    elif signal == "opportunity":
        directive = "Delayed Discretionary Spot Clearance"
        bracket = f"staged entry band **{low_30d:.4f}–{spot:.4f}**"
    else:
        directive = "Rolling Layered Forward + Meta/Google Ad Spend Asset Matching"
        bracket = f"immunize between **{low_30d:.4f}** and **{high_30d:.4f}**"

    conclusion_line = (
        f"- **{directive}** at **{hedge_ratio:.0%} layered hedge**; {bracket}."
    )

    score_note = ""
    if metrics.get("final_risk_score") is not None:
        score_note = (
            f"\n\n**Composite risk score:** **{float(metrics.get('final_risk_score', 0)):.1f}/100** "
            f"({metrics.get('multi_factor_risk_rating', 'MEDIUM')}) — "
            f"35% market · 25% macro · 25% policy · 15% knowledge."
        )

    body = (
        "### ⚡ NOVA Multi-Factor Risk Advisory\n\n"
        "**1. Quantitative Ingestion & Market Signals [DATA]**\n"
        f"{data_line}\n\n"
        "**2. Transmission Channel & Treasury Risk Impact [ANALYSIS]**\n"
        f"{analysis_line}\n\n"
        "**3. Optimized Execution Framework [CONCLUSION]**\n"
        f"{conclusion_line}{score_note}"
    )
    return _trim_to_word_budget(body, max_words=220)
def _merge_retrieved_docs(
    *doc_lists: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Deduplicate RAG hits by document id while preserving rank order."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for docs in doc_lists:
        for doc in docs:
            doc_id = doc.get("id", doc.get("title", ""))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(doc)
            if len(merged) >= top_k:
                return merged
    return merged
def fx_rag_build_response_parts(
    user_query: str,
    metrics: dict[str, Any] | None = None,
    pair: str = DEFAULT_PAIR,
    live_context: str | None = None,
    prompt_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured RAG payload for Streamlit expander UI (market-aware copilot)."""
    user_query = sanitize_user_query(user_query)
    m = metrics or {}
    active_pair = m.get("pair", pair)

    docs_pass1 = _simple_semantic_search(user_query, pair=active_pair, top_k=3)
    bundle = prompt_bundle or generate_dynamic_agent_prompt(
        user_query, metrics=m, retrieved_docs=docs_pass1
    )
    macro_block = bundle.get("macro_layer", "")
    live_desk_block = live_context or bundle.get("live_desk_layer", "")
    retrieval_query = bundle["agent_prompt"]

    tokens = _tokenize(retrieval_query)
    docs_pass2 = _simple_semantic_search(retrieval_query, pair=active_pair, top_k=3)
    docs = _merge_retrieved_docs(docs_pass1, docs_pass2, top_k=3)

    risk_refresh = compute_nova_multidimensional_risk_assessment(
        m,
        market_factor_score=float(m.get("multi_factor_score", 0.0)),
        imf_indicators=bundle.get("imf_macro_indicators")
        if isinstance(bundle.get("imf_macro_indicators"), dict)
        else None,
        china_money=bundle.get("china_money"),
        tnx_proxy=bundle.get("tnx_proxy"),
        retrieved_documents=docs,
        knowledge_query=user_query,
    )
    m = {
        **m,
        **risk_refresh,
        "risk_score_explanation": build_risk_score_explanation({**m, **risk_refresh}),
    }

    playbooks_layer = assemble_retrieved_playbooks_context(docs)
    context_block = bundle["agent_prompt"]
    if playbooks_layer and playbooks_layer not in context_block:
        marker = f"User question:\n{user_query}"
        if marker in context_block:
            context_block = context_block.replace(
                marker,
                f"{playbooks_layer}\n\n{marker}",
                1,
            )
        else:
            context_block = f"{context_block}\n\n{playbooks_layer}"

    spot = m.get("spot")
    ma5 = float(m.get("ma5", spot or 0.0))
    ma30 = float(m.get("ma30", spot or 0.0))
    bollinger_upper = float(m.get("bollinger_upper", spot or 0.0))
    risk = m.get("multi_factor_risk_rating", m.get("risk_level", "MEDIUM"))
    hedge_ratio = _resolved_hedge_ratio(m)
    final_score = float(m.get("final_risk_score", 0.0))

    market_block = ""
    if spot is not None:
        market_block = (
            f"---\n**Live desk snapshot — {active_pair}** "
            f"(`{m.get('data_source', 'unknown')}`)\n"
            f"- Real-time Spot: **{float(spot):.4f}** | MA5: **{ma5:.4f}** | "
            f"MA30: **{ma30:.4f}** | Bollinger upper: **{bollinger_upper:.4f}**\n"
            f"- Final composite risk: **{risk}** | Score: **{final_score:.1f}/100** | "
            f"Hedge ratio: **{hedge_ratio:.1%}**\n"
            f"- Dimension scores — Market: **{float(m.get('market_score', 0)):.1f}** · "
            f"Macro: **{float(m.get('macro_score', 0)):.1f}** · "
            f"Policy: **{float(m.get('policy_score', 0)):.1f}** · "
            f"Knowledge: **{float(m.get('knowledge_score', 0)):.1f}**\n"
            f"- Desk note: {m.get('risk_advice', '')}"
        )

    execution_block = _build_execution_advisory(user_query, m)
    executive_action = build_executive_treasury_action(
        m,
        active_pair,
        imf_indicators=bundle.get("imf_macro_indicators")
        if isinstance(bundle.get("imf_macro_indicators"), dict)
        else None,
        china_money=bundle.get("china_money"),
        tnx_proxy=bundle.get("tnx_proxy"),
    )
    action = executive_action

    imf_retrieved = any(doc.get("id") == IMF_MACRO_KB_ID for doc in docs) or bool(
        bundle.get("imf_macro_indicators")
    )

    disclaimer = (
        "Disclaimer: Educational simulation only—not investment advice. "
        "Execute trades with licensed institutions and verified trade documentation."
    )

    return {
        "docs": docs,
        "user_query": user_query,
        "active_pair": active_pair,
        "agent_prompt": context_block,
        "live_context": live_desk_block,
        "macro_layer": macro_block,
        "macro_available": bundle.get("macro_available", False),
        "imf_macro_block": bundle.get("imf_macro_block", ""),
        "imf_macro_indicators": bundle.get("imf_macro_indicators"),
        "imf_ppp_spreads": bundle.get("imf_ppp_spreads"),
        "imf_retrieved": imf_retrieved,
        "china_money": bundle.get("china_money"),
        "tnx_proxy": bundle.get("tnx_proxy"),
        "market_block": market_block,
        "execution_block": execution_block,
        "executive_action": executive_action,
        "action": action,
        "risk_score_explanation": m.get("risk_score_explanation", ""),
        "final_risk_score": m.get("final_risk_score"),
        "market_score": m.get("market_score"),
        "macro_score": m.get("macro_score"),
        "policy_score": m.get("policy_score"),
        "knowledge_score": m.get("knowledge_score"),
        "disclaimer": disclaimer,
        "playbook_count": len(docs),
        "playbooks_layer": playbooks_layer,
    }
def fx_rag_answer(
    user_query: str,
    metrics: dict[str, Any] | None = None,
    pair: str = DEFAULT_PAIR,
) -> str:
    """Backward-compatible markdown answer (playbooks omitted—use expanders in app)."""
    parts = fx_rag_build_response_parts(user_query, metrics=metrics, pair=pair)
    active_pair = parts["active_pair"]

    return (
        f"### NOVA FX Risk Copilot · {active_pair}\n\n"
        f"**Your question:** {parts['user_query']}\n\n"
        f"{parts.get('executive_action', parts['action'])}\n\n"
        f"> {parts['disclaimer']}"
    )
_AMOUNT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "labeled_total",
        re.compile(
            r"(?:(?:grand\s+)?total|invoice\s+total|amount\s+due|balance\s+due|"
            r"contract\s+value|notional)[:\s]*(?:USD|EUR|HKD|US\$|\$|€)?\s*"
            r"([\d,]+(?:\.\d{1,2})?)",
            re.IGNORECASE,
        ),
    ),
    (
        "amount_label",
        re.compile(
            r"(?:amount|sum|value)[:\s]*(?:USD|EUR|HKD|\$|€)?\s*([\d,]+(?:\.\d{1,2})?)",
            re.IGNORECASE,
        ),
    ),
    (
        "currency_prefix",
        re.compile(
            r"(?:USD|EUR|HKD)\s*([\d,]+(?:\.\d{1,2})?)",
            re.IGNORECASE,
        ),
    ),
    (
        "dollar_sign",
        re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)"),
    ),
]
_TENOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "due_in_days",
        re.compile(
            r"(?:due|payable|payment|settlement|deliver(?:y)?|paid)\s+(?:in|within)\s+"
            r"(\d{1,3})\s+days?",
            re.IGNORECASE,
        ),
    ),
    (
        "net_days",
        re.compile(r"net\s+(\d{1,3})\s+days?", re.IGNORECASE),
    ),
    (
        "terms_days",
        re.compile(
            r"(?:payment\s+terms|terms)[:\s]*(\d{1,3})\s+days?",
            re.IGNORECASE,
        ),
    ),
]
_MARGIN_PATTERN = re.compile(
    r"(?:gross\s+)?margin[:\s]*(\d{1,2}(?:\.\d{1,2})?)\s*%",
    re.IGNORECASE,
)
def _normalize_amount(raw: str) -> float:
    return float(raw.replace(",", "").strip())
def _extract_pdf_text(file_bytes: bytes) -> str:
    """LangChain PyPDFLoader-style workflow: bytes → PdfReader → page text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        pages: list[str] = []
        for page in reader.pages:
            chunk = page.extract_text()
            if chunk:
                pages.append(chunk)
        return "\n".join(pages).strip()
    except Exception:
        return ""
def _extract_plaintext(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""
MAX_UPLOAD_DOCUMENT_CHARS = 12_000
def _truncate_uploaded_document_text(text: str) -> str:
    """Token protection: cap extracted upload text before regex/LLM processing."""
    cleaned = str(text or "")
    if len(cleaned) <= MAX_UPLOAD_DOCUMENT_CHARS:
        return cleaned
    return cleaned[:MAX_UPLOAD_DOCUMENT_CHARS]
def _extract_document_text(filename: str, file_bytes: bytes | None) -> tuple[str, str]:
    """
    Returns (text, source_kind).
    source_kind: pdf | text | empty
    """
    if not file_bytes:
        return "", "empty"

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _truncate_uploaded_document_text(_extract_pdf_text(file_bytes)), "pdf"
    if name.endswith((".txt", ".csv")):
        return _truncate_uploaded_document_text(_extract_plaintext(file_bytes)), "text"
    return "", "empty"
def read_upload_document_text(filename: str, file_bytes: bytes | None) -> str:
    """Public helper for optional AI contract review (text capped at 12,000 chars)."""
    text, _ = _extract_document_text(filename, file_bytes)
    return text
def _parse_amounts_from_text(text: str) -> list[tuple[str, float, str]]:
    """Returns list of (pattern_name, value, raw_match)."""
    hits: list[tuple[str, float, str]] = []
    for name, pattern in _AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            try:
                value = _normalize_amount(raw)
                if value >= 100:
                    hits.append((name, value, raw))
            except ValueError:
                continue
    return hits
def _parse_tenor_from_text(text: str) -> tuple[int | None, str | None, str | None]:
    """Returns (days, pattern_name, raw_match)."""
    for name, pattern in _TENOR_PATTERNS:
        match = pattern.search(text)
        if match:
            days = int(match.group(1))
            if 1 <= days <= 365:
                return days, name, match.group(0)
    return None, None, None
def _parse_margin_from_text(text: str) -> float | None:
    match = _MARGIN_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
def _choose_best_amount(hits: list[tuple[str, float, str]]) -> tuple[float | None, str | None, str | None]:
    if not hits:
        return None, None, None
    priority = {"labeled_total": 0, "amount_label": 1, "currency_prefix": 2, "dollar_sign": 3}
    hits.sort(key=lambda h: (priority.get(h[0], 9), -h[1]))
    name, value, raw = hits[0]
    return value, name, raw
def _simulate_audit_payload(name: str, pair: str) -> dict[str, Any]:
    """Deterministic fallback when PDF/regex extraction is unavailable."""
    seed = sum(ord(c) for c in name + pair) % 10_000
    rng = np.random.default_rng(seed)
    meta = PAIR_META[pair]

    notional = float(rng.integers(50_000, 500_000))
    tenor_days = int(rng.integers(30, 120))
    margin_pct = float(rng.integers(8, 18))
    recommended_hedge = min(0.85, 0.45 + notional / 1_000_000)

    return {
        "filename": name,
        "pair": pair,
        "currency": meta["base_ccy"],
        "symbol": meta["symbol"],
        "notional": notional,
        "expected_settlement_days": tenor_days,
        "gross_margin_pct": margin_pct,
        "recommended_forward_ratio": round(recommended_hedge, 2),
        "estimated_cnh_at_risk": round(notional * 0.03, 2),
        "status": "PASS",
        "extraction_mode": "simulated_fallback",
        "document_chars": 0,
        "text_preview": "",
        "regex_hits": {},
    }
def _parse_contract_from_text(text: str, pair: str) -> dict[str, Any] | None:
    """Regex extraction layer; returns None if mandatory fields are missing."""
    amount_hits = _parse_amounts_from_text(text)
    notional, amount_rule, amount_raw = _choose_best_amount(amount_hits)
    tenor_days, tenor_rule, tenor_raw = _parse_tenor_from_text(text)

    if notional is None or tenor_days is None:
        return None

    meta = PAIR_META[pair]
    margin_pct = _parse_margin_from_text(text)
    if margin_pct is None:
        margin_pct = float(min(25.0, max(8.0, 12.0 + (notional % 7))))

    recommended_hedge = min(0.85, 0.45 + notional / 1_000_000)

    return {
        "filename": "",
        "pair": pair,
        "currency": meta["base_ccy"],
        "symbol": meta["symbol"],
        "notional": notional,
        "expected_settlement_days": tenor_days,
        "gross_margin_pct": margin_pct,
        "recommended_forward_ratio": round(recommended_hedge, 2),
        "estimated_cnh_at_risk": round(notional * 0.03, 2),
        "status": "PASS",
        "extraction_mode": "pdf_regex",
        "document_chars": len(text),
        "text_preview": text[:280].replace("\n", " "),
        "regex_hits": {
            "amount_rule": amount_rule,
            "amount_raw": amount_raw,
            "tenor_rule": tenor_rule,
            "tenor_raw": tenor_raw,
            "amount_candidates": len(amount_hits),
        },
    }
def audit_contract_file(
    filename: str,
    file_bytes: bytes | None = None,
    pair: str = DEFAULT_PAIR,
) -> dict[str, Any]:
    """
    Audit uploaded contract/invoice:
    1) PDF → pypdf text extraction (PyPDFLoader-style)
    2) Regex for notional + tenor (+ margin when present)
    3) Fallback to deterministic simulation if needed
    """
    name = filename or "contract.pdf"
    sym = PAIR_META[pair]["symbol"]
    base = PAIR_META[pair]["base_ccy"]

    text, source_kind = _extract_document_text(name, file_bytes)
    parsed: dict[str, Any] | None = None

    if text:
        parsed = _parse_contract_from_text(text, pair)
        if parsed is not None:
            parsed["filename"] = name
            if source_kind == "pdf":
                parsed["extraction_mode"] = "pdf_regex"
            else:
                parsed["extraction_mode"] = "text_regex"

    if parsed is None:
        parsed = _simulate_audit_payload(name, pair)
        if text and source_kind == "pdf":
            parsed["extraction_mode"] = "pdf_fallback_simulated"
            parsed["document_chars"] = len(text)
            parsed["text_preview"] = text[:280].replace("\n", " ")
            parsed["regex_hits"] = {
                "note": "PDF text extracted but no amount/tenor regex match",
            }
        elif text and source_kind == "text":
            parsed["extraction_mode"] = "text_fallback_simulated"
            parsed["document_chars"] = len(text)
            parsed["text_preview"] = text[:280].replace("\n", " ")

    notional = parsed["notional"]
    tenor_days = parsed["expected_settlement_days"]
    hedge = parsed["recommended_forward_ratio"]
    mode = parsed["extraction_mode"]

    if mode in ("pdf_regex", "text_regex"):
        hits = parsed.get("regex_hits", {})
        parsed["summary"] = (
            f"**Document read successfully** — extracted **{parsed['document_chars']:,}** "
            f"characters from **{name}** ({mode}).\n\n"
            f"- **Amount detected** via `{hits.get('amount_rule')}`: "
            f"**{hits.get('amount_raw')}** → {base} **{sym}{notional:,.2f}**\n"
            f"- **Tenor detected** via `{hits.get('tenor_rule')}`: "
            f"\"{hits.get('tenor_raw')}\" → **{tenor_days} days**\n"
            f"- Recommended forward hedge: **{hedge:.0%}** | "
            f"CNH at risk (sim.): **¥{parsed['estimated_cnh_at_risk']:,.0f}**\n\n"
            f"_Preview:_ \"{parsed.get('text_preview', '')}…\""
        )
    elif mode == "pdf_fallback_simulated":
        parsed["summary"] = (
            f"**PDF parsed ({parsed.get('document_chars', 0):,} chars)** from **{name}**, "
            f"but no confident amount/tenor regex match — using **deterministic fallback**.\n\n"
            f"- Fallback {base} exposure: **{sym}{notional:,.0f}** | Tenor: **{tenor_days} days** | "
            f"Hedge: **{hedge:.0%}**\n\n"
            f"_Preview:_ \"{parsed.get('text_preview', '')}…\""
        )
    else:
        parsed["summary"] = (
            f"**Simulation mode** for **{name}** ({pair}) — non-PDF or empty upload; "
            f"deterministic seed-based audit.\n\n"
            f"- {base} exposure: **{sym}{notional:,.0f}** | Tenor: **{tenor_days} days** | "
            f"Hedge: **{hedge:.0%}** | CNH at risk: **¥{parsed['estimated_cnh_at_risk']:,.0f}**"
        )

    return parsed
