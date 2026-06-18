"""
Treasury Copilot — enterprise intelligence, hedge-ratio reasoning, and knowledge matching.

Deterministic scores remain authoritative; this module enriches interpretation only.
"""

from __future__ import annotations

import json
import re
from typing import Any

_ARCHETYPE_MARKERS: tuple[tuple[str, str], ...] = (
    ("amazon", "Marketplace Seller"),
    ("shopify", "Marketplace Seller"),
    ("seller", "Marketplace Seller"),
    ("export", "Export SME"),
    ("exporter", "Export SME"),
    ("import", "Import SME"),
    ("importer", "Import SME"),
    ("factory", "Manufacturer"),
    ("manufactur", "Manufacturer"),
    ("distribut", "Distributor"),
    ("wholesale", "Distributor"),
)


def build_similar_knowledge_cases(
    retrieved_documents: list[dict[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rank retrieved playbooks by semantic retrieval weight."""
    ranked: list[tuple[float, dict[str, Any]]] = []
    for doc in retrieved_documents:
        weight = doc.get("similarity", doc.get("score", 0.0))
        try:
            score = float(weight)
        except (TypeError, ValueError):
            score = 0.0
        ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    cases: list[dict[str, Any]] = []
    for score, doc in ranked[:top_k]:
        cases.append(
            {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "domain": doc.get("domain", ""),
                "business_type": doc.get("business_type", ""),
                "similarity_score": round(min(1.0, max(0.0, score)), 3),
                "risk_domains": doc.get("risk_domains", []),
            }
        )
    return cases


def _coerce_similarity_score(value: Any) -> float:
    try:
        return round(min(1.0, max(0.0, float(value))), 3)
    except (TypeError, ValueError):
        return 0.0


def normalize_knowledge_case(entry: Any, *, index: int = 0) -> dict[str, Any]:
    """Coerce LLM or legacy payloads into a render-safe case dict."""
    if isinstance(entry, dict):
        return {
            "id": str(entry.get("id") or f"case-{index + 1}"),
            "title": str(entry.get("title") or entry.get("name") or "Untitled"),
            "domain": str(entry.get("domain") or "General"),
            "business_type": str(entry.get("business_type") or ""),
            "similarity_score": _coerce_similarity_score(
                entry.get("similarity_score", entry.get("similarity", 0.0))
            ),
            "risk_domains": entry.get("risk_domains") or [],
        }
    if isinstance(entry, str):
        text = entry.strip()
        return {
            "id": f"case-{index + 1}",
            "title": text or "Untitled",
            "domain": "General",
            "business_type": "",
            "similarity_score": 0.0,
            "risk_domains": [],
        }
    return {
        "id": f"case-{index + 1}",
        "title": str(entry or "Untitled"),
        "domain": "General",
        "business_type": "",
        "similarity_score": 0.0,
        "risk_domains": [],
    }


def normalize_knowledge_case_list(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        return []
    return [normalize_knowledge_case(case, index=i) for i, case in enumerate(cases)]


def infer_local_risk_archetype(scenario: str) -> str:
    text = str(scenario or "").lower()
    for marker, archetype in _ARCHETYPE_MARKERS:
        if marker in text:
            return archetype
    return "Cross-Border Treasury"


def infer_exporter_importer(scenario: str) -> str:
    text = str(scenario or "").lower()
    export_hit = any(w in text for w in ("export", "exporter", "seller", "amazon", "payout", "receivable"))
    import_hit = any(w in text for w in ("import", "importer", "payable", "supplier", "factory"))
    if export_hit and import_hit:
        return "mixed"
    if export_hit:
        return "exporter"
    if import_hit:
        return "importer"
    return "unknown"


def build_enterprise_intelligence_context(
    metrics: dict[str, Any],
    retrieved_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scenario = str(metrics.get("enterprise_scenario", ""))
    pair = str(metrics.get("pair", "USD/CNH"))
    base_ccy = pair.split("/")[0] if "/" in pair else "USD"
    docs = retrieved_documents or []
    similar_cases = build_similar_knowledge_cases(docs)
    contributions = metrics.get("risk_contributions", {}) or {}

    return {
        "pair": pair,
        "enterprise_scenario": scenario,
        "deterministic_scores": {
            "market_score": metrics.get("market_score"),
            "macro_score": metrics.get("macro_score"),
            "policy_score": metrics.get("policy_score"),
            "knowledge_score": metrics.get("knowledge_score"),
            "final_risk_score": metrics.get("final_risk_score"),
            "recommended_hedge_ratio": metrics.get("recommended_hedge_ratio"),
            "weighted_contributions": contributions,
        },
        "local_inference": {
            "risk_archetype": infer_local_risk_archetype(scenario),
            "exporter_importer": infer_exporter_importer(scenario),
        },
        "similar_knowledge_cases": similar_cases,
        "knowledge_domains": (metrics.get("knowledge_breakdown") or {}).get(
            "detected_risk_domains", []
        ),
        "exposure_hint": {
            "primary_currency": base_ccy,
            "settlement_cycle_days": metrics.get("expected_settlement_days"),
        },
    }


def build_hedge_ratio_explanation_context(
    metrics: dict[str, Any],
    retrieved_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    contributions = metrics.get("risk_contributions", {}) or {}
    knowledge_breakdown = metrics.get("knowledge_breakdown") or {}
    similar_cases = build_similar_knowledge_cases(retrieved_documents or [], top_k=2)
    hedge_ratio = metrics.get("recommended_hedge_ratio")

    return {
        "pair": metrics.get("pair"),
        "enterprise_scenario": metrics.get("enterprise_scenario", ""),
        "deterministic_hedge_ratio": hedge_ratio,
        "deterministic_hedge_ratio_pct": (
            f"{float(hedge_ratio):.0%}" if hedge_ratio is not None else "N/A"
        ),
        "final_risk_score": metrics.get("final_risk_score"),
        "multi_factor_risk_rating": metrics.get("multi_factor_risk_rating"),
        "weighted_contributions": {
            "market_pts": contributions.get("market"),
            "macro_pts": contributions.get("macro"),
            "policy_pts": contributions.get("policy"),
            "knowledge_pts": contributions.get("knowledge"),
        },
        "component_scores": {
            "market_score": metrics.get("market_score"),
            "macro_score": metrics.get("macro_score"),
            "policy_score": metrics.get("policy_score"),
            "knowledge_score": metrics.get("knowledge_score"),
        },
        "knowledge_domains": knowledge_breakdown.get("detected_risk_domains", []),
        "most_similar_cases": similar_cases,
        "deterministic_formula_note": (
            "Hedge ratio is derived from final composite risk score tiers "
            "(not modified by AI)."
        ),
    }


def build_local_hedge_ratio_explanation(
    metrics: dict[str, Any],
    retrieved_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministic fallback bullets explaining hedge ratio drivers."""
    ctx = build_hedge_ratio_explanation_context(
        metrics,
        retrieved_documents=retrieved_documents,
    )
    contributions = ctx["weighted_contributions"]
    bullets: list[str] = []
    for label, pts in (
        ("Market", contributions.get("market_pts")),
        ("Macro", contributions.get("macro_pts")),
        ("Policy", contributions.get("policy_pts")),
        ("Knowledge", contributions.get("knowledge_pts")),
    ):
        if pts is not None:
            bullets.append(f"{label} environment contributes {float(pts):.1f} weighted points")
    cases = ctx.get("most_similar_cases") or []
    if cases:
        top = cases[0]
        bullets.append(
            f"Exposure profile resembles **{top.get('title', 'treasury case')}** "
            f"(similarity {top.get('similarity_score', 0):.2f})"
        )
    ratio = ctx.get("deterministic_hedge_ratio_pct", "N/A")
    bullets.append(
        f"Therefore maintaining **{ratio}** hedge coverage aligns with the deterministic composite "
        f"score **{ctx.get('final_risk_score', 'N/A')}**."
    )
    return {
        "deterministic_hedge_ratio": ctx.get("deterministic_hedge_ratio_pct"),
        "interpretation_bullets": bullets,
        "mode": "local",
    }


def build_local_enterprise_intelligence(
    metrics: dict[str, Any],
    retrieved_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scenario = str(metrics.get("enterprise_scenario", ""))
    pair = str(metrics.get("pair", "USD/CNH"))
    base_ccy = pair.split("/")[0] if "/" in pair else "USD"
    archetype = infer_local_risk_archetype(scenario)
    flow = infer_exporter_importer(scenario)
    return {
        "business_profile": {
            "company_type": archetype,
            "exporter_importer": flow,
            "margin_sensitivity": "medium",
        },
        "exposure_mapping": {
            "receivable_currencies": [base_ccy] if flow in ("exporter", "mixed", "unknown") else [],
            "payable_currencies": [base_ccy] if flow in ("importer", "mixed") else [],
            "settlement_cycle": "14-90 days (inferred from scenario)",
            "hedge_urgency": metrics.get("multi_factor_risk_rating", "MEDIUM"),
        },
        "risk_archetype": archetype,
        "most_similar_knowledge_cases": build_similar_knowledge_cases(
            retrieved_documents or []
        ),
        "mode": "local",
    }


def build_local_strategic_copilot_report(
    metrics: dict[str, Any],
    retrieved_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministic fallback strategic copilot when API unavailable."""
    hedge_ratio = metrics.get("recommended_hedge_ratio")
    hedge_pct = f"{float(hedge_ratio):.0%}" if hedge_ratio is not None else "50%"
    rating = metrics.get("multi_factor_risk_rating", "MEDIUM")
    cases = build_similar_knowledge_cases(retrieved_documents or [], top_k=3)
    return {
        "executive_summary": (
            f"Composite risk **{rating}** on **{metrics.get('pair', 'USD/CNH')}** "
            f"with deterministic hedge coverage **{hedge_pct}**. "
            "Rule-based knowledge scoring drives the 15% knowledge weight."
        ),
        "primary_risk_drivers": [
            f"Market layer: {float(metrics.get('market_score', 0)):.1f}",
            f"Macro layer: {float(metrics.get('macro_score', 0)):.1f}",
            f"Policy layer: {float(metrics.get('policy_score', 0)):.1f}",
            f"Knowledge layer: {float(metrics.get('knowledge_score', 0)):.1f}",
        ],
        "similar_treasury_cases": [c.get("title", "") for c in cases],
        "recommended_hedge_actions": [
            f"Maintain **{hedge_pct}** deterministic hedge coverage",
            "Review settlement timing against forward tenor windows",
        ],
        "recommended_instruments": ["Forward Contract", "Natural Hedging"],
        "monitoring_priorities": ["Spot vs MA5/MA30", "CFETS parity deviation", "FedWatch proxy"],
        "immediate_actions": [f"Hedge {hedge_pct} of identified receivables"],
        "thirty_day_plan": ["Layer additional forwards if volatility expands"],
        "watchlist_30_day": ["FOMC", "CPI releases", "USD yield trend", "PBOC guidance"],
        "mode": "local",
    }


def build_local_hedge_strategy_review(metrics: dict[str, Any]) -> dict[str, Any]:
    ratio = metrics.get("recommended_hedge_ratio")
    pct = f"{float(ratio):.0%}" if ratio is not None else "N/A"
    return {
        "deterministic_hedge_ratio": pct,
        "agreement_level": "MODERATE",
        "confidence": 0.5,
        "narrative": (
            f"**{pct}** hedge coverage aligns with composite score "
            f"**{metrics.get('final_risk_score', 'N/A')}** / 100 (deterministic)."
        ),
        "strengths": ["Derived from transparent composite risk tiers"],
        "weaknesses": ["AI review unavailable — rule-based ratio only"],
        "mode": "local",
    }


def serialize_for_cache(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def format_copilot_item(item: Any) -> str:
    """Render LLM JSON objects as readable treasury prose."""
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item)

    if item.get("risk") or item.get("driver"):
        label = str(item.get("risk") or item.get("driver"))
        severity = item.get("severity")
        rationale = str(item.get("rationale") or item.get("reason") or "").strip()
        if severity is not None:
            head = f"**{label}** (severity {severity}/10)"
        else:
            head = f"**{label}**"
        return f"{head} — {rationale}" if rationale else head

    if item.get("action"):
        action = str(item["action"]).strip()
        rationale = str(item.get("rationale") or item.get("reason") or "").strip()
        return f"{action} — {rationale}" if rationale else action

    if item.get("week") is not None and item.get("actions") is not None:
        week = item["week"]
        actions = item["actions"]
        if isinstance(actions, list):
            action_text = "; ".join(format_copilot_item(a) for a in actions)
        else:
            action_text = str(actions)
        return f"**Week {week}:** {action_text}"

    if item.get("instrument"):
        inst = str(item["instrument"])
        bits: list[str] = [f"**{inst}**"]
        if item.get("effectiveness") is not None:
            bits.append(f"effectiveness {item['effectiveness']}/10")
        if item.get("coverage"):
            bits.append(f"coverage: {item['coverage']}")
        line = " · ".join(bits)
        rationale = str(item.get("rationale") or "").strip()
        return f"{line} — {rationale}" if rationale else line

    if item.get("item"):
        label = str(item["item"])
        rationale = str(item.get("rationale") or item.get("reason") or "").strip()
        return f"**{label}** — {rationale}" if rationale else f"**{label}**"

    if item.get("title"):
        title = str(item["title"])
        rationale = str(item.get("rationale") or item.get("reason") or "").strip()
        return f"**{title}** — {rationale}" if rationale else title

    parts: list[str] = []
    for key, value in item.items():
        if key in {"rationale", "reason"}:
            continue
        if isinstance(value, list):
            parts.append(
                f"{key}: {', '.join(format_copilot_item(v) for v in value)}"
            )
        else:
            parts.append(f"{key}: {value}")
    rationale = str(item.get("rationale") or item.get("reason") or "").strip()
    body = " · ".join(parts)
    if rationale and body:
        return f"{body} — {rationale}"
    return rationale or body or str(item)
