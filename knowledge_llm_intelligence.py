"""
Professional AI Treasury Intelligence Layer.

Rule-based knowledge scoring remains authoritative; the LLM applies bounded,
domain-transparent adjustments (-15 to +15) when an OpenAI API key is available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable

from llm_service import (
    analyze_knowledge_with_llm,
    interpret_knowledge_cases_with_llm,
    rerank_knowledge_documents_with_llm,
    review_hedge_strategy_with_llm,
    run_strategic_treasury_copilot_with_llm,
)

logger = logging.getLogger(__name__)

KNOWLEDGE_LLM_CACHE_TTL_SEC = 30 * 60
LLM_KNOWLEDGE_ADJUSTMENT_MIN = -15.0
LLM_KNOWLEDGE_ADJUSTMENT_MAX = 15.0
DOMAIN_ADJUSTMENT_MIN = -5.0
DOMAIN_ADJUSTMENT_MAX = 5.0

_LLM_INTELLIGENCE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

ALLOWED_DOMAIN_NAMES = frozenset(
    {
        "Exposure Risk",
        "Liquidity Risk",
        "Compliance Risk",
        "Macro Sensitivity",
        "Operational Risk",
    }
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_domain_adjustment_total(risk_domains: list[Any]) -> tuple[float, list[dict[str, Any]]]:
    """
    Sum per-domain adjustments (each clamped -5..+5), then clamp total to -15..+15.
    """
    normalized: list[dict[str, Any]] = []
    total = 0.0
    for item in risk_domains or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name not in ALLOWED_DOMAIN_NAMES:
            continue
        try:
            raw_adj = float(item.get("adjustment", 0.0))
        except (TypeError, ValueError):
            raw_adj = 0.0
        try:
            severity = int(item.get("severity", 0))
        except (TypeError, ValueError):
            severity = 0
        clamped_adj = round(_clamp(raw_adj, DOMAIN_ADJUSTMENT_MIN, DOMAIN_ADJUSTMENT_MAX), 1)
        total += clamped_adj
        normalized.append(
            {
                "name": name,
                "severity": _clamp(severity, 0, 10),
                "adjustment": clamped_adj,
                "reason": str(item.get("reason", ""))[:300],
            }
        )
    overall = round(_clamp(total, LLM_KNOWLEDGE_ADJUSTMENT_MIN, LLM_KNOWLEDGE_ADJUSTMENT_MAX), 1)
    return overall, normalized


def apply_confidence_weighted_adjustment(
    raw_adjustment: float,
    confidence: float,
) -> float:
    """Scale bounded LLM adjustment by analyst confidence."""
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < 0.3:
        return 0.0
    try:
        adjustment = float(raw_adjustment)
    except (TypeError, ValueError):
        adjustment = 0.0
    clamped = _clamp(adjustment, LLM_KNOWLEDGE_ADJUSTMENT_MIN, LLM_KNOWLEDGE_ADJUSTMENT_MAX)
    weighted = clamped * conf
    if conf < 0.5:
        weighted *= 0.5
    return round(weighted, 1)


def resolve_overall_adjustment(llm_result: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Prefer transparent domain-sum; fall back to overall_adjustment field."""
    domain_items = llm_result.get("risk_domains") or []
    if domain_items:
        return compute_domain_adjustment_total(domain_items)
    try:
        overall = float(llm_result.get("overall_adjustment", llm_result.get("risk_adjustment", 0.0)))
    except (TypeError, ValueError):
        overall = 0.0
    overall = round(_clamp(overall, LLM_KNOWLEDGE_ADJUSTMENT_MIN, LLM_KNOWLEDGE_ADJUSTMENT_MAX), 1)
    return overall, []


def _compact_doc_fingerprint(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc.get("id", ""),
        "title": doc.get("title", ""),
        "domain": doc.get("domain", ""),
        "score": doc.get("score"),
        "similarity": doc.get("similarity"),
    }


def _cache_get(key: str) -> dict[str, Any] | None:
    entry = _LLM_INTELLIGENCE_CACHE.get(key)
    if not entry:
        return None
    expires_at, payload = entry
    if time.time() > expires_at:
        _LLM_INTELLIGENCE_CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: dict[str, Any]) -> None:
    _LLM_INTELLIGENCE_CACHE[key] = (time.time() + KNOWLEDGE_LLM_CACHE_TTL_SEC, payload)


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rerank_retrieved_documents(
    candidates: list[dict[str, Any]],
    enterprise_scenario: str,
    currency_pair: str,
    *,
    api_key: str | None = None,
    budget_gate: Callable[[], bool] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """GPT rerank top-N vector hits; fallback to original order when API unavailable."""
    if not candidates:
        return []
    pool = candidates[:10]
    key = str(api_key or "").strip()
    if not key:
        return pool[:top_k]

    cache_key = f"rerank:{_hash_payload({'pair': currency_pair, 'scenario': enterprise_scenario, 'docs': [_compact_doc_fingerprint(d) for d in pool]})}"
    cached = _cache_get(cache_key)
    if cached and cached.get("ranked_ids"):
        id_order = {doc_id: idx for idx, doc_id in enumerate(cached["ranked_ids"])}
        reranked = sorted(
            pool,
            key=lambda d: id_order.get(str(d.get("id", "")), 999),
        )
        for doc in reranked[:top_k]:
            doc["llm_reranked"] = True
        return reranked[:top_k]

    try:
        result = rerank_knowledge_documents_with_llm(
            pool,
            enterprise_scenario=enterprise_scenario,
            currency_pair=currency_pair,
            api_key=key,
            budget_gate=budget_gate,
        )
    except Exception as exc:
        logger.warning("LLM rerank failed: %s", str(exc)[:200])
        return pool[:top_k]

    if not result or not result.get("ranked_ids"):
        return pool[:top_k]

    _cache_set(cache_key, {"ranked_ids": result["ranked_ids"]})
    id_order = {doc_id: idx for idx, doc_id in enumerate(result["ranked_ids"])}
    reranked = sorted(pool, key=lambda d: id_order.get(str(d.get("id", "")), 999))
    for doc in reranked[:top_k]:
        doc["llm_reranked"] = True
    return reranked[:top_k]


def build_ai_knowledge_assessment_payload(
    llm_result: dict[str, Any],
    *,
    rule_based_score: float,
    applied_adjustment: float,
    final_score: float,
    domain_breakdown: list[dict[str, Any]],
) -> dict[str, Any]:
    """User-facing AI Knowledge Intelligence block — no raw OpenAI response."""
    try:
        confidence = float(llm_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    raw_overall, _ = resolve_overall_adjustment(llm_result)
    consensus = llm_result.get("knowledge_consensus") or {}
    if not isinstance(consensus, dict):
        consensus = {}

    mitigation = llm_result.get("mitigation_assessment") or []
    if not isinstance(mitigation, list):
        mitigation = []

    hidden = llm_result.get("hidden_risks") or []
    if not isinstance(hidden, list):
        hidden = [str(hidden)]

    top_risks = [
        f"{d.get('name')}: {d.get('reason', '')}"
        for d in domain_breakdown
        if d.get("adjustment", 0) > 0
    ][:5]

    adj_sign = f"{applied_adjustment:+.1f}".replace(".0", "")
    return {
        "confidence": round(confidence, 3),
        "confidence_pct": f"{confidence * 100:.0f}%",
        "raw_adjustment": round(raw_overall, 1),
        "adjustment_applied": applied_adjustment,
        "adjustment_display": adj_sign,
        "rule_based_knowledge_score": rule_based_score,
        "final_knowledge_score": final_score,
        "risk_domains": domain_breakdown,
        "mitigation_assessment": mitigation[:6],
        "mitigation_strength": mitigation[:6],
        "hidden_risks": [str(r) for r in hidden[:8]],
        "top_risks": top_risks,
        "executive_summary": str(llm_result.get("executive_summary", ""))[:800],
        "knowledge_consensus": {
            "consensus": str(consensus.get("consensus", llm_result.get("consensus", "MEDIUM"))),
            "consensus_score": consensus.get(
                "consensus_score", llm_result.get("consensus_score")
            ),
        },
        "enabled": True,
    }


def enhance_knowledge_score_with_llm(
    rule_based_score: float,
    retrieved_documents: list[dict[str, Any]],
    currency_pair: str,
    market_context: dict[str, Any],
    macro_context: dict[str, Any],
    policy_context: dict[str, Any],
    *,
    enterprise_scenario: str = "",
    api_key: str | None = None,
    budget_gate: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Apply optional bounded LLM adjustment; fallback to rule-based score on failure."""
    rule_score = round(_clamp(float(rule_based_score), 0.0, 100.0), 1)
    base_result: dict[str, Any] = {
        "rule_based_knowledge_score": rule_score,
        "final_knowledge_score": rule_score,
        "llm_adjustment_raw": 0.0,
        "llm_adjustment_applied": 0.0,
        "ai_knowledge_assessment": None,
        "case_interpretations": [],
        "hedge_strategy_review": None,
        "treasury_copilot_report": None,
        "llm_enhanced": False,
        "documents_reranked": any(d.get("llm_reranked") for d in (retrieved_documents or [])),
    }

    key = str(api_key or "").strip()
    if not key:
        return base_result

    docs = retrieved_documents or []
    cache_key = _hash_payload(
        {
            "pair": currency_pair,
            "scenario": enterprise_scenario,
            "docs": [_compact_doc_fingerprint(d) for d in docs],
            "market": market_context,
            "macro": macro_context,
            "policy": policy_context,
        }
    )

    bundle = _cache_get(f"knowledge:{cache_key}")
    if bundle is None:
        bundle = _run_knowledge_analysis_bundle(
            docs,
            currency_pair,
            market_context,
            macro_context,
            policy_context,
            enterprise_scenario=enterprise_scenario,
            api_key=key,
            budget_gate=budget_gate,
        )
        if bundle:
            _cache_set(f"knowledge:{cache_key}", bundle)

    if not bundle or not bundle.get("llm_result"):
        return base_result

    llm_result = bundle["llm_result"]
    raw_overall, domain_breakdown = resolve_overall_adjustment(llm_result)
    try:
        confidence = float(llm_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    applied = apply_confidence_weighted_adjustment(raw_overall, confidence)
    final_score = round(_clamp(rule_score + applied, 0.0, 100.0), 1)

    assessment = build_ai_knowledge_assessment_payload(
        llm_result,
        rule_based_score=rule_score,
        applied_adjustment=applied,
        final_score=final_score,
        domain_breakdown=domain_breakdown,
    )

    base_result.update(
        {
            "final_knowledge_score": final_score,
            "llm_adjustment_raw": round(raw_overall, 1),
            "llm_adjustment_applied": applied,
            "ai_knowledge_assessment": assessment,
            "case_interpretations": bundle.get("case_interpretations") or [],
            "llm_enhanced": applied != 0.0 or bool(assessment.get("executive_summary")),
        }
    )
    return base_result


def enrich_post_assessment_intelligence(
    metrics: dict[str, Any],
    retrieved_documents: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    budget_gate: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Hedge strategy review + strategic copilot report (after composite is computed).
    Informational only — never modifies deterministic scores.
    """
    result: dict[str, Any] = {
        "hedge_strategy_review": None,
        "treasury_copilot_report": None,
    }
    key = str(api_key or "").strip()
    if not key:
        return result

    cache_key = _hash_payload(
        {
            "pair": metrics.get("pair"),
            "scenario": metrics.get("enterprise_scenario"),
            "final_risk_score": metrics.get("final_risk_score"),
            "hedge_ratio": metrics.get("recommended_hedge_ratio"),
            "knowledge_score": metrics.get("knowledge_score"),
            "docs": [_compact_doc_fingerprint(d) for d in (retrieved_documents or [])[:3]],
        }
    )
    cached = _cache_get(f"post:{cache_key}")
    if cached:
        return cached

    pair = str(metrics.get("pair", "USD/CNH"))
    scenario = str(metrics.get("enterprise_scenario", ""))
    docs = retrieved_documents or []

    try:
        hedge_ctx = {
            "pair": pair,
            "enterprise_scenario": scenario,
            "deterministic_hedge_ratio": metrics.get("recommended_hedge_ratio"),
            "deterministic_hedge_ratio_pct": (
                f"{float(metrics['recommended_hedge_ratio']):.0%}"
                if metrics.get("recommended_hedge_ratio") is not None
                else "N/A"
            ),
            "rule_based_knowledge_score": metrics.get("rule_based_knowledge_score"),
            "final_knowledge_score": metrics.get("knowledge_score"),
            "market_score": metrics.get("market_score"),
            "macro_score": metrics.get("macro_score"),
            "policy_score": metrics.get("policy_score"),
            "final_risk_score": metrics.get("final_risk_score"),
            "multi_factor_risk_rating": metrics.get("multi_factor_risk_rating"),
            "ai_knowledge_assessment": metrics.get("ai_knowledge_assessment"),
        }
        hedge_review = review_hedge_strategy_with_llm(
            hedge_ctx,
            api_key=key,
            budget_gate=budget_gate,
        )
        if hedge_review:
            result["hedge_strategy_review"] = hedge_review

        copilot_ctx = {
            "pair": pair,
            "enterprise_scenario": scenario,
            "deterministic_scores": {
                "market_score": metrics.get("market_score"),
                "macro_score": metrics.get("macro_score"),
                "policy_score": metrics.get("policy_score"),
                "rule_based_knowledge_score": metrics.get("rule_based_knowledge_score"),
                "final_knowledge_score": metrics.get("knowledge_score"),
                "recommended_hedge_ratio": metrics.get("recommended_hedge_ratio"),
                "final_risk_score": metrics.get("final_risk_score"),
                "multi_factor_risk_rating": metrics.get("multi_factor_risk_rating"),
            },
            "ai_knowledge_assessment": metrics.get("ai_knowledge_assessment"),
            "case_interpretations": metrics.get("case_interpretations"),
            "retrieved_documents": [
                {"id": d.get("id"), "title": d.get("title"), "domain": d.get("domain")}
                for d in docs[:3]
            ],
        }
        copilot = run_strategic_treasury_copilot_with_llm(
            copilot_ctx,
            api_key=key,
            budget_gate=budget_gate,
        )
        if copilot:
            result["treasury_copilot_report"] = copilot
    except Exception as exc:
        logger.warning("Post-assessment intelligence failed: %s", str(exc)[:200])

    _cache_set(f"post:{cache_key}", result)
    return result


def _run_knowledge_analysis_bundle(
    docs: list[dict[str, Any]],
    currency_pair: str,
    market_context: dict[str, Any],
    macro_context: dict[str, Any],
    policy_context: dict[str, Any],
    *,
    enterprise_scenario: str,
    api_key: str,
    budget_gate: Callable[[], bool] | None,
) -> dict[str, Any] | None:
    """Knowledge domain analysis + case interpretation."""
    bundle: dict[str, Any] = {}
    try:
        llm_result = analyze_knowledge_with_llm(
            docs,
            currency_pair,
            market_context,
            macro_context,
            policy_context,
            enterprise_scenario=enterprise_scenario,
            api_key=api_key,
            budget_gate=budget_gate,
        )
        if llm_result:
            bundle["llm_result"] = llm_result

        case_interp = interpret_knowledge_cases_with_llm(
            docs[:3],
            enterprise_scenario,
            api_key=api_key,
            budget_gate=budget_gate,
        )
        if case_interp:
            bundle["case_interpretations"] = case_interp.get("cases") or []
    except Exception as exc:
        logger.warning("Knowledge analysis bundle failed: %s", str(exc)[:200])
        return bundle or None

    return bundle if bundle else None
