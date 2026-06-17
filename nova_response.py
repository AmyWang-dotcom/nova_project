"""
Dual-mode NOVA response router: Local (deterministic) vs AI Enhanced (optional OpenAI).

MODE A — no API key: identical to legacy rule-engine + RAG behaviour.
MODE B — API key in session_state: GPT narrative layer on top; scores stay deterministic.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from llm_service import LLMConfig, LLMService, mask_secrets_in_text

logger = logging.getLogger(__name__)

MAX_UPLOAD_DOCUMENT_CHARS = 12_000


def truncate_document_text(text: str, limit: int = MAX_UPLOAD_DOCUMENT_CHARS) -> str:
    """Cap uploaded document text for token protection."""
    cleaned = str(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit]


def is_ai_enhanced_mode(api_key: str | None) -> bool:
    return LLMConfig.from_api_key(api_key).enabled


def _simplify_retrieved_docs(docs: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for doc in docs[:limit]:
        simplified.append(
            {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "domain": doc.get("domain", ""),
                "tags": doc.get("tags", ""),
                "business_type": doc.get("business_type", ""),
                "risk_domains": doc.get("risk_domains", []),
                "pain_points": (doc.get("pain_points") or [])[:4],
                "actionable_playbook": str(doc.get("actionable_playbook", ""))[:600],
            }
        )
    return simplified


def build_copilot_llm_context(
    *,
    user_query: str,
    rag_parts: dict[str, Any],
    metrics: dict[str, Any],
    prompt_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle = prompt_bundle or {}
    return {
        "user_query": user_query,
        "pair": rag_parts.get("active_pair") or metrics.get("pair"),
        "live_metrics": {
            "spot": metrics.get("spot"),
            "ma5": metrics.get("ma5"),
            "ma30": metrics.get("ma30"),
            "bollinger_upper": metrics.get("bollinger_upper"),
            "pct_change": metrics.get("pct_change"),
            "rsi": metrics.get("net_payout_rsi"),
            "macd_histogram": metrics.get("net_payout_macd_histogram"),
            "market_score": metrics.get("market_score"),
            "macro_score": metrics.get("macro_score"),
            "policy_score": metrics.get("policy_score"),
            "knowledge_score": metrics.get("knowledge_score"),
            "final_risk_score": metrics.get("final_risk_score"),
            "recommended_hedge_ratio": metrics.get("recommended_hedge_ratio"),
            "multi_factor_risk_rating": metrics.get("multi_factor_risk_rating"),
        },
        "macro_indicators": bundle.get("imf_macro_indicators"),
        "imf_macro_block": bundle.get("imf_macro_block", ""),
        "china_money": bundle.get("china_money"),
        "tnx_proxy": bundle.get("tnx_proxy"),
        "enterprise_assessment": {
            "ready": metrics.get("enterprise_assessment_ready", False),
            "scenario": metrics.get("enterprise_scenario", ""),
            "knowledge_breakdown_domains": (metrics.get("knowledge_breakdown") or {}).get(
                "detected_risk_domains", []
            ),
        },
        "retrieved_documents": _simplify_retrieved_docs(rag_parts.get("docs", [])),
        "deterministic_executive_action": rag_parts.get("executive_action", ""),
        "deterministic_execution_block": rag_parts.get("execution_block", ""),
        "deterministic_risk_explanation": rag_parts.get("risk_score_explanation", ""),
    }


def build_contract_audit_llm_context(
    audit_result: dict[str, Any],
    *,
    pair: str,
    document_text: str = "",
) -> dict[str, Any]:
    preview = truncate_document_text(document_text, 4_000)
    return {
        "pair": pair,
        "filename": audit_result.get("filename", ""),
        "extraction_mode": audit_result.get("extraction_mode", ""),
        "notional": audit_result.get("notional"),
        "expected_settlement_days": audit_result.get("expected_settlement_days"),
        "gross_margin_pct": audit_result.get("gross_margin_pct"),
        "recommended_forward_ratio": audit_result.get("recommended_forward_ratio"),
        "estimated_cnh_at_risk": audit_result.get("estimated_cnh_at_risk"),
        "regex_hits": audit_result.get("regex_hits", {}),
        "document_preview": preview,
        "deterministic_summary": audit_result.get("summary", ""),
    }


def build_risk_explanation_llm_context(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": metrics.get("pair"),
        "enterprise_scenario": metrics.get("enterprise_scenario", ""),
        "deterministic_explanation": metrics.get("risk_score_explanation", ""),
        "scores": {
            "market_score": metrics.get("market_score"),
            "macro_score": metrics.get("macro_score"),
            "policy_score": metrics.get("policy_score"),
            "knowledge_score": metrics.get("knowledge_score"),
            "final_risk_score": metrics.get("final_risk_score"),
            "recommended_hedge_ratio": metrics.get("recommended_hedge_ratio"),
            "multi_factor_risk_rating": metrics.get("multi_factor_risk_rating"),
        },
        "market_breakdown": metrics.get("market_breakdown", {}),
        "macro_breakdown": metrics.get("macro_breakdown", {}),
        "policy_breakdown": metrics.get("policy_breakdown", {}),
        "knowledge_breakdown": metrics.get("knowledge_breakdown", {}),
        "risk_contributions": metrics.get("risk_contributions", {}),
    }


def generate_nova_response(
    *,
    response_type: str,
    api_key: str | None,
    context: dict[str, Any],
    local_builder: Callable[[], dict[str, Any]],
    budget_gate: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Central dual-mode wrapper.

    response_type: copilot | risk_explanation | contract_audit | knowledge
    Always returns local_builder() payload; adds AI fields only when key works.
    """
    local_payload = local_builder()
    result: dict[str, Any] = {
        "mode": "local",
        "response_type": response_type,
        **local_payload,
    }

    config = LLMConfig.from_api_key(api_key)
    if not config.enabled:
        return result

    llm = LLMService(config, budget_gate=budget_gate)
    try:
        if response_type == "copilot":
            narrative = llm.generate_copilot_narrative(context)
            if narrative:
                result["mode"] = "ai_enhanced"
                result["ai_narrative"] = narrative
            knowledge = llm.analyze_knowledge_base(
                context.get("retrieved_documents", []),
                user_scenario=str(
                    (context.get("enterprise_assessment") or {}).get("scenario", "")
                ),
            )
            if knowledge:
                result["ai_knowledge_reasoning"] = knowledge

        elif response_type == "risk_explanation":
            narrative = llm.generate_risk_narrative(context)
            if narrative:
                result["mode"] = "ai_enhanced"
                result["ai_risk_narrative"] = narrative

        elif response_type == "contract_audit":
            review = llm.generate_contract_audit_review(context)
            if review:
                result["mode"] = "ai_enhanced"
                result["ai_contract_review"] = review

        elif response_type == "knowledge":
            docs = context.get("retrieved_documents", [])
            scenario = context.get("user_scenario", "")
            reasoning = llm.analyze_knowledge_base(docs, user_scenario=scenario)
            if reasoning:
                result["mode"] = "ai_enhanced"
                result["ai_knowledge_reasoning"] = reasoning

    except Exception as exc:
        logger.warning(
            "generate_nova_response AI layer failed (%s): %s",
            response_type,
            mask_secrets_in_text(str(exc)),
        )

    return result
