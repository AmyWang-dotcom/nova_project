"""Multi-dimensional risk scoring engine."""
from __future__ import annotations

from typing import Any

from .constants import (
    DEFAULT_PAIR,
    NOVA_RISK_WEIGHTS,
    _KNOWLEDGE_DOMAIN_BASE_SCORES,
    _KNOWLEDGE_DOMAIN_REASONING,
    _KNOWLEDGE_MITIGATION_FRAMEWORK_STRENGTHS,
    _KNOWLEDGE_MITIGATION_REASONING,
    _KNOWLEDGE_SIGNAL_WEIGHT,
    _LEGACY_DOMAIN_KEYWORD_PATTERNS,
    _LEGACY_MITIGATION_KEYWORD_PATTERNS,
    _SCORE_AMPLIFIER_TIERS,
    _EXPORT_ENTITY_MARKERS,
    _IMPORT_ENTITY_MARKERS,
)
from .knowledge_base import (
    _doc_case_metadata,
    _doc_retrieval_weight,
    _knowledge_domain_base_score,
    _knowledge_mitigation_strength,
    compute_imf_ppp_spreads,
    format_imf_macro_indicators_block,
    get_imf_macro_knowledge_entry,
)
from .macro_engine import compute_macro_risk_score, compute_policy_risk_score
from .market_data import fetch_china_money_central_parity, fetch_tnx_fedwatch_proxy
from .retrieval import retrieve_knowledge_with_llm_rerank

def _build_knowledge_corpus(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    parts = [query.lower()]
    for doc in retrieved_docs:
        meta = _doc_case_metadata(doc)
        parts.extend(
            [
                str(doc.get("title", "")),
                str(doc.get("domain", "")),
                meta["business_type"],
                " ".join(meta["pain_points"]),
                " ".join(meta["risk_domains"]),
                " ".join(meta["hedging_solutions"]),
                str(doc.get("tags", "")),
            ]
        )
    return " ".join(parts).lower()
def _signal_density_score(
    patterns: tuple[str, ...],
    corpus: str,
    *,
    cap: float,
) -> tuple[float, list[str]]:
    """Count pattern hits and convert to capped density score."""
    matched = [pattern for pattern in patterns if pattern in corpus]
    raw_score = min(len(matched) * _KNOWLEDGE_SIGNAL_WEIGHT, cap)
    return raw_score, matched
def _graduated_tier_points(value: float, tiers: tuple[tuple[float, float], ...]) -> float:
    for threshold, points in tiers:
        if value >= threshold:
            return points
    return 0.0
def _metadata_confidence_factor(
    *,
    matched_signals: list[str],
    total_signals: int,
    severity: int,
    retrieval_weight: float,
) -> float:
    """Blend evidence density, case severity, and retrieval relevance."""
    if total_signals > 0:
        density = len(matched_signals) / total_signals
    else:
        density = 0.55
    severity_factor = min(1.0, max(0.2, severity / 5.0))
    confidence = density * 0.55 + severity_factor * 0.25 + retrieval_weight * 0.20
    return round(min(1.0, max(0.35, confidence)), 3)
def _retrieved_affected_entities(retrieved_docs: list[dict[str, Any]]) -> set[str]:
    entities: set[str] = set()
    for doc in retrieved_docs:
        for entity in _doc_case_metadata(doc).get("affected_entities", []):
            entities.add(str(entity).lower())
    return entities
def _entities_indicate_exporter(entities: set[str]) -> bool:
    return any(
        any(marker in entity for marker in _EXPORT_ENTITY_MARKERS)
        for entity in entities
    )
def _entities_indicate_importer(entities: set[str]) -> bool:
    return any(
        any(marker in entity for marker in _IMPORT_ENTITY_MARKERS)
        for entity in entities
    )
def _legacy_domain_signal_scores(corpus: str) -> dict[str, tuple[float, list[str]]]:
    """Keyword-density fallback for documents without structured metadata."""
    scores: dict[str, tuple[float, list[str]]] = {}
    for domain_name, patterns in _LEGACY_DOMAIN_KEYWORD_PATTERNS.items():
        cap = float(_knowledge_domain_base_score(domain_name))
        scores[domain_name] = _signal_density_score(patterns, corpus, cap=cap)
    return scores
def _legacy_mitigation_signal_scores(corpus: str) -> dict[str, tuple[float, list[str]]]:
    scores: dict[str, tuple[float, list[str]]] = {}
    for framework_name, patterns in _LEGACY_MITIGATION_KEYWORD_PATTERNS.items():
        cap = float(_knowledge_mitigation_strength(framework_name))
        scores[framework_name] = _signal_density_score(patterns, corpus, cap=cap)
    return scores
def _aggregate_semantic_knowledge_risk(
    retrieved_docs: list[dict[str, Any]],
    *,
    corpus: str,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[str],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Extract risk domains and mitigations from structured case metadata, applying
    retrieval weight and evidence-confidence scaling. Legacy keyword density is
    used only when metadata is absent.
    """
    domain_totals: dict[str, float] = {}
    mitigation_totals: dict[str, float] = {}
    domain_contributions: list[dict[str, Any]] = []
    mitigation_contributions: list[dict[str, Any]] = []
    domain_evidence_map: dict[str, dict[str, Any]] = {}
    mitigation_evidence_map: dict[str, dict[str, Any]] = {}
    metadata_docs_present = False

    for doc in retrieved_docs:
        meta = _doc_case_metadata(doc)
        weight = _doc_retrieval_weight(doc)
        doc_id = str(doc.get("id", ""))
        doc_title = str(doc.get("title", doc_id))
        pain_points = [str(point).lower() for point in meta["pain_points"]]
        matched_pain_points = [point for point in pain_points if point in corpus]

        if meta["risk_domains"]:
            metadata_docs_present = True

        for domain_name in meta["risk_domains"]:
            domain = str(domain_name)
            base_points = float(_knowledge_domain_base_score(domain))
            confidence = _metadata_confidence_factor(
                matched_signals=matched_pain_points,
                total_signals=max(len(pain_points), 1),
                severity=meta["severity"],
                retrieval_weight=weight,
            )
            weighted_points = round(base_points * weight * confidence, 2)
            domain_totals[domain] = round(domain_totals.get(domain, 0.0) + weighted_points, 2)
            domain_contributions.append(
                {
                    "label": domain,
                    "points": weighted_points,
                    "type": "domain",
                    "weight": round(weight, 3),
                    "confidence_factor": confidence,
                    "base_points": base_points,
                    "source_id": doc_id,
                    "source_title": doc_title,
                }
            )
            evidence = domain_evidence_map.setdefault(
                domain,
                {
                    "domain": domain,
                    "signals": [],
                    "reason": _KNOWLEDGE_DOMAIN_REASONING.get(domain, ""),
                    "confidence_factor": 0.0,
                    "contribution": 0.0,
                },
            )
            evidence["signals"].extend(matched_pain_points or pain_points[:3])
            evidence["confidence_factor"] = max(evidence["confidence_factor"], confidence)
            evidence["contribution"] = round(domain_totals[domain], 2)

        for framework_name in meta["hedging_solutions"]:
            framework = str(framework_name)
            base_strength = float(_knowledge_mitigation_strength(framework))
            confidence = _metadata_confidence_factor(
                matched_signals=matched_pain_points,
                total_signals=max(len(pain_points), 1),
                severity=meta["severity"],
                retrieval_weight=weight,
            )
            weighted_points = round(base_strength * weight * confidence, 2)
            mitigation_totals[framework] = round(
                mitigation_totals.get(framework, 0.0) + weighted_points,
                2,
            )
            mitigation_contributions.append(
                {
                    "label": framework,
                    "points": weighted_points,
                    "type": "mitigation",
                    "weight": round(weight, 3),
                    "confidence_factor": confidence,
                    "base_points": base_strength,
                    "source_id": doc_id,
                    "source_title": doc_title,
                }
            )
            mitigation_evidence = mitigation_evidence_map.setdefault(
                framework,
                {
                    "framework": framework,
                    "reason": _KNOWLEDGE_MITIGATION_REASONING.get(
                        framework,
                        "Reduces identified treasury risk through structured hedging.",
                    ),
                    "contribution": 0.0,
                },
            )
            mitigation_evidence["contribution"] = round(mitigation_totals[framework], 2)

    if not metadata_docs_present:
        for domain_name, (density_score, matched) in _legacy_domain_signal_scores(corpus).items():
            if density_score <= 0:
                continue
            base_points = float(_knowledge_domain_base_score(domain_name))
            confidence = round(min(1.0, density_score / base_points), 3)
            weighted_points = round(density_score, 2)
            domain_totals[domain_name] = round(
                domain_totals.get(domain_name, 0.0) + weighted_points,
                2,
            )
            domain_contributions.append(
                {
                    "label": domain_name,
                    "points": weighted_points,
                    "type": "domain",
                    "weight": 1.0,
                    "confidence_factor": confidence,
                    "base_points": base_points,
                    "source_id": "legacy_keyword_fallback",
                    "source_title": "Legacy keyword density",
                }
            )
            evidence = domain_evidence_map.setdefault(
                domain_name,
                {
                    "domain": domain_name,
                    "signals": matched,
                    "reason": _KNOWLEDGE_DOMAIN_REASONING.get(domain_name, ""),
                    "confidence_factor": confidence,
                    "contribution": 0.0,
                },
            )
            evidence["signals"].extend(matched)
            evidence["contribution"] = round(domain_totals[domain_name], 2)

        for framework_name, (density_score, matched) in _legacy_mitigation_signal_scores(
            corpus
        ).items():
            if density_score <= 0:
                continue
            base_strength = float(_knowledge_mitigation_strength(framework_name))
            confidence = round(min(1.0, density_score / base_strength), 3)
            weighted_points = round(density_score, 2)
            mitigation_totals[framework_name] = round(
                mitigation_totals.get(framework_name, 0.0) + weighted_points,
                2,
            )
            mitigation_contributions.append(
                {
                    "label": framework_name,
                    "points": weighted_points,
                    "type": "mitigation",
                    "weight": 1.0,
                    "confidence_factor": confidence,
                    "base_points": base_strength,
                    "source_id": "legacy_keyword_fallback",
                    "source_title": "Legacy keyword density",
                }
            )
            mitigation_evidence = mitigation_evidence_map.setdefault(
                framework_name,
                {
                    "framework": framework_name,
                    "reason": _KNOWLEDGE_MITIGATION_REASONING.get(framework_name, ""),
                    "contribution": 0.0,
                },
            )
            mitigation_evidence["contribution"] = round(mitigation_totals[framework_name], 2)

    for evidence in domain_evidence_map.values():
        evidence["signals"] = list(dict.fromkeys(evidence.get("signals", [])))[:6]

    detected_domains = sorted(
        domain_totals.keys(),
        key=lambda name: domain_totals[name],
        reverse=True,
    )
    detected_mitigations = sorted(
        mitigation_totals.keys(),
        key=lambda name: mitigation_totals[name],
        reverse=True,
    )
    domain_evidence = [domain_evidence_map[name] for name in detected_domains if name in domain_evidence_map]
    mitigation_evidence = [
        mitigation_evidence_map[name]
        for name in detected_mitigations
        if name in mitigation_evidence_map
    ]
    return (
        domain_totals,
        mitigation_totals,
        detected_domains,
        detected_mitigations,
        domain_contributions,
        mitigation_contributions,
        domain_evidence,
        mitigation_evidence,
    )
def _infer_base_ccy_trend(
    metrics: dict[str, Any],
    tnx_proxy: dict[str, Any] | None = None,
) -> str:
    """Infer whether the pair's base currency is bullish, bearish, or neutral vs CNH."""
    spot = float(metrics.get("spot", 0.0))
    ma5 = float(metrics.get("ma5", spot))
    ma30 = float(metrics.get("ma30", spot))
    base_ccy = str(metrics.get("base_ccy", "USD")).upper()

    if spot > ma5 and spot > ma30:
        trend = "bullish"
    elif spot < ma5 and spot < ma30:
        trend = "bearish"
    else:
        trend = "neutral"

    if base_ccy == "USD" and tnx_proxy and tnx_proxy.get("dovish_signal"):
        return "bearish"
    if base_ccy == "USD" and tnx_proxy and tnx_proxy.get("status") == "ok":
        ma5_y = float(tnx_proxy.get("ma5_yield", 0.0))
        ma30_y = float(tnx_proxy.get("ma30_yield", 0.0))
        current_y = float(tnx_proxy.get("current_yield", 0.0))
        bb_mid = float(tnx_proxy.get("bollinger_middle", current_y))
        if ma5_y > ma30_y and current_y > bb_mid and trend == "neutral":
            return "bullish"

    return trend
def _market_volatility_severity(metrics: dict[str, Any]) -> float:
    """Graduated 0–100 volatility severity for contextual amplification."""
    severity = 0.0
    pct = abs(float(metrics.get("pct_change", 0.0)))
    if pct >= 0.5:
        severity += 40.0
    elif pct >= 0.35:
        severity += 25.0
    elif pct >= 0.2:
        severity += 10.0

    spot = float(metrics.get("spot", 0.0))
    low_30d = float(metrics.get("low_30d", spot))
    high_30d = float(metrics.get("high_30d", spot))
    if spot > 0 and high_30d > low_30d:
        range_pct = (high_30d - low_30d) / spot * 100.0
        if range_pct >= 4.0:
            severity += 35.0
        elif range_pct >= 3.0:
            severity += 25.0
        elif range_pct >= 2.0:
            severity += 10.0

    boll_upper = float(metrics.get("bollinger_upper", spot))
    if spot > 0 and spot >= boll_upper * 0.995:
        severity += 20.0

    if metrics.get("risk_level") == "HIGH":
        severity += 25.0

    return min(100.0, severity)
def _market_volatility_elevated(metrics: dict[str, Any]) -> bool:
    return _market_volatility_severity(metrics) >= 40.0
def _trend_deviation_severity(metrics: dict[str, Any]) -> float:
    """Graduated trend strength from spot distance to MA5/MA30."""
    spot = float(metrics.get("spot", 0.0))
    if spot <= 0:
        return 0.0
    ma5 = float(metrics.get("ma5", spot))
    ma30 = float(metrics.get("ma30", spot))
    dev5 = abs(spot - ma5) / spot * 100.0
    dev30 = abs(spot - ma30) / spot * 100.0
    return min(100.0, (dev5 + dev30) * 12.0)
def _compute_knowledge_context_amplifiers(
    *,
    retrieved_docs: list[dict[str, Any]],
    detected_domains: set[str],
    metrics: dict[str, Any],
    macro_score: float,
    policy_score: float,
    tnx_proxy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """
    Jointly infer graduated contextual amplifiers from detected knowledge risk
    domains, retrieved knowledge entities, and live market/macro/policy state.
    """
    contextual_matches: list[dict[str, Any]] = []
    amplifiers: list[dict[str, Any]] = []
    total_points = 0.0
    base_ccy = str(metrics.get("base_ccy", "USD")).upper()
    trend = _infer_base_ccy_trend(metrics, tnx_proxy)
    trend_severity = _trend_deviation_severity(metrics)
    volatility_severity = _market_volatility_severity(metrics)
    affected_entities = _retrieved_affected_entities(retrieved_docs)

    if "Exposure Risk" in detected_domains:
        if trend == "bearish" and _entities_indicate_exporter(affected_entities):
            points = _graduated_tier_points(
                trend_severity,
                ((60.0, 15.0), (35.0, 10.0), (15.0, 5.0)),
            )
            if points > 0:
                total_points += points
                contextual_matches.append(
                    {
                        "domain": "Exposure Risk",
                        "context": (
                            f"{base_ccy} bearish trend (severity {trend_severity:.0f}) "
                            "with exporter/receivable knowledge exposure"
                        ),
                        "matched_signals": sorted(affected_entities)[:4],
                    }
                )
                amplifiers.append(
                    {
                        "label": f"{base_ccy} Bearish Export Exposure",
                        "points": points,
                        "type": "context",
                    }
                )
        elif trend == "bullish" and _entities_indicate_importer(affected_entities):
            points = _graduated_tier_points(
                trend_severity,
                ((60.0, 15.0), (35.0, 10.0), (15.0, 5.0)),
            )
            if points > 0:
                total_points += points
                contextual_matches.append(
                    {
                        "domain": "Exposure Risk",
                        "context": (
                            f"{base_ccy} bullish trend (severity {trend_severity:.0f}) "
                            "with import/payable knowledge exposure"
                        ),
                        "matched_signals": sorted(affected_entities)[:4],
                    }
                )
                amplifiers.append(
                    {
                        "label": f"{base_ccy} Bullish Import Exposure",
                        "points": points,
                        "type": "context",
                    }
                )

    if "Macro Event Risk" in detected_domains:
        points = _graduated_tier_points(float(macro_score), _SCORE_AMPLIFIER_TIERS)
        if points > 0:
            total_points += points
            contextual_matches.append(
                {
                    "domain": "Macro Event Risk",
                    "context": (
                        f"Macro score {macro_score:.1f} aligns with macro playbook exposure"
                    ),
                    "matched_signals": ["elevated macro regime"],
                }
            )
            amplifiers.append(
                {
                    "label": "Elevated Macro Regime Alignment",
                    "points": points,
                    "type": "context",
                }
            )

    if "Compliance Risk" in detected_domains:
        points = _graduated_tier_points(float(policy_score), _SCORE_AMPLIFIER_TIERS)
        if points > 0:
            total_points += points
            contextual_matches.append(
                {
                    "domain": "Compliance Risk",
                    "context": (
                        f"Policy score {policy_score:.1f} stresses compliance knowledge exposure"
                    ),
                    "matched_signals": sorted(affected_entities)[:4]
                    or ["policy-stressed compliance"],
                }
            )
            amplifiers.append(
                {
                    "label": "Policy-Stressed Compliance Exposure",
                    "points": points,
                    "type": "context",
                }
            )

    if "Liquidity Risk" in detected_domains:
        points = _graduated_tier_points(volatility_severity, _SCORE_AMPLIFIER_TIERS)
        if points > 0:
            total_points += points
            contextual_matches.append(
                {
                    "domain": "Liquidity Risk",
                    "context": (
                        f"Market volatility severity {volatility_severity:.0f} "
                        "amplifies liquidity knowledge exposure"
                    ),
                    "matched_signals": sorted(affected_entities)[:4] or ["elevated volatility"],
                }
            )
            amplifiers.append(
                {
                    "label": "Volatility-Stressed Liquidity Exposure",
                    "points": points,
                    "type": "context",
                }
            )

    return contextual_matches, amplifiers, total_points
def _format_knowledge_risk_explanation(knowledge: dict[str, Any], score: float) -> str:
    """Human-readable knowledge-driven risk reasoning breakdown."""
    sources = knowledge.get("retrieved_knowledge_sources", [])
    domain_evidence = knowledge.get("domain_evidence", [])
    mitigation_evidence = knowledge.get("mitigation_evidence", [])
    contextual_matches = knowledge.get("contextual_matches", [])
    amplifiers = knowledge.get("risk_amplifiers", [])
    contributions = knowledge.get("weighted_contributions", knowledge.get("risk_contributions", []))

    lines = [
        f"**Knowledge Risk Score:** {score:.0f}",
        "",
        "**Retrieved Knowledge Sources:**",
    ]
    if sources:
        for source in sources:
            weight = source.get("weight", 1.0)
            business_type = source.get("business_type", "")
            suffix = f" · {business_type}" if business_type else ""
            lines.append(
                f"- {source.get('title', source.get('id', 'Unknown'))}{suffix} "
                f"(weight {float(weight):.2f})"
            )
    else:
        lines.append("- None retrieved for current query")

    lines.append("")
    lines.append("**Risk Domains Inferred:**")
    if domain_evidence:
        for item in domain_evidence:
            lines.append(f"- **{item.get('domain', 'Risk')}**")
            signals = item.get("signals", [])
            if signals:
                lines.append("  - Signals:")
                for signal in signals[:5]:
                    lines.append(f"    - {signal}")
            reason = item.get("reason", "")
            if reason:
                lines.append(f"  - Reason: {reason}")
            confidence = item.get("confidence_factor")
            contribution = item.get("contribution")
            if confidence is not None or contribution is not None:
                lines.append(
                    f"  - Contribution: +{float(contribution or 0.0):.1f} "
                    f"(confidence {float(confidence or 0.0):.2f})"
                )
    else:
        lines.append("- None inferred from retrieved knowledge metadata")

    lines.append("")
    lines.append("**Mitigation Frameworks Identified:**")
    if mitigation_evidence:
        for item in mitigation_evidence:
            lines.append(f"- **{item.get('framework', 'Mitigation')}**")
            reason = item.get("reason", "")
            if reason:
                lines.append(f"  - Reason: {reason}")
            contribution = item.get("contribution")
            if contribution is not None:
                lines.append(f"  - Contribution: -{float(contribution):.1f} risk points")
    else:
        lines.append("- None identified from retrieved knowledge metadata")

    lines.append("")
    lines.append("**Contextual Amplifiers:**")
    if amplifiers:
        for item in amplifiers:
            lines.append(f"- +{item.get('points', 0)} {item.get('label', 'Context Amplifier')}")
    elif contextual_matches:
        for match in contextual_matches:
            lines.append(f"- {match.get('domain', 'Risk')}: {match.get('context', '')}")
    else:
        lines.append("- No contextual amplification under current market/macro/policy state")

    lines.append("")
    lines.append("**Weighted Contributions:**")
    if contributions:
        for item in contributions:
            pts = float(item.get("points", 0.0))
            sign = "+" if pts >= 0 else ""
            weight = item.get("weight")
            confidence = item.get("confidence_factor")
            detail_parts: list[str] = []
            if weight is not None:
                detail_parts.append(f"retrieval {float(weight):.2f}")
            if confidence is not None:
                detail_parts.append(f"confidence {float(confidence):.2f}")
            detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
            lines.append(
                f"{sign}{pts:.1f} {item.get('label', 'Unknown')}{detail_suffix}"
            )
    else:
        lines.append("+0.0 No active weighted exposure")

    lines.extend(
        [
            "",
            f"**Final Knowledge Score:** {score:.0f}",
        ]
    )
    return "\n".join(lines)
def _risk_rating_from_score(score: float) -> str:
    if score <= 33:
        return "LOW"
    if score <= 66:
        return "MEDIUM"
    return "HIGH"
def _hedge_ratio_from_final_score(
    score: float,
    mapping: dict[str, Any] | None = None,
) -> float:
    from risk_constants import hedge_ratio_from_score

    return hedge_ratio_from_score(score, mapping)
def _normalize_component_score(raw_points: float, max_points: float) -> float:
    if max_points <= 0:
        return 0.0
    return round(min(100.0, max(0.0, (raw_points / max_points) * 100.0)), 1)
def compute_market_risk_score(
    metrics: dict[str, Any],
    market_factor_score: float,
) -> dict[str, Any]:
    """Scale the legacy 0–1 technical factor score to a 0–100 market risk score."""
    market_score = round(min(100.0, max(0.0, float(market_factor_score) * 100.0)), 1)
    spot = float(metrics.get("spot", 0.0))
    ma5 = float(metrics.get("ma5", spot))
    ma30 = float(metrics.get("ma30", spot))
    boll_upper = float(metrics.get("bollinger_upper", spot))
    rsi = float(metrics.get("net_payout_rsi", 50.0))
    macd_hist = float(metrics.get("net_payout_macd_histogram", 0.0))

    return {
        "market_score": market_score,
        "components": {
            "factor_trend_ma5": int(metrics.get("factor_trend", 0)),
            "factor_bollinger": int(metrics.get("factor_bollinger", 0)),
            "factor_ma30": int(metrics.get("factor_macro", 0)),
            "spot_vs_ma5": "above" if spot >= ma5 else "below",
            "spot_vs_ma30": "above" if spot >= ma30 else "below",
            "spot_vs_bollinger_upper": "above" if spot >= boll_upper else "below",
            "rsi": round(rsi, 2),
            "macd_histogram": round(macd_hist, 6),
            "legacy_factor_score_0_1": round(float(market_factor_score), 3),
        },
    }
def compute_knowledge_risk_score(
    retrieved_documents: list[dict[str, Any]] | None = None,
    *,
    query: str = "",
    metrics: dict[str, Any] | None = None,
    macro_score: float = 0.0,
    policy_score: float = 0.0,
    tnx_proxy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Knowledge-driven risk reasoning: structured case metadata drives domain and
    mitigation inference, evidence density sets confidence, and graduated context
    amplifiers adjust the final explainable knowledge score.
    """
    docs = retrieved_documents or []
    desk_metrics = metrics or {}
    corpus = _build_knowledge_corpus(query, docs)

    (
        domain_totals,
        mitigation_totals,
        detected_domain_names,
        detected_mitigation_names,
        domain_contributions,
        mitigation_contributions,
        domain_evidence,
        mitigation_evidence,
    ) = _aggregate_semantic_knowledge_risk(docs, corpus=corpus)

    contextual_matches, risk_amplifiers, contextual_points = _compute_knowledge_context_amplifiers(
        retrieved_docs=docs,
        detected_domains=set(detected_domain_names),
        metrics=desk_metrics,
        macro_score=float(macro_score),
        policy_score=float(policy_score),
        tnx_proxy=tnx_proxy,
    )

    domain_points = round(sum(domain_totals.values()), 2)
    mitigation_points = round(sum(mitigation_totals.values()), 2)
    raw_points = domain_points + contextual_points - mitigation_points
    knowledge_score = round(min(100.0, max(0.0, float(raw_points))), 1)

    weighted_contributions: list[dict[str, Any]] = list(domain_contributions)
    weighted_contributions.extend(
        {
            "label": item["label"],
            "points": item["points"],
            "type": "context",
        }
        for item in risk_amplifiers
    )
    weighted_contributions.extend(
        {
            "label": item["label"],
            "points": -item["points"],
            "type": "mitigation",
            "weight": item.get("weight"),
            "base_points": item.get("base_points"),
            "source_id": item.get("source_id"),
            "source_title": item.get("source_title"),
        }
        for item in mitigation_contributions
    )

    risk_contributions: list[dict[str, Any]] = [
        {
            "label": name,
            "points": round(domain_totals[name], 1),
            "type": "domain",
        }
        for name in detected_domain_names
    ]
    risk_contributions.extend(
        {"label": item["label"], "points": item["points"], "type": "context"}
        for item in risk_amplifiers
    )
    risk_contributions.extend(
        {
            "label": name,
            "points": -round(mitigation_totals[name], 1),
            "type": "mitigation",
        }
        for name in detected_mitigation_names
    )

    retrieved_knowledge_sources = [
        {
            "id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "domain": doc.get("domain", ""),
            "business_type": _doc_case_metadata(doc).get("business_type", ""),
            "weight": round(_doc_retrieval_weight(doc), 3),
            "score": doc.get("score"),
            "similarity": doc.get("similarity"),
        }
        for doc in docs
    ]

    explanation_payload = {
        "retrieved_knowledge_sources": retrieved_knowledge_sources,
        "detected_risk_domains": detected_domain_names,
        "detected_mitigation_frameworks": detected_mitigation_names,
        "domain_evidence": domain_evidence,
        "mitigation_evidence": mitigation_evidence,
        "contextual_matches": contextual_matches,
        "risk_amplifiers": risk_amplifiers,
        "weighted_contributions": weighted_contributions,
        "risk_contributions": risk_contributions,
    }

    components = {
        **explanation_payload,
        "domain_points": domain_points,
        "contextual_points": round(contextual_points, 2),
        "mitigation_points": mitigation_points,
        "raw_points": raw_points,
        "market_trend": _infer_base_ccy_trend(desk_metrics, tnx_proxy),
        "market_volatility_elevated": _market_volatility_elevated(desk_metrics),
        "macro_score_context": round(float(macro_score), 1),
        "policy_score_context": round(float(policy_score), 1),
        "documents_scanned": len(docs),
        "supporting_playbooks": [
            {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "domain": doc.get("domain", ""),
            }
            for doc in docs
        ],
        "explanation": _format_knowledge_risk_explanation(explanation_payload, knowledge_score),
    }

    return {
        "rule_based_knowledge_score": knowledge_score,
        "knowledge_score": knowledge_score,
        "knowledge_risk_level": _risk_rating_from_score(knowledge_score),
        "components": components,
    }
def compute_environment_risk_assessment(
    metrics: dict[str, Any],
    *,
    market_factor_score: float,
    imf_indicators: dict[str, Any] | None = None,
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Market + macro + policy layers only — no enterprise knowledge composite."""
    pair = metrics.get("pair", DEFAULT_PAIR)
    spot = float(metrics.get("spot", 0.0))

    market = compute_market_risk_score(metrics, market_factor_score)
    macro = compute_macro_risk_score(pair, imf_indicators)
    policy = compute_policy_risk_score(spot, china_money, tnx_proxy)

    market_score = float(market["market_score"])
    macro_score = float(macro["macro_score"])
    policy_score = float(policy["policy_score"])

    contributions = {
        "market": round(market_score * NOVA_RISK_WEIGHTS["market"], 1),
        "macro": round(macro_score * NOVA_RISK_WEIGHTS["macro"], 1),
        "policy": round(policy_score * NOVA_RISK_WEIGHTS["policy"], 1),
        "knowledge": 0.0,
    }

    return {
        "market_score": market_score,
        "macro_score": macro_score,
        "policy_score": policy_score,
        "knowledge_score": None,
        "final_risk_score": None,
        "multi_factor_risk_rating": None,
        "recommended_hedge_ratio": None,
        "enterprise_assessment_ready": False,
        "knowledge_risk_level": None,
        "macro_risk_level": macro["macro_risk_level"],
        "policy_risk_level": policy["policy_risk_level"],
        "market_risk_level": _risk_rating_from_score(market_score),
        "risk_weights": dict(NOVA_RISK_WEIGHTS),
        "risk_contributions": contributions,
        "market_breakdown": market["components"],
        "macro_breakdown": macro["components"],
        "policy_breakdown": policy["components"],
        "knowledge_breakdown": {},
        "risk_score_explanation": "",
    }
def compute_nova_multidimensional_risk_assessment(
    metrics: dict[str, Any],
    *,
    market_factor_score: float,
    imf_indicators: dict[str, Any] | None = None,
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
    retrieved_documents: list[dict[str, Any]] | None = None,
    knowledge_query: str = "",
    api_key: str | None = None,
    budget_gate: Any = None,
) -> dict[str, Any]:
    """Aggregate market, macro, policy, and knowledge scores into final desk risk."""
    from knowledge_llm_intelligence import enhance_knowledge_score_with_llm

    pair = metrics.get("pair", DEFAULT_PAIR)
    spot = float(metrics.get("spot", 0.0))

    market = compute_market_risk_score(metrics, market_factor_score)
    macro = compute_macro_risk_score(pair, imf_indicators)
    policy = compute_policy_risk_score(spot, china_money, tnx_proxy)
    knowledge = compute_knowledge_risk_score(
        retrieved_documents,
        query=knowledge_query,
        metrics=metrics,
        macro_score=float(macro["macro_score"]),
        policy_score=float(policy["policy_score"]),
        tnx_proxy=tnx_proxy,
    )

    market_score = float(market["market_score"])
    macro_score = float(macro["macro_score"])
    policy_score = float(policy["policy_score"])
    rule_based_knowledge_score = float(
        knowledge.get("rule_based_knowledge_score", knowledge["knowledge_score"])
    )

    market_context = {
        **market["components"],
        "market_score": market_score,
        "spot": metrics.get("spot"),
        "rsi": metrics.get("net_payout_rsi"),
        "macd_histogram": metrics.get("net_payout_macd_histogram"),
    }
    macro_context = {**macro["components"], "macro_score": macro_score}
    policy_context = {**policy["components"], "policy_score": policy_score}

    llm_enhancement = enhance_knowledge_score_with_llm(
        rule_based_knowledge_score,
        retrieved_documents or [],
        pair,
        market_context,
        macro_context,
        policy_context,
        enterprise_scenario=knowledge_query,
        api_key=api_key,
        budget_gate=budget_gate,
    )
    knowledge_score = float(llm_enhancement["final_knowledge_score"])
    ai_knowledge_assessment = llm_enhancement.get("ai_knowledge_assessment")
    case_interpretations = llm_enhancement.get("case_interpretations") or []

    final_risk_score = round(
        market_score * NOVA_RISK_WEIGHTS["market"]
        + macro_score * NOVA_RISK_WEIGHTS["macro"]
        + policy_score * NOVA_RISK_WEIGHTS["policy"]
        + knowledge_score * NOVA_RISK_WEIGHTS["knowledge"],
        1,
    )
    final_risk_rating = _risk_rating_from_score(final_risk_score)
    recommended_hedge_ratio = _hedge_ratio_from_final_score(final_risk_score)

    contributions = {
        "market": round(market_score * NOVA_RISK_WEIGHTS["market"], 1),
        "macro": round(macro_score * NOVA_RISK_WEIGHTS["macro"], 1),
        "policy": round(policy_score * NOVA_RISK_WEIGHTS["policy"], 1),
        "knowledge": round(knowledge_score * NOVA_RISK_WEIGHTS["knowledge"], 1),
    }

    knowledge_breakdown = {
        **knowledge["components"],
        "rule_based_knowledge_score": rule_based_knowledge_score,
        "final_knowledge_score": knowledge_score,
        "llm_knowledge_adjustment": llm_enhancement.get("llm_adjustment_applied", 0.0),
        "ai_knowledge_assessment": ai_knowledge_assessment,
        "case_interpretations": case_interpretations,
        "documents_reranked": llm_enhancement.get("documents_reranked", False),
    }

    return {
        "market_score": market_score,
        "macro_score": macro_score,
        "policy_score": policy_score,
        "rule_based_knowledge_score": rule_based_knowledge_score,
        "knowledge_score": knowledge_score,
        "llm_knowledge_adjustment": llm_enhancement.get("llm_adjustment_applied", 0.0),
        "ai_knowledge_assessment": ai_knowledge_assessment,
        "case_interpretations": case_interpretations,
        "documents_reranked": llm_enhancement.get("documents_reranked", False),
        "final_risk_score": final_risk_score,
        "multi_factor_risk_rating": final_risk_rating,
        "recommended_hedge_ratio": recommended_hedge_ratio,
        "enterprise_assessment_ready": True,
        "knowledge_risk_level": _risk_rating_from_score(knowledge_score),
        "rule_based_knowledge_risk_level": knowledge["knowledge_risk_level"],
        "macro_risk_level": macro["macro_risk_level"],
        "policy_risk_level": policy["policy_risk_level"],
        "market_risk_level": _risk_rating_from_score(market_score),
        "risk_weights": dict(NOVA_RISK_WEIGHTS),
        "risk_contributions": contributions,
        "market_breakdown": market["components"],
        "macro_breakdown": macro["components"],
        "policy_breakdown": policy["components"],
        "knowledge_breakdown": knowledge_breakdown,
    }
def build_risk_score_explanation(metrics: dict[str, Any]) -> str:
    """Transparent explainability block for treasury decision support."""
    from .recommendation import _resolved_hedge_ratio

    pair = metrics.get("pair", DEFAULT_PAIR)
    market = metrics.get("market_breakdown", {})
    macro = metrics.get("macro_breakdown", {})
    policy = metrics.get("policy_breakdown", {})
    knowledge = metrics.get("knowledge_breakdown", {})
    contributions = metrics.get("risk_contributions", {})

    lines = [
        "**Why this score was generated:**",
        "",
        f"**Market factors ({NOVA_RISK_WEIGHTS['market']:.0%} weight · "
        f"{contributions.get('market', metrics.get('market_score', 0)):.1f} pts contributed):** "
        f"Legacy technical score {market.get('legacy_factor_score_0_1', 'N/A')} "
        f"→ market score **{metrics.get('market_score', 0):.1f}**. "
        f"Spot is {market.get('spot_vs_ma5', 'N/A')} MA5 and {market.get('spot_vs_ma30', 'N/A')} MA30; "
        f"RSI **{market.get('rsi', 'N/A')}**, MACD hist **{market.get('macd_histogram', 'N/A')}**.",
        "",
        f"**Macro factors ({NOVA_RISK_WEIGHTS['macro']:.0%} weight · "
        f"{contributions.get('macro', metrics.get('macro_score', 0)):.1f} pts contributed):** "
        f"IMF CPI spread **{macro.get('inflation_spread_pp', 'N/A')}pp** "
        f"(+{macro.get('inflation_points', 0)} pts), GDP spread "
        f"**{macro.get('gdp_spread_pp', 'N/A')}pp** (+{macro.get('gdp_points', 0)} pts), "
        f"{macro.get('reserve_currency', 'USD')} COFER share "
        f"**{macro.get('reserve_share_pct', 'N/A')}%** (+{macro.get('cofer_points', 0)} pts).",
        "",
        f"**Policy factors ({NOVA_RISK_WEIGHTS['policy']:.0%} weight · "
        f"{contributions.get('policy', metrics.get('policy_score', 0)):.1f} pts contributed):** "
        f"FedWatch state **{policy.get('fedwatch_state', 'neutral')}** "
        f"(+{policy.get('fedwatch_points', 0)} pts); CFETS deviation "
        f"**{policy.get('cfets_deviation_bps', 0):.0f} bps** "
        f"(+{policy.get('cfets_points', 0)} pts).",
        "",
        f"**Knowledge factors ({NOVA_RISK_WEIGHTS['knowledge']:.0%} weight · "
        f"{contributions.get('knowledge', metrics.get('knowledge_score', 0)):.1f} pts contributed):** "
        f"Risk-domain exposure model across retrieved treasury playbooks "
        f"(+{knowledge.get('domain_points', 0)} domain · "
        f"+{knowledge.get('contextual_points', 0)} contextual · "
        f"-{knowledge.get('mitigation_points', 0)} mitigation).",
        "",
        knowledge.get(
            "explanation",
            _format_knowledge_risk_explanation(
                knowledge,
                float(
                    knowledge.get(
                        "rule_based_knowledge_score",
                        metrics.get("rule_based_knowledge_score", metrics.get("knowledge_score", 0.0)),
                    )
                ),
            ),
        ),
    ]

    ai_assessment = metrics.get("ai_knowledge_assessment") or knowledge.get("ai_knowledge_assessment")
    rule_kb = metrics.get("rule_based_knowledge_score")
    llm_adj = metrics.get("llm_knowledge_adjustment")
    if ai_assessment and rule_kb is not None:
        lines.extend(
            [
                "",
                "**AI Knowledge Assessment (advisory adjustment layer):**",
                f"- Rule-based knowledge score: **{float(rule_kb):.1f}**",
                f"- AI adjustment applied: **{ai_assessment.get('adjustment_display', llm_adj)}** "
                f"(confidence **{ai_assessment.get('confidence_pct', 'N/A')}**)",
                f"- Final knowledge score used in composite: **{float(metrics.get('knowledge_score', 0)):.1f}**",
            ]
        )
        for reason in (ai_assessment.get("reasoning") or [])[:5]:
            lines.append(f"  - {reason}")

    lines.extend(
        [
        "",
        f"**Final composite:** **{metrics.get('final_risk_score', 0):.1f} / 100** → "
        f"**{metrics.get('multi_factor_risk_rating', 'MEDIUM')}** on **{pair}** "
        f"with recommended hedge **{_resolved_hedge_ratio(metrics):.0%}**.",
        ]
    )
    return "\n".join(lines)
def apply_enterprise_risk_assessment(
    metrics: dict[str, Any],
    enterprise_query: str,
    *,
    retrieved_documents: list[dict[str, Any]] | None = None,
    imf_indicators: dict[str, Any] | None = None,
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
    api_key: str | None = None,
    budget_gate: Any = None,
) -> dict[str, Any]:
    """Layer full enterprise knowledge composite onto existing environment metrics."""
    from knowledge_llm_intelligence import enrich_post_assessment_intelligence

    market_factor_score = float(metrics.get("multi_factor_score", 0.0))
    pair = metrics.get("pair", DEFAULT_PAIR)
    docs = retrieved_documents
    if docs is None:
        docs = retrieve_knowledge_with_llm_rerank(
            enterprise_query,
            pair=pair,
            api_key=api_key,
            budget_gate=budget_gate,
            top_k=3,
            rerank_pool=10,
        )

    risk_assessment = compute_nova_multidimensional_risk_assessment(
        metrics,
        market_factor_score=market_factor_score,
        imf_indicators=imf_indicators,
        china_money=china_money,
        tnx_proxy=tnx_proxy,
        retrieved_documents=docs,
        knowledge_query=enterprise_query,
        api_key=api_key,
        budget_gate=budget_gate,
    )
    merged = {
        **metrics,
        **risk_assessment,
        "enterprise_assessment_ready": True,
        "enterprise_scenario": enterprise_query,
    }
    post_intel = enrich_post_assessment_intelligence(
        merged,
        docs,
        api_key=api_key,
        budget_gate=budget_gate,
    )
    merged.update(post_intel)
    merged["risk_score_explanation"] = build_risk_score_explanation(merged)
    return merged
