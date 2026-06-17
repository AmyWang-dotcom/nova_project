"""Vector retrieval and RAG indexing."""
from __future__ import annotations

import json
import threading
from typing import Any

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment,misc]
    FAISS_AVAILABLE = False

try:
    import streamlit as st
except ImportError:  # pragma: no cover — optional when backend imported outside Streamlit
    st = None  # type: ignore[assignment]

import numpy as np

from .constants import DEFAULT_PAIR
from .knowledge_base import (
    _FX_KNOWLEDGE_BASE,
    _PAIR_CURRENCY_TOKEN,
    _doc_macro_indicators_text,
    format_imf_macro_indicators_block,
    get_imf_macro_knowledge_entry,
)
from .utils import _serialize_structural_doc_payload, _tokenize, sanitize_user_query

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment,misc]
    FAISS_AVAILABLE = False
try:
    import streamlit as st
except ImportError:  # pragma: no cover — optional when backend imported outside Streamlit
    st = None  # type: ignore[assignment]
EMBED_DIM = 384
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_vector_index: Any = None
_indexed_docs_corpus: list[dict[str, str]] = []
_embedding_matrix: np.ndarray | None = None
_rag_lock = threading.RLock()
def retrieve_knowledge_with_llm_rerank(
    query: str,
    pair: str = DEFAULT_PAIR,
    *,
    api_key: str | None = None,
    budget_gate: Any = None,
    top_k: int = 3,
    rerank_pool: int = 10,
) -> list[dict[str, Any]]:
    """Vector retrieve top-N, optionally GPT-rerank to top-K; fallback to vector order."""
    pool = _simple_semantic_search(query, pair=pair, top_k=rerank_pool)
    if not pool:
        return []
    key = str(api_key or "").strip()
    if not key:
        return pool[:top_k]
    try:
        from knowledge_llm_intelligence import rerank_retrieved_documents

        return rerank_retrieved_documents(
            pool,
            query,
            pair,
            api_key=key,
            budget_gate=budget_gate,
            top_k=top_k,
        )
    except Exception:
        return pool[:top_k]
def _english_retrieval_anchor_text(doc: dict[str, Any]) -> str:
    """English-only dense retrieval anchor for MiniLM FAISS indexing."""
    blocks = [
        str(doc.get("domain", "") or "").strip(),
        str(doc.get("title", "") or "").strip(),
        str(doc.get("professional_definition", "") or "").strip(),
        str(doc.get("actionable_playbook", "") or "").strip(),
        str(doc.get("vector_description", "") or "").strip(),
    ]
    for key in (
        "core_anchors",
        "no_arbitrage_condition",
        "mathematical_representation",
        "nova_imf_macro_reasoning_rules",
    ):
        payload = doc.get(key)
        if payload is None:
            continue
        if isinstance(payload, (dict, list)):
            blocks.append(json.dumps(payload, ensure_ascii=False))
        else:
            blocks.append(str(payload).strip())
    macro = _doc_macro_indicators_text(doc)
    if macro:
        blocks.append(macro)
    tags = doc.get("tags")
    if tags:
        blocks.append(str(tags))
    return " || ".join(b for b in blocks if b)
def _load_embedding_model_impl():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)
if st is not None:

    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        return _load_embedding_model_impl()

else:

    def load_embedding_model():
        return _load_embedding_model_impl()
def _indexed_vector_count() -> int:
    """Active row count in the FAISS index or NumPy embedding fallback."""
    if FAISS_AVAILABLE and _vector_index is not None:
        return int(_vector_index.ntotal)
    if _embedding_matrix is not None:
        return int(_embedding_matrix.shape[0])
    return 0
def _vector_corpus_stale(knowledge_base: list[dict[str, Any]], force_rebuild: bool) -> bool:
    """True when corpus length or index rows diverge from the live knowledge base."""
    if force_rebuild:
        return True
    kb_len = len(knowledge_base)
    if kb_len == 0:
        return False
    if len(_indexed_docs_corpus) != kb_len:
        return True
    if _indexed_vector_count() != kb_len:
        return True
    return False
def init_vector_index(
    force_rebuild: bool = False,
    knowledge_base: list[dict[str, Any]] | None = None,
) -> None:
    """
    Build or refresh a local FAISS IndexFlatIP over L2-normalized MiniLM embeddings.
    Automatically re-indexes when _FX_KNOWLEDGE_BASE grows or _indexed_docs_corpus drifts.
    """
    global _vector_index, _indexed_docs_corpus, _embedding_matrix

    with _rag_lock:
        kb = knowledge_base if knowledge_base is not None else _FX_KNOWLEDGE_BASE
        if not _vector_corpus_stale(kb, force_rebuild) and _vector_index_ready():
            return

        _indexed_docs_corpus = list(kb)
        _vector_index = None
        _embedding_matrix = None

        if not kb:
            return

        try:
            texts = [_english_retrieval_anchor_text(doc) for doc in kb]
            model = load_embedding_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(embeddings)
            else:
                _l2_normalize_inplace(embeddings)

            if FAISS_AVAILABLE:
                index = faiss.IndexFlatIP(EMBED_DIM)
                index.add(embeddings)
                _vector_index = index
            else:
                _embedding_matrix = embeddings

        except Exception:
            _vector_index = None
            _embedding_matrix = None
            _indexed_docs_corpus = list(kb)
def initialize_vector_db(knowledge_base: list[dict[str, Any]]) -> None:
    """Backward-compatible alias — always syncs corpus size with the knowledge base."""
    init_vector_index(force_rebuild=True, knowledge_base=knowledge_base)
def _l2_normalize_inplace(vectors: np.ndarray) -> None:
    """NumPy L2 normalize (FAISS-free fallback)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
def _dense_vector_scores(query_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity scores vs all indexed docs (query_vec must be L2-normalized)."""
    with _rag_lock:
        if FAISS_AVAILABLE and _vector_index is not None and _vector_index.ntotal > 0:
            scores, _ = _vector_index.search(query_vec, _vector_index.ntotal)
            return scores[0]

        if _embedding_matrix is not None and _embedding_matrix.size > 0:
            return (_embedding_matrix @ query_vec[0]).astype(np.float32)

    return np.array([], dtype=np.float32)
def format_data_sources_block(doc: dict[str, Any]) -> str:
    """
    Format optional `data_sources` for LLM context / UI.
    Returns empty string when the key is absent or empty — never raises KeyError.
    """
    sources = doc.get("data_sources")
    if not sources or not isinstance(sources, dict):
        return ""

    lines: list[str] = []
    for label, url in sources.items():
        if not url or not isinstance(url, str):
            continue
        human_label = str(label).replace("_", " ").strip().title()
        lines.append(f"  * {human_label}: {url}")

    if not lines:
        return ""

    return "**Authoritative data sources:**\n" + "\n".join(lines)
def _data_sources_corpus_text(doc: dict[str, Any]) -> str:
    """Flatten data_sources labels/URLs for sparse + dense retrieval."""
    sources = doc.get("data_sources")
    if not sources or not isinstance(sources, dict):
        return ""
    parts = [f"{k} {v}" for k, v in sources.items() if v]
    return "official references " + " ".join(parts) if parts else ""
def _doc_corpus(doc: dict[str, Any]) -> str:
    structural_parts: list[str] = []
    for key in (
        "core_anchors",
        "no_arbitrage_condition",
        "mathematical_representation",
        "nova_imf_macro_reasoning_rules",
        "vector_description",
    ):
        payload = doc.get(key)
        if payload is None:
            continue
        if isinstance(payload, (dict, list)):
            structural_parts.append(json.dumps(payload, ensure_ascii=False))
        else:
            structural_parts.append(str(payload))
    return " ".join(
        p
        for p in (
            doc.get("title", ""),
            doc.get("domain", ""),
            doc.get("professional_definition", ""),
            doc.get("actionable_playbook", ""),
            doc.get("tags", ""),
            doc.get("pairs", ""),
            _doc_macro_indicators_text(doc),
            _data_sources_corpus_text(doc),
            *structural_parts,
        )
        if p
    )
def _format_doc_for_context(doc: dict[str, Any]) -> str:
    """Single playbook block for RAG / LLM context assembly (English-only telemetry)."""
    domain = doc.get("domain", "FX Risk")
    title = doc.get("title", "Playbook")
    parts = [
        f"**[{domain}] {title}** ({doc.get('pairs', 'ALL')})",
        f"**Professional definition:** {doc.get('professional_definition', '')}",
    ]
    playbook = doc.get("actionable_playbook", "")
    if playbook:
        parts.append(f"**Actionable playbook:** {playbook}")
    vector_desc = doc.get("vector_description", "")
    if vector_desc:
        parts.append(f"**Vector description:** {vector_desc}")

    for key, label in (
        ("core_anchors", "Core anchors"),
        ("no_arbitrage_condition", "No-arbitrage condition"),
        ("mathematical_representation", "Mathematical representation"),
        ("nova_imf_macro_reasoning_rules", "NOVA IMF macro reasoning rules"),
    ):
        block = _serialize_structural_doc_payload(label, doc.get(key))
        if block:
            parts.append(block)

    body = "\n\n".join(p for p in parts if p and str(p).strip())
    imf_block = format_imf_macro_indicators_block(doc.get("macro_indicators"))
    if imf_block:
        body += f"\n\n**IMF quantitative baseline (structured):**\n{imf_block}"

    sources_block = format_data_sources_block(doc)
    if sources_block:
        body += f"\n\n{sources_block}"
    return body
def assemble_retrieved_playbooks_context(
    docs: list[dict[str, Any]] | None,
    *,
    header: str = "[RETRIEVED CORPORATE FX PLAYBOOKS — RAG]",
) -> str:
    """Concatenate matched KB articles for prompt injection; skips empty retrieval."""
    if not docs:
        return ""

    blocks = [_format_doc_for_context(doc) for doc in docs if doc]
    blocks = [b for b in blocks if b.strip()]
    if not blocks:
        return ""

    return f"{header}\n\n" + "\n\n---\n\n".join(blocks)
def _doc_matches_pair(doc: dict[str, str], pair: str) -> bool:
    applicable = doc.get("pairs", "")
    return pair in applicable or applicable == "ALL"
def _vector_index_ready() -> bool:
    with _rag_lock:
        kb_len = len(_FX_KNOWLEDGE_BASE)
        if kb_len == 0:
            return False
        if len(_indexed_docs_corpus) != kb_len:
            return False
        if FAISS_AVAILABLE and _vector_index is not None:
            return _vector_index.ntotal == kb_len
        return (
            _embedding_matrix is not None
            and _embedding_matrix.shape[0] == kb_len
            and bool(_indexed_docs_corpus)
        )
def _enrich_retrieved_doc(doc: dict[str, Any], score: float) -> dict[str, Any]:
    """Attach retrieval scoring fields used by the semantic knowledge risk engine."""
    weight = min(1.0, max(0.0, float(score)))
    return {
        **doc,
        "score": round(float(score), 4),
        "similarity": round(weight, 4),
    }
def _repository_fallback(pair: str, top_k: int) -> list[dict[str, Any]]:
    """Safe slice when FAISS index is unavailable or returns no pair-matched hits."""
    fallback = [d for d in _FX_KNOWLEDGE_BASE if _doc_matches_pair(d, pair)]
    selected = (fallback or _FX_KNOWLEDGE_BASE)[:top_k]
    return [
        _enrich_retrieved_doc(doc, max(0.35, 0.85 - index * 0.15))
        for index, doc in enumerate(selected)
    ]
def _compute_sparse_keyword_score(
    query: str,
    q_tokens: set[str],
    doc: dict[str, str],
    pair: str,
) -> float:
    """Sparse track: token overlap, tag hits, domain boosts, currency precision."""
    q_lower = query.lower()
    score = 0.0
    doc_tokens = _tokenize(_doc_corpus(doc))
    tags = doc.get("tags", "")

    overlap = q_tokens & doc_tokens
    score += len(overlap) * 2.0

    for tok in q_tokens:
        if tok in tags:
            score += 3.0

    domain_boosts: dict[str, list[str]] = {
        "hedge": ["forward_locking", "cross_border_case_amazon", "eur_vat_remittance"],
        "forward": ["forward_locking", "policy_compliance"],
        "locking": ["forward_locking", "policy_compliance"],
        "macro": ["q2_2026_macro_usd", "eur_macro_q2"],
        "q2": ["q2_2026_macro_usd", "eur_macro_q2"],
        "amazon": ["cross_border_case_amazon"],
        "tiktok": ["cross_border_case_tiktok"],
        "vat": ["eur_vat_remittance"],
        "euro": ["eur_vat_remittance", "eur_macro_q2"],
        "eur": ["eur_vat_remittance", "eur_macro_q2"],
        "hkd": ["hkd_peg_defense", "hkd_treasury_case"],
        "peg": ["hkd_peg_defense"],
        "depreciation": ["hkd_peg_defense", "hkd_treasury_case"],
        "loss": ["forward_locking", "cross_border_case_amazon", "hkd_treasury_case"],
        "bleeding": ["forward_locking", "cross_border_case_amazon", "hkd_treasury_case"],
    }
    pair_doc_boost = {
        "USD/CNH": ["q2_2026_macro_usd", "cross_border_case_amazon"],
        "EUR/CNH": ["eur_vat_remittance", "eur_macro_q2"],
        "HKD/CNH": ["hkd_peg_defense", "hkd_treasury_case"],
        "GBP/CNH": ["net_payout_rolling_hedge_eu_vat", "high_frequency_micro_payouts"],
    }

    for kw, ids in domain_boosts.items():
        if kw in q_tokens and doc["id"] in ids:
            score += 4.0

    if doc["id"] in pair_doc_boost.get(pair, []):
        score += 2.5

    pair_ccy = _PAIR_CURRENCY_TOKEN.get(pair, "")
    if pair_ccy and pair_ccy in q_tokens:
        if pair_ccy in tags or pair_ccy in doc_tokens:
            score += 2.0

    if "amazon" in q_lower and doc["id"] == "cross_border_case_amazon":
        score += 3.0
    if any(p in q_lower for p in ("losing money", "bleed", "bleeding", "stop bleeding")):
        if doc["id"] in ("forward_locking", "cross_border_case_amazon", "hkd_treasury_case"):
            score += 5.0
    if pair == "EUR/CNH" and any(w in q_lower for w in ("vat", "euro", "europe")):
        if doc["id"] == "eur_vat_remittance":
            score += 6.0
    if pair == "GBP/CNH" and any(w in q_lower for w in ("vat", "uk", "gbp", "british", "pound", "amazon")):
        if doc["id"] == "net_payout_rolling_hedge_eu_vat":
            score += 6.0
    if pair == "HKD/CNH" and any(w in q_lower for w in ("peg", "hkd", "hong kong", "depreciation")):
        if doc["id"] == "hkd_peg_defense":
            score += 6.0
    if any(z in query for z in ("锁汇", "远期", "结售汇")) and doc["id"] == "forward_locking":
        score += 6.0
    if any(z in query for z in ("亏损", "对冲")) and doc["id"] in (
        "forward_locking",
        "cross_border_case_amazon",
    ):
        score += 4.0

    return score
def _ensure_vector_index() -> None:
    """Retry or resize index when corpus and knowledge base lengths diverge."""
    if _FX_KNOWLEDGE_BASE:
        init_vector_index(force_rebuild=not _vector_index_ready())
def _simple_semantic_search(
    query: str,
    pair: str = DEFAULT_PAIR,
    top_k: int = 3,
) -> list[dict[str, str]]:
    """
    Hybrid retriever: dense FAISS cosine (IndexFlatIP + L2 norm) +
    sparse keyword overlap fusion.
    final_score = vector_score + (keyword_score * 0.15)
    """
    query = sanitize_user_query(query)
    if not query:
        return _repository_fallback(pair, top_k)

    _ensure_vector_index()
    if not _vector_index_ready():
        return _repository_fallback(pair, top_k)

    try:
        model = load_embedding_model()
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if FAISS_AVAILABLE:
            faiss.normalize_L2(q_emb)
        else:
            _l2_normalize_inplace(q_emb)

        vector_scores = _dense_vector_scores(q_emb)
        if vector_scores.size == 0:
            return _repository_fallback(pair, top_k)

        q_tokens = _tokenize(query)
        scored: list[tuple[float, dict[str, str]]] = []

        with _rag_lock:
            corpus_snapshot = list(_indexed_docs_corpus)

        for idx, vector_score in enumerate(vector_scores):
            if idx >= len(corpus_snapshot):
                break
            doc = corpus_snapshot[int(idx)]
            if not _doc_matches_pair(doc, pair):
                continue

            keyword_score = _compute_sparse_keyword_score(query, q_tokens, doc, pair)
            final_score = float(vector_score) + keyword_score * 0.15
            scored.append((final_score, doc))

        if not scored:
            return _repository_fallback(pair, top_k)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            _enrich_retrieved_doc(doc, final_score)
            for final_score, doc in scored[:top_k]
        ]
    except Exception:
        return _repository_fallback(pair, top_k)
def retrieve_knowledge(
    query: str,
    pair: str = DEFAULT_PAIR,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Semantic knowledge retrieval with scored documents for risk weighting."""
    return _simple_semantic_search(query, pair=pair, top_k=top_k)
