"""
NOVA FX Risk Copilot — Streamlit Executive Dashboard
Multi-pair: USD/CNH · EUR/CNH · HKD/CNH · GBP/CNH
Run: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
from typing import Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend import (
    CHINA_MACRO_ANCHOR,
    FAISS_AVAILABLE,
    OFFSHORE_QUOTE_CCY,
    ONSHORE_QUOTE_CCY,
    REQUESTS_AVAILABLE,
    SUPPORTED_PAIRS,
    advanced_quantitative_analysis,
    apply_enterprise_risk_assessment,
    audit_contract_file,
    compute_imf_ppp_spreads,
    fetch_fx_kline,
    fetch_real_market_data,
    format_data_sources_block,
    format_imf_macro_indicators_block,
    fx_rag_build_response_parts,
    generate_dynamic_agent_prompt,
    get_imf_macro_knowledge_entry,
    build_nova_risk_advisory,
    read_upload_document_text,
    fetch_china_money_central_parity,
    fetch_tnx_fedwatch_proxy,
    get_live_desk_multi_factor_metrics,
    retrieve_knowledge,
    resolve_30d_range_from_kline,
    sanitize_user_query,
    _MAX_USER_QUERY_LEN,
)
import importlib
import llm_service as llm_service_module

importlib.reload(llm_service_module)

LLMConfig = llm_service_module.LLMConfig
LLMService = llm_service_module.LLMService
run_llm_enhancement_bundle = llm_service_module.run_llm_enhancement_bundle
from nova_response import (
    build_contract_audit_llm_context,
    build_copilot_llm_context,
    build_risk_explanation_llm_context,
    generate_nova_response,
    is_ai_enhanced_mode,
)
import hedge_preference_ui as hedge_ui

importlib.reload(hedge_ui)

get_applied_risk_mapping = hedge_ui.get_applied_risk_mapping
render_hedge_recommendation_block = hedge_ui.render_hedge_recommendation_block
render_risk_preference_mapping_panel = hedge_ui.render_risk_preference_mapping_panel


def _reset_risk_preference_panel_guard() -> None:
    """Compatibility: older hot-reload copies may miss the guard function."""
    fn = getattr(hedge_ui, "reset_risk_preference_panel_guard", None)
    if callable(fn):
        fn()
from risk_constants import resolve_display_hedge_ratio
from treasury_copilot import (
    build_local_enterprise_intelligence,
    build_local_hedge_ratio_explanation,
    build_local_hedge_strategy_review,
    build_local_strategic_copilot_report,
    build_similar_knowledge_cases,
    format_copilot_item,
    normalize_knowledge_case_list,
    serialize_for_cache,
)

MAX_AI_CALLS_PER_SESSION = 60

DESK_QUOTE_LABEL = f"{OFFSHORE_QUOTE_CCY} (Offshore Yuan)"
MACRO_YUAN_LABEL = f"{ONSHORE_QUOTE_CCY} (Onshore Yuan Anchor)"

# UI-facing provider labels (endpoint/model resolution lives in llm_service.py)
LLM_PROVIDER_OPTIONS: dict[str, dict[str, str]] = {
    "openai": {"label": "OpenAI", "model": "gpt-4o-mini"},
    "deepseek": {"label": "DeepSeek", "model": "deepseek-chat"},
}


def _apply_llm_provider(provider: str) -> None:
    llm_service_module._session_provider = str(provider or "openai").lower()


def resolve_llm_config(
    api_key: str | None = None,
    *,
    provider: str | None = None,
) -> LLMConfig:
    """Build LLMConfig; provider is applied via session hook (no extra kwargs)."""
    prov = str(
        provider or st.session_state.get("llm_provider", "openai")
    ).lower()
    _apply_llm_provider(prov)
    key = (
        api_key
        if api_key is not None
        else st.session_state.get("openai_api_key")
    )
    return LLMConfig.from_api_key(key)

PAIR_TO_COUNTRY: dict[str, str] = {
    "USD/CNH": "US",
    "EUR/CNH": "Eurozone",
    "GBP/CNH": "UK",
    "HKD/CNH": "Hong Kong",
}

PAIR_PROMPTS: dict[str, list[str]] = {
    "USD/CNH": [
        "I'm an Amazon US seller. My sales tax is auto-withheld, so how does this zero-VAT baseline affect my cash flow forecast right now?",
        "We have a 1 million USD payout next month — what do our technical indicators suggest for locking the rate today?",
    ],
    "EUR/CNH": [
        "EU VAT is eating into my margins. If I set my Germany VAT slider to 19%, does the RSI show it's a good time to hedge?",
        "The Euro has been moving fast lately. Does the MACD confirm if this trend is about to reverse for my upcoming payout?",
    ],
    "HKD/CNH": [
        "Hong Kong is a zero-VAT port, but our CNH payroll is feeling the squeeze. What risk alerts does NOVA have for us today?",
        "Based on the latest 2026 macro growth forecasts, should we convert our HKD revenues to CNH now or wait?",
    ],
    "GBP/CNH": [
        "We are losing 20% to HMRC VAT in the UK. How does this deduction impact our net payout momentum and cash flow?",
        "The Sterling is on a single-direction rally. Is the Net Payout RSI showing overbought saturation, or should we hedge now?",
    ],
}

st.set_page_config(
    page_title="NOVA FX Risk Copilot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; max-width: 1400px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(165deg, #0c1929 0%, #1e3a5f 45%, #334155 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    .nova-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(148,163,184,0.3);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .nova-label {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .nova-value {
        font-size: clamp(1.05rem, 2.2vw, 1.5rem);
        font-weight: 700;
        color: #f8fafc;
        white-space: nowrap;
        overflow: visible;
        cursor: help;
    }
    .traffic-light {
        border-radius: 14px;
        padding: 16px;
        margin: 12px 0;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.88; }
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    .bloomberg-header {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        border-left: 4px solid #00f2fe;
        padding-left: 12px;
        margin-bottom: 0.25rem;
    }
    /* Streamlit truncates st.metric values with ellipsis in narrow columns */
    [data-testid="stHorizontalBlock"] [data-testid="column"] {
        overflow: visible !important;
        min-width: 0 !important;
    }
    [data-testid="stMetric"] {
        overflow: visible !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] label,
    [data-testid="stMetricValue"] p {
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
        max-width: none !important;
        font-size: clamp(0.78rem, 1.6vw, 1.15rem) !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        overflow: visible !important;
        text-overflow: unset !important;
        font-size: clamp(0.68rem, 1.2vw, 0.82rem) !important;
    }
    .desk-metric {
        padding: 0.15rem 0 0.35rem 0;
    }
    .desk-metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569 !important;
        line-height: 1.2;
        margin-bottom: 0.2rem;
        letter-spacing: 0.02em;
    }
    .desk-metric-value {
        font-size: clamp(0.95rem, 2vw, 1.35rem);
        font-weight: 700;
        color: #0f172a !important;
        line-height: 1.25;
        white-space: nowrap;
        overflow: visible;
        cursor: help;
    }
    .desk-metric-value:hover {
        color: #0369a1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* 1. Target ONLY the actual labels inside clickable sidebar buttons to be high-contrast and clear */
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"] p,
    .stSidebar button[data-testid="baseButton-secondary"] p {
        color: #1A202C !important; /* Sharp Onyx Black for buttons */
        font-weight: 700 !important; /* Strong bold for buttons */
        font-size: 0.92rem !important;
    }

    /* 2. Style the custom clickable buttons container with precise styling */
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
    .stSidebar button[data-testid="baseButton-secondary"] {
        border: 1.5px solid #CBD5E0 !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
    .stSidebar button[data-testid="baseButton-secondary"]:hover {
        border-color: #4A5568 !important;
        background-color: #F7FAFC !important;
    }

    /* 3. RESTORE secondary hierarchy for ordinary text labels, sliders, captions and sub-headers */
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] span[data-testid="stWidgetLabel"],
    .stSidebar div[data-testid="stMarkdownContainer"] p:not(button p),
    .stSidebar .stSlider label,
    .stSidebar .stSelectbox label,
    .stSidebar span[data-testid="stWidgetLabel"] {
        color: #A0AEC0 !important; /* Sophisticated muted label tier on dark sidebar */
        font-weight: 500 !important; /* Standard clean weight */
    }

    /* 4. Ensure tiny background metadata captions remain elegant but fully legible */
    [data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
    .stSidebar div[data-testid="stCaptionContainer"] {
        color: #718096 !important; /* Medium Slate Gray */
        font-weight: 400 !important;
    }

    /* 5. Force colloquial prompt examples to crisp white on the dark sidebar */
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] ul,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li p,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] blockquote,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p em,
    .stSidebar div[data-testid="stMarkdownContainer"] li,
    .stSidebar div[data-testid="stMarkdownContainer"] ul,
    .stSidebar div[data-testid="stMarkdownContainer"] li p,
    .stSidebar div[data-testid="stMarkdownContainer"] blockquote,
    .stSidebar div[data-testid="stMarkdownContainer"] p em {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p:has(code),
    .stSidebar div[data-testid="stMarkdownContainer"] p:has(code) {
        color: #FFFFFF !important;
    }

    .ai-enhanced-section {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 55%, #1e293b 100%);
        border: 1px solid rgba(129, 140, 248, 0.45);
        border-radius: 16px;
        padding: 18px 20px;
        margin: 16px 0;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.12);
    }
    .ai-enhanced-section h4 {
        color: #c7d2fe !important;
        margin-top: 0 !important;
    }
    .ai-enhanced-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        color: #e0e7ff;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
    }
    /* Fallback sweep for bullet prompt lines in the sidebar */
    [data-testid="stSidebar"] li p,
    [data-testid="stSidebar"] li span,
    .stSidebar li p,
    .stSidebar li span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _empty_fx_kline_frame(pair: str) -> pd.DataFrame:
    """Safe empty OHLCV shell when live ingestion fails."""
    df = pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "volatility_pct", "ma5"]
    )
    df.attrs["pair"] = pair
    df.attrs["data_source"] = "empty_fallback"
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_fx_data(pair: str, *, _range_v: int = 3):
    try:
        return fetch_fx_kline(pair=pair, days=30)
    except Exception:
        st.warning("FX Market Data ingestion failed. Reverting to empty fallback frame.")
        return _empty_fx_kline_frame(pair)


@st.cache_data(show_spinner=False, ttl=3600)
def load_backtest_analysis(monthly_revenue: float) -> dict:
    """Ingest live Yahoo USD/CNH history, then run MPT analytics engine."""
    history = fetch_real_market_data(days=180)
    return advanced_quantitative_analysis(history, monthly_revenue=monthly_revenue)


def _metric_value(analysis: dict, metric_key: str, track: str) -> float:
    """Read a scalar metric from metrics_comparison with benchmark/nova fallback."""
    comparison = analysis.get("metrics_comparison", {})
    if metric_key in comparison and track in comparison[metric_key]:
        return float(comparison[metric_key][track])
    track_block = analysis.get(track, {})
    return float(track_block.get(metric_key, 0.0))


def _parse_timeline(analysis: dict) -> pd.DatetimeIndex:
    """Coerce backend timeline list to datetime for Plotly X-axis."""
    raw = analysis.get("timeline") or []
    if raw:
        return pd.to_datetime(raw)

    n = len(analysis.get("benchmark", {}).get("cumulative_equity", []))
    if n == 0:
        n = int(analysis.get("observation_days", 180))
    end = pd.Timestamp.today().normalize()
    return pd.bdate_range(end=end, periods=n)


def build_fx_chart(df, pair: str) -> go.Figure:
    chart_dates = pd.to_datetime(df["date"], errors="coerce")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            f"{pair} Daily Candlestick + MA5 Trend Trigger",
            "FX Volume Proxy & Intraday Volatility (%)",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_dates,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=pair,
            increasing_line_color="#38bdf8",
            increasing_fillcolor="#38bdf8",
            decreasing_line_color="#f87171",
            decreasing_fillcolor="#f87171",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_dates,
            y=df["ma5"],
            mode="lines",
            name="MA5",
            line=dict(color="#fbbf24", width=2.5),
        ),
        row=1,
        col=1,
    )

    bar_colors = [
        "#38bdf8" if c >= o else "#f87171"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=chart_dates,
            y=df["volume"],
            name="Volume Proxy",
            marker_color=bar_colors,
            opacity=0.55,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=chart_dates,
            y=df["volatility_pct"],
            mode="lines",
            name="Volatility %",
            line=dict(color="#a78bfa", width=1.5, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=680,
        margin=dict(l=8, r=8, t=48, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        paper_bgcolor="#060b14",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", type="date")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
    return fig


def build_equity_curve_chart(analysis: dict) -> go.Figure:
    timeline = _parse_timeline(analysis)
    nova_equity = analysis.get("nova", {}).get("cumulative_equity", [])
    bench_equity = analysis.get("benchmark", {}).get("cumulative_equity", [])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline,
            y=nova_equity,
            mode="lines",
            name=analysis.get("nova", {}).get("label", "NOVA Dynamic Overlay"),
            line=dict(color="#00f2fe", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeline,
            y=bench_equity,
            mode="lines",
            name=analysis.get("benchmark", {}).get("label", "Unhedged Spot Benchmark"),
            line=dict(color="#ff4b4b", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title=f"Cumulative Capital Growth · Net Asset Value ({OFFSHORE_QUOTE_CCY})",
        template="plotly_dark",
        height=520,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="#060b14",
        plot_bgcolor="#0f172a",
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_title="Trading Date",
        yaxis_title=f"Cumulative {OFFSHORE_QUOTE_CCY} (¥)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", type="date")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
    return fig


def build_quant_metrics_table(analysis: dict) -> pd.DataFrame:
    """Render side-by-side grid aligned to backend metrics_comparison schema."""
    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    bench_total = _metric_value(analysis, "total_value", "benchmark")
    nova_total = _metric_value(analysis, "total_value", "nova")
    bench_vol = _metric_value(analysis, "annualized_volatility", "benchmark")
    nova_vol = _metric_value(analysis, "annualized_volatility", "nova")
    bench_var = _metric_value(analysis, "var_95", "benchmark")
    nova_var = _metric_value(analysis, "var_95", "nova")
    bench_mdd = _metric_value(analysis, "max_drawdown", "benchmark")
    nova_mdd = _metric_value(analysis, "max_drawdown", "nova")
    bench_sharpe = _metric_value(analysis, "sharpe_ratio", "benchmark")
    nova_sharpe = _metric_value(analysis, "sharpe_ratio", "nova")

    return pd.DataFrame(
        [
            {
                "📊 金融量化评级指标 (Quant Metrics)": f"Total Value ({OFFSHORE_QUOTE_CCY})",
                "传统零对冲 (Benchmark)": f"¥{bench_total:,.0f}",
                "NOVA 动态策略 (Proposed)": f"¥{nova_total:,.0f}",
            },
            {
                "📊 金融量化评级指标 (Quant Metrics)": "Annualized Volatility",
                "传统零对冲 (Benchmark)": pct(bench_vol),
                "NOVA 动态策略 (Proposed)": pct(nova_vol),
            },
            {
                "📊 金融量化评级指标 (Quant Metrics)": (
                    f"95% Historical VaR (Daily {OFFSHORE_QUOTE_CCY})"
                ),
                "传统零对冲 (Benchmark)": f"¥{bench_var:,.0f}",
                "NOVA 动态策略 (Proposed)": f"¥{nova_var:,.0f}",
            },
            {
                "📊 金融量化评级指标 (Quant Metrics)": "Maximum Drawdown (MDD)",
                "传统零对冲 (Benchmark)": pct(bench_mdd),
                "NOVA 动态策略 (Proposed)": pct(nova_mdd),
            },
            {
                "📊 金融量化评级指标 (Quant Metrics)": "Sharpe Ratio (rf=2%)",
                "传统零对冲 (Benchmark)": f"{bench_sharpe:.3f}",
                "NOVA 动态策略 (Proposed)": f"{nova_sharpe:.3f}",
            },
        ]
    )


def format_pct_change(pct: float) -> tuple[str, str]:
    """Compact display string + full-precision tooltip for day change."""
    value = float(pct)
    full = f"{value:+.6f}%"
    if abs(value) < 0.0001:
        display = f"{value:+.4f}%"
    elif abs(value) < 0.1:
        display = f"{value:+.3f}%"
    else:
        display = f"{value:+.2f}%"
    return display, full


def render_desk_metric(
    container,
    label: str,
    value: str,
    *,
    tooltip: str | None = None,
) -> None:
    """Desk metric with hover tooltip — avoids Streamlit st.metric ellipsis clipping."""
    tip = html.escape(tooltip or value, quote=True)
    with container:
        st.markdown(
            f'<div class="desk-metric">'
            f'<div class="desk-metric-label">{html.escape(label)}</div>'
            f'<div class="desk-metric-value" title="{tip}">{html.escape(value)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def sidebar_card(label: str, value: str, sub: str = "", *, tooltip: str | None = None):
    sub_html = (
        f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">{sub}</div>'
        if sub
        else ""
    )
    tip = html.escape(tooltip or value, quote=True)
    st.sidebar.markdown(
        f"""
        <div class="nova-card">
            <div class="nova-label">{html.escape(label)}</div>
            <div class="nova-value" title="{tip}">{html.escape(value)}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_traffic_light(metrics: dict):
    color = metrics["risk_color"]
    level = metrics["risk_level"]
    advice = metrics["risk_advice"]
    st.sidebar.markdown(
        f"""
        <div class="traffic-light" style="background:{color}22;border:2px solid {color};">
            <div style="font-size:0.75rem;color:#cbd5e1;">FX RISK TRAFFIC LIGHT</div>
            <div style="font-size:1.4rem;font-weight:800;color:{color};">{level}</div>
            <div style="font-size:0.85rem;color:#f1f5f9;line-height:1.4;margin-top:8px;">
                {advice}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_playbook_in_expander(doc: dict) -> None:
    st.markdown(
        f"**Professional definition:** {doc.get('professional_definition', '')}"
    )
    executive = doc.get("layman_translation", "")
    if executive:
        st.markdown(f"**Executive summary:** {executive}")
    st.markdown(f"**Actionable playbook:** {doc.get('actionable_playbook', '')}")

    sources_block = format_data_sources_block(doc)
    if sources_block:
        st.markdown(sources_block)


def render_assistant_rag_message(msg: dict) -> None:
    parts = msg.get("rag_parts")
    if not parts:
        st.markdown(msg.get("content", ""))
        return

    st.markdown(f"**Your question:** {parts['user_query']}")

    if parts.get("nova_mode") == "ai_enhanced" and parts.get("ai_narrative"):
        st.markdown(
            '<span class="ai-enhanced-badge">🧠 AI Enhanced Copilot Response</span>',
            unsafe_allow_html=True,
        )
        st.markdown(parts["ai_narrative"])
        st.caption(
            "Deterministic executive action and scores below remain authoritative."
        )

    st.markdown(parts.get("executive_action", parts.get("action", "")))

    if parts.get("ai_knowledge_reasoning"):
        with st.expander("🧠 AI Knowledge Reasoning", expanded=False):
            kb = parts["ai_knowledge_reasoning"]
            if kb.get("executive_summary"):
                st.markdown(kb["executive_summary"])
            if kb.get("primary_risks"):
                st.markdown("**Primary risks:** " + "; ".join(kb["primary_risks"]))
            if kb.get("vulnerabilities"):
                st.markdown("**Vulnerabilities:** " + "; ".join(kb["vulnerabilities"]))
            if kb.get("mitigations"):
                st.markdown("**Mitigations:** " + "; ".join(kb["mitigations"]))

    if parts.get("execution_block"):
        with st.expander("⚡ Numeric execution ticket", expanded=False):
            st.success(parts["execution_block"])

    playbook_count = parts.get("playbook_count", len(parts.get("docs", [])))
    if playbook_count:
        with st.expander(
            f"🔍 Retrieved playbooks ({playbook_count}) — on-demand detail",
            expanded=False,
        ):
            for idx, doc in enumerate(parts.get("docs", [])):
                if idx:
                    st.divider()
                st.markdown(f"**[{doc['domain']}] {doc['title']}**")
                render_playbook_in_expander(doc)

    st.caption(parts.get("disclaimer", ""))


def _init_chat(pair: str):
    if "messages_by_pair" not in st.session_state:
        st.session_state.messages_by_pair = {}

    if pair not in st.session_state.messages_by_pair:
        prompts = PAIR_PROMPTS.get(pair, PAIR_PROMPTS["USD/CNH"])
        st.session_state.messages_by_pair[pair] = [
            {
                "role": "assistant",
                "content": (
                    f"Welcome to **NOVA FX Risk Copilot** — desk view: **{pair}**.\n\n"
                    "I am wired to your **live multi-factor desk** (MA5 · Bollinger · MA30) "
                    "and will output **execution tickets** when you state a payout size.\n\n"
                    "Try a plain-English question, for example:\n"
                    f'- *"{prompts[0]}"*\n'
                    f'- *"We have 1 million USD payout next month — what do I lock today?"*\n'
                    f'- *"{prompts[1]}"*\n\n'
                    "I cross-reference the Corporate FX Risk Manual with real-time spot, "
                    "hedge ratio, and CNH math."
                ),
            }
        ]


def _merge_enterprise_assessment(metrics: dict, selected_pair: str) -> dict:
    """Overlay stored enterprise assessment onto environment-only desk metrics."""
    stored = st.session_state.get("enterprise_assessment_by_pair", {}).get(selected_pair)
    if not stored or not stored.get("ready"):
        return {**metrics, "enterprise_assessment_ready": False}

    assessment = stored.get("assessment", {})
    merged = {**metrics, **assessment, "enterprise_assessment_ready": True}
    if stored.get("scenario"):
        merged["enterprise_scenario"] = stored["scenario"]
    # Live market snapshots must always win over stale session assessment copies.
    for field in (
        "spot",
        "ma5",
        "ma30",
        "pct_change",
        "low_30d",
        "high_30d",
        "bollinger_upper",
        "data_source",
        "latest_date",
        "ticker",
        "ma_signal",
        "risk_level",
        "risk_color",
        "risk_advice",
    ):
        if metrics.get(field) is not None:
            merged[field] = metrics[field]
    return merged


def _sync_live_desk_session(selected_pair: str, desk_metrics: dict) -> dict:
    """Persist latest multi-factor desk snapshot for the active pair."""
    if "live_desk_by_pair" not in st.session_state:
        st.session_state.live_desk_by_pair = {}
    st.session_state.live_desk_by_pair[selected_pair] = desk_metrics
    st.session_state.active_desk_metrics = desk_metrics
    return desk_metrics


def _extract_live_desk_vars(desk_metrics: dict) -> dict:
    """Normalized live parameters passed into the market-aware copilot pipeline."""
    return {
        "selected_pair": desk_metrics.get("pair"),
        "current_spot": float(desk_metrics.get("spot", 0.0)),
        "ma5_val": float(desk_metrics.get("ma5", 0.0)),
        "ma30_val": float(desk_metrics.get("ma30", desk_metrics.get("ma5", 0.0))),
        "bollinger_upper": float(
            desk_metrics.get("bollinger_upper", desk_metrics.get("spot", 0.0))
        ),
        "risk_level": desk_metrics.get(
            "multi_factor_risk_rating", desk_metrics.get("risk_level", "MEDIUM")
        ),
        "recommended_hedge_ratio": float(
            resolve_display_hedge_ratio(
                desk_metrics,
                get_applied_risk_mapping(),
            )
            or desk_metrics.get("recommended_hedge_ratio")
            or 0.0
        ),
        "enterprise_assessment_ready": bool(
            desk_metrics.get("enterprise_assessment_ready", False)
        ),
        "multi_factor_score": float(desk_metrics.get("multi_factor_score", 0.0)),
        "market_score": float(desk_metrics.get("market_score", 0.0)),
        "macro_score": float(desk_metrics.get("macro_score", 0.0)),
        "policy_score": float(desk_metrics.get("policy_score", 0.0)),
        "knowledge_score": float(desk_metrics.get("knowledge_score") or 0.0),
        "final_risk_score": float(desk_metrics.get("final_risk_score") or 0.0),
        "net_payout_rsi": float(desk_metrics.get("net_payout_rsi", 50.0)),
        "net_payout_macd_histogram": float(
            desk_metrics.get("net_payout_macd_histogram", 0.0)
        ),
        "gross_sales_base": float(desk_metrics.get("gross_sales_base", 0.0)),
        "vat_deduction_rate": float(desk_metrics.get("vat_deduction_rate", 0.0)),
    }


def init_llm_config() -> LLMConfig:
    """Optional LLM API key — session_state only; never persisted to disk."""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "openai"
    return resolve_llm_config()


def _init_ai_budget() -> None:
    if "ai_call_count" not in st.session_state:
        st.session_state.ai_call_count = 0


def _ai_budget_remaining() -> int:
    _init_ai_budget()
    return max(0, MAX_AI_CALLS_PER_SESSION - int(st.session_state.ai_call_count))


def _ai_budget_exhausted() -> bool:
    return _ai_budget_remaining() <= 0


def _try_consume_ai_budget() -> bool:
    _init_ai_budget()
    if int(st.session_state.ai_call_count) >= MAX_AI_CALLS_PER_SESSION:
        return False
    st.session_state.ai_call_count = int(st.session_state.ai_call_count) + 1
    return True


def _refund_ai_budget() -> None:
    _init_ai_budget()
    count = int(st.session_state.ai_call_count)
    if count > 0:
        st.session_state.ai_call_count = count - 1


def _message_indicates_api_quota_exhausted(message: str) -> bool:
    blob = str(message or "").lower()
    markers = (
        "quota/billing exhausted",
        "quota exceeded",
        "insufficient_quota",
        "exceeded your current quota",
        "insufficient funds",
        "insufficient balance",
        "payment required",
        "out of credits",
        "no credit",
        "余额不足",
    )
    return any(marker in blob for marker in markers)


def _record_openai_api_error(message: str) -> None:
    safe = str(message)[:500]
    st.session_state.openai_api_error = safe
    if _message_indicates_api_quota_exhausted(safe):
        st.session_state.api_quota_exhausted = True


def _clear_openai_api_error() -> None:
    st.session_state.pop("openai_api_error", None)
    st.session_state.pop("api_quota_exhausted", None)


def _arm_api_quota_retry() -> None:
    """Clear a prior quota warning so the next user-initiated AI call can probe billing again."""
    _clear_openai_api_error()


def _record_openai_api_success() -> None:
    _clear_openai_api_error()


def _ai_quota_exhausted() -> bool:
    return bool(st.session_state.get("api_quota_exhausted"))


def _ai_budget_block_reason() -> str:
    if _ai_budget_exhausted():
        return (
            f"Session AI call limit reached ({MAX_AI_CALLS_PER_SESSION}/session). "
            "Refresh the page to reset, or continue in Local Mode."
        )
    return ""


def _ai_calls_blocked() -> bool:
    return _ai_budget_exhausted()


def _install_llm_session_hooks() -> None:
    """Wire refund + error surfacing for LLMService instances created outside app.py."""
    llm_service_module._session_budget_refund = _refund_ai_budget
    llm_service_module._session_error_sink = _record_openai_api_error
    llm_service_module._session_budget_block_reason = _ai_budget_block_reason
    llm_service_module._session_success_sink = _record_openai_api_success


_install_llm_session_hooks()


def make_llm_service(config: LLMConfig) -> LLMService:
    """Construct LLMService; tolerate older llm_service builds during Streamlit hot reload."""
    _install_llm_session_hooks()
    kwargs: dict[str, Any] = {"budget_gate": _try_consume_ai_budget}
    try:
        params = inspect.signature(LLMService.__init__).parameters
        if "budget_refund" in params:
            kwargs["budget_refund"] = _refund_ai_budget
        if "error_sink" in params:
            kwargs["error_sink"] = _record_openai_api_error
    except (TypeError, ValueError):
        pass
    return LLMService(config, **kwargs)


def render_openai_api_error_banner(*, sidebar: bool = False) -> None:
    """Surface the most recent OpenAI failure (quota, invalid key, etc.)."""
    err = st.session_state.get("openai_api_error")
    if not err or not init_llm_config().enabled:
        return
    target = st.sidebar if sidebar else st
    if _ai_quota_exhausted():
        target.error(err)
    else:
        target.warning(err)


def _api_key_fingerprint(api_key: str | None) -> str:
    key = str(api_key or "").strip()
    provider = str(st.session_state.get("llm_provider", "openai")).lower()
    if not key:
        return f"{provider}:no-key"
    return f"{provider}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def render_ai_mode_badge() -> None:
    """Dual-mode status indicator for sidebar."""
    if is_ai_enhanced_mode(st.session_state.get("openai_api_key")):
        remaining = _ai_budget_remaining()
        if _ai_quota_exhausted():
            st.sidebar.markdown(
                '<div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.45);'
                'border-radius:10px;padding:10px 12px;margin:8px 0;">'
                '<span style="color:#fcd34d;font-weight:700;">🟠 API Quota Previously Exhausted</span>'
                f'<br><span style="color:#94a3b8;font-size:0.85em;">'
                f"AI calls remaining: {remaining}/{MAX_AI_CALLS_PER_SESSION} · "
                "after topping up, click <b>Refresh AI Analysis</b> or send a copilot message to retry</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<div style="background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.5);'
                'border-radius:10px;padding:10px 12px;margin:8px 0;">'
                '<span style="color:#86efac;font-weight:700;">🟢 Treasury Copilot</span>'
                f'<br><span style="color:#94a3b8;font-size:0.85em;">'
                f"AI calls remaining: {remaining}/{MAX_AI_CALLS_PER_SESSION}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            if _ai_budget_exhausted():
                st.sidebar.warning(
                    f"AI call limit reached ({MAX_AI_CALLS_PER_SESSION}/session). "
                    "Deterministic desk remains fully available."
                )
        render_openai_api_error_banner(sidebar=True)
    else:
        st.sidebar.markdown(
            '<div style="background:rgba(148,163,184,0.12);border:1px solid rgba(148,163,184,0.35);'
            'border-radius:10px;padding:10px 12px;margin:8px 0;">'
            '<span style="color:#cbd5e1;font-weight:700;">⚪ Local Mode</span>'
            '<br><span style="color:#94a3b8;font-size:0.85em;">'
            "Deterministic treasury engine — no OpenAI costs</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def get_llm_service() -> LLMService:
    return make_llm_service(init_llm_config())


def render_llm_sidebar_config() -> LLMConfig:
    """User-facing optional LLM API key — session_state only, never logged or saved."""
    st.sidebar.divider()
    st.sidebar.markdown("### 🏛 Treasury Copilot (Optional)")
    st.sidebar.caption(
        "Leave blank for Local Mode. Supports OpenAI and DeepSeek (OpenAI-compatible). "
        "Key stays in this browser session only."
    )
    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=list(LLM_PROVIDER_OPTIONS.keys()),
        format_func=lambda p: LLM_PROVIDER_OPTIONS[p]["label"],
        index=list(LLM_PROVIDER_OPTIONS.keys()).index(
            st.session_state.get("llm_provider", "openai")
        )
        if st.session_state.get("llm_provider", "openai") in LLM_PROVIDER_OPTIONS
        else 0,
        help="DeepSeek uses the same chat-completions API shape as OpenAI.",
    )
    st.session_state.llm_provider = provider
    _apply_llm_provider(provider)
    preset = LLM_PROVIDER_OPTIONS[provider]
    api_key = st.sidebar.text_input(
        f"{preset['label']} API Key (optional)",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        help=f"Model: {preset['model']}. Never stored to disk.",
        placeholder="sk-…",
    )
    st.session_state.openai_api_key = api_key.strip()
    fp = _api_key_fingerprint(api_key.strip()) if api_key.strip() else None
    prev_fp = st.session_state.get("_llm_api_key_fingerprint")
    if fp and fp != prev_fp:
        _clear_openai_api_error()
        st.session_state._llm_api_key_fingerprint = fp
    elif not fp:
        st.session_state.pop("_llm_api_key_fingerprint", None)
        _clear_openai_api_error()
    config = resolve_llm_config(st.session_state.openai_api_key, provider=provider)
    render_ai_mode_badge()
    return config


def _ai_insights_cache_key(metrics: dict, selected_pair: str) -> str:
    scenario = str(metrics.get("enterprise_scenario", ""))
    return (
        f"{selected_pair}|{scenario}|"
        f"{float(metrics.get('market_score', 0)):.1f}|"
        f"{float(metrics.get('macro_score', 0)):.1f}|"
        f"{float(metrics.get('policy_score', 0)):.1f}|"
        f"{float(metrics.get('knowledge_score') or 0):.1f}|"
        f"{float(metrics.get('final_risk_score') or 0):.1f}"
    )


@st.cache_data(ttl=3600)
def _cached_treasury_ai_bundle(
    cache_key: str,
    _api_fingerprint: str,
    _api_key: str,
    _provider: str,
    metrics_json: str,
    retrieved_docs_json: str,
) -> str:
    """Run full Treasury Copilot AI bundle; cached for 1 hour per scenario fingerprint."""
    metrics = json.loads(metrics_json)
    retrieved_docs = json.loads(retrieved_docs_json) if retrieved_docs_json else None
    _apply_llm_provider(_provider)
    llm = make_llm_service(LLMConfig.from_api_key(_api_key))
    if not llm.enabled:
        return "{}"
    insights = run_llm_enhancement_bundle(
        llm,
        metrics,
        retrieved_documents=retrieved_docs,
    )
    if not insights:
        return json.dumps({"enabled": True, "errors": ["all_sections_failed"]})
    return json.dumps(insights)


@st.cache_data(ttl=3600)
def _cached_ai_executive_report(
    cache_key: str,
    _api_fingerprint: str,
    _api_key: str,
    _provider: str,
    metrics_json: str,
    retrieved_docs_json: str,
) -> str:
    bundle = json.loads(
        _cached_treasury_ai_bundle(
            cache_key,
            _api_fingerprint,
            _api_key,
            _provider,
            metrics_json,
            retrieved_docs_json,
        )
    )
    return json.dumps(bundle.get("executive_summary") or {})


@st.cache_data(ttl=3600)
def _cached_ai_risk_analysis(
    cache_key: str,
    _api_fingerprint: str,
    _api_key: str,
    _provider: str,
    metrics_json: str,
    retrieved_docs_json: str,
) -> str:
    bundle = json.loads(
        _cached_treasury_ai_bundle(
            cache_key,
            _api_fingerprint,
            _api_key,
            _provider,
            metrics_json,
            retrieved_docs_json,
        )
    )
    return json.dumps(bundle.get("risk_reasoning") or {})


@st.cache_data(ttl=3600)
def _cached_ai_hedge_recommendations(
    cache_key: str,
    _api_fingerprint: str,
    _api_key: str,
    _provider: str,
    metrics_json: str,
    retrieved_docs_json: str,
) -> str:
    bundle = json.loads(
        _cached_treasury_ai_bundle(
            cache_key,
            _api_fingerprint,
            _api_key,
            _provider,
            metrics_json,
            retrieved_docs_json,
        )
    )
    return json.dumps(bundle.get("hedge_strategy") or {})


def _run_and_cache_ai_insights(
    metrics: dict,
    selected_pair: str,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    llm_config = init_llm_config()
    if not llm_config.enabled:
        return None
    if _ai_budget_exhausted() and force:
        st.warning(
            f"AI call budget exhausted ({MAX_AI_CALLS_PER_SESSION}/session). "
            "Refresh the page to reset, or continue in Local Mode."
        )
        return st.session_state.get("ai_insights_by_pair", {}).get(selected_pair, {}).get(
            "insights"
        )

    if force:
        _arm_api_quota_retry()

    cache_key = _ai_insights_cache_key(metrics, selected_pair)
    api_fp = _api_key_fingerprint(st.session_state.get("openai_api_key"))
    metrics_json = serialize_for_cache(metrics)

    retrieved_docs = None
    scenario = str(metrics.get("enterprise_scenario", "")).strip()
    if scenario:
        retrieved_docs = retrieve_knowledge(scenario, pair=selected_pair, top_k=5)
    retrieved_docs_json = serialize_for_cache(retrieved_docs or [])

    if force:
        _cached_treasury_ai_bundle.clear()
        _cached_ai_executive_report.clear()
        _cached_ai_risk_analysis.clear()
        _cached_ai_hedge_recommendations.clear()

    provider = str(st.session_state.get("llm_provider", "openai")).lower()
    api_key_str = str(st.session_state.get("openai_api_key") or "")

    with st.spinner("Running Treasury Copilot AI layer…"):
        bundle_raw = _cached_treasury_ai_bundle(
            cache_key,
            api_fp,
            api_key_str,
            provider,
            metrics_json,
            retrieved_docs_json,
        )
        _cached_ai_executive_report(
            cache_key, api_fp, api_key_str, provider, metrics_json, retrieved_docs_json
        )
        _cached_ai_risk_analysis(
            cache_key, api_fp, api_key_str, provider, metrics_json, retrieved_docs_json
        )
        _cached_ai_hedge_recommendations(
            cache_key, api_fp, api_key_str, provider, metrics_json, retrieved_docs_json
        )
    try:
        insights = json.loads(bundle_raw)
    except json.JSONDecodeError:
        insights = {}

    if not insights or insights.get("errors") == ["all_sections_failed"]:
        render_openai_api_error_banner()
        if not st.session_state.get("openai_api_error"):
            st.warning(
                "Treasury Copilot AI layer could not reach OpenAI — rule-based desk output is unchanged. "
                "Check your API key billing/quota, then click **Refresh AI Analysis**."
            )
        if force:
            _cached_treasury_ai_bundle.clear()
        return None
    cache_bucket = st.session_state.setdefault("ai_insights_by_pair", {})
    cache_bucket[selected_pair] = {"cache_key": cache_key, "insights": insights}
    return insights


def render_ai_knowledge_intelligence(metrics: dict) -> None:
    """AI-only knowledge insights — scores/rule breakdown live in 'Why this score was generated'."""
    assessment = metrics.get("ai_knowledge_assessment")
    if not assessment:
        if init_llm_config().enabled:
            render_openai_api_error_banner()
            if st.session_state.get("openai_api_error"):
                st.caption("Rule-based knowledge score used — see warning above for LLM status.")
            else:
                st.caption("AI knowledge intelligence unavailable — rule-based score used.")
        return

    consensus = assessment.get("knowledge_consensus") or {}
    top_risks = assessment.get("top_risks") or []
    hidden_risks = assessment.get("hidden_risks") or []
    mitigation = assessment.get("mitigation_assessment") or assessment.get("mitigation_strength") or []
    executive_summary = assessment.get("executive_summary")
    cases = metrics.get("case_interpretations") or []

    if not any(
        [
            consensus.get("consensus"),
            top_risks,
            hidden_risks,
            mitigation,
            executive_summary,
            cases,
        ]
    ):
        return

    st.markdown(
        '<div class="ai-enhanced-section">'
        '<span class="ai-enhanced-badge">🧠 AI Knowledge Intelligence</span>'
        "<p style='color:#94a3b8;margin-top:0;'>"
        "Supplemental AI read — rule-based score breakdown is in "
        "<em>Why this score was generated</em> above."
        "</p></div>",
        unsafe_allow_html=True,
    )

    if consensus.get("consensus"):
        st.markdown(
            f"**Knowledge Consensus:** {consensus.get('consensus', 'N/A')} "
            f"({consensus.get('consensus_score', '—')} score) — informational only"
        )

    if top_risks:
        st.markdown("**Top Risks**")
        for risk in top_risks[:5]:
            st.markdown(f"- {risk}")

    if hidden_risks:
        st.markdown("**Hidden Risks**")
        for risk in hidden_risks[:5]:
            st.markdown(f"- {risk}")

    if mitigation:
        st.markdown("**Mitigation Strength**")
        for mit in mitigation[:5]:
            if isinstance(mit, dict):
                st.markdown(
                    f"- **{mit.get('name', 'Framework')}** · effectiveness "
                    f"{mit.get('effectiveness', '—')}/10"
                )
            else:
                st.markdown(f"- {mit}")

    if executive_summary:
        st.markdown(f"**Executive Summary:** {executive_summary}")

    if cases:
        st.markdown("**Case Interpretation**")
        for case in cases[:3]:
            st.markdown(f"- **{case.get('title', 'Case')}**")
            st.caption(case.get("relevance_reason", ""))


def render_ai_knowledge_assessment(metrics: dict) -> None:
    """Backward-compatible alias for AI Knowledge Intelligence panel."""
    render_ai_knowledge_intelligence(metrics)


def _resolve_retrieved_docs(metrics: dict, selected_pair: str) -> list[dict[str, Any]]:
    scenario = str(metrics.get("enterprise_scenario", "")).strip()
    if not scenario:
        return []
    return retrieve_knowledge(scenario, pair=selected_pair, top_k=5)


def render_treasury_copilot_intelligence(
    metrics: dict,
    selected_pair: str,
    *,
    insights: dict[str, Any] | None = None,
) -> None:
    """Enterprise intelligence layer — local deterministic fallback or AI enrichment."""
    ai_enabled = init_llm_config().enabled
    retrieved_docs = _resolve_retrieved_docs(metrics, selected_pair)

    strategic_report = metrics.get("treasury_copilot_report")
    hedge_review = metrics.get("hedge_strategy_review")

    if ai_enabled and (strategic_report or insights):
        enterprise = (insights or {}).get("enterprise_intelligence") or {}
        hedge_why = (insights or {}).get("hedge_ratio_explanation") or {}
        local_enterprise = build_local_enterprise_intelligence(metrics, retrieved_docs)
        local_hedge_why = build_local_hedge_ratio_explanation(metrics, retrieved_docs)
        if not enterprise:
            enterprise = local_enterprise
        if not (
            hedge_why.get("interpretation_bullets") or hedge_why.get("conclusion")
        ):
            hedge_why = local_hedge_why
        if not strategic_report:
            strategic_report = build_local_strategic_copilot_report(metrics, retrieved_docs)
        if not hedge_review:
            hedge_review = build_local_hedge_strategy_review(metrics)
        mode_label = "Treasury Copilot"
        if ai_enabled and not insights:
            render_openai_api_error_banner()
            if not st.session_state.get("openai_api_error"):
                st.caption(
                    "AI enrichment unavailable — showing rule-based enterprise intelligence. "
                    "Use **Refresh AI Analysis** after fixing API quota, or check sidebar for details."
                )
    else:
        enterprise = build_local_enterprise_intelligence(metrics, retrieved_docs)
        hedge_why = build_local_hedge_ratio_explanation(metrics, retrieved_docs)
        strategic_report = build_local_strategic_copilot_report(metrics, retrieved_docs)
        hedge_review = build_local_hedge_strategy_review(metrics)
        mode_label = "Local Treasury Engine"

    st.markdown(
        '<div class="ai-enhanced-section">'
        f'<span class="ai-enhanced-badge">🏛 {mode_label}</span>'
        "<h4>Enterprise Intelligence</h4>"
        "<p style='color:#94a3b8;margin-top:0;'>"
        "Structured business intelligence — deterministic scores and hedge ratios remain authoritative."
        "</p></div>",
        unsafe_allow_html=True,
    )

    profile = enterprise.get("business_profile") or {}
    exposure = enterprise.get("exposure_mapping") or {}
    archetype = enterprise.get("risk_archetype", "Cross-Border Treasury")
    similar_cases = normalize_knowledge_case_list(
        enterprise.get("most_similar_knowledge_cases")
    ) or build_similar_knowledge_cases(retrieved_docs)

    with st.expander("📊 Business Profile & Exposure Mapping", expanded=False):
        col_bp, col_exp = st.columns(2)
        with col_bp:
            st.markdown("**Business Profile**")
            st.markdown(f"- **Company type:** {profile.get('company_type', archetype)}")
            st.markdown(f"- **Exporter / Importer:** {profile.get('exporter_importer', 'unknown')}")
            st.markdown(f"- **Margin sensitivity:** {profile.get('margin_sensitivity', 'medium')}")
            st.markdown(f"- **Risk archetype:** {archetype}")
        with col_exp:
            st.markdown("**Exposure Mapping**")
            recv = exposure.get("receivable_currencies") or []
            pay = exposure.get("payable_currencies") or []
            st.markdown(
                f"- **Receivable currencies:** {', '.join(recv) if recv else '—'}"
            )
            st.markdown(f"- **Payable currencies:** {', '.join(pay) if pay else '—'}")
            st.markdown(f"- **Settlement cycle:** {exposure.get('settlement_cycle', '—')}")
            st.markdown(f"- **Hedge urgency:** {exposure.get('hedge_urgency', '—')}")

    if similar_cases:
        with st.expander("🔍 Most Similar Knowledge Cases", expanded=False):
            for case in similar_cases[:3]:
                st.markdown(
                    f"- **{case.get('title', 'Untitled')}** "
                    f"({case.get('domain', 'General')}) · "
                    f"similarity **{case.get('similarity_score', 0):.2f}**"
                )

    det_ratio = metrics.get("recommended_hedge_ratio")
    det_label = f"{float(det_ratio):.0%}" if det_ratio is not None else "N/A"

    with st.expander("📐 Why the Hedge Ratio Exists", expanded=False):
        st.markdown(f"**Deterministic Hedge Ratio:** {det_label}")
        bullets = hedge_why.get("interpretation_bullets") or []
        if bullets:
            for bullet in bullets:
                st.markdown(f"- {bullet}")
        elif hedge_why.get("conclusion"):
            st.markdown(hedge_why["conclusion"])
        else:
            st.caption(
                "Hedge ratio explanation unavailable — deterministic ratio above is authoritative."
            )

    final_score = metrics.get("final_risk_score")
    final_score_f = float(final_score) if final_score is not None else None
    if final_score_f is not None:
        render_risk_preference_mapping_panel(
            metrics,
            final_score=final_score_f,
            key_prefix=selected_pair.replace("/", "_"),
        )

    if enterprise.get("executive_insight"):
        st.markdown(f"**Executive insight:** {enterprise['executive_insight']}")

    if hedge_review:
        with st.expander("⚙️ Hedge Strategy Review", expanded=False):
            st.markdown(
                f"**Deterministic Hedge Ratio:** "
                f"{hedge_review.get('deterministic_hedge_ratio', det_label)}"
            )
            if hedge_review.get("agreement_level"):
                st.markdown(f"- **Agreement:** {hedge_review.get('agreement_level', 'N/A')}")
            if hedge_review.get("confidence") is not None:
                st.markdown(
                    f"- **Confidence:** {float(hedge_review.get('confidence', 0)) * 100:.0f}%"
                )
            if hedge_review.get("narrative"):
                st.markdown(hedge_review["narrative"])
            if hedge_review.get("strengths"):
                st.caption(
                    "Strengths: " + "; ".join(str(s) for s in hedge_review["strengths"][:3])
                )
            if hedge_review.get("weaknesses"):
                st.caption(
                    "Weaknesses: " + "; ".join(str(s) for s in hedge_review["weaknesses"][:3])
                )

    if strategic_report:
        with st.expander("🎯 Strategic Treasury Copilot", expanded=False):
            if strategic_report.get("executive_summary"):
                st.markdown(strategic_report["executive_summary"])
            if strategic_report.get("primary_risk_drivers"):
                st.markdown("**Primary Risk Drivers**")
                for driver in strategic_report["primary_risk_drivers"][:5]:
                    st.markdown(f"- {format_copilot_item(driver)}")
            if strategic_report.get("immediate_actions"):
                st.markdown("**Immediate Action**")
                for action in strategic_report["immediate_actions"][:4]:
                    st.markdown(f"- {format_copilot_item(action)}")
            if strategic_report.get("thirty_day_plan"):
                st.markdown("**30-Day Plan**")
                for step in strategic_report["thirty_day_plan"][:4]:
                    st.markdown(f"- {format_copilot_item(step)}")
            if strategic_report.get("recommended_instruments"):
                st.markdown("**Recommended Instruments**")
                for instrument in strategic_report["recommended_instruments"][:5]:
                    st.markdown(f"- {format_copilot_item(instrument)}")
            if strategic_report.get("watchlist_30_day"):
                st.markdown("**30-Day Watchlist**")
                for watch_item in strategic_report["watchlist_30_day"][:6]:
                    st.markdown(f"- {format_copilot_item(watch_item)}")


def render_ai_enhanced_analysis(
    metrics: dict,
    selected_pair: str,
    *,
    insights: dict[str, Any] | None,
) -> None:
    """Display optional AI interpretations beside deterministic scores."""
    llm_config = init_llm_config()
    enterprise_ready = bool(metrics.get("enterprise_assessment_ready"))

    if enterprise_ready:
        render_treasury_copilot_intelligence(
            metrics,
            selected_pair,
            insights=insights if llm_config.enabled else None,
        )

    if not llm_config.enabled or not insights:
        return

    st.markdown(
        '<div class="ai-enhanced-section">'
        '<span class="ai-enhanced-badge">🧠 AI Enhanced Analysis</span>'
        "<h4>AI Treasury Intelligence Layer</h4>"
        "<p style='color:#94a3b8;margin-top:0;'>"
        "Advisory interpretation only — deterministic NOVA scores above remain the source of truth."
        "</p></div>",
        unsafe_allow_html=True,
    )

    market = insights.get("market_intelligence") or {}
    macro = insights.get("macro_intelligence") or {}
    policy = insights.get("policy_intelligence") or {}
    knowledge = insights.get("knowledge_intelligence") or {}
    hedge = insights.get("hedge_strategy") or {}
    executive = insights.get("executive_summary") or {}

    col_m, col_x = st.columns(2)
    with col_m:
        st.markdown("**Market Intelligence**")
        if market:
            st.markdown(f"- **Regime:** {market.get('regime', 'N/A')}")
            st.markdown(f"- **Confidence:** {market.get('confidence', 'N/A')}")
            st.markdown(f"- **Reasoning:** {market.get('reasoning', 'N/A')}")
            st.caption(
                f"Exporter: {market.get('exporter_impact', 'N/A')} · "
                f"Importer: {market.get('importer_impact', 'N/A')}"
            )
        else:
            st.caption("AI market interpretation unavailable.")

        st.markdown("**Macro Intelligence**")
        if macro:
            st.markdown(f"- **Macro regime:** {macro.get('macro_regime', 'N/A')}")
            st.markdown(f"- **Key risk:** {macro.get('key_risk', 'N/A')}")
            if macro.get("bullish_factors"):
                st.markdown("- **Bullish:** " + "; ".join(macro["bullish_factors"][:3]))
            if macro.get("bearish_factors"):
                st.markdown("- **Bearish:** " + "; ".join(macro["bearish_factors"][:3]))
        else:
            st.caption("AI macro interpretation unavailable.")

    with col_x:
        st.markdown("**Policy Intelligence**")
        if policy:
            st.markdown(f"- **Stance:** {policy.get('policy_stance', 'N/A')}")
            st.markdown(f"- **FX consequences:** {policy.get('fx_consequences', 'N/A')}")
            st.markdown(f"- **Key risk:** {policy.get('key_risk', 'N/A')}")
        else:
            st.caption("AI policy interpretation unavailable.")

        st.markdown("**Knowledge Intelligence**")
        if knowledge:
            st.markdown(f"- **Summary:** {knowledge.get('executive_summary', 'N/A')}")
            if knowledge.get("primary_risks"):
                st.markdown("- **Primary risks:** " + "; ".join(knowledge["primary_risks"][:4]))
            if knowledge.get("vulnerabilities"):
                st.markdown(
                    "- **Vulnerabilities:** " + "; ".join(knowledge["vulnerabilities"][:4])
                )
        elif metrics.get("enterprise_assessment_ready"):
            st.caption("AI knowledge analysis unavailable — rule-based score shown above.")
        else:
            st.caption("Submit an enterprise scenario to unlock knowledge AI analysis.")

    st.markdown("**Executive Recommendation**")
    if executive:
        narrative = executive.get("narrative") or executive.get("overall_risk_view", "")
        if narrative:
            st.markdown(narrative)
        if executive.get("recommended_actions"):
            st.markdown("**Actions:**")
            for action in executive["recommended_actions"][:4]:
                st.markdown(f"- {action}")
    elif insights.get("risk_reasoning"):
        rr = insights["risk_reasoning"]
        st.markdown(rr.get("overall_view", ""))
    else:
        st.caption("AI executive summary unavailable.")

    if hedge:
        det_ratio = metrics.get("recommended_hedge_ratio")
        det_label = f"{float(det_ratio):.0%}" if det_ratio is not None else "N/A"
        st.markdown("**AI Hedge Recommendation**")
        st.markdown(
            f"- **AI suggested ratio:** {hedge.get('recommended_hedge_ratio', 'N/A')} · "
            f"**Deterministic ratio (authoritative):** {det_label}"
        )
        if hedge.get("preferred_instruments"):
            st.markdown("- **Instruments:** " + ", ".join(hedge["preferred_instruments"]))
        st.markdown(f"- **Tenor:** {hedge.get('hedge_tenor', 'N/A')}")
        st.markdown(f"- **Rationale:** {hedge.get('rationale', 'N/A')}")
        if hedge.get("risks"):
            st.markdown(f"- **Risks:** {hedge['risks']}")

    errors = insights.get("errors", [])
    if errors:
        st.caption(f"Partial AI response — unavailable sections: {', '.join(errors)}")


def render_ai_report_controls(metrics: dict, selected_pair: str) -> None:
    """Generate AI board report and export as markdown."""
    llm_config = init_llm_config()
    if not llm_config.enabled:
        return

    exhausted = _ai_calls_blocked()
    st.markdown("#### AI Board Report")
    c1, c2 = st.columns([1, 2])
    with c1:
        run_ai = st.button(
            "Generate AI Report",
            type="primary",
            key="generate_ai_report",
            disabled=exhausted,
        )
        refresh_ai = st.button(
            "Refresh AI Analysis",
            key="refresh_ai_analysis",
            disabled=exhausted,
        )
    with c2:
        remaining = _ai_budget_remaining()
        st.caption(
            f"Produces Executive Summary, Market/Macro/Policy/Knowledge outlook, "
            f"hedge strategy, and 30-day watchlist. "
            f"AI calls remaining: **{remaining}/{MAX_AI_CALLS_PER_SESSION}** (1h cache)."
        )
        if _ai_quota_exhausted():
            render_openai_api_error_banner()
            st.caption(
                "Last API call reported quota/billing exhaustion. "
                "After topping up, click **Refresh AI Analysis** to retry — no need to re-enter your API key."
            )
        elif exhausted:
            st.warning(
                f"AI call limit reached ({MAX_AI_CALLS_PER_SESSION}/session). "
                "Buttons disabled to protect against runaway OpenAI costs."
            )

    insights = st.session_state.get("ai_insights_by_pair", {}).get(selected_pair, {}).get(
        "insights"
    )

    if refresh_ai or run_ai:
        insights = _run_and_cache_ai_insights(metrics, selected_pair, force=True)
        if refresh_ai and not run_ai:
            st.rerun()

    if run_ai and not insights:
        render_openai_api_error_banner()
        if not st.session_state.get("openai_api_error"):
            st.warning(
                "AI report could not be generated — OpenAI returned no insights. "
                "Check billing/quota, then try **Refresh AI Analysis** first."
            )

    if run_ai and insights:
        llm = get_llm_service()
        report_context = {
            "pair": selected_pair,
            "enterprise_scenario": metrics.get("enterprise_scenario", ""),
            "deterministic_scores": {
                "market_score": metrics.get("market_score"),
                "macro_score": metrics.get("macro_score"),
                "policy_score": metrics.get("policy_score"),
                "knowledge_score": metrics.get("knowledge_score"),
                "final_risk_score": metrics.get("final_risk_score"),
                "recommended_hedge_ratio": metrics.get("recommended_hedge_ratio"),
            },
            "market_intelligence": insights.get("market_intelligence"),
            "macro_intelligence": insights.get("macro_intelligence"),
            "policy_intelligence": insights.get("policy_intelligence"),
            "knowledge_intelligence": insights.get("knowledge_intelligence"),
            "hedge_strategy": insights.get("hedge_strategy"),
            "executive_summary": insights.get("executive_summary"),
            "enterprise_intelligence": insights.get("enterprise_intelligence"),
            "hedge_ratio_explanation": insights.get("hedge_ratio_explanation"),
        }
        markdown_report = llm.generate_board_report(report_context)
        if markdown_report:
            st.session_state[f"ai_board_report_{selected_pair}"] = markdown_report
        else:
            st.warning("AI report generation failed — deterministic desk remains fully available.")

    report_md = st.session_state.get(f"ai_board_report_{selected_pair}")
    if report_md:
        st.download_button(
            label="Download AI Report (Markdown)",
            data=report_md,
            file_name=f"nova_ai_report_{selected_pair.replace('/', '_')}.md",
            mime="text/markdown",
        )
        with st.expander("Preview AI Report", expanded=False):
            st.markdown(report_md)


def _submit_copilot_message(
    prompt: str,
    selected_pair: str,
    desk_metrics: dict,
    messages: list[dict],
) -> None:
    """Inject macro + live desk context, run RAG retrieval, and append chat turns."""
    safe_prompt = sanitize_user_query(prompt)
    if not safe_prompt:
        return

    live_vars = _extract_live_desk_vars(desk_metrics)
    ai_enabled = init_llm_config().enabled
    if ai_enabled:
        _arm_api_quota_retry()
    spinner_msg = (
        "Running enterprise assessment + LLM layers (knowledge, copilot, risk narrative)… "
        "This may take 30–90s on slow networks."
        if ai_enabled
        else "Running enterprise assessment + knowledge retrieval…"
    )

    with st.spinner(spinner_msg):
        prompt_bundle = generate_dynamic_agent_prompt(safe_prompt, metrics=desk_metrics)

        messages.append(
            {
                "role": "user",
                "content": safe_prompt,
                "agent_prompt": prompt_bundle["agent_prompt"],
                "macro_layer": prompt_bundle.get("macro_layer", ""),
                "live_context": prompt_bundle.get("live_desk_layer", ""),
                "desk_snapshot": live_vars,
                "china_money": prompt_bundle.get("china_money"),
                "tnx_proxy": prompt_bundle.get("tnx_proxy"),
            }
        )

        enterprise_metrics = apply_enterprise_risk_assessment(
            desk_metrics,
            safe_prompt,
            imf_indicators=prompt_bundle.get("imf_macro_indicators")
            if isinstance(prompt_bundle.get("imf_macro_indicators"), dict)
            else None,
            china_money=prompt_bundle.get("china_money"),
            tnx_proxy=prompt_bundle.get("tnx_proxy"),
            api_key=st.session_state.get("openai_api_key"),
            budget_gate=_try_consume_ai_budget,
        )

        rag_parts = fx_rag_build_response_parts(
            user_query=safe_prompt,
            metrics=enterprise_metrics,
            pair=selected_pair,
            live_context=prompt_bundle.get("live_desk_layer"),
            prompt_bundle=prompt_bundle,
        )

        copilot_context = build_copilot_llm_context(
            user_query=safe_prompt,
            rag_parts=rag_parts,
            metrics=enterprise_metrics,
            prompt_bundle=prompt_bundle,
        )
        nova_copilot = generate_nova_response(
            response_type="copilot",
            api_key=st.session_state.get("openai_api_key"),
            context=copilot_context,
            local_builder=lambda: dict(rag_parts),
            budget_gate=_try_consume_ai_budget,
        )
        if nova_copilot.get("mode") != "ai_enhanced" and ai_enabled:
            render_openai_api_error_banner()
            if not st.session_state.get("openai_api_error"):
                st.caption(
                    "LLM narrative unavailable — showing rule-based desk answer. "
                    "Slow local mode is often caused by HuggingFace embedding downloads "
                    "or SSL blocks to the LLM API (not Alpha Vantage spot data)."
                )

        rag_parts.update(
            {
                "nova_mode": nova_copilot.get("mode", "local"),
                "ai_narrative": nova_copilot.get("ai_narrative"),
                "ai_knowledge_reasoning": nova_copilot.get("ai_knowledge_reasoning"),
            }
        )

        risk_context = build_risk_explanation_llm_context(enterprise_metrics)
        nova_risk = generate_nova_response(
            response_type="risk_explanation",
            api_key=st.session_state.get("openai_api_key"),
            context=risk_context,
            local_builder=lambda: {
                "risk_score_explanation": enterprise_metrics.get("risk_score_explanation", "")
            },
            budget_gate=_try_consume_ai_budget,
        )
        if nova_risk.get("ai_risk_narrative"):
            rag_parts["ai_risk_narrative"] = nova_risk["ai_risk_narrative"]
            enterprise_metrics["ai_risk_narrative"] = nova_risk["ai_risk_narrative"]

    if "enterprise_assessment_by_pair" not in st.session_state:
        st.session_state.enterprise_assessment_by_pair = {}
    if "ai_insights_by_pair" in st.session_state:
        st.session_state.ai_insights_by_pair.pop(selected_pair, None)
    st.session_state.enterprise_assessment_by_pair[selected_pair] = {
        "ready": True,
        "scenario": safe_prompt,
        "assessment": {
            key: enterprise_metrics.get(key)
            for key in (
                "knowledge_score",
                "rule_based_knowledge_score",
                "llm_knowledge_adjustment",
                "ai_knowledge_assessment",
                "case_interpretations",
                "hedge_strategy_review",
                "treasury_copilot_report",
                "documents_reranked",
                "final_risk_score",
                "multi_factor_risk_rating",
                "recommended_hedge_ratio",
                "knowledge_risk_level",
                "rule_based_knowledge_risk_level",
                "risk_contributions",
                "knowledge_breakdown",
                "risk_score_explanation",
                "ai_risk_narrative",
                "enterprise_assessment_ready",
                "enterprise_scenario",
            )
        },
    }
    st.session_state.active_desk_metrics = enterprise_metrics
    if "live_desk_by_pair" not in st.session_state:
        st.session_state.live_desk_by_pair = {}
    st.session_state.live_desk_by_pair[selected_pair] = enterprise_metrics

    messages.append(
        {
            "role": "assistant",
            "content": "",
            "rag_parts": rag_parts,
            "desk_snapshot": live_vars,
        }
    )
    st.session_state.messages_by_pair[selected_pair] = messages


def render_data_source_badge(metrics: dict, df) -> None:
    source = metrics.get("data_source") or df.attrs.get("data_source", "unknown")
    ticker = metrics.get("ticker") or df.attrs.get("ticker", "")
    if source == "Alpha Vantage":
        st.caption(
            f"🟢 Data Source: Alpha Vantage (Live) · `{ticker}` · quote **{DESK_QUOTE_LABEL}**"
        )
    elif source == "yfinance":
        st.caption(
            f"🟢 Data Source: Yahoo Finance (Live) · `{ticker}` · quote **{DESK_QUOTE_LABEL}**"
        )
    else:
        st.caption("🟡 Data Source: Cached Local Fallback (Network Timeout)")


def _imf_metric_value(block: dict[str, Any], country: str) -> str:
    """Format a single IMF CPI/GDP reading with defensive lookup."""
    raw = block.get(country)
    if raw is None:
        return "N/A"
    try:
        return f"{float(raw):.2f}"
    except (TypeError, ValueError):
        return str(raw)


def build_live_desk_metrics_table(metrics: dict) -> pd.DataFrame:
    """On-demand live desk × multi-factor matrix for sidebar inspection."""
    rows: list[dict[str, str]] = [
        {"Metric": "Trading Pair", "Value": metrics.get("pair", "N/A")},
        {"Metric": "Spot", "Value": f"{float(metrics.get('spot', 0.0)):.4f}"},
        {"Metric": "MA5", "Value": f"{float(metrics.get('ma5', 0.0)):.4f}"},
        {"Metric": "MA30", "Value": f"{float(metrics.get('ma30', 0.0)):.4f}"},
        {"Metric": "Bollinger Upper", "Value": f"{float(metrics.get('bollinger_upper', 0.0)):.4f}"},
        {"Metric": "Legacy Market Factor (0–1)", "Value": f"{float(metrics.get('multi_factor_score', 0.0)):.3f}"},
        {"Metric": "Market Risk Score", "Value": f"{float(metrics.get('market_score', 0.0)):.1f}"},
        {"Metric": "Macro Risk Score", "Value": f"{float(metrics.get('macro_score', 0.0)):.1f}"},
        {"Metric": "Policy Risk Score", "Value": f"{float(metrics.get('policy_score', 0.0)):.1f}"},
    ]
    if metrics.get("enterprise_assessment_ready"):
        rows.extend(
            [
                {
                    "Metric": "Knowledge Risk Score",
                    "Value": f"{float(metrics.get('knowledge_score', 0.0)):.1f}",
                },
                {
                    "Metric": "Final Composite Score",
                    "Value": f"{float(metrics.get('final_risk_score', 0.0)):.1f} / 100",
                },
                {
                    "Metric": "Enterprise Risk Rating",
                    "Value": metrics.get("multi_factor_risk_rating", "N/A"),
                },
                {
                    "Metric": "Hedge Recommendation",
                    "Value": (
                        f"{float(resolve_display_hedge_ratio(metrics, get_applied_risk_mapping()) or 0.0):.1%}"
                        if metrics.get("recommended_hedge_ratio") is not None
                        or metrics.get("final_risk_score") is not None
                        else "N/A"
                    ),
                },
            ]
        )
    else:
        rows.append(
            {
                "Metric": "Enterprise Assessment",
                "Value": "Waiting for enterprise profile input",
            }
        )
    rows.extend(
        [
            {"Metric": "VAT Net Payout RSI", "Value": f"{float(metrics.get('net_payout_rsi', 50.0)):.2f}"},
            {"Metric": "MACD Histogram", "Value": f"{float(metrics.get('net_payout_macd_histogram', 0.0)):.6f}"},
            {"Metric": "Applied VAT Rate", "Value": f"{float(metrics.get('vat_deduction_rate', 0.0)) * 100:.1f}%"},
            {"Metric": "Gross Sales Base", "Value": f"{float(metrics.get('gross_sales_base', 0.0)):,.2f}"},
        ]
    )
    return pd.DataFrame(rows)


def render_institutional_data_ingestion_sidebar(
    selected_pair: str,
    metrics: dict,
) -> None:
    """On-demand macro / ingestion inspect — does not touch chat session state."""
    if "intel_panel" not in st.session_state:
        st.session_state.intel_panel = None

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Institutional Data & Ingestion Inspect")

    if st.sidebar.button(
        "🌐 IMF Long-Term Macro Anchor (June 2026)",
        use_container_width=True,
        key="intel_btn_imf",
    ):
        st.session_state.intel_panel = "imf"
    if st.sidebar.button(
        "🏛️ China Monetary Policy Boundary (CFETS)",
        use_container_width=True,
        key="intel_btn_cfets",
    ):
        st.session_state.intel_panel = "cfets"
    if st.sidebar.button(
        "📈 CME FedWatch Proxy & US 10Y Treasury",
        use_container_width=True,
        key="intel_btn_fedwatch",
    ):
        st.session_state.intel_panel = "fedwatch"
    if st.sidebar.button(
        "🔬 Live Desk Snapshot & Macro×Multi-Factor Synthesis",
        use_container_width=True,
        key="intel_btn_live_desk",
    ):
        st.session_state.intel_panel = "live_desk"

    panel = st.session_state.get("intel_panel")
    if panel == "imf":
        entry = get_imf_macro_knowledge_entry()
        indicators = (entry or {}).get("macro_indicators", {})
        spreads = compute_imf_ppp_spreads(selected_pair, indicators)
        st.sidebar.info("### IMF Macro Anchor Insights")
        cpi = indicators.get("inflation_cpi_yoy", {}) if isinstance(indicators, dict) else {}
        gdp = indicators.get("gdp_growth_forecast", {}) if isinstance(indicators, dict) else {}
        cofer = (
            indicators.get("cofer_sovereign_reserves_share", {}) if isinstance(indicators, dict) else {}
        )

        dest = PAIR_TO_COUNTRY.get(selected_pair, "US")
        base_ccy = (selected_pair.split("/", 1)[0] if "/" in selected_pair else "USD").strip().upper()
        reserve_ccy = "USD" if base_ccy == "HKD" else base_ccy

        def _pct(val: Any) -> str:
            try:
                return f"{float(val):.2f}%"
            except (TypeError, ValueError):
                return "N/A"

        def _pp(val: Any) -> str:
            try:
                return f"{float(val):+.2f}pp"
            except (TypeError, ValueError):
                return "N/A"

        us_cpi = cpi.get(dest)
        cn_cpi = cpi.get(CHINA_MACRO_ANCHOR, cpi.get("China"))
        cpi_spread = spreads.get("cpi_spread_vs_china")

        dest_gdp = gdp.get(dest)
        cn_gdp = gdp.get(CHINA_MACRO_ANCHOR, gdp.get("China"))
        try:
            gdp_diff = (
                float(dest_gdp) - float(cn_gdp)
                if dest_gdp is not None and cn_gdp is not None
                else None
            )
        except (TypeError, ValueError):
            gdp_diff = None

        reserve_share = cofer.get(reserve_ccy)

        with st.sidebar.container():
            st.sidebar.markdown(f"- **{dest} CPI**: {_pct(us_cpi)}")
            st.sidebar.markdown(f"- **{CHINA_MACRO_ANCHOR} CPI**: {_pct(cn_cpi)}")
            st.sidebar.metric("CPI Spread (PPP)", _pp(cpi_spread))

            st.sidebar.markdown("")
            st.sidebar.markdown(f"- **{dest} GDP Forecast**: {_pct(dest_gdp)}")
            st.sidebar.markdown(f"- **{CHINA_MACRO_ANCHOR} GDP Forecast**: {_pct(cn_gdp)}")
            st.sidebar.metric(
                "GDP Growth Differential",
                _pp(gdp_diff) if gdp_diff is not None else "N/A",
            )

            st.sidebar.markdown("")
            st.sidebar.metric(
                f"{reserve_ccy} Reserve Share (COFER)",
                _pct(reserve_share),
            )

            st.sidebar.markdown("---")
            st.sidebar.markdown("**Executive Interpretation:**")

            inflation_line = (
                f"Higher {dest} inflation than {CHINA_MACRO_ANCHOR} implies mild long-term "
                f"{reserve_ccy} purchasing-power erosion under PPP."
                if isinstance(cpi_spread, (int, float)) and float(cpi_spread) > 0
                else (
                    f"{dest} inflation is not materially above {CHINA_MACRO_ANCHOR}; PPP pressure is muted."
                )
            )
            growth_line = (
                f"{CHINA_MACRO_ANCHOR} growth is structurally stronger, favoring a gradual long-cycle "
                f"re-anchoring toward CNH on 90–180 day horizons."
                if gdp_diff is not None and gdp_diff < 0
                else f"Growth differentials are balanced; macro carry pressure is moderate."
            )
            reserve_line = (
                f"{reserve_ccy} retains structural support from reserve-currency dominance (COFER)."
                if reserve_share is not None
                else "Reserve-currency support is stable; maintain disciplined hedge governance."
            )

            if selected_pair == "USD/CNH":
                stance = "Maintain **moderate hedge coverage (55–70%)** for 90–180 day receivables via layered window forwards."
            elif selected_pair in ("EUR/CNH", "GBP/CNH"):
                stance = "Maintain **moderate hedge coverage (50–65%)** for 90–180 day receivables; ladder forwards to avoid timing risk."
            else:
                stance = "Maintain **baseline hedge coverage (40–55%)** and rebalance monthly as the policy corridor evolves."

            st.sidebar.markdown(
                f"{inflation_line} {reserve_line} {growth_line}\n\n**Recommended long-term hedge stance:** {stance}"
            )
    elif panel == "cfets":
        china_money = fetch_china_money_central_parity()
        st.sidebar.info("### CFETS Central Parity Limits")
        if china_money.get("status") == "ok":
            st.sidebar.metric(
                label="USD/CNY Official Parity (CFETS)",
                value=f"{float(china_money.get('usd_cny', 0.0)):.4f}",
            )
            st.sidebar.caption(
                f"Parity date: {china_money.get('parity_date', 'N/A')} · "
                f"{china_money.get('policy_note', '')}"
            )
        else:
            st.sidebar.metric(
                label="CFETS Anchor Level",
                value="Dynamic Fixing Rule Active",
            )
            st.sidebar.caption("Live China Money feed unavailable — showing policy placeholder.")
    elif panel == "fedwatch":
        tnx_proxy = fetch_tnx_fedwatch_proxy()
        st.sidebar.info("### Yield Curve Analytics (Alpha Vantage / ^TNX Ingestion)")
        if tnx_proxy.get("status") == "ok":
            st.sidebar.metric(
                label="US 10Y Treasury Yield",
                value=f"{float(tnx_proxy.get('current_yield', 0.0)):.4f}%",
            )
            st.sidebar.metric(
                label="FedWatch Cut Probability (proxy)",
                value=f"{float(tnx_proxy.get('rate_cut_probability', 0.0)):.0%}",
            )
            st.sidebar.caption(tnx_proxy.get("signal_label", ""))
        else:
            st.sidebar.metric(label="US 10Y Treasury Yield", value="TREASURY_YIELD_10Y")
            st.sidebar.caption("Live yield ingestion unavailable — structural simulator fallback.")
    elif panel == "live_desk":
        st.sidebar.info("### Automated Micro & Macro Matrix Synthesis")
        st.sidebar.dataframe(build_live_desk_metrics_table(metrics), use_container_width=True)


def render_pair_imf_macro_panel(selected_pair: str) -> None:
    """Desk macro layer: active region vs China anchor (IMF 2026 WEO baseline)."""
    entry = get_imf_macro_knowledge_entry()
    if entry is None:
        return

    indicators = entry.get("macro_indicators")
    if not isinstance(indicators, dict):
        return

    cpi = indicators.get("inflation_cpi_yoy", {})
    gdp = indicators.get("gdp_growth_forecast", {})
    if not isinstance(cpi, dict) or not isinstance(gdp, dict):
        return

    country = PAIR_TO_COUNTRY.get(selected_pair, "US")
    as_of = str(indicators.get("as_of_date", "2026-06"))
    spreads = compute_imf_ppp_spreads(selected_pair, indicators)

    region_cpi = _imf_metric_value(cpi, country)
    china_cpi = _imf_metric_value(cpi, CHINA_MACRO_ANCHOR)
    region_gdp = _imf_metric_value(gdp, country)
    china_gdp = _imf_metric_value(gdp, CHINA_MACRO_ANCHOR)
    cpi_spread = spreads.get("cpi_spread_vs_china")

    with st.expander(
        f"📊 IMF 2026 Macro Anchor · {country} vs China (PPP desk overlay)",
        expanded=True,
    ):
        st.caption(
            f"Forward-looking IMF WEO baseline · As of {as_of} · "
            f"Active desk **{selected_pair}** maps to **{country}**"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            f"{country} CPI YoY (2026)",
            f"{region_cpi}%",
            delta=f"vs China {china_cpi}%",
            delta_color="inverse",
        )
        m2.metric(
            f"China CPI YoY ({MACRO_YUAN_LABEL} anchor)",
            f"{china_cpi}%",
            help=(
                f"Home-jurisdiction PPP anchor ({CHINA_MACRO_ANCHOR}) vs desk "
                f"{DESK_QUOTE_LABEL} trading layer."
            ),
        )
        m3.metric(
            f"{country} GDP Forecast (2026)",
            f"{region_gdp}%",
            delta=f"vs China {china_gdp}%",
        )
        m4.metric(
            "PPP CPI Spread vs China",
            f"{cpi_spread:+.2f} pp" if cpi_spread is not None else "N/A",
            help="Destination inflation minus China — inputs multi-factor long-cycle hedge sleeve.",
        )

        imf_block = format_imf_macro_indicators_block(indicators)
        if imf_block:
            st.markdown("**[REAL-TIME MACRO INTELLIGENCE LAYER]**")
            st.markdown(imf_block)


def render_multifactor_risk_breakdown(metrics: dict, selected_pair: str) -> None:
    """Environment monitoring before user input; enterprise assessment after."""
    _reset_risk_preference_panel_guard()
    enterprise_ready = bool(metrics.get("enterprise_assessment_ready"))

    if enterprise_ready:
        st.markdown("#### Enterprise Risk Assessment")
    else:
        st.markdown("#### Environment Monitoring")

    cached_insights = (
        st.session_state.get("ai_insights_by_pair", {}).get(selected_pair, {}).get("insights")
        if init_llm_config().enabled
        else None
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Risk", f"{float(metrics.get('market_score', 0.0)):.1f}")
    c2.metric("Macro Risk", f"{float(metrics.get('macro_score', 0.0)):.1f}")
    c3.metric("Policy Risk", f"{float(metrics.get('policy_score', 0.0)):.1f}")
    if cached_insights:
        market_ai = cached_insights.get("market_intelligence") or {}
        macro_ai = cached_insights.get("macro_intelligence") or {}
        policy_ai = cached_insights.get("policy_intelligence") or {}
        if market_ai.get("regime"):
            c1.caption(f"🧠 AI regime: **{market_ai['regime']}**")
        if macro_ai.get("macro_regime"):
            c2.caption(f"🧠 AI macro: **{macro_ai['macro_regime']}**")
        if policy_ai.get("policy_stance"):
            c3.caption(f"🧠 AI policy: **{policy_ai['policy_stance']}**")

    if not enterprise_ready:
        st.info(
            "Provide your FX exposure scenario to generate an enterprise risk assessment. "
            "Use the copilot below to describe receivables, payables, hedge status, or treasury context."
        )
        contributions = metrics.get("risk_contributions", {})
        st.caption(
            "Active environment layers · "
            f"**35% Market** ({contributions.get('market', 0):.1f} pts) · "
            f"**25% Macro** ({contributions.get('macro', 0):.1f} pts) · "
            f"**25% Policy** ({contributions.get('policy', 0):.1f} pts) · "
            "**15% Knowledge** (pending user scenario)"
        )
        insights = (
            st.session_state.get("ai_insights_by_pair", {})
            .get(selected_pair, {})
            .get("insights")
            if init_llm_config().enabled
            else None
        )
        render_ai_enhanced_analysis(metrics, selected_pair, insights=insights)
        if init_llm_config().enabled:
            render_ai_report_controls(metrics, selected_pair)
        return

    c4, f1, f2 = st.columns(3)
    rule_kb = metrics.get("rule_based_knowledge_score")
    final_kb = float(metrics.get("knowledge_score", 0.0))
    llm_adj = metrics.get("llm_knowledge_adjustment")
    if rule_kb is not None and llm_adj not in (None, 0, 0.0):
        c4.metric(
            "Knowledge Risk",
            f"{final_kb:.1f}",
            delta=f"{float(llm_adj):+.1f} AI adj.",
            help="Final knowledge score after optional LLM adjustment (rule-based shown below).",
        )
        c4.caption(f"Rule-based: **{float(rule_kb):.1f}**")
    else:
        c4.metric("Knowledge Risk", f"{final_kb:.1f}")
    f1.metric(
        "Final Score",
        f"{float(metrics.get('final_risk_score', 0.0)):.1f} / 100",
    )
    f2.metric(
        "Risk Rating",
        metrics.get("multi_factor_risk_rating", "MEDIUM"),
    )

    render_hedge_recommendation_block(metrics)

    scenario = metrics.get("enterprise_scenario")
    if scenario:
        st.caption(f"Enterprise scenario: *{scenario}*")

    contributions = metrics.get("risk_contributions", {})
    st.caption(
        "Contribution mix · "
        f"**35% Market** ({contributions.get('market', 0):.1f} pts) · "
        f"**25% Macro** ({contributions.get('macro', 0):.1f} pts) · "
        f"**25% Policy** ({contributions.get('policy', 0):.1f} pts) · "
        f"**15% Knowledge** ({contributions.get('knowledge', 0):.1f} pts)"
    )

    explanation = metrics.get("risk_score_explanation", "")
    ai_risk_narrative = metrics.get("ai_risk_narrative", "")
    if explanation or ai_risk_narrative:
        with st.expander("Why this score was generated", expanded=False):
            if explanation:
                st.markdown(explanation)
            if ai_risk_narrative:
                st.markdown("**🧠 AI Institutional Narrative**")
                st.markdown(ai_risk_narrative)

    render_ai_knowledge_intelligence(metrics)

    insights = (
        st.session_state.get("ai_insights_by_pair", {})
        .get(selected_pair, {})
        .get("insights")
        if init_llm_config().enabled
        else None
    )
    render_ai_enhanced_analysis(metrics, selected_pair, insights=insights)
    if init_llm_config().enabled:
        render_ai_report_controls(metrics, selected_pair)


def render_nova_risk_advisory(metrics: dict, selected_pair: str) -> None:
    """Executive NOVA Risk Advisory — VAT net payout RSI/MACD saturation overlay."""
    pair = metrics.get("pair", selected_pair)
    advisory = build_nova_risk_advisory(
        pair,
        float(metrics.get("net_payout_rsi", 50.0)),
        float(metrics.get("net_payout_macd_histogram", 0.0)),
        vat_rate=metrics.get("vat_deduction_rate"),
        gross_sales_base=metrics.get("gross_sales_base"),
        macd_line=float(metrics.get("net_payout_macd_line", 0.0)),
        macd_signal=float(metrics.get("net_payout_macd_signal", 0.0)),
    )
    signal_class = advisory.get(
        "signal_class", metrics.get("nova_risk_signal_class", "normal")
    )
    text = advisory.get("text", metrics.get("nova_risk_advisory", ""))
    if not text:
        return

    st.markdown("#### NOVA Risk Advisory · VAT Net Payout Technical Overlay")
    if signal_class == "risk_alert":
        st.error(text)
    elif signal_class == "opportunity":
        st.success(text)
    else:
        st.info(text)


def render_live_desk_factor_strip(live_vars: dict) -> None:
    """Compact multi-factor readout above the copilot chat."""
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Spot", f"{live_vars['current_spot']:.4f}")
    r1c2.metric("MA5", f"{live_vars['ma5_val']:.4f}")
    r1c3.metric("MA30", f"{live_vars['ma30_val']:.4f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Bollinger ↑", f"{live_vars['bollinger_upper']:.4f}")
    if live_vars.get("enterprise_assessment_ready"):
        r2c2.metric("Final Score", f"{live_vars.get('final_risk_score', 0.0):.1f}")
        r2c3.metric("Hedge Ratio", f"{live_vars['recommended_hedge_ratio']:.1%}")
    else:
        r2c2.metric("Assessment", "Pending")
        r2c3.metric("Hedge Ratio", "—")
    if live_vars.get("net_payout_rsi") is not None:
        st.caption(
            f"VAT Net Payout RSI: **{live_vars['net_payout_rsi']:.2f}** · "
            f"MACD Hist: **{live_vars.get('net_payout_macd_histogram', 0.0):.4f}**"
        )


def render_live_risk_desk(selected_pair: str, metrics: dict, df) -> None:
    st.markdown(
        '<p class="bloomberg-header">🌐 Live Risk Desk</p>',
        unsafe_allow_html=True,
    )
    if init_llm_config().enabled and st.session_state.get("openai_api_error"):
        render_openai_api_error_banner()
    render_data_source_badge(metrics, df)

    c1, c2, c3, c4 = st.columns([1.05, 1.05, 1.05, 1.35])
    day_display, day_full = format_pct_change(metrics["pct_change"])
    render_desk_metric(c1, f"{selected_pair} Spot", f"{metrics['spot']:.4f}")
    render_desk_metric(c2, "Day Change", day_display, tooltip=day_full)
    render_desk_metric(c3, "MA5", f"{metrics['ma5']:.4f}")
    desk_mode = "Enterprise" if metrics.get("enterprise_assessment_ready") else "Monitoring"
    with c4:
        st.caption("Desk Mode")
        st.markdown(
            f'<p style="font-size:1.75rem;font-weight:600;margin:0;line-height:1.2;">'
            f"{desk_mode}</p>",
            unsafe_allow_html=True,
        )

    render_multifactor_risk_breakdown(metrics, selected_pair)
    render_nova_risk_advisory(metrics, selected_pair)

    st.plotly_chart(build_fx_chart(df, selected_pair), width="stretch")

    st.markdown("### Contract & Invoice Audit Agent")
    st.caption(
        f"Upload a sample contract or invoice—demo extracts {metrics['base_ccy']} exposure "
        f"and a recommended forward hedge ratio for **{selected_pair}**."
    )
    uploaded = st.file_uploader(
        "Upload contract or invoice (XLSX, PDF, CSV, TXT)",
        type=["xlsx", "xls", "pdf", "csv", "txt"],
        help="Deterministic demo parser for competition judging.",
    )
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        result = audit_contract_file(uploaded.name, file_bytes, pair=selected_pair)
        st.success(result["summary"])
        a, b, c, d = st.columns(4)
        sym = metrics["symbol"]
        a.metric(f"{metrics['base_ccy']} Exposure", f"{sym}{result['notional']:,.0f}")
        b.metric("Settlement Horizon", f"{result['expected_settlement_days']} days")
        c.metric("Gross Margin", f"{result['gross_margin_pct']:.0f}%")
        d.metric("Rec. Forward Hedge", f"{result['recommended_forward_ratio']:.0%}")

        doc_text = read_upload_document_text(uploaded.name, file_bytes)
        nova_audit = generate_nova_response(
            response_type="contract_audit",
            api_key=st.session_state.get("openai_api_key"),
            context=build_contract_audit_llm_context(
                result,
                pair=selected_pair,
                document_text=doc_text,
            ),
            local_builder=lambda: {"audit": result},
            budget_gate=_try_consume_ai_budget,
        )
        if nova_audit.get("mode") == "ai_enhanced" and nova_audit.get("ai_contract_review"):
            review = nova_audit["ai_contract_review"]
            with st.expander("🧠 AI Contract Risk Review", expanded=False):
                if review.get("executive_summary"):
                    st.markdown(review["executive_summary"])
                if review.get("risk_review"):
                    st.markdown(f"**Risk review:** {review['risk_review']}")
                if review.get("settlement_analysis"):
                    st.markdown(f"**Settlement:** {review['settlement_analysis']}")
                if review.get("hedge_recommendations"):
                    st.markdown(
                        "**Hedge recommendations:** "
                        + "; ".join(review["hedge_recommendations"])
                    )
                if review.get("compliance_flags"):
                    st.markdown(
                        "**Compliance flags:** " + "; ".join(review["compliance_flags"])
                    )

    st.markdown("### Colloquial FX Risk Copilot")
    desk_metrics = _sync_live_desk_session(selected_pair, metrics)
    live_vars = _extract_live_desk_vars(desk_metrics)
    render_live_desk_factor_strip(live_vars)
    st.caption(
        f"Market-aware RAG · China Money parity + ^TNX Fed proxy + live {selected_pair} desk · "
        "state a payout (e.g. *1 million USD*) for an instant execution ticket."
    )

    _init_chat(selected_pair)
    messages = st.session_state.messages_by_pair[selected_pair]

    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("rag_parts"):
                render_assistant_rag_message(msg)
            else:
                st.markdown(msg.get("content", ""))

    prompt = st.chat_input(
        f"Ask about {selected_pair} — hedging, VAT, peg stress, payouts…"
    )
    if prompt:
        safe_prompt = sanitize_user_query(prompt)
        if not safe_prompt:
            st.warning(
                "Your question was empty or blocked after security sanitization "
                f"(max {_MAX_USER_QUERY_LEN} characters; system-override phrases removed)."
            )
        else:
            _submit_copilot_message(safe_prompt, selected_pair, desk_metrics, messages)
            st.rerun()


def render_strategy_backtest_tab() -> None:
    st.markdown(
        '<p class="bloomberg-header">🏛️ Portfolio Risk & Quant Performance Evaluation</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "MPT / FRM analytics desk · 180-day USD/CNH via Yahoo Finance (`USDCNH=X`) · "
        "Benchmark (zero hedge) vs NOVA multi-factor dynamic overlay · "
        "Volatility · 95% VaR · MDD · Sharpe (rf=2%)."
    )

    monthly_revenue = st.slider(
        "Monthly USD Revenue Exposure ($)",
        min_value=50_000,
        max_value=500_000,
        value=100_000,
        step=10_000,
    )

    analysis = load_backtest_analysis(monthly_revenue)
    source = analysis.get("data_source", "unknown")
    ticker = analysis.get("ticker", "USDCNH=X")
    obs_days = int(analysis.get("observation_days", 0))

    if source == "Alpha Vantage":
        st.success(
            f"Live market data · **{ticker}** · "
            f"{obs_days} trading days ingested from Alpha Vantage."
        )
    elif source == "yfinance":
        st.success(
            f"Live market data · **{ticker}** · "
            f"{obs_days} trading days ingested from Yahoo Finance."
        )
    else:
        st.warning(
            "Live market feeds unavailable — deterministic fallback simulator active "
            "(bounded pair-specific ranges). Analytics engine remains fully operational."
        )

    nova_total = _metric_value(analysis, "total_value", "nova")
    bench_sharpe = _metric_value(analysis, "sharpe_ratio", "benchmark")
    nova_sharpe = _metric_value(analysis, "sharpe_ratio", "nova")
    daily_usd = float(analysis.get("daily_usd_exposure", monthly_revenue / 30.0))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Backtest Horizon", f"{obs_days} Days")
    k2.metric("Daily USD Flow", f"${daily_usd:,.0f}")
    k3.metric(f"NOVA Total {OFFSHORE_QUOTE_CCY}", f"¥{nova_total:,.0f}")
    k4.metric("Sharpe Uplift", f"{nova_sharpe - bench_sharpe:+.3f}")

    st.plotly_chart(build_equity_curve_chart(analysis), width="stretch")

    st.markdown("#### Quantitative Performance Comparison")
    st.table(build_quant_metrics_table(analysis))

    with st.expander("Strategy specification (academic reference)", expanded=False):
        st.markdown(
            "**Data pipeline:** `fetch_real_market_data()` → Alpha Vantage FX_DAILY "
            f"for `{ticker}` (yfinance backup) → Spot + MA5; network-safe fallback if blocked.\n\n"
            "**Strategy A — Unhedged Spot Benchmark:** "
            f"Daily {OFFSHORE_QUOTE_CCY} = (USD {daily_usd:,.0f}) × Spot (USD/CNH desk).\n\n"
            "**Strategy B — NOVA Multi-Factor Dynamic Overlay:** "
            "Integrated risk score = 0.4×F₁ (Spot≥MA5) + 0.3×F₂ (Spot≥20d Bollinger upper) "
            "+ 0.3×F₃ (Spot≥MA30). "
            "Score ≥0.7 → 80% hedge; 0.4–0.7 → 55%; <0.4 → 30% baseline. "
            "Effective rate = (hedge × MA5) + ((1 − hedge) × Spot).\n\n"
            "**Metrics (from `metrics_comparison`):** "
            "Ann. Vol = σ(daily returns) × √365 · "
            f"VaR₉₅ = 5th percentile daily {OFFSHORE_QUOTE_CCY} cash flow · "
            f"MDD = peak-to-trough on cumulative {OFFSHORE_QUOTE_CCY} equity · "
            "Sharpe = (μ − rf/365) / σ × √365."
        )


def main():
    st.sidebar.markdown("## NOVA FX Copilot")
    st.sidebar.caption(
        f"Cross-Border SME · Desk quotes {DESK_QUOTE_LABEL} · Macro anchor {MACRO_YUAN_LABEL}"
    )

    selected_pair = st.sidebar.selectbox(
        "Select Currency Pair",
        SUPPORTED_PAIRS,
        index=0,
    )

    if st.session_state.get("active_pair") != selected_pair:
        st.session_state.active_pair = selected_pair

    st.sidebar.divider()
    st.sidebar.markdown("**VAT Net Payout Desk**")
    gross_sales_base = st.sidebar.number_input(
        "Store Daily Gross Revenue Base",
        min_value=0.0,
        value=10_000.0,
        step=500.0,
        format="%.2f",
    )
    custom_vat_rate: float | None = None
    if selected_pair == "EUR/CNH":
        custom_vat = st.sidebar.slider(
            "EU Member State VAT Override (%)",
            min_value=15,
            max_value=25,
            value=20,
        )
        custom_vat_rate = float(custom_vat) / 100.0
        st.sidebar.caption(
            f"Applied EUR VAT: **{custom_vat}%** · Net payout base scales with gross revenue "
            f"and {OFFSHORE_QUOTE_CCY} desk close."
        )

    if not FAISS_AVAILABLE:
        st.sidebar.info(
            "RAG: NumPy cosine + keyword routing (FAISS optional — install faiss-cpu for speed)."
        )
    if not REQUESTS_AVAILABLE:
        st.sidebar.caption("HTTP: urllib fallback (requests optional).")

    df = load_fx_data(selected_pair)
    metrics = get_live_desk_multi_factor_metrics(
        df,
        pair=selected_pair,
        gross_sales_base=gross_sales_base,
        custom_vat_rate=custom_vat_rate,
    )
    metrics = _merge_enterprise_assessment(metrics, selected_pair)
    if not df.empty:
        low_30d, high_30d = resolve_30d_range_from_kline(df)
        metrics["low_30d"] = low_30d
        metrics["high_30d"] = high_30d

    sidebar_card(f"{selected_pair} Spot", f"{metrics['spot']:.4f}")
    day_display, day_full = format_pct_change(metrics["pct_change"])
    sidebar_card("Day Change", day_display, tooltip=day_full)
    sidebar_card("MA5 Factor", f"{metrics['ma5']:.4f}", metrics["ma_signal"])
    sidebar_card(
        "30-Day Range",
        f"{metrics['low_30d']:.4f} – {metrics['high_30d']:.4f}",
    )
    render_traffic_light(metrics)

    st.sidebar.divider()
    st.sidebar.markdown("**Try asking (colloquial English)**")
    for p in PAIR_PROMPTS[selected_pair]:
        st.sidebar.markdown(f'- "{p}"')

    llm_config = render_llm_sidebar_config()
    render_institutional_data_ingestion_sidebar(selected_pair, metrics)

    st.markdown('<p class="hero-title">NOVA FX Risk Copilot</p>', unsafe_allow_html=True)
    mode_label = "🟢 AI Enhanced Mode" if llm_config.enabled else "⚪ Local Mode"
    st.markdown(
        f'<p class="hero-sub">HKUST NOVA AI Competition · Active desk: <b>{selected_pair}</b> · '
        f"USD / EUR / HKD / GBP offshore yuan risk for cross-border SMEs · <b>{mode_label}</b></p>",
        unsafe_allow_html=True,
    )

    tab_live, tab_backtest = st.tabs(["🌐 Live Risk Desk", "📊 Strategy Backtest"])

    with tab_live:
        render_live_risk_desk(selected_pair, metrics, df)

    with tab_backtest:
        render_strategy_backtest_tab()


if __name__ == "__main__":
    main()
