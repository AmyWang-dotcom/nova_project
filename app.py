"""
NOVA FX Risk Copilot — Streamlit Executive Dashboard
Multi-pair: USD/CNH · EUR/CNH · HKD/CNH · GBP/CNH
Run: streamlit run app.py
"""

from __future__ import annotations

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
    fetch_china_money_central_parity,
    fetch_tnx_fedwatch_proxy,
    get_live_desk_multi_factor_metrics,
    get_secure_alpha_vantage_key,
    get_secure_openai_key,
    sanitize_user_query,
    _MAX_USER_QUERY_LEN,
)

DESK_QUOTE_LABEL = f"{OFFSHORE_QUOTE_CCY} (Offshore Yuan)"
MACRO_YUAN_LABEL = f"{ONSHORE_QUOTE_CCY} (Onshore Yuan Anchor)"

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
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
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


@st.cache_data(show_spinner=False)
def load_fx_data(pair: str):
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


def sidebar_card(label: str, value: str, sub: str = ""):
    sub_html = (
        f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">{sub}</div>'
        if sub
        else ""
    )
    st.sidebar.markdown(
        f"""
        <div class="nova-card">
            <div class="nova-label">{label}</div>
            <div class="nova-value">{value}</div>
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
    st.markdown(parts.get("executive_action", parts.get("action", "")))

    if parts.get("execution_block"):
        with st.expander("⚡ Numeric execution ticket", expanded=False):
            st.success(parts["execution_block"])

    playbook_count = parts.get("playbook_count", len(parts.get("docs", [])))
    if playbook_count:
        with st.expander(
            f"🔍 Retrieved playbooks ({playbook_count}) — on-demand detail",
            expanded=False,
        ):
            for doc in parts.get("docs", []):
                label = f"[{doc['domain']}] {doc['title']}"
                with st.expander(label, expanded=False):
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
            desk_metrics.get("recommended_hedge_ratio", 0.55)
        ),
        "multi_factor_score": float(desk_metrics.get("multi_factor_score", 0.0)),
        "net_payout_rsi": float(desk_metrics.get("net_payout_rsi", 50.0)),
        "net_payout_macd_histogram": float(
            desk_metrics.get("net_payout_macd_histogram", 0.0)
        ),
        "gross_sales_base": float(desk_metrics.get("gross_sales_base", 0.0)),
        "vat_deduction_rate": float(desk_metrics.get("vat_deduction_rate", 0.0)),
    }


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

    rag_parts = fx_rag_build_response_parts(
        user_query=safe_prompt,
        metrics=desk_metrics,
        pair=selected_pair,
        live_context=prompt_bundle.get("live_desk_layer"),
        prompt_bundle=prompt_bundle,
    )

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
    return pd.DataFrame(
        [
            {"Metric": "Trading Pair", "Value": metrics.get("pair", "N/A")},
            {"Metric": "Spot", "Value": f"{float(metrics.get('spot', 0.0)):.4f}"},
            {"Metric": "MA5", "Value": f"{float(metrics.get('ma5', 0.0)):.4f}"},
            {"Metric": "MA30", "Value": f"{float(metrics.get('ma30', 0.0)):.4f}"},
            {"Metric": "Bollinger Upper", "Value": f"{float(metrics.get('bollinger_upper', 0.0)):.4f}"},
            {"Metric": "Multi-Factor Score", "Value": f"{float(metrics.get('multi_factor_score', 0.0)):.3f}"},
            {"Metric": "MF Risk Rating", "Value": metrics.get("multi_factor_risk_rating", "N/A")},
            {"Metric": "Hedge Ratio", "Value": f"{float(metrics.get('recommended_hedge_ratio', 0.0)):.1%}"},
            {"Metric": "VAT Net Payout RSI", "Value": f"{float(metrics.get('net_payout_rsi', 50.0)):.2f}"},
            {"Metric": "MACD Histogram", "Value": f"{float(metrics.get('net_payout_macd_histogram', 0.0)):.6f}"},
            {"Metric": "Applied VAT Rate", "Value": f"{float(metrics.get('vat_deduction_rate', 0.0)) * 100:.1f}%"},
            {"Metric": "Gross Sales Base", "Value": f"{float(metrics.get('gross_sales_base', 0.0)):,.2f}"},
        ]
    )


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
        st.sidebar.json(
            {
                "pair": selected_pair,
                "china_macro_anchor": CHINA_MACRO_ANCHOR,
                "macro_indicators": indicators,
                "ppp_spreads": spreads,
                "formatted_block": format_imf_macro_indicators_block(indicators),
            }
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
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.metric("Spot", f"{live_vars['current_spot']:.4f}")
    f2.metric("MA5", f"{live_vars['ma5_val']:.4f}")
    f3.metric("MA30", f"{live_vars['ma30_val']:.4f}")
    f4.metric("Bollinger ↑", f"{live_vars['bollinger_upper']:.4f}")
    f5.metric("MF Risk", live_vars["risk_level"])
    f6.metric("Hedge Ratio", f"{live_vars['recommended_hedge_ratio']:.1%}")
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
    render_data_source_badge(metrics, df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{selected_pair} Spot", f"{metrics['spot']:.4f}")
    c2.metric("Day Change", f"{metrics['pct_change']:+.3f}%")
    c3.metric("MA5", f"{metrics['ma5']:.4f}")
    c4.metric("Risk Level", metrics["risk_level"])

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
        result = audit_contract_file(uploaded.name, uploaded.getvalue(), pair=selected_pair)
        st.success(result["summary"])
        a, b, c, d = st.columns(4)
        sym = metrics["symbol"]
        a.metric(f"{metrics['base_ccy']} Exposure", f"{sym}{result['notional']:,.0f}")
        b.metric("Settlement Horizon", f"{result['expected_settlement_days']} days")
        c.metric("Gross Margin", f"{result['gross_margin_pct']:.0f}%")
        d.metric("Rec. Forward Hedge", f"{result['recommended_forward_ratio']:.0%}")

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


def _runtime_secrets_status() -> dict[str, bool]:
    """Non-cached probe of optional ingestion keys from st.secrets / environment."""
    return {
        "alpha_vantage": bool(get_secure_alpha_vantage_key()),
        "openai": bool(get_secure_openai_key()),
    }


def main():
    secrets_status = _runtime_secrets_status()

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

    if secrets_status["alpha_vantage"] or secrets_status["openai"]:
        loaded = [
            name
            for name, ok in (
                ("Alpha Vantage", secrets_status["alpha_vantage"]),
                ("OpenAI", secrets_status["openai"]),
            )
            if ok
        ]
        st.sidebar.success(
            f"Credentials loaded via st.secrets: {', '.join(loaded)}"
        )
    else:
        st.sidebar.success(
            "Local-first RAG · Add optional keys in `.streamlit/secrets.toml`"
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

    sidebar_card(f"{selected_pair} Spot", f"{metrics['spot']:.4f}")
    sidebar_card("Day Change", f"{metrics['pct_change']:+.3f}%")
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

    render_institutional_data_ingestion_sidebar(selected_pair, metrics)

    st.markdown('<p class="hero-title">NOVA FX Risk Copilot</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="hero-sub">HKUST NOVA AI Competition · Active desk: <b>{selected_pair}</b> · '
        "USD / EUR / HKD / GBP offshore yuan risk for cross-border SMEs · Zero API keys</p>",
        unsafe_allow_html=True,
    )

    tab_live, tab_backtest = st.tabs(["🌐 Live Risk Desk", "📊 Strategy Backtest"])

    with tab_live:
        render_live_risk_desk(selected_pair, metrics, df)

    with tab_backtest:
        render_strategy_backtest_tab()


if __name__ == "__main__":
    main()
