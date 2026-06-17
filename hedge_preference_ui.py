"""
Hedge Recommendation UI — tooltip, segmented bar, and risk preference mapping panel.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from risk_constants import (
    DEFAULT_MAPPING,
    HEDGE_RATIO_HELP_TEXT,
    SEGMENT_COLORS,
    format_mapping_rules,
    hedge_ratio_from_score,
    normalize_mapping,
    resolve_display_hedge_ratio,
)

_SESSION_APPLIED_KEY = "risk_preference_mapping_applied"
_SESSION_TEMP_KEY = "risk_preference_mapping_temp"
_rpm_panel_rendered = False


def init_risk_preference_state() -> None:
    if _SESSION_APPLIED_KEY not in st.session_state:
        st.session_state[_SESSION_APPLIED_KEY] = dict(DEFAULT_MAPPING)
    if _SESSION_TEMP_KEY not in st.session_state:
        st.session_state[_SESSION_TEMP_KEY] = dict(
            st.session_state[_SESSION_APPLIED_KEY]
        )


def get_applied_risk_mapping() -> dict[str, Any]:
    init_risk_preference_state()
    return normalize_mapping(st.session_state[_SESSION_APPLIED_KEY])


def get_temp_risk_mapping() -> dict[str, Any]:
    init_risk_preference_state()
    return normalize_mapping(st.session_state[_SESSION_TEMP_KEY])


def apply_temp_risk_mapping() -> None:
    init_risk_preference_state()
    st.session_state[_SESSION_APPLIED_KEY] = dict(
        normalize_mapping(st.session_state[_SESSION_TEMP_KEY])
    )


def reset_risk_preference_mapping() -> None:
    st.session_state[_SESSION_APPLIED_KEY] = dict(DEFAULT_MAPPING)
    st.session_state[_SESSION_TEMP_KEY] = dict(DEFAULT_MAPPING)


def build_segmented_score_bar(
    final_score: float,
    mapping: dict[str, Any],
    *,
    height: int = 96,
) -> go.Figure:
    """0–100 segmented progress bar with red / yellow / green tiers."""
    m = normalize_mapping(mapping)
    fig = go.Figure()

    segments = (
        (0, m["low_threshold"], SEGMENT_COLORS["low"], "Low"),
        (m["low_threshold"] + 0.5, m["high_threshold"] - 0.5, SEGMENT_COLORS["medium"], "Medium"),
        (m["high_threshold"], 100, SEGMENT_COLORS["high"], "High"),
    )
    for x0, x1, color, label in segments:
        if x1 < x0:
            continue
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0.25,
            y1=0.75,
            fillcolor=color,
            line=dict(color="rgba(15,23,42,0.5)", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.5,
            text=label,
            showarrow=False,
            font=dict(color="#0f172a", size=11, family="Arial Black"),
        )

    clamped = max(0.0, min(100.0, float(final_score)))
    fig.add_shape(
        type="line",
        x0=clamped,
        x1=clamped,
        y0=0.1,
        y1=0.9,
        line=dict(color="#f8fafc", width=3),
    )
    fig.add_annotation(
        x=clamped,
        y=1.05,
        text=f"Score {clamped:.0f}",
        showarrow=False,
        font=dict(color="#e2e8f0", size=12),
    )

    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        xaxis=dict(
            range=[0, 100],
            tickvals=[0, m["low_threshold"], m["high_threshold"], 100],
            title="Composite Score",
            gridcolor="rgba(148,163,184,0.15)",
        ),
        yaxis=dict(visible=False, range=[0, 1.15]),
        font=dict(color="#cbd5e1", size=12),
    )
    return fig


def reset_risk_preference_panel_guard() -> None:
    """Allow one mapping panel render per Streamlit script run."""
    global _rpm_panel_rendered
    _rpm_panel_rendered = False


def render_risk_preference_mapping_panel(
    metrics: dict[str, Any],
    *,
    final_score: float | None,
    key_prefix: str | None = None,
) -> None:
    """Collapsible advanced mapping — temp preview until Apply."""
    global _rpm_panel_rendered
    if _rpm_panel_rendered:
        return
    _rpm_panel_rendered = True

    prefix = key_prefix or str(metrics.get("pair", "desk")).replace("/", "_")
    init_risk_preference_state()
    applied = get_applied_risk_mapping()
    temp = get_temp_risk_mapping()
    pending = temp != applied

    with st.expander("⚙️ Risk Preference Mapping (Advanced)", expanded=False):
        st.caption(
            "Adjust score thresholds and hedge ratios. Changes preview instantly; "
            "click **Apply Mapping** to update desk recommendations."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Score Thresholds**")
            high_th = st.slider(
                "High-risk threshold (score ≥)",
                min_value=50,
                max_value=95,
                value=int(temp["high_threshold"]),
                key=f"{prefix}_rpm_high_threshold",
            )
            low_th = st.slider(
                "Low-risk ceiling (score ≤)",
                min_value=5,
                max_value=49,
                value=int(temp["low_threshold"]),
                key=f"{prefix}_rpm_low_threshold",
            )
        with col_b:
            st.markdown("**Hedge Ratios**")
            high_ratio_pct = st.slider(
                "High-tier hedge ratio",
                min_value=50,
                max_value=100,
                value=int(round(float(temp["high_ratio"]) * 100)),
                step=5,
                format="%d%%",
                key=f"{prefix}_rpm_high_ratio",
            )
            medium_ratio_pct = st.slider(
                "Medium-tier hedge ratio",
                min_value=20,
                max_value=90,
                value=int(round(float(temp["medium_ratio"]) * 100)),
                step=5,
                format="%d%%",
                key=f"{prefix}_rpm_medium_ratio",
            )
            low_ratio_pct = st.slider(
                "Low-tier hedge ratio",
                min_value=10,
                max_value=60,
                value=int(round(float(temp["low_ratio"]) * 100)),
                step=5,
                format="%d%%",
                key=f"{prefix}_rpm_low_ratio",
            )

        st.session_state[_SESSION_TEMP_KEY] = normalize_mapping(
            {
                "high_threshold": high_th,
                "low_threshold": low_th,
                "high_ratio": high_ratio_pct / 100.0,
                "medium_ratio": medium_ratio_pct / 100.0,
                "low_ratio": low_ratio_pct / 100.0,
            }
        )
        temp = get_temp_risk_mapping()

        st.markdown("**Current Mapping Preview**")
        for rule in format_mapping_rules(temp):
            st.markdown(f"- {rule}")

        if final_score is not None:
            preview_ratio = hedge_ratio_from_score(final_score, temp)
            st.info(
                f"With final score **{final_score:.1f}**, preview hedge ratio = "
                f"**{preview_ratio:.0%}**"
                + (" _(pending Apply)_" if pending else "")
            )
            st.plotly_chart(
                build_segmented_score_bar(final_score, temp),
                use_container_width=True,
                key=f"{prefix}_rpm_segmented_bar",
            )

        btn_apply, btn_reset = st.columns(2)
        with btn_apply:
            if st.button(
                "Apply Mapping",
                type="primary",
                use_container_width=True,
                disabled=not pending,
                key=f"{prefix}_rpm_apply",
            ):
                apply_temp_risk_mapping()
                st.rerun()
        with btn_reset:
            if st.button(
                "Reset to Defaults",
                use_container_width=True,
                key=f"{prefix}_rpm_reset",
            ):
                reset_risk_preference_mapping()
                st.rerun()


def render_hedge_recommendation_block(metrics: dict[str, Any]) -> None:
    """Hedge Recommendation metric and mapping rules caption."""
    init_risk_preference_state()
    applied = get_applied_risk_mapping()
    final_score = metrics.get("final_risk_score")
    final_score_f = float(final_score) if final_score is not None else None

    hedge_ratio = resolve_display_hedge_ratio(metrics, applied)
    if hedge_ratio is None:
        return

    default_ratio = (
        hedge_ratio_from_score(final_score_f, DEFAULT_MAPPING)
        if final_score_f is not None
        else hedge_ratio
    )
    customized = applied != normalize_mapping(DEFAULT_MAPPING)
    delta = None
    if customized and final_score_f is not None and abs(hedge_ratio - default_ratio) > 1e-6:
        delta = f"{hedge_ratio - default_ratio:+.0%} vs default"

    st.metric(
        "Hedge Recommendation",
        f"{hedge_ratio:.0%}",
        delta=delta,
        help=HEDGE_RATIO_HELP_TEXT,
    )
    st.caption(
        "ℹ️ " + " · ".join(format_mapping_rules(applied))
        + (" · _Custom mapping active_" if customized else "")
    )
