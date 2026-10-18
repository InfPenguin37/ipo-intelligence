"""Streamlit research dashboard. Start with `make run` or `ipo app`."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ipo_intel import pipeline
from ipo_intel.config import scrapers_enabled
from ipo_intel.report import DISCLAIMER
from ipo_intel.schema import ALL_KEYS

st.set_page_config(page_title="IPO Intelligence", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #182b43;
        --muted: #697b90;
        --line: #e1e8f0;
        --teal: #087f8c;
        --paper: #f5f8fb;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2.25rem; }
    [data-testid="stSidebar"] {
        background: #edf3f7;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    h1 { font-size: 2.15rem !important; }
    h2 { font-size: 1.35rem !important; margin-bottom: .35rem !important; }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 2px 8px rgba(24, 43, 67, .035);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 650; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff;
        border-color: var(--line) !important;
        border-radius: 16px;
    }
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {
        background: var(--teal);
        border: 1px solid var(--teal);
        border-radius: 10px;
        color: white;
        font-weight: 650;
        min-height: 2.75rem;
    }
    .stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {
        background: #066b76;
        border-color: #066b76;
        color: white;
    }
    [data-testid="stDataFrame"], [data-testid="stAlert"] { border-radius: 12px; }
    .eyebrow {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .lede { color: var(--muted); font-size: 1rem; margin-top: -.45rem; }
    .section-note { color: var(--muted); font-size: .88rem; margin: -.1rem 0 1rem; }
    .verdict {
        color: var(--teal);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: 0;
    }
    .status {
        display: inline-block;
        background: #e6f4f3;
        border: 1px solid #c7e8e5;
        border-radius: 999px;
        color: #086a71;
        font-size: .78rem;
        font-weight: 650;
        padding: .3rem .65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="eyebrow">IPO INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Research desk")
    st.caption("Build a listing outlook from live signals, your inputs, and the trained model.")

    with st.form("analysis_form"):
        name = st.text_input("IPO name", value="Jio Platforms")
        live = st.checkbox(
            "Fetch live data",
            value=False,
            disabled=not scrapers_enabled(),
            help="Fetch current GMP and subscription data. Requires IPO_ENABLE_SCRAPERS=1.",
        )
        submitted = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

    if not scrapers_enabled():
        st.caption("Live fetching is off. Set `IPO_ENABLE_SCRAPERS=1` before starting the app to enable it.")

    with st.expander("Manual overrides", expanded=False):
        overrides_text = st.text_area(
            "One key=value per line",
            placeholder="sub_retail=2.4\ngmp_final=250",
            help="Use this for subscription, GMP, issue details, or other values missing from the sources.",
        )
        st.caption("Fields: " + ", ".join(ALL_KEYS))

st.markdown('<div class="eyebrow">LISTING OUTLOOK · RESEARCH DASHBOARD</div>', unsafe_allow_html=True)
st.title("IPO Intelligence")
st.markdown(
    '<p class="lede">A clear view of listing momentum, model expectations, and the data behind them.</p>',
    unsafe_allow_html=True,
)

if submitted:
    if not name.strip():
        st.error("Enter an IPO name to run an analysis.")
        st.stop()
    try:
        overrides = pipeline.parse_overrides(overrides_text.splitlines())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Collecting data and scoring..."):
        rep = pipeline.run(name.strip(), scrape=live and scrapers_enabled(), overrides=overrides)

    for warning in rep["warnings"]:
        st.warning(warning)

    provenance = pd.DataFrame(rep["provenance"])
    observed = {row["field"]: row["value"] for row in rep["provenance"]}
    fetched_at = pd.to_datetime(provenance["fetched_at"], errors="coerce", utc=True).max() if not provenance.empty else None
    last_updated = fetched_at.strftime("%d %b %Y, %H:%M UTC") if pd.notna(fetched_at) else "No source timestamps"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"## {escape(rep['name'])}")
        st.caption("Live and saved information is combined with seed data and user overrides.")
    with right:
        st.markdown(
            f'<div style="text-align:right"><span class="status">Updated · {escape(last_updated)}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Short term")
    st.markdown('<p class="section-note">Expected listing performance and the signals behind the estimate.</p>', unsafe_allow_html=True)
    if rep["missing_short_term"]:
        st.info(
            "A listing outlook needs "
            + ", ".join(rep["missing_short_term"])
            + ". Add them under Manual overrides or enable live fetching."
        )
    else:
        with st.container(border=True):
            headline, range_col, verdict_col = st.columns([1.2, 1, 1])
            with headline:
                st.metric("Model / headline estimate", f"{rep['headline_gain']:+.1f}%")
                st.caption(rep["basis"])
            with range_col:
                if rep.get("ml"):
                    model = rep["ml"]
                    st.metric("Model range · 80%", f"{model['q10']:+.1f}% to {model['q90']:+.1f}%")
                    st.caption(f"{model['n_train']} IPOs in the {model['tier']} tier")
                else:
                    st.metric("Rule-based estimate", f"{rep['heuristic_gain']:+.1f}%")
                    st.caption("Model estimate unavailable; using the research heuristic.")
            with verdict_col:
                st.markdown('<p class="verdict">Signal</p>', unsafe_allow_html=True)
                st.subheader(rep["short_verdict"])
                st.caption("Research indicator, not a recommendation.")

            st.divider()
            raw_gmp, heuristic = st.columns(2)
            raw_gmp.metric("GMP-implied return", f"{rep['gmp_pct']:+.1f}%")
            heuristic.metric("Heuristic estimate", f"{rep['heuristic_gain']:+.1f}%")

        subscriptions = [
            ("sub_overall", "Overall"),
            ("sub_qib", "QIB"),
            ("sub_nii", "NII"),
            ("sub_retail", "Retail"),
        ]
        available_subscriptions = [(field, label, observed[field]) for field, label in subscriptions if observed.get(field) is not None]
        if available_subscriptions:
            st.markdown("#### Subscription")
            subscription_cols = st.columns(len(available_subscriptions))
            for column, (_, label, value) in zip(subscription_cols, available_subscriptions):
                column.metric(label, f"{value:g}×")

        if rep.get("scenarios"):
            st.markdown("#### Listing scenarios")
            scenario_rows = [
                {
                    "Scenario": scenario["scenario"],
                    "Estimated gain": scenario["est_gain_pct"],
                    "Signal": scenario["call"],
                }
                for scenario in rep["scenarios"]
            ]
            st.dataframe(
                pd.DataFrame(scenario_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Estimated gain": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )

        for note in rep["notes"]:
            st.info(note)

    st.subheader("Long term")
    st.markdown('<p class="section-note">A fundamentals-based view over a 1–3 year horizon.</p>', unsafe_allow_html=True)
    long_term = rep.get("long_term")
    if long_term:
        score_col, verdict_col, metrics_col = st.columns([1, 1.7, 2])
        score_col.metric("Fundamental Score", f"{long_term['score']:.0f} / 100")
        verdict_col.markdown("**Assessment**")
        verdict_col.write(long_term["verdict"])
        metrics_col.markdown("**Valuation & growth**")
        metrics_col.write(
            f"Blended growth **{long_term['blended_growth_pct']}%** · "
            f"PEG **{long_term['peg']}** · Peer discount **{long_term['peer_discount_pct']}%**"
        )
        parts = pd.DataFrame(
            [{"Factor": factor.title(), "Score": score} for factor, score in long_term["parts"].items()]
        )
        st.dataframe(
            parts,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score / 10", min_value=0, max_value=10, format="%d"),
            },
        )
    else:
        missing = ", ".join(rep.get("long_term_missing", []))
        st.info(f"Long-term score is not available yet. Missing fundamentals: {missing}")

    with st.expander("Data sources & freshness"):
        if provenance.empty:
            st.caption("No source fields were available for this analysis.")
        else:
            details = provenance.rename(
                columns={
                    "field": "Field",
                    "value": "Value",
                    "source": "Source",
                    "fetched_at": "Retrieved (UTC)",
                }
            )
            details["Value"] = details["Value"].astype(str)
            st.dataframe(details, hide_index=True, use_container_width=True)

    st.caption(DISCLAIMER)
else:
    with st.container(border=True):
        st.markdown("### Start with an IPO")
        st.write("Choose a company in the Research desk, then run an analysis to review listing signals and fundamentals.")
        st.caption("Your saved training model is used automatically. You can fetch live data or enter values manually.")
