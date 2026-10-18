"""Streamlit research dashboard. Start with `make run` or `ipo app`."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ipo_intel import pipeline
from ipo_intel.config import scrapers_enabled
from ipo_intel.report import DISCLAIMER
from ipo_intel.schema import ALL_KEYS, MARKET_KEYS

st.set_page_config(page_title="IPO Intelligence", page_icon="📈", layout="wide")

_MARKET_LABELS = {
    "issue_size_cr": "Issue size",
    "price_band_high": "Upper price band",
    "structure": "Issue structure",
    "ofs_share": "Offer-for-sale share",
    "gmp_open": "Opening GMP",
    "gmp_peak": "Peak GMP",
    "gmp_final": "Latest GMP",
    "sub_overall": "Overall subscription",
    "sub_qib": "QIB subscription",
    "sub_nii": "NII subscription",
    "sub_retail": "Retail subscription",
    "listing_date": "Listing date",
}


def _format_market_value(field: str, value: object) -> str:
    if isinstance(value, (int, float)):
        if field == "issue_size_cr":
            return f"₹{value:,.2f} Cr"
        if field == "ofs_share":
            return f"{value * 100:.1f}%"
        if field.startswith("sub_"):
            return f"{value:g}×"
        if field in {"price_band_high", "gmp_open", "gmp_peak", "gmp_final"}:
            return f"₹{value:,.2f}"
    return str(value)


st.markdown(
    """
    <style>
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 2.25rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { opacity: .76; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h1 { font-size: 2.15rem !important; }
    h2 { font-size: 1.35rem !important; margin-bottom: .35rem !important; }
    [data-testid="stMetric"] {
        border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, .08);
    }
    [data-testid="stMetricLabel"] p { opacity: .76; font-size: .82rem; }
    [data-testid="stMetricValue"] { font-weight: 650; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: color-mix(in srgb, currentColor 16%, transparent) !important;
        border-radius: 16px;
    }
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.75rem;
    }
    [data-testid="stFormSubmitButton"] button p { color: inherit; }
    [data-testid="stDataFrame"], [data-testid="stAlert"] { border-radius: 12px; }
    .eyebrow {
        opacity: .76;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .lede { opacity: .76; font-size: 1rem; margin-top: -.45rem; }
    .section-note { opacity: .76; font-size: .88rem; margin: -.1rem 0 1rem; }
    .verdict {
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: 0;
        opacity: .82;
    }
    .status {
        display: inline-block;
        background: color-mix(in srgb, currentColor 8%, transparent);
        border: 1px solid color-mix(in srgb, currentColor 24%, transparent);
        border-radius: 999px;
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
        screener_url = st.text_input(
            "Screener company URL (optional)",
            placeholder="https://www.screener.in/company/...",
            help="Paste a direct company page URL. Automated Screener search is not supported.",
        )
        live = st.checkbox(
            "Fetch live data",
            value=True,
            disabled=not scrapers_enabled(),
            help="Fetch live GMP and subscription data, plus fundamentals from the supplied Screener page.",
        )
        submitted = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

    if not scrapers_enabled():
        st.caption("Live fetching is disabled by `IPO_ENABLE_SCRAPERS=0`.")

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
        rep = pipeline.run(
            name.strip(),
            scrape=live and scrapers_enabled(),
            screener_url=screener_url.strip() or None,
            overrides=overrides,
        )

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
        available_subscriptions = [
            (field, label, observed[field])
            for field, label in subscriptions
            if observed.get(field) is not None
        ]
        if available_subscriptions:
            st.markdown("#### Reported subscription")
            st.caption(
                "Latest available demand snapshot, not a projection. Subscription can change until bidding closes."
            )
            subscription_cols = st.columns(len(available_subscriptions))
            for column, (_, label, value) in zip(subscription_cols, available_subscriptions):
                column.metric(label, f"{value:g}×")

        market_rows = [
            {
                "Data point": _MARKET_LABELS[row["field"]],
                "Value": _format_market_value(row["field"], row["value"]),
                "Source": row["source"],
                "Snapshot time (UTC)": pd.to_datetime(row["fetched_at"], utc=True).strftime(
                    "%d %b %Y, %H:%M"
                ),
            }
            for row in rep["provenance"]
            if row["field"] in MARKET_KEYS
        ]
        if market_rows:
            st.markdown("#### Current market data used")
            st.dataframe(pd.DataFrame(market_rows), hide_index=True, use_container_width=True)

        gmp_history = rep.get("gmp_history", [])
        if gmp_history:
            st.markdown("#### GMP history")
            history_frame = pd.DataFrame(gmp_history).rename(
                columns={
                    "observed_at": "Observed at (UTC)",
                    "gmp": "GMP",
                    "source": "Source",
                }
            )
            history_frame["Observed at (UTC)"] = pd.to_datetime(
                history_frame["Observed at (UTC)"], utc=True
            )
            st.line_chart(
                history_frame.set_index("Observed at (UTC)")["GMP"],
                y_label="GMP (₹ per share)",
            )
            history_frame["Observed at (UTC)"] = history_frame["Observed at (UTC)"].dt.strftime(
                "%d %b %Y, %H:%M"
            )
            history_frame["GMP"] = history_frame["GMP"].map(lambda value: f"₹{value:,.2f}")
            st.dataframe(history_frame, hide_index=True, use_container_width=True)

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
