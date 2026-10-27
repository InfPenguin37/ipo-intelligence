"""Streamlit research dashboard. Start with `make run` or `ipo app`."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ipo_intel import pipeline
from ipo_intel.config import scrapers_enabled
from ipo_intel.fetchers import ChittorgarhFetcher, FetchError
from ipo_intel.forecast_ledger import (
    DISPLAY_COLUMNS,
    append_forecast,
    load_forecasts,
    merge_uploaded_forecasts,
)
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
    "gmp_latest": "Latest GMP",
    "gmp_final": "Final pre-listing GMP",
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
        if field in {"price_band_high", "gmp_open", "gmp_peak", "gmp_latest", "gmp_final"}:
            return f"₹{value:,.2f}"
    return str(value)


def _short_verdict_class(verdict: str) -> str:
    if verdict.startswith("STRONG APPLY"):
        return "verdict-strong"
    if verdict == "APPLY":
        return "verdict-apply"
    if verdict.startswith("NEUTRAL"):
        return "verdict-neutral"
    if verdict == "AVOID":
        return "verdict-avoid"
    return "verdict-unknown"


def _live_fetch_enabled() -> bool:
    if not scrapers_enabled():
        return False
    try:
        setting = st.secrets.get("IPO_ENABLE_SCRAPERS")
    except StreamlitSecretNotFoundError:
        return True
    return setting is None or str(setting).strip() == "1"


@st.cache_data(ttl=300, show_spinner=False)
def _active_ipo_schedules() -> tuple[tuple[str, str | None, str | None, str | None], ...]:
    schedules = ChittorgarhFetcher().fetch_active_ipo_schedule()
    return tuple(
        (
            item["name"],
            item["opening_date"],
            item["closing_date"],
            item["listing_date"],
        )
        for item in schedules
    )


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
    .verdict-badge {
        display: inline-block;
        border: 1px solid currentColor;
        border-radius: 999px;
        padding: .4rem .8rem;
        font-size: 1.02rem;
        font-weight: 750;
        line-height: 1.25;
    }
    .verdict-strong { border-color: #16803c; background: color-mix(in srgb, #16803c 14%, transparent); }
    .verdict-apply { border-color: #25834a; background: color-mix(in srgb, #25834a 10%, transparent); }
    .verdict-neutral { border-color: #d99a19; background: color-mix(in srgb, #d99a19 14%, transparent); }
    .verdict-avoid { border-color: #c23838; background: color-mix(in srgb, #c23838 12%, transparent); }
    .verdict-unknown { color: inherit; background: color-mix(in srgb, currentColor 8%, transparent); }
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

live_fetch_enabled = _live_fetch_enabled()

with st.sidebar:
    st.title("Research desk")
    st.caption("Build a listing outlook from live signals, your inputs, and the trained model.")

    with st.expander("Manual overrides", expanded=False):
        overrides_text = st.text_area(
            "One key=value per line",
            placeholder="sub_retail=2.4\ngmp_latest=250",
            help="Use this for subscription, GMP, issue details, or other values missing from the sources.",
        )
        st.caption("Fields: " + ", ".join(ALL_KEYS))

st.markdown('<div class="eyebrow">LISTING OUTLOOK · RESEARCH DASHBOARD</div>', unsafe_allow_html=True)
st.title("IPO Intelligence")
st.markdown(
    '<p class="lede">A clear view of listing momentum, model expectations, and the data behind them.</p>',
    unsafe_allow_html=True,
)

ipo_schedules: tuple[tuple[str, str | None, str | None, str | None], ...] = ()
suggestion_error: str | None = None
if live_fetch_enabled:
    try:
        ipo_schedules = _active_ipo_schedules()
    except FetchError as error:
        suggestion_error = str(error)

if suggestion_error:
    st.error(f"Could not load IPO suggestions from Chittorgarh: {suggestion_error}")
elif not live_fetch_enabled:
    st.info("IPO suggestions require live scraping, which is disabled; enter a name manually.")

if ipo_schedules:
    ipo_names = tuple(schedule[0] for schedule in ipo_schedules)
    schedule_by_name = {
        name: (opening, closing, listing)
        for name, opening, closing, listing in ipo_schedules
    }
    name = st.selectbox(
        "IPO name",
        options=ipo_names,
        index=None,
        placeholder="Search active and upcoming mainboard IPOs",
    )
else:
    ipo_names = ()
    name = st.text_input("IPO name", placeholder="Enter an IPO name")

if name and name in ipo_names:
    opening_date, closing_date, listing_date = schedule_by_name[name]
    st.subheader("IPO schedule")
    date_columns = st.columns(3)
    for column, label, value in zip(
        date_columns,
        ("Opens", "Closes", "Expected listing"),
        (opening_date, closing_date, listing_date),
        strict=True,
    ):
        formatted_date = pd.Timestamp(value).strftime("%d %b %Y") if value else "Not announced"
        column.metric(label, formatted_date)

with st.form("analysis_form"):
    live = st.checkbox(
        "Fetch live data",
        value=live_fetch_enabled,
        disabled=not live_fetch_enabled,
        help=(
            "Fetch IPOWatch GMP and overview data, plus Chittorgarh issue, subscription and financials."
        ),
    )
    submitted = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

if submitted:
    if not name or not name.strip():
        st.error(
            "Select an IPO name to run an analysis."
            if ipo_schedules
            else "Enter an IPO name to run an analysis."
        )
        st.stop()
    try:
        overrides = pipeline.parse_overrides(overrides_text.splitlines())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Collecting data and scoring..."):
        rep = pipeline.run(
            name.strip(),
            scrape=live and live_fetch_enabled,
            overrides=overrides,
        )

    if rep.get("headline_gain") is not None:
        try:
            append_forecast(rep)
        except (OSError, ValueError) as error:
            st.error(f"Could not save this forecast to the prediction history: {error}")

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
            headline, range_col = st.columns([1, 1])
            with headline:
                st.metric("Estimated listing gain", f"{rep['headline_gain']:+.1f}%")
                st.caption(rep["basis"])
            with range_col:
                if rep.get("ml"):
                    model = rep["ml"]
                    st.metric(
                        "Model uncertainty range",
                        f"{model['q10']:+.1f}% to {model['q90']:+.1f}%",
                    )
                    st.caption(
                        f"Central 80% prediction range (10th–90th percentile); "
                        f"{model['n_train']} historical IPOs in the {model['tier']} tier."
                    )
                else:
                    st.metric("Model uncertainty range", "Not available")
                    st.caption(rep.get("ml_status") or "The estimate uses the research heuristic.")

            verdict = escape(rep["short_verdict"])
            verdict_class = _short_verdict_class(rep["short_verdict"])
            st.markdown(
                f'<div class="verdict {verdict_class}">Short-term signal</div>'
                f'<div style="margin-top:.35rem"><span class="verdict-badge {verdict_class}">{verdict}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Research indicator based on estimated listing-day gain; not an investment recommendation.")

            st.divider()
            raw_gmp, heuristic = st.columns(2)
            raw_gmp.metric("Raw GMP-implied return", f"{rep['gmp_pct']:+.1f}%")
            gmp_info = rep["gmp_explanation"]
            raw_gmp.caption(
                f"₹{gmp_info['gmp_rupees']:,.2f} unofficial grey-market premium ÷ "
                f"₹{gmp_info['offer_price_rupees']:,.2f} offer price. This is a market-indicator "
                "calculation, not a forecast."
            )
            heuristic.metric("Rule-based listing estimate", f"{rep['heuristic_gain']:+.1f}%")
            factors = rep["heuristic_breakdown"]
            retail_note = (
                f"retail demand {factors['demand_factor']:.0%} "
                f"(reported {observed['sub_retail']:.2f}×)"
                if rep["subscription_known"]
                else "retail demand 85% (assumed 1–3×; subscription unavailable)"
            )
            trend_note = (
                f"GMP trend {factors['trend_factor']:.0%} "
                f"(latest/open ratio {factors['gmp_ratio']:.2f}×)"
                if observed.get("gmp_open") is not None and observed["gmp_open"] > 0
                else "GMP trend neutral (opening GMP unavailable)"
            )
            surge_note = (
                " A rising-GMP adjustment raised the issue-size factor."
                if factors["surge_adjustment"]
                else ""
            )
            heuristic.caption(
                f"GMP-implied return × issue-size factor {factors['size_factor']:.0%} × "
                f"{retail_note} × {trend_note} = estimate.{surge_note}"
            )

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
        score_col, verdict_col, coverage_col = st.columns([1, 1.7, 1])
        score = long_term["score"]
        score_col.metric("Fundamental Score", f"{score:.0f} / 100" if score is not None else "N/A")
        verdict_col.markdown("**Assessment**")
        verdict_col.write(long_term["verdict"])
        coverage_col.metric("Fundamental coverage", f"{long_term['coverage_pct']}%")
        if long_term["provisional"]:
            st.warning(
                "Provisional score only: the score is normalized over the categories with usable data. "
                "It is not a complete investment verdict."
            )
        valuation = []
        for label, key, suffix in (
            ("Blended growth", "blended_growth_pct", "%"),
            ("PEG", "peg", ""),
            ("Discount to best peer", "peer_discount_pct", "%"),
        ):
            value = long_term[key]
            if value is not None:
                valuation.append(f"{label} **{value}{suffix}**")
        if valuation:
            st.markdown(" · ".join(valuation))
        parts = pd.DataFrame(
            [{"Factor": factor.title(), "Score": score} for factor, score in long_term["parts"].items()]
        )
        if not parts.empty:
            st.dataframe(
                parts,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score / 10", min_value=0, max_value=10, format="%d"),
                },
            )
        for item in long_term["evidence"]:
            st.write(f"- {item}")
        for item in long_term["unusable_fields"]:
            st.caption(item)
        if long_term["missing_fields"]:
            st.caption("Still needed: " + "; ".join(long_term["missing_field_descriptions"]))
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

st.subheader("Prediction history")
st.caption(
    "Forecasts are recorded with their inputs and sources. Download this CSV regularly: "
    "the hosted app's local files can be reset. After an IPO lists, fill in actual listing prices "
    "and upload the updated CSV; gains are calculated from the offer price."
)
history = load_forecasts()
display_columns = [column for column in DISPLAY_COLUMNS if column in history]
if history.empty:
    st.info("No forecasts recorded yet.")
else:
    st.dataframe(
        history[display_columns],
        hide_index=True,
        use_container_width=True,
        column_config={
            "predicted_open_gain_pct": st.column_config.NumberColumn("Predicted open gain", format="%+.1f%%"),
            "heuristic_gain_pct": st.column_config.NumberColumn("Heuristic gain", format="%+.1f%%"),
            "ml_q10": st.column_config.NumberColumn("ML 10th percentile", format="%+.1f%%"),
            "ml_q50": st.column_config.NumberColumn("ML median", format="%+.1f%%"),
            "ml_q90": st.column_config.NumberColumn("ML 90th percentile", format="%+.1f%%"),
            "actual_listing_open_gain_pct": st.column_config.NumberColumn("Actual open gain", format="%+.1f%%"),
            "actual_listing_close_gain_pct": st.column_config.NumberColumn("Actual close gain", format="%+.1f%%"),
        },
    )
st.download_button(
    "Download prediction history CSV",
    data=history.to_csv(index=False).encode("utf-8"),
    file_name="ipo_forecast_history.csv",
    mime="text/csv",
)

with st.form("forecast_outcome_import_form"):
    uploaded_ledger = st.file_uploader("Upload an updated prediction history CSV", type="csv")
    import_submitted = st.form_submit_button("Import and update history")

if import_submitted:
    if uploaded_ledger is None:
        st.error("Choose an updated prediction history CSV to import.")
    else:
        try:
            imported = pd.read_csv(uploaded_ledger)
            updated_history = merge_uploaded_forecasts(imported)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not import prediction history: {error}")
        except OSError as error:
            st.error(f"Could not save the updated prediction history: {error}")
        else:
            st.success(f"Prediction history updated with {len(updated_history)} forecast rows.")
            st.rerun()
