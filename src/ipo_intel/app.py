"""Streamlit UI. Start with `make run` or `ipo app`."""
import pandas as pd
import streamlit as st

from ipo_intel import pipeline
from ipo_intel.config import scrapers_enabled
from ipo_intel.report import DISCLAIMER
from ipo_intel.schema import ALL_KEYS

st.set_page_config(page_title="IPO Intelligence", layout="wide")
st.title("IPO Intelligence")
st.caption("Name an IPO, fetch the latest data, get a short-term and a long-term read.")

name = st.text_input("IPO name", value="Jio Platforms")
live = st.checkbox(
    "Fetch live data (GMP and subscription scrapers)",
    value=False,
    disabled=not scrapers_enabled(),
    help="Off unless IPO_ENABLE_SCRAPERS=1 is set in your shell. See the README before enabling.",
)
with st.expander("Manual overrides (final subscription, latest GMP, anything the fetchers missed)"):
    overrides_text = st.text_area("One key=value per line", placeholder="sub_retail=2.4\ngmp_final=250")
    st.caption("Valid fields: " + ", ".join(ALL_KEYS))

if st.button("Run analysis", type="primary"):
    try:
        overrides = pipeline.parse_overrides(overrides_text.splitlines())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Collecting data and scoring..."):
        rep = pipeline.run(name, scrape=live and scrapers_enabled(), overrides=overrides)

    for w in rep["warnings"]:
        st.warning(w)

    st.subheader("Short term")
    if rep["missing_short_term"]:
        st.info("Not enough data yet. Missing: " + ", ".join(rep["missing_short_term"]))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw GMP implies", f"{rep['gmp_pct']:+.1f}%")
        c2.metric("Heuristic estimate", f"{rep['heuristic_gain']:+.1f}%")
        if rep.get("ml"):
            m = rep["ml"]
            c3.metric(f"ML estimate ({m['tier']})", f"{m['q50']:+.1f}%", f"band {m['q10']:+.0f}% to {m['q90']:+.0f}%", delta_color="off")
        c4.metric("Verdict", rep["short_verdict"])
        st.caption(f"Headline {rep['headline_gain']:+.1f}% from the {rep['basis']}.")
        if rep.get("scenarios"):
            st.dataframe(pd.DataFrame(rep["scenarios"]), hide_index=True, use_container_width=True)
        for n in rep["notes"]:
            st.info(n)

    st.subheader("Long term")
    lt = rep.get("long_term")
    if lt:
        st.metric("Score", f"{lt['score']:.0f} / 100", lt["verdict"], delta_color="off")
        st.bar_chart(pd.Series(lt["parts"], name="score out of 10"))
        st.caption(
            f"Blended growth {lt['blended_growth_pct']}%, PEG {lt['peg']}, discount to best peer {lt['peer_discount_pct']}%."
        )
    else:
        st.info("Not scored. Missing: " + ", ".join(rep.get("long_term_missing", [])))

    with st.expander("Data used, with sources and timestamps"):
        st.dataframe(pd.DataFrame(rep["provenance"]).astype(str), hide_index=True, use_container_width=True)
    st.caption(DISCLAIMER)
