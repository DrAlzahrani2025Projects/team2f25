"""
Internship Finder Agent – Team2f25 (port 5002)
Streamlit UI:
- Fetch ALL internships from CSUSB (Playwright)
- Toggle demo/sample vs. live data
- Natural language search -> structured filters (LangChain optional)
- Results table with clickable links + CSV export
"""

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from scraper import scrape_csusb_listings, CSUSB_CSE_URL
from query_to_filter import parse_query_to_filter

# ------------------------- Paths -------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")
PARQUET_PATH = os.path.join(DATA_DIR, "internships.parquet")
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_data.csv")

# ------------------------- UI Setup ----------------------
st.set_page_config(
    page_title="Internship Finder Agent – Team2f25",
    page_icon="🧭",
    layout="wide",
)

st.title("Internship Finder Agent – Team2f25")
st.caption(f"Base URL path: `/team2f25` • Source: {CSUSB_CSE_URL}")

# ------------------------- Sidebar -----------------------
with st.sidebar:
    st.header("Actions")

    fetch_label = "🔄 Fetch ALL internships from CSUSB (refresh)"
    fetch_help = (
        "Visits the CSUSB CSE Internships & Careers page and extracts every "
        "internship opening found. Replaces the current dataset."
    )

    if st.button(fetch_label, help=fetch_help):
        try:
            with st.spinner("Fetching all internships from CSUSB…"):
                df_live = scrape_csusb_listings()
                os.makedirs(DATA_DIR, exist_ok=True)
                if df_live.empty:
                    st.warning("No internships were found on the page right now. Try again later.")
                else:
                    df_live.to_parquet(PARQUET_PATH, index=False)
                    st.session_state["df"] = df_live
                    st.success(f"Loaded ALL internships from CSUSB — {len(df_live)} found. "
                               f"Saved to data/internships.parquet.")
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    if st.button("📂 Load saved parquet (if any)"):
        if os.path.exists(PARQUET_PATH):
            st.session_state["df"] = pd.read_parquet(PARQUET_PATH)
            st.success(f"Loaded {len(st.session_state['df'])} rows from parquet")
        else:
            st.warning("No saved parquet yet. Use the fetch button above first.")

# --------------------- Data Source Choice ----------------
st.subheader("Search internships")

use_sample = st.toggle(
    "Use sample data (demo mode)",
    value=False,
    help="When on, loads local sample CSV instead of scraped/saved data.",
)

# Decide the working dataframe `df` exactly once
if use_sample and os.path.exists(SAMPLE_CSV_PATH):
    df = pd.read_csv(SAMPLE_CSV_PATH)
    st.warning("Demo mode is ON — showing sample data instead of live scraped results.")
else:
    df = st.session_state.get("df")
    if df is None:
        if os.path.exists(PARQUET_PATH):
            df = pd.read_parquet(PARQUET_PATH)
        else:
            df = pd.DataFrame(
                columns=["title", "company", "location", "posted_date", "tags", "link", "source"]
            )

# -------------------- NL Query -> Filters ----------------
query = st.text_input(
    "Describe what you're looking for (e.g., 'software internships in San Bernardino requiring Python')",
    value="",
)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    do_filter = st.button("Search")
with col_b:
    clear_filter = st.button("Clear")
with col_c:
    show_debug = st.toggle("Show filter info")

if clear_filter:
    st.session_state.pop("filters", None)

if do_filter:
    st.session_state["filters"] = parse_query_to_filter(query)

filters = st.session_state.get("filters", {})

if show_debug and filters:
    st.info(f"Filters: `{json.dumps(filters, indent=2)}`")

# --------------------- Apply Filters ---------------------
filtered = df.copy()

if filters and not filtered.empty:
    # role -> match in title
    role = (filters.get("role") or "").strip()
    if role:
        filtered = filtered[filtered["title"].str.contains(role, case=False, na=False)]

    # location -> match in title or location column
    location = (filters.get("location") or "").strip()
    if location:
        in_title = filtered["title"].str.contains(location, case=False, na=False)
        in_loc = filtered["location"].fillna("").str.contains(location, case=False, na=False)
        filtered = filtered[in_title | in_loc]

    # skills -> match in title or tags
    for skill in (filters.get("skills") or []):
        s = str(skill).strip()
        if not s:
            continue
        in_title = filtered["title"].str.contains(s, case=False, na=False)
        in_tags = filtered["tags"].fillna("").str.contains(s, case=False, na=False)
        filtered = filtered[in_title | in_tags]

    # keywords -> match in title or tags (limit a few)
    for kw in (filters.get("keywords") or [])[:5]:
        k = str(kw).strip()
        if not k:
            continue
        in_title = filtered["title"].str.contains(k, case=False, na=False)
        in_tags = filtered["tags"].fillna("").str.contains(k, case=False, na=False)
        filtered = filtered[in_title | in_tags]

# ---------------- Results Table + Export -----------------
st.subheader("Results")
st.write(f"Showing **{len(filtered)}** of **{len(df)}** rows")

# clickable hyperlinks for `link` column
try:
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn("link", help="Open posting"),
        },
    )
except Exception:
    # Fallback for older Streamlit versions
    st.dataframe(filtered, use_container_width=True, hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Export CSV",
    data=csv_bytes,
    file_name=f"internships_filtered_{datetime.utcnow().date().isoformat()}.csv",
    mime="text/csv",
)

st.divider()
st.caption("Streamlit + Playwright + pandas + (optional) LangChain • Team2f25 • port 5002")
