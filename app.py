# app.py
# CSUSB Internship Finder – Orchestrator
# - LLM-first intent routing via query_to_filter.classify_intent
# - Internship: CSUSB-only links (no deep search)
# - Resume: upload + deterministic Q&A using your resume_parser
# - General: concise non-LLM reply

from __future__ import annotations

import os
import re
import json
import time
import math
import query_to_filter
from collections import deque
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from urllib.parse import urlparse

# Local modules
import ui  # all UI helpers live here
from scraper import scrape_csusb_listings, CSUSB_CSE_URL
from query_to_filter import parse_query_to_filter
from query_to_filter import classify_intent
# Keep your existing resume helpers:
from resume_parser import extract_resume_text, llm_resume_extract, save_resume, answer_from_resume

# -----------------------------------------------------------------------------
# Constants / Paths
# -----------------------------------------------------------------------------
APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"

# -----------------------------------------------------------------------------
# Page setup + CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
ui.inject_css("styles.css")
ui.inject_badge_css()
ui.header(APP_TITLE, CSUSB_CSE_URL)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hi! I can list internships from the CSUSB CSE site, "
            "answer simple résumé questions, and handle general questions. "
            "What can I do for you?"
        )
    }]

# Rate-limit (10/min)
if "q_times" not in st.session_state:
    st.session_state.q_times = deque()

# Résumé state + lazy autoload
st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("resume_data", {})
try:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    p_json = DATA_DIR / "resume.json"
    if p_json.exists() and not st.session_state["resume_data"]:
        st.session_state["resume_data"] = json.loads(p_json.read_text(encoding="utf-8"))
        rt = DATA_DIR / "resume.txt"
        st.session_state["resume_text"] = rt.read_text(encoding="utf-8") if rt.exists() else ""
except Exception:
    pass

# -----------------------------------------------------------------------------
# Rate limit helper (10/min)
# -----------------------------------------------------------------------------
def allow_query() -> bool:
    now = time.time()
    while st.session_state.q_times and (now - st.session_state.q_times[0]) > 60:
        st.session_state.q_times.popleft()
    if len(st.session_state.q_times) >= 10:
        st.error("⏱️ Rate limit reached: 10 queries/min. Please wait ~60 seconds.")
        return False
    st.session_state.q_times.append(now)
    return True

# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_df() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        try:
            return pd.read_parquet(PARQUET_PATH)
        except Exception:
            pass
    return pd.DataFrame()

def cache_age_hours() -> float:
    if not PARQUET_PATH.exists():
        return math.inf
    return (time.time() - PARQUET_PATH.stat().st_mtime) / 3600.0

@st.cache_data(show_spinner=False, ttl=6*60*60)
def fetch_csusb_df() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # CSUSB-only; no deep crawl
    df = scrape_csusb_listings(deep=False, max_pages=1)
    df.to_parquet(PARQUET_PATH, index=False)
    return df

# -----------------------------------------------------------------------------
# Sidebar: résumé uploader (UI) – persists into session_state
# -----------------------------------------------------------------------------
ui.show_resume_sidebar(
    on_extract=lambda file: extract_resume_text(file),
    on_llm_extract=lambda text: llm_resume_extract(text),
    on_save=lambda data, text: save_resume(data, text),
)

# -----------------------------------------------------------------------------
# Render existing chat history
# -----------------------------------------------------------------------------
ui.render_history(st.session_state.messages)

# -----------------------------------------------------------------------------
# Chat input
# -----------------------------------------------------------------------------
user_msg = st.chat_input("Type your question…")
if not user_msg:
    st.stop()
if not allow_query():
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_msg})
ui.render_msg("user", user_msg)

# -----------------------------------------------------------------------------
# LLM-first routing (no heuristic hardcoding)
# -----------------------------------------------------------------------------
def normalize_intent(label: str) -> str:
    """
    Map any model-specific labels to our canonical set.
    The LLM *decides* the label; this just normalizes synonyms.
    """
    if not label:
        return "general_question"
    l = label.strip().lower()
    if l in {"internship", "internship_search", "internships"}:
        return "internship_search"
    if l in {"resume", "résumé", "resume_question"}:
        return "resume_question"
    return "general_question"

raw_intent = classify_intent(user_msg)  # LLM decides (temperature set in query_to_filter.py)
intent = normalize_intent(raw_intent)
st.sidebar.caption(f"🎯 Intent (LLM): {intent}")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
if intent == "resume_question":
    # Use existing résumé upload + Q&A
    data = st.session_state.get("resume_data") or {}
    if not data:
        reply = "Please upload your résumé (PDF/DOCX/TXT) using the sidebar, then ask your question."
        ui.render_msg("assistant", reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.stop()
    try:
        # Keep your helper signature: answer_from_resume(question, resume_json)
        reply = answer_from_resume(user_msg, data)
    except Exception:
        reply = "I ran into an issue analyzing your résumé. Please re-upload it and try again."
    ui.render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

elif intent == "internship_search":
    # 1) Load/cached scrape (CSUSB page only; no deep search)
    need_refresh = (cache_age_hours() > 24) or any(w in user_msg.lower() for w in ["refresh", "reload", "latest"])
    csusb_df = load_cached_df()
    if csusb_df.empty or need_refresh:
        if need_refresh:
            fetch_csusb_df.clear()
        with st.spinner("📡 Fetching CSUSB CSE links..."):
            csusb_df = fetch_csusb_df()

    # 2) Keep all links extracted from the CSUSB CSE page (external career sites included)
    df_all = csusb_df.copy()

    # only http(s)
    if "link" in df_all.columns:
        df_all = df_all[df_all["link"].astype(str).str.startswith(("http://", "https://"))]

    # ensure they came from the CSUSB CSE page (if 'source' exists)
    if "source" in df_all.columns:
        df_all = df_all[df_all["source"].astype(str).str.contains("csusb.edu/cse/internships-careers", na=False)]

    # normalize expected columns
    for col in ["title", "company", "link"]:
        if col not in df_all.columns:
            df_all[col] = ""

    # ensure host column for domain matching
    from urllib.parse import urlparse
    if "host" not in df_all.columns:
        df_all["host"] = df_all["link"].map(lambda u: urlparse(u).netloc if isinstance(u, str) else "")

    # 3) Ask LLM for filters; apply ONLY if present
    try:
        filt: Dict = parse_query_to_filter(user_msg) or {}
    except Exception:
        filt = {}

    show_all = bool(filt.get("show_all"))
    df = df_all.copy()
    applied_any_filter = False

    def _low(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().fillna("")

    # company filter
    company = str(filt.get("company_name") or "").strip().lower()
    if company:
        import re as _re
        pat = _re.escape(company)
        df = df[
            _low(df["company"]).str.contains(pat) |
            _low(df["title"]).str.contains(pat)   |
            _low(df["host"]).str.contains(pat)
        ]
        applied_any_filter = True

    # title keywords
    for kw in (filt.get("title_keywords") or []):
        kw = (kw or "").strip().lower()
        if kw:
            import re as _re
            pat = _re.escape(kw)
            if "title" in df.columns:
                df = df[_low(df["title"]).str.contains(pat)]
                applied_any_filter = True

    # skills -> use title as proxy
    for sk in (filt.get("skills") or []):
        sk = (sk or "").strip().lower()
        if sk and "title" in df.columns:
            import re as _re
            pat = _re.escape(sk)
            df = df[_low(df["title"]).str.contains(pat)]
            applied_any_filter = True

    # 4) Decide result set:
    # - if user explicitly asked for "all", show everything
    # - otherwise show ONLY filtered matches (no automatic fallback to all)
    results = df_all if show_all else df

    if results.empty:
        msg = "I couldn’t find any matching links on the CSUSB CSE page."
        if not show_all:
            msg += " Say **“show all internships”** to list everything."
        ui.render_msg("assistant", msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.stop()

    # 5) Keep essentials and dedupe by link
    keep_cols = [c for c in ["title", "company", "link"] if c in results.columns]
    if "link" in keep_cols:
        results = results[keep_cols].drop_duplicates(subset=["link"], keep="first")
    else:
        results = results[keep_cols].drop_duplicates()

    # 6) Chat summary + bullet list + table
    if show_all:
        summary = f"Here are **all {len(results)}** links listed on the CSUSB CSE page."
    elif applied_any_filter:
        summary = f"Here are **{len(results)}** matching link(s) from the CSUSB CSE page."
    else:
        # No filters detected (e.g., user said "find internships"); be explicit
        summary = f"I found **{len(results)}** link(s). Ask for a company (e.g., 'nasa') or say 'show all internships'."

    ui.render_msg("assistant", summary)
    st.session_state.messages.append({"role": "assistant", "content": summary})

    # links inside the chat + full table below
    ui.render_links_in_chat(results, limit=50)
    ui.render_found_links_table(results)
    st.stop()

else:
    # General question: concise, deterministic reply (no LLM needed)
    t = user_msg.strip().lower()
    if any(g in t for g in ["hi", "hello", "hey"]):
        reply = "Hi! Ask me about internships (company/tech/term) or upload your résumé for questions about your experience."
    elif "what is this" in t or "about" in t:
        reply = ("This app lists internship-related links from the CSUSB CSE website "
                 "and lets you ask simple résumé questions. No deep web search is performed.")
    elif "download" in t or "csv" in t:
        reply = "Use the “Download Results (CSV)” button below the table to save the links."
    elif "refresh" in t or "latest" in t:
        reply = "Type ‘refresh’ in an internship request and I’ll pull the latest CSUSB CSE links."
    else:
        reply = ("I can help with internships (company, technology, term) or simple résumé questions. "
                 "Try: ‘summer software internships’, ‘Nasa internships’, or ‘What skills are on my résumé?’")
    ui.render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
