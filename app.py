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
from collections import deque
from pathlib import Path
from typing import Dict, List
import pandas as pd
import streamlit as st
from urllib.parse import urlparse

import ui  # all UI helpers live here
from scraper import scrape_csusb_listings, CSUSB_CSE_URL
from query_to_filter import parse_query_to_filter, classify_intent
from resume_parser import extract_resume_text, llm_resume_extract, save_resume, answer_from_resume
from cover_letter.cl_state import init_cover_state, set_target_url
from cover_letter.cl_flow import (
    offer_cover_letter, handle_user_message as cl_flow_handle_user_message, 
    start_collection, ask_next_question
)
from resume_manager import read_file_to_text, llm_is_resume_question

APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"

st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
ui.inject_css("styles.css")
ui.inject_badge_css()
ui.header(APP_TITLE, CSUSB_CSE_URL)
init_cover_state()

def render_links_as_assistant_message(results):
    """Render a markdown chat message with links/title/company, and save it to chat history."""
    # Remove duplicate links!
    if results.empty or "link" not in results.columns:
        return
    df_deduped = results.drop_duplicates(subset=["link"], keep="first")
    formatted_links = []
    for _, row in df_deduped.iterrows():
        url = row.get("link", "")
        comp = row.get("company", "") or ""
        title = row.get("title", "") or ""
        # Markdown format: [title] — company, or with fallback
        if title and comp:
            formatted_links.append(f"[{title}]({url}) — {comp}")
        elif title:
            formatted_links.append(f"[{title}]({url})")
        elif comp:
            formatted_links.append(f"[{comp}]({url})")
        elif url:
            formatted_links.append(f"{url}")
    if formatted_links:
        links_chat = "\n".join(formatted_links)
        ui.render_msg("assistant", links_chat)
        st.session_state.messages.append({"role": "assistant", "content": links_chat})


# --- PERSISTENT INTERNSHIP LINK DISPLAY --- #

# Initialize "all links" on first page load, if not present in session_state
if "csusb_links" not in st.session_state:
    # You’d replace this with your real scraping call, but this guarantees it’s always present
    st.session_state.csusb_links = []

# Always show current filtered links or all links, BEFORE chat, EVERY rerun
def show_persistent_links():
    filtered_links = st.session_state.get("filtered_links")
    filter_desc = st.session_state.get("current_filter", None)
    if filtered_links:
        st.markdown(
            f"### Internship Links: {filter_desc if filter_desc else 'Filtered Results'}"
        )
        for link in filtered_links:
            st.markdown(f"- [{link}]({link})")
    else:
        st.markdown("### All Internship Links")
        for link in st.session_state.get("csusb_links", []):
            st.markdown(f"- [{link}]({link})")
# show_persistent_links()


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hi! I can list internships from the CSUSB CSE site, "
            "answer résumé questions, and handle general questions. "
            "What can I do for you?"
        )
    }]

# Sidebar Resume Upload
with st.sidebar:
    st.subheader("Resume")
    up = st.file_uploader(
        "Upload PDF/DOCX/TXT",
        type=["pdf", "docx", "txt"],
        key="resume_upl_single",
        accept_multiple_files=False,
        help="Upload your resume file (PDF/DOCX/TXT)."
    )
    if up is not None:
        file_changed = up.name != st.session_state.get("last_resume_file", "")
        if file_changed:
            try:
                with st.spinner("Processing resume..."):
                    text = read_file_to_text(up) or ""
                    max_chars = 2000
                    if len(text) > max_chars:
                        text = text[:max_chars]
                    parsed = llm_resume_extract(text) or {}
                    st.session_state["resume_text"] = text
                    st.session_state["resume_json"] = parsed
                    st.session_state["resume_data"] = parsed
                    st.session_state["cover_profile"] = {**st.session_state.get("cover_profile", {}), **{
                        "full_name": parsed.get("name"),
                        "email": parsed.get("email"),
                        "phone": parsed.get("phone")
                    }}
                    st.success("Resume processed & parsed successfully. I’ll use it for your cover letter.")
                    st.session_state["resume_just_uploaded"] = True
                    st.session_state["last_resume_file"] = up.name
            except Exception as e:
                import traceback as _tb
                st.error(f"Resume processing failed: {e}")
                st.code(_tb.format_exc())
                st.session_state["last_resume_file"] = ""

if "q_times" not in st.session_state:
    st.session_state.q_times = deque()

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

def allow_query() -> bool:
    now = time.time()
    while st.session_state.q_times and (now - st.session_state.q_times[0]) > 60:
        st.session_state.q_times.popleft()
    if len(st.session_state.q_times) >= 10:
        st.error("⏱️ Rate limit reached: 10 queries/min. Please wait ~60 seconds.")
        return False
    st.session_state.q_times.append(now)
    return True

def show_results_and_wire_cover_letter(df):
    if df is None or df.empty:
        with st.chat_message("assistant"):
            st.write("No results found.")
        return
    st.session_state["last_results_df"] = df
    st.dataframe(df, use_container_width=True, hide_index=True)
    for i, row in df.head(20).iterrows():
        url = str(row.get("link") or row.get("url") or "")
        c1, c2 = st.columns([6, 1])
        with c1:
            st.write(f"**{row.get('title','(no title)')}** — {row.get('company','')}")
            if url:
                st.write(url)
        with c2:
            if st.button("Cover Letter", key=f"cl_btn_{i}"):
                set_target_url(url)
                offer_cover_letter(render=ui.render_msg)
    offer_cover_letter(render=ui.render_msg)

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
    df = scrape_csusb_listings(deep=False, max_pages=1)
    df.to_parquet(PARQUET_PATH, index=False)
    return df

ui.render_history(st.session_state.messages)

if st.session_state.pop("resume_just_uploaded", False) and st.session_state.get("collecting_cover_profile"):
    with st.spinner("Reading resume and continuing…"):
        ask_next_question(render=ui.render_msg)

pend = st.session_state.pop("pending_company_query", None)
if pend:
    try:
        df_all = fetch_csusb_df()
        low = pend.lower()
        df = df_all.copy()
        mask = (
            df.get("company", "").astype(str).str.lower().str.contains(low, na=False) |
            df.get("title", "").astype(str).str.lower().str.contains(low, na=False)
        )
        df = df[mask].reset_index(drop=True)
        if not df.empty and len(df) == 1:
            url_col = "link" if "link" in df.columns else ("url" if "url" in df.columns else None)
            if url_col:
                set_target_url(str(df.iloc[0][url_col]))
            ask_next_question(render=ui.render_msg)
        st.session_state["last_results_df"] = df
        url_to_use = None
        if not df.empty:
            url_col = "link" if "link" in df.columns else ("url" if "url" in df.columns else None)
            if url_col and len(df) == 1:
                url_to_use = str(df.iloc[0][url_col])
        if url_to_use:
            set_target_url(url_to_use)
        if df.empty:
            ui.render_msg("assistant", f"I couldn’t find any CSUSB CSE links for **{pend}**. "
                                       f"You can paste a direct job link and I’ll continue.")
        else:
            ui.render_msg("assistant", f"Here are {len(df)} matching link(s) for **{pend}**.")
            ui.render_links_in_chat(df, limit=20)
        ask_next_question(render=ui.render_msg)
    except Exception as e:
        import traceback as _tb
        ui.render_msg("assistant", f"Sorry — fetching links for **{pend}** failed: {e}")
        st.code(_tb.format_exc())
        ask_next_question(render=ui.render_msg)

user_msg = st.chat_input("Type your question…")
if not user_msg:
    st.stop()
if not allow_query():
    st.stop()
st.session_state.messages.append({"role": "user", "content": user_msg})
ui.render_msg("user", user_msg)



# --- PROACTIVE COVER LETTER FLOW OVERRIDE ---
# Check if the user's message either:
#  1) Mentions "cover letter" in a variety of phrasings (regex catches "cover letter",
#     "make ... cover letter", "create ... cover letter", "draft ... cover letter"), OR
#  2) Contains any URL (basic "http(s)://" detection).
if re.search(r"\b(cover\s*letter|make.*cover\s*letter|create.*cover\s*letter|draft.*cover\s*letter)\b", user_msg.lower()) \
   or re.search(r"https?://", user_msg.lower()):
    # If a link is present anywhere in the message, treat it as the target job posting.
    # This lets users paste a URL and immediately kick off the cover-letter flow.
    if re.search(r"https?://", user_msg.lower()):
        set_target_url(user_msg)
        # Begin the guided cover-letter information collection flow
    # (asks the user for details/resume bits if needed).
    start_collection(render=ui.render_msg)
       # Stop further Streamlit processing in this run so we don't
    # fall through to other intent handlers or render duplicate UI.
    st.stop()

    # --- COVER LETTER PROFILE COLLECTION OVERRIDE ---
if st.session_state.get("collecting_cover_profile", False):
    cl_flow_handle_user_message(user_msg, ui.render_msg)
    st.stop()


def handle_user_message(message_text, render):
    if llm_is_resume_question(message_text):
        resume_json = st.session_state.get("resume_json", {})
        reply = answer_from_resume(message_text, resume_json)
        render("assistant", reply or "No resume data available.")
        return True
    consumed = cl_flow_handle_user_message(message_text, render)
    return consumed

t = (user_msg or "").lower().strip()
if re.search(r"\b(cover\s*letter|make.*cover\s*letter|create.*cover\s*letter|draft.*cover\s*letter)\b", t):
    df_sel = st.session_state.get("last_results_df")
    if df_sel is not None and len(df_sel) > 0:
        matches = df_sel[
            df_sel["company"].astype(str).str.lower().str.contains(t, na=False) |
            df_sel["title"].astype(str).str.lower().str.contains(t, na=False)
        ]
        if len(matches) == 1:
            url_col = "link" if "link" in matches.columns else ("url" if "url" in matches.columns else None)
            if url_col:
                set_target_url(str(matches.iloc[0][url_col]))
    start_collection(render=ui.render_msg)
    st.stop()

raw_intent = classify_intent(user_msg)
intent = raw_intent.strip().lower()
st.sidebar.caption(f"🎯 Intent (LLM): {intent}")

if intent == "resume_question":
    data = st.session_state.get("resume_data") or {}
    if not data:
        reply = "Please upload your résumé (PDF/DOCX/TXT) using the sidebar, then ask your question."
        ui.render_msg("assistant", reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    try:
        reply = answer_from_resume(user_msg, data)
    except Exception:
        reply = "I ran into an issue analyzing your résumé. Please re-upload it and try again."
    ui.render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

elif intent == "internship_search":
    need_refresh = (cache_age_hours() > 24) or any(w in user_msg.lower() for w in ["refresh", "reload", "latest"])
    csusb_df = load_cached_df()
    if csusb_df.empty or need_refresh:
        if need_refresh:
            fetch_csusb_df.clear()
        with st.spinner("📡 Fetching CSUSB CSE links..."):
            csusb_df = fetch_csusb_df()
    df_all = csusb_df.copy()
    if "link" in df_all.columns:
        df_all = df_all[df_all["link"].astype(str).str.startswith(("http://", "https://"))]
    if "source" in df_all.columns:
        df_all = df_all[df_all["source"].astype(str).str.contains("csusb.edu/cse/internships-careers", na=False)]
    for col in ["title", "company", "link"]:
        if col not in df_all.columns:
            df_all[col] = ""
    if "host" not in df_all.columns:
        df_all["host"] = df_all["link"].map(lambda u: urlparse(u).netloc if isinstance(u, str) else "")
    try:
        filt: Dict = parse_query_to_filter(user_msg) or {}
    except Exception:
        filt = {}
    show_all = bool(filt.get("show_all"))
    df = df_all.copy()
    applied_any_filter = False
    def _low(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().fillna("")
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
    for kw in (filt.get("title_keywords") or []):
        kw = (kw or "").strip().lower()
        if kw:
            import re as _re
            pat = _re.escape(kw)
            if "title" in df.columns:
                df = df[_low(df["title"]).str.contains(pat)]
                applied_any_filter = True
    for sk in (filt.get("skills") or []):
        sk = (sk or "").strip().lower()
        if sk and "title" in df.columns:
            import re as _re
            pat = _re.escape(sk)
            df = df[_low(df["title"]).str.contains(pat)]
            applied_any_filter = True
    results = df_all if show_all else df
    if results.empty:
        msg = "I couldn’t find any matching links on the CSUSB CSE page."
        if not show_all:
            msg += " Say **“show all internships”** to list everything."
        ui.render_msg("assistant", msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
    keep_cols = [c for c in ["title", "company", "link"] if c in results.columns]
    if "link" in keep_cols:
        results = results[keep_cols].drop_duplicates(subset=["link"], keep="first")
    else:
        results = results[keep_cols].drop_duplicates()
    # --- PERSISTENT FILTERED LINKS: STORE IN SESSION_STATE FOR DISPLAY AT TOP ---
    if results.empty:
     st.session_state.filtered_links = []
     st.session_state.current_filter = None 
    elif show_all:
     st.session_state.filtered_links = list(results["link"]) if "link" in results.columns else []
     st.session_state.current_filter = "All"
    elif applied_any_filter:
     st.session_state.filtered_links = list(results["link"]) if "link" in results.columns else []
    # Pick a useful filter name/label
     st.session_state.current_filter = (
        str(filt.get("company_name")) or
        " & ".join(filt.get("title_keywords") or []) or
        " & ".join(filt.get("skills") or []) or "Filtered"
    )
    else:
     st.session_state.filtered_links = []
     st.session_state.current_filter = None

    if show_all:
        summary = f"Here are **all {len(results)}** links listed on the CSUSB CSE page."
    elif applied_any_filter:
        summary = f"Here are **{len(results)}** matching link(s) from the CSUSB CSE page."
    else:
        summary = f"I found **{len(results)}** link(s). Ask for a company (e.g., 'nasa') or say 'show all internships'."
            

    ui.render_msg("assistant", summary)
    st.session_state.messages.append({"role": "assistant", "content": summary})

    render_links_as_assistant_message(results)
    
    ui.render_found_links_table(results)
else:
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
