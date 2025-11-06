import os, time, math
from resume_parser import  answer_from_resume

from collections import deque
from pathlib import Path

import pandas as pd
import streamlit as st
# from streamlit import session_state as ss


from llm import initialize_llm, llm_query, ensure_ollama_ready, llm_general_reply, llm_internship_search_directed, extract_preference_from_response
from scraper import scrape_csusb_listings
from query_to_filter import classify_intent
from ui import setup_ui, get_max_hops, get_mode, is_deep_mode, setup_session_states, render_msg, render_message_history, init_resume_ui
# === NEW: résumé helpers ===
from resume_parser import (
    answer_from_resume,
)

APP_TITLE = "CSUSB Internship Finder Agent"

DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------- Page setup ----------
setup_ui(APP_TITLE)

ensure_ollama_ready()

setup_session_states(DATA_DIR)

llm = initialize_llm()

# ---------- Rate limit (10/min) ----------
if "q_times" not in st.session_state:
    st.session_state.q_times = deque()

def allow_query() -> bool:
    now = time.time()
    while st.session_state.q_times and (now - st.session_state.q_times[0]) > 60:
        st.session_state.q_times.popleft()
    if len(st.session_state.q_times) >= 10:
        st.error("⏱️ Rate limit reached: 10 queries per minute. Please wait ~60 seconds.")
        return False
    st.session_state.q_times.append(now)
    return True

# ---------- Cache helpers ----------
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


# Preference questions - one at a time
PREF_QUESTIONS = [
    {"key": "interests", "question": "What are your interests or fields you'd like to work in? (e.g., software development, data science, product management, marketing)"},
    {"key": "roles", "question": "What specific job roles are you looking for? (e.g., Software Engineer Intern, Data Analyst Intern, Product Manager Intern)"},
    {"key": "location", "question": "Do you have a location preference? (e.g., Remote, On-site, specific city/region)"},
    {"key": "skills", "question": "What are your key technical or professional skills? (e.g., Python, JavaScript, problem-solving, communication)"}
]

render_message_history()

def history_text(last_n: int = 8) -> str:
    msgs = st.session_state.messages[-last_n:]
    return "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in msgs])

init_resume_ui()

# Chat input
user_msg = st.chat_input(placeholder="Type your question…", accept_file=True, file_type=["docx", "pdf"])

# Stop if no input
if not user_msg:
    st.stop()
if not allow_query():
    st.stop()

render_msg("user", user_msg.text)

with st.spinner("Thinking..."):
    response = llm_query(llm, user_msg.text)

render_msg("llm", response)
st.rerun()


# ---------- Routing ----------
intent = classify_intent(user_msg)
txt_lo = user_msg.lower()

# === NEW: detect résumé questions BEFORE routing ===
RESUME_KEYS = [
    "resume","résumé","cv","experience","experiences","work history","employment",
    "skills","projects","education","school","degree","certifications","gpa",
    "linkedin","github","portfolio","website","email","phone","summary","address","name"
]

def is_resume_question(q: str) -> bool:
    if any(k in q for k in ["resume","résumé","cv"]):
        return True
    if st.session_state.get("resume_data"):
        if any(k in q for k in RESUME_KEYS):
            return (" my " in f" {q} ") or (" me " in f" {q} ") or True
    return False

if is_resume_question(txt_lo):
    data = st.session_state.get("resume_data") or {}
    if data:
        reply = answer_from_resume(user_msg, data)
    else:
        reply = "I don't have a résumé saved yet. Click the **➕** button beside the chat box to upload one."
    render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# --- Mode + Intent routing ---
txt_lo = user_msg.lower()
intent = classify_intent(user_msg)
has_intern_kw = "intern" in txt_lo

st.sidebar.caption(f"🎯 Intent: {intent}")

mode = get_mode()
if mode == "General chat":
    route = "general"
elif mode == "Internships":
    route = "internships"
else:
    if intent == "internship_search" or has_intern_kw:
        route = "internships"
    else:
        route = "general"

# ---------- GENERAL CHAT ----------
if route == "general":
    with st.spinner("Thinking..."):
        reply = llm_general_reply(user_msg, history_text())
    render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# ---------- INTERNSHIP SEARCH: PREFERENCE COLLECTION (ONE QUESTION AT A TIME) ----------
# Start preference collection if not already doing so
if not st.session_state.get("collecting_prefs"):
    st.session_state.collecting_prefs = True
    st.session_state.current_pref_step = 0
    st.session_state.initial_query = user_msg
    st.session_state.user_preferences = {}

# Check if we're still collecting preferences
if st.session_state.get("collecting_prefs"):
    current_step = st.session_state.get("current_pref_step", 0)
    
    # If this is not the first message, extract the user's answer to the previous question
    if current_step > 0:
        prev_pref_key = PREF_QUESTIONS[current_step - 1]["key"]
        extracted = extract_preference_from_response(user_msg, prev_pref_key)
        
        # Store extracted preferences
        if extracted:
            st.session_state.user_preferences[prev_pref_key] = extracted
    
    # Check if we've asked all questions
    if current_step >= len(PREF_QUESTIONS):
        # Done collecting preferences - now do the search
        st.session_state.collecting_prefs = False
        # Proceed to internship search below
    else:
        # Ask the next question
        question = PREF_QUESTIONS[current_step]["question"]
        
        render_msg("assistant", question)
        st.session_state.messages.append({"role": "assistant", "content": question})
        
        # Increment step for next iteration
        st.session_state.current_pref_step += 1
        st.stop()

# ---------- INTERNSHIP SEARCH (LLM-GUIDED NAVIGATION) ----------
need_refresh = cache_age_hours() > 24 or any(w in txt_lo for w in ["refresh", "reload", "latest"])
csusb_df = load_cached_df()

if csusb_df.empty or need_refresh:
    if need_refresh:
        fetch_csusb_df.clear()
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    with status_placeholder:
        st.info("📡 Fetching company career pages from CSUSB...")
    
    progress_bar.progress(50)
    csusb_df = fetch_csusb_df()
    progress_bar.progress(100)
    status_placeholder.empty()
    progress_bar.empty()

# LLM decides which companies to navigate based on preferences, backend navigates them
user_prefs = st.session_state.get("user_preferences", {})
initial_query = st.session_state.get("initial_query", user_msg)

with st.spinner("Searching for internships matching your preferences..."):
    answer_md, results_df = llm_internship_search_directed(initial_query, csusb_df, get_max_hops(), user_prefs)

render_msg("assistant", answer_md)
st.session_state.messages.append({"role": "assistant", "content": answer_md})

# Show results table if links were found
if not results_df.empty:
    st.markdown("### 📊 Found Links")
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Links", len(results_df))
    with col2:
        unique_companies = results_df["company"].dropna().nunique()
        st.metric("Companies", unique_companies)
    
    # Display table
    cols = ["title", "company", "url"]
    display_cols = [c for c in cols if c in results_df.columns and results_df[c].notna().any()]
    st.dataframe(
        results_df[display_cols],
        width='stretch',
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Visit", help="Click to open link"),
            "title": st.column_config.TextColumn("Link Text", width="medium"),
            "company": st.column_config.TextColumn("Company", width="medium"),
        },
    )
    
    # Download option
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"csusb_internships_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
