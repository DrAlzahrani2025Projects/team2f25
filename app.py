import os, time, math, re
from collections import deque
from pathlib import Path
from typing import List, Dict
import pandas as pd
import streamlit as st
from flask import Flask, request, jsonify
from intents import detect_intent

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_text = request.json.get("message", "").strip()
    intent = detect_intent(user_text)

    if intent == "empty":
        reply = "Say hi or tell me what kind of internship you're looking for 😊"
    elif intent == "greeting":
        reply = "Hey! I'm your internship finder bot. Try: 'Find software internships in Bangalore for January.'"
    elif intent == "search_internships":
        reply = f"Got it — searching internships related to '{user_text}'..."
    else:
        reply = "Sorry, I didn’t get that. Try: 'Marketing internships in Delhi (paid)'."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)

from scraper import (
    scrape_csusb_listings,
    CSUSB_CSE_URL,
    quick_company_links_playwright,
)
from query_to_filter import parse_query_to_filter, classify_intent

APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"

st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
st.title(APP_TITLE)
st.caption(
    f"Source: {CSUSB_CSE_URL} • Ask anything. For internships, try: "
    "“nasa internships”, “google internships”, “only java developer internships”, "
    "“python qa remote”, or “show all internships”."
)

# ---------------- Rate limit (10/min) ----------------
if "q_times" not in st.session_state:
    st.session_state.q_times = deque()

def check_rate_limit() -> bool:
    now = time.time()
    while st.session_state.q_times and (now - st.session_state.q_times[0]) > 60:
        st.session_state.q_times.popleft()
    if len(st.session_state.q_times) >= 10:
        st.error("You’ve reached the limit of 10 questions per minute. Try again shortly.")
        return False
    st.session_state.q_times.append(now)
    return True

# ---------------- Cache ----------------
@st.cache_data(show_spinner=False)
def load_cached_df() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        try:
            return pd.read_parquet(PARQUET_PATH)
        except Exception:
            pass
    return pd.DataFrame()

def _cache_age_hours() -> float:
    if not PARQUET_PATH.exists():
        return math.inf
    return (time.time() - PARQUET_PATH.stat().st_mtime) / 3600.0

@st.cache_data(show_spinner=True)
def fetch_all_internships() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = scrape_csusb_listings(deep=False, max_pages=int(os.getenv("MAX_PAGES", "80")))
    df.to_parquet(PARQUET_PATH, index=False)
    return df

# ---------------- LLM helpers (small talk) ----------------
def _llm_reply(system: str, user: str) -> str:
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            model=os.getenv("MODEL_NAME", "qwen2.5:0.5b"),
            temperature=0.3,
            streaming=False,
            model_kwargs={"num_ctx": 1024, "num_predict": 180},
        )
        return (
            (ChatPromptTemplate.from_messages([("system", system), ("human", "{q}")]) | llm)
            .invoke({"q": user})
            .content.strip()
        )
    except Exception:
        return ""

def _general_chat_reply(user_text: str) -> str:
    sys = ("You are the CSUSB Internship Finder Agent. "
           "If the user is NOT asking for internships, answer helpfully in 1–3 sentences. "
           "Stay friendly and concise.")
    return _llm_reply(sys, user_text) or "Hello! How can I help?"

def _describe_filters(f: dict) -> str:
    parts = []
    if f.get("company_name"):
        parts.append(f.get("company_name"))
    if f.get("title_keywords"):
        parts.append(" ".join(f["title_keywords"]))
    if f.get("skills"):
        parts.append(" ".join(f["skills"]))
    loc = " ".join([f.get("city",""), f.get("state",""), f.get("country","")]).strip()
    if loc:
        parts.append(loc)
    return ", ".join([p for p in parts if p])

def _search_summary(user_text: str, count: int, df: pd.DataFrame, filters: dict) -> str:
    desc = _describe_filters(filters)
    if count == 0:
        what = f" for **{desc}**" if desc else ""
        return f"Sorry, I couldn’t find any internships on the CSUSB page{what}."
    suffix = f" for **{desc}**" if desc else ""
    return f"Here are **{count}** internships from the CSUSB page{suffix}."

def _links_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No links available._"
    lines = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        title = (getattr(r, "title", "") or "Internship").strip()
        company = (getattr(r, "company", "") or "").strip()
        label = f"{title}" + (f" — {company}" if company else "")
        link = (getattr(r, "link", "") or "").strip()
        lines.append(f"{i}. [{label}]({link}) — **Apply**")
    return "**Apply links:**\n\n" + "\n".join(lines)

# ---------------- Chat history ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I’m the CSUSB Internship Finder Agent. Ask anything. "
                   "For internships, tell me the role, skills, or location you want."
    }]

# Render prior history ONCE (prevents duplicate/“extra” top messages)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Input ----------------
user_msg = st.chat_input("Type your question…")
if not user_msg:
    st.stop()
if not check_rate_limit():
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_msg})
with st.chat_message("user"):
    st.markdown(user_msg)

# ---------------- Intent / filters ----------------
filters = parse_query_to_filter(user_msg)
intent = classify_intent(user_msg)

force_search = bool(
    filters.get("company_name")
    or filters.get("title_keywords")
    or filters.get("skills")
    or filters.get("show_all")
)

if intent == "general_question" and not force_search:
    reply = _general_chat_reply(user_msg)
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# ---------------- Data ----------------
ask_all = bool(filters.get("show_all"))
need_refresh = _cache_age_hours() > 6 or any(w in user_msg.lower() for w in ["refresh", "reload", "latest", "new"])

df = load_cached_df()
if df.empty or need_refresh or ("details" not in df.columns):
    with st.spinner("Fetching internships from the CSUSB CSE Internships & Careers page…"):
        df = fetch_all_internships()

table = df.copy()

# ---------------- Matching mode ----------------
role_mode_env = os.getenv("ROLE_MATCH_MODE", "").strip().lower()
role_mode = filters.get("role_match", "broad")
if role_mode_env in ("strict", "broad"):
    role_mode = role_mode_env
STRICT = (role_mode == "strict") or bool(re.search(r"\b(only|strict|exact)\b", user_msg, re.I))

ROLE_SYNONYMS: Dict[str, List[str]] = {
    "qa": ["qa", "quality assurance", "test", "testing", "software tester"],
    "developer": ["developer", "software developer", "software engineer", "sde", "programmer"],
    "java": ["java", "java developer", "java engineer"],
    "business analyst": ["business analyst", "ba", "requirements analyst"],
    "data analyst": ["data analyst", "analytics", "business intelligence", "bi analyst"],
}

def _role_pattern(tokens: List[str], strict: bool) -> str:
    if not tokens:
        return ""
    if strict:
        alts = []
        for t in tokens:
            alts += [fr"\b{re.escape(s)}\b" for s in ROLE_SYNONYMS.get(t.lower(), [t])]
        return "(" + "|".join(sorted(set(alts))) + ")"
    return "(" + "|".join([re.escape(t) for t in tokens]) + ")"

def _any_match(df_: pd.DataFrame, cols: List[str], pattern: str, strict: bool = False):
    mask = None
    for c in cols:
        if c not in df_.columns:
            continue
        m = df_[c].fillna("").str.contains(
            pattern,
            flags=re.I if strict else 0,
            regex=strict,
            case=not strict,
            na=False,
        )
        mask = m if mask is None else (mask | m)
    return mask if mask is not None else pd.Series([False] * len(df_))

# ---------------- Apply filters (CSUSB-only) ----------------
if not ask_all:
    # 1) Company first
    comp = (filters.get("company_name") or "").strip().lower()
    if comp:
        def _match_company(df_: pd.DataFrame, token: str):
            pats = [re.escape(token)]
            if " " in token:
                pats += [re.escape(p) for p in token.split() if len(p) > 2]
            pat = "(" + "|".join(pats) + ")"
            return (
                df_.get("company", pd.Series([""] * len(df_))).fillna("").str.lower().str.contains(pat, regex=True)
                | df_.get("title", pd.Series([""] * len(df_))).fillna("").str.lower().str.contains(pat, regex=True)
                | df_.get("details", pd.Series([""] * len(df_))).fillna("").str.lower().str.contains(pat, regex=True)
                | df_.get("host", pd.Series([""] * len(df_))).fillna("").str.lower().str.contains(pat, regex=True)
            )
        table = table[_match_company(table, comp)]

        if len(table) == 0:
            fb = quick_company_links_playwright(comp)
            if len(fb) > 0:
                table = fb

    # 2) Role / title keywords
    title_keywords = [t for t in (filters.get("title_keywords") or []) if str(t).strip()]
    if title_keywords:
        pat = _role_pattern(title_keywords, STRICT)
        if pat:
            table = table[_any_match(table, ["title", "details"], pat, strict=STRICT)]

    # 3) Skills / tech stack
    for s in (filters.get("skills") or []):
        s = str(s).strip()
        if not s:
            continue
        table = table[_any_match(table, ["title", "details"], re.escape(s), strict=False)]

    # 4) Remote
    rtype = (filters.get("remote_type") or "").strip().lower()
    if rtype:
        if "remote" in table.columns:
            table = table[table["remote"].fillna("").str.contains(rtype, case=False, na=False)]
        else:
            table = table[_any_match(table, ["details"], rtype)]

    # 5) Location
    for key in ("city", "state", "country", "zipcode"):
        val = (filters.get(key) or "").strip()
        if val:
            table = table[_any_match(table, ["title", "location", "details"], re.escape(val))]

    # 6) Education / experience / salary
    if filters.get("education_level"):
        table = table[_any_match(table, ["details"], re.escape(str(filters["education_level"])))]
    if filters.get("experience_level"):
        table = table[_any_match(table, ["details"], re.escape(str(filters["experience_level"])))]
    if filters.get("salary_min") and "salary" in table.columns:
        try:
            minv = int(str(filters["salary_min"]).replace("$", "").replace(",", ""))
            amt = table["salary"].fillna("").str.replace(",", "").str.extract(r"(\d+)")[0].astype(float)
            table = table[amt >= minv]
        except Exception:
            pass

# ---------------- Render ----------------
cols = ["title","company","location","posted_date","salary","education","remote","host","link","source","details"]
for c in cols:
    if c not in table.columns:
        table[c] = None

summary = _search_summary(user_msg, len(table), table, filters)
links_md = _links_md(table)
assistant_md = summary + "\n\n" + links_md

# Render the current reply and store it once (prevents “extra message at top”)
with st.chat_message("assistant"):
    st.markdown(assistant_md)
    if not table.empty:
        st.dataframe(
            table[cols],
            use_container_width=True,
            hide_index=True,
            column_config={"link": st.column_config.LinkColumn("link", help="Open posting")},
        )

st.session_state.messages.append({"role": "assistant", "content": assistant_md})
