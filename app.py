import os, time, math, re
from collections import deque
from pathlib import Path
from typing import List, Dict
import streamlit as st
import pandas as pd

from scraper import scrape_csusb_listings, CSUSB_CSE_URL
from query_to_filter import parse_query_to_filter, classify_intent

APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"

st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
st.title(APP_TITLE)
st.caption(
    f"Source: {CSUSB_CSE_URL} • Ask anything. For internships, try: "
    "“strict qa internships remote”, “only java developer internships”, “nasa internships”, “show all internships”."
)

# ---------------- Rate limit 10/min ----------------
if "q_times" not in st.session_state:
    st.session_state.q_times = deque()
def check_rate_limit():
    now = time.time()
    while st.session_state.q_times and (now - st.session_state.q_times[0]) > 60:
        st.session_state.q_times.popleft()
    if len(st.session_state.q_times) >= 10:
        st.error("You’ve reached the limit of 10 questions per minute because the server has limited resources. Please try again in 3 minutes.")
        return False
    st.session_state.q_times.append(now); return True

# ---------------- Cached data ----------------
@st.cache_data(show_spinner=False)
def load_cached_df() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        try:
            return pd.read_parquet(PARQUET_PATH)
        except Exception:
            pass
    return pd.DataFrame()

def _cache_age_hours() -> float:
    if not PARQUET_PATH.exists(): return math.inf
    return (time.time() - PARQUET_PATH.stat().st_mtime) / 3600.0

@st.cache_data(show_spinner=True)
def fetch_all_internships(deep: bool=True) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = scrape_csusb_listings(deep=deep, max_pages=int(os.getenv("MAX_PAGES", "80")))
    df.to_parquet(PARQUET_PATH, index=False)
    return df

# ---------------- LLM helpers ----------------
def _llm_reply(system: str, user: str) -> str:
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_HOST","http://127.0.0.1:11434"),
            model=os.getenv("MODEL_NAME","qwen2.5:0.5b"),
            temperature=0.3, streaming=False,
            model_kwargs={"num_ctx":1024, "num_predict":180},
        )
        return (ChatPromptTemplate.from_messages([("system",system),("human","{q}")]) | llm).invoke({"q":user}).content.strip()
    except Exception:
        return ""

def _general_chat_reply(user_text: str) -> str:
    sys = "You are the CSUSB Internship Finder Agent. When the user is NOT asking for internships, answer helpfully in 1–3 sentences."
    return _llm_reply(sys, user_text) or "I’m the CSUSB Internship Finder Agent."

def _search_summary(user_text: str, count: int, df: pd.DataFrame) -> str:
    sample = "; ".join(df["title"].head(5).fillna("").tolist()) if not df.empty else ""
    sys = "Summarize matches in 1–2 short sentences. Don’t list items; I will render links separately."
    return _llm_reply(sys, f"Query: {user_text}\nMatches: {count}\nExamples: {sample}") or "Here are internships from the CSUSB page."

def _links_md(df: pd.DataFrame) -> str:
    if df.empty: return "_No links available._"
    lines = []
    for i,r in enumerate(df.itertuples(index=False), start=1):
        title = (getattr(r,"title","") or "Internship").strip()
        company = (getattr(r,"company","") or "").strip()
        label = f"{title}" + (f" — {company}" if company else "")
        link = (getattr(r,"link","") or "").strip()
        lines.append(f"{i}. [{label}]({link}) — **Apply**")
    return "**Apply links:**\n\n" + "\n".join(lines)

# ---------------- Chat history (ordered) ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":"assistant",
        "content":"Hi! I’m the CSUSB Internship Finder Agent. Ask anything. For internships, tell me the role, skills, or location you want."
    }]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Input ----------------
user_msg = st.chat_input("Type your question…")
if not user_msg: st.stop()
if not check_rate_limit(): st.stop()

st.session_state.messages.append({"role":"user","content":user_msg})
with st.chat_message("user"): st.markdown(user_msg)

# ---------------- Parse filters / intent ----------------
filters = parse_query_to_filter(user_msg)
intent = classify_intent(user_msg)

# Force search if company or role hint is present
force_search = bool(filters.get("company_name")) or bool(filters.get("title_keywords")) or bool(filters.get("skills"))
if intent == "general_question" and force_search:
    intent = "internship_search"

if intent == "general_question":
    reply = _general_chat_reply(user_msg)
    with st.chat_message("assistant"): st.markdown(reply)
    st.session_state.messages.append({"role":"assistant","content":reply})
    st.stop()

# ---------------- Fetch data ----------------
ask_all = bool(filters.get("show_all"))
need_refresh = _cache_age_hours() > 6 or any(w in user_msg.lower() for w in ["refresh","reload","latest","new"])

df = load_cached_df()
if df.empty or need_refresh or ("details" not in df.columns):
    with st.spinner("Fetching internships from the CSUSB CSE Internships & Careers page…"):
        df = fetch_all_internships(deep=True)

table = df.copy()

# ---------------- Strict / Broad role matching ----------------
role_mode_env = os.getenv("ROLE_MATCH_MODE","").strip().lower()
role_mode = filters.get("role_match","broad")
if role_mode_env in ("strict","broad"):
    role_mode = role_mode_env
STRICT = (role_mode == "strict") or bool(re.search(r"\b(only|strict|exact)\b", user_msg, re.I))

ROLE_SYNONYMS: Dict[str, List[str]] = {
    "qa": ["qa", "quality assurance", "test", "testing", "software tester"],
    "developer": ["developer","software developer","software engineer","sde","programmer"],
    "java": ["java","java developer","java engineer"],
    "business analyst": ["business analyst","ba","requirements analyst"],
    "data analyst": ["data analyst","analytics","business intelligence","bi analyst"],
}

def _role_pattern(tokens: List[str], strict: bool) -> str:
    if not tokens: return ""
    if strict:
        alts = []
        for t in tokens:
            syns = ROLE_SYNONYMS.get(t.lower(), [t])
            alts.extend([fr"\b{re.escape(s)}\b" for s in syns])
        return "(" + "|".join(sorted(set(alts))) + ")"
    return "(" + "|".join([re.escape(t) for t in tokens]) + ")"

def _any_match(df: pd.DataFrame, cols: List[str], pattern: str, strict=False):
    mask = None
    for c in cols:
        if c not in df.columns: continue
        m = df[c].fillna("").str.contains(pattern, flags=re.I if strict else 0, regex=strict, case=not strict, na=False)
        mask = m if mask is None else (mask | m)
    return mask if mask is not None else pd.Series([False]*len(df))

# ---------------- Apply filters (CSUSB-only) ----------------
if not ask_all:
    # Role / keywords
    title_keywords = [t for t in (filters.get("title_keywords") or []) if str(t).strip()]
    if title_keywords:
        pat = _role_pattern(title_keywords, STRICT)
        if pat:
            mask = _any_match(table, ["title","details"], pat, strict=STRICT)
            table = table[mask]

    # Skills
    for s in (filters.get("skills") or []):
        s = str(s).strip()
        if not s: continue
        mask = _any_match(table, ["title","details"], re.escape(s), strict=False)
        table = table[mask]

    # Remote type
    rtype = (filters.get("remote_type") or "").strip().lower()
    if rtype:
        if "remote" in table.columns:
            table = table[table["remote"].fillna("").str.contains(rtype, case=False, na=False)]
        else:
            table = table[_any_match(table, ["details"], rtype)]

    # Location: city/state/country/zip best-effort
    for key in ("city","state","country","zipcode"):
        val = (filters.get(key) or "").strip()
        if val:
            table = table[_any_match(table, ["title","location","details"], re.escape(val))]

    # Education / Experience / Salary
    if filters.get("education_level"):
        table = table[_any_match(table, ["details"], re.escape(str(filters["education_level"])))]
    if filters.get("experience_level"):
        table = table[_any_match(table, ["details"], re.escape(str(filters["experience_level"])))]
    if filters.get("salary_min") and "salary" in table.columns:
        try:
            minv = int(str(filters["salary_min"]).replace("$","").replace(",",""))
            amt = table["salary"].fillna("").str.replace(",","").str.extract(r"(\d+)")[0].astype(float)
            table = table[amt >= minv]
        except Exception:
            pass

    # Company filter (title/details/company/host)
    comp = (filters.get("company_name") or "").strip().lower()
    if comp:
        def _match_company(df: pd.DataFrame, token: str):
            pats = [re.escape(token)]
            if " " in token:
                parts=[p for p in token.split() if len(p) > 2]
                pats.extend([re.escape(p) for p in parts])
            pat = "(" + "|".join(pats) + ")"
            m = (
                df["company"].fillna("").str.lower().str.contains(pat, regex=True) |
                df["title"].fillna("").str.lower().str.contains(pat, regex=True) |
                df["details"].fillna("").str.lower().str.contains(pat, regex=True) |
                df["host"].fillna("").str.lower().str.contains(pat, regex=True)
            )
            return m
        table = table[_match_company(table, comp)]

# Fallback: if the user demanded STRICT/ONLY, do NOT flood with all
if len(table) == 0 and len(df) > 0 and not STRICT:
    table = df.copy()

# ---------------- Render ----------------
cols = ["title","company","location","posted_date","salary","education","remote","host","link","source","details"]
for c in cols:
    if c not in table.columns: table[c] = None

summary = _search_summary(user_msg, len(table), table)
links_md = _links_md(table)

with st.chat_message("assistant"):
    st.markdown(summary)
    st.markdown(links_md)
    st.dataframe(table[cols], use_container_width=True, hide_index=True,
                 column_config={"link": st.column_config.LinkColumn("link", help="Open posting")})

# store only a short echo in history (prevents “jumping up” on rerun)
st.session_state.messages.append({"role":"assistant","content":f"Listed {len(table)} internships."})
