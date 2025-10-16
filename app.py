import os, time, math, re
from collections import deque
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from scraper import (
    scrape_csusb_listings,
    CSUSB_CSE_URL,
    quick_company_links_playwright,
)
from query_to_filter import parse_query_to_filter, classify_intent

APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"

# ---------- Page setup ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")

from pathlib import Path
def inject_css(path: str = "styles.css"):
    p = Path(path)
    if p.exists():
        mtime = p.stat().st_mtime
        key = "css_mtime"
        if st.session_state.get(key) != mtime:
            st.session_state[key] = mtime
            st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

inject_css() 

st.title(APP_TITLE)
st.caption(
    f"Source: {CSUSB_CSE_URL} • Ask anything. For internships try: "
    "“nasa internships”, “google internships”, “only java developer internships”, "
    "“python qa remote”, or “show all internships”."
)

# Load optional CSS
try:
    with open("styles.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- Mode toggle ---
mode = st.sidebar.radio("Mode", ["Auto", "General chat", "Internships"], index=0)

# ---------- Rate limit (10/min) ----------
if "q_times" not in st.session_state:
    st.session_state.q_times = deque()

def allow_query() -> bool:
    now = time.time()
    while st.session_state.q_times and (now - st.session_state.q_times[0]) > 60:
        st.session_state.q_times.popleft()
    if len(st.session_state.q_times) >= 10:
        st.error(
            "You’ve reached the limit of 10 questions per minute because the server has limited resources. "
            "Please try again in about a minute."
        )
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

@st.cache_data(show_spinner=True, ttl=6*60*60)  # auto-refresh every 6 hours
def fetch_csusb_df(max_pages: int = 80) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = scrape_csusb_listings(deep=False, max_pages=max_pages)
    df.to_parquet(PARQUET_PATH, index=False)
    return df

# ---------- Small talk intents & patterns ----------
SMALLTALK_PATTERNS = re.compile(
    r"\b(hi|hello|hey|yo|sup|what'?s up|how are you|how’s it going|"
    r"your name|who are you|name\??|thanks|thank you|ty|thx|"
    r"your age|how old are you|tell me a joke|joke|bye|goodbye|"
    r"what can you do|capabilities|help|idk|i don'?t know)\b",
    re.I,
)

def is_smalltalk(txt: str) -> bool:
    return bool(SMALLTALK_PATTERNS.search(txt))

# ---------- Conversational memory ----------
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "messages" not in st.session_state:
    # seed with a single greeting; we won't repeat it again
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey there! I’m Chatbot. How can I help today?"
    }]
    st.session_state.greeted = True  # mark greeted once

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def history_text(last_n: int = 8) -> str:
    msgs = st.session_state.messages[-last_n:]
    lines = []
    for m in msgs:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)

# ---------- Rule-based small-talk replies (varied, short, no re-greeting) ----------
def quick_smalltalk_reply(txt: str) -> str | None:
    s = txt.strip().lower()

    # greetings (follow-up)
    if re.search(r"\b(hi|hello|hey|yo|sup|what'?s up)\b", s):
        return "Hey again! What’s up?"

    if "how are you" in s or "how’s it going" in s:
        return "I’m doing well, thanks! How about you?"

    if "your name" in s or re.search(r"\bwho are you\b|\bname\??\b", s):
        return "You can call me Chatbot. I’m your AI assistant."

    if "your age" in s or "how old are you" in s:
        return "I don’t have an age—just lots of energy to help. 🙂"

    if "thanks" in s or "thank you" in s or s in {"ty", "thx"}:
        return "You’re welcome!"

    if "sorry" in s:
        return "No worries!"

    if "joke" in s:
        return "Why did the developer go broke? They used up all their cache."

    if s in {"bye", "goodbye"}:
        return "Take care! Ping me anytime."

    if "what can you do" in s or "capabilities" in s or s == "help":
        return ("I can chat, answer questions, brainstorm, draft or edit text, "
                "summarize content, and find CSUSB internship links. What do you need?")

    if s in {"idk", "i dont know", "i don't know"}:
        return "No problem—want to: (1) learn something, (2) write or edit text, or (3) plan a task?"

    return None

# ---------- General chat with LLM fallback (uses history, short & human) ----------
def llm_general_reply(user_text: str) -> str:
    # 1) small-talk rules first (most deterministic)
    quick = quick_smalltalk_reply(user_text)
    if quick:
        return quick

    # 2) LLM for everything else
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            model=os.getenv("MODEL_NAME", "qwen2.5:0.5b"),
            temperature=0.3,
            streaming=False,
            model_kwargs={"num_ctx": 1536, "num_predict": 180},
        )

        sys = (
            "You are a helpful, friendly assistant. Reply contextually to each message, "
            "keeping replies short (1–4 sentences), natural, and human. "
            "Do NOT greet repeatedly; greet only once per session. "
            "Use first person and remember prior turns in the given chat history. "
            "If you don’t know, say so briefly and ask one focused follow-up. "
            "Avoid filler. Vary phrasing."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys),
            ("human", "Chat history:\n{history}\n\nUser: {q}\nAssistant (concise):")
        ])
        return (prompt | llm).invoke({"history": history_text(), "q": user_text}).content.strip()
    except Exception:
        # conservative fallback (never generic greeting)
        return "Got it. I’ll keep it short—what outcome are you aiming for?"

# ---------- Utility helpers ----------
def nontrivial_filters(f: dict) -> int:
    """
    Count real job filters; ignore random words so small-talk doesn't trigger internship mode.
    """
    n = 0
    if f.get("show_all"): n += 1
    if f.get("company_name") and len(str(f["company_name"]).strip()) > 2: n += 1
    if any(len(str(s).strip()) > 1 for s in (f.get("skills") or [])): n += 1

    JOBISH = {"intern","developer","engineer","analyst","qa","data","software","security","cloud","ml","ai"}
    tks = [str(t).lower().strip() for t in (f.get("title_keywords") or [])]
    if any(t in JOBISH for t in tks):
        n += 1

    if any(f.get(k) for k in ("city","state","country","zipcode","remote_type","education_level","experience_level","salary_min")):
        n += 1
    return n

def describe_filters(f: dict) -> str:
    parts = []
    if f.get("company_name"): parts.append(f["company_name"])
    if f.get("title_keywords"): parts.append(" ".join(f["title_keywords"]))
    if f.get("skills"): parts.append(" ".join(f["skills"]))
    loc = " ".join(filter(None, [f.get("city"), f.get("state"), f.get("country")]))
    if loc: parts.append(loc)
    return ", ".join(parts)

def links_md(df: pd.DataFrame) -> str:
    if df.empty: return "_No links available._"
    out = ["**Apply links:**\n"]
    for i, r in enumerate(df.itertuples(index=False), start=1):
        title = (getattr(r, "title", "") or "Internship").strip()
        company = (getattr(r, "company", "") or "").strip()
        label = f"{title}" + (f" — {company}" if company else "")
        link = (getattr(r, "link", "") or "").strip()
        out.append(f"{i}. [{label}]({link}) — **Apply**")
    return "\n".join(out)

def any_match(df_: pd.DataFrame, cols: List[str], pattern: str, regex=True, flags=re.I):
    mask = None
    for c in cols:
        if c not in df_.columns: continue
        m = df_[c].fillna("").str.contains(pattern, regex=regex, flags=flags, na=False)
        mask = m if mask is None else (mask | m)
    if mask is None:
        return pd.Series([False] * len(df_), index=df_.index)
    return mask

# ---------- Input ----------
user_msg = st.chat_input("Type your question…")
if not user_msg:
    st.stop()
if not allow_query():
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_msg})
with st.chat_message("user"):
    st.markdown(user_msg)

# ---------- Routing ----------
filters = parse_query_to_filter(user_msg)
intent  = classify_intent(user_msg)
txt_lo  = user_msg.lower()

has_intern_kw = "intern" in txt_lo
has_smalltalk = is_smalltalk(txt_lo)
has_filters   = nontrivial_filters(filters) > 0

# Debug chip
st.sidebar.caption(f"Intent: {intent} | intern_kw={has_intern_kw} | smalltalk={has_smalltalk} | filters={has_filters}")

if mode == "General chat":
    route = "general"
elif mode == "Internships":
    route = "internships"
else:
    # AUTO:
    if has_smalltalk and intent != "internship_search" and not has_intern_kw:
        route = "general"
    elif intent == "internship_search":
        route = "internships"
    elif has_intern_kw or has_filters:
        route = "internships"
    else:
        route = "general"

# ---------- GENERAL CHAT ----------
if route == "general":
    reply = llm_general_reply(user_msg)
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# ---------- INTERNSHIP SEARCH ----------
need_refresh = cache_age_hours() > 6 or any(w in txt_lo for w in ["refresh", "reload", "latest", "new"])
df = load_cached_df()
if df.empty or need_refresh or ("details" not in df.columns):
    if need_refresh:
        fetch_csusb_df.clear()  # force re-scrape on user request
    with st.spinner("Fetching internships from the CSUSB CSE Internships & Careers page…"):
        df = fetch_csusb_df(max_pages=int(os.getenv("MAX_PAGES", "80")))

table = df.copy()

# Apply filters unless “show all”
if not filters.get("show_all"):
    # Company
    comp = (filters.get("company_name") or "").strip()
    if comp:
        pat = "(" + "|".join(map(re.escape, [comp] + ([p for p in comp.split() if len(p) > 2] if " " in comp else []))) + ")"
        mask = any_match(table, ["company","title","details","host"], pat, regex=True, flags=re.I)
        table = table[mask]
        if len(table) == 0:
            fb = quick_company_links_playwright(comp)
            if isinstance(fb, pd.DataFrame) and len(fb) > 0:
                table = fb

    # Role/title
    title_tokens = [t for t in (filters.get("title_keywords") or []) if str(t).strip()]
    if title_tokens:
        strict = bool(re.search(r"\b(only|strict|exact)\b", user_msg, re.I))
        if strict:
            SYN = {
                "qa": ["qa","quality assurance","test","testing","software tester"],
                "developer": ["developer","software developer","software engineer","sde","programmer"],
                "java": ["java","java developer","java engineer"],
                "business analyst": ["business analyst","ba","requirements analyst"],
                "data analyst": ["data analyst","analytics","business intelligence","bi analyst"],
            }
            alts = []
            for t in title_tokens:
                alts += SYN.get(t.lower(), [t])
            pat = r"(" + "|".join(map(re.escape, sorted(set(alts)))) + r")"
            table = table[any_match(table, ["title","details"], pat, regex=True, flags=re.I)]
        else:
            pat = r"(" + "|".join(map(re.escape, title_tokens)) + r")"
            table = table[any_match(table, ["title","details"], pat, regex=True, flags=re.I)]

    # Skills
    for s in (filters.get("skills") or []):
        s = str(s).strip()
        if s:
            table = table[any_match(table, ["title","details"], re.escape(s), regex=True, flags=re.I)]

    # Remote / location / misc
    if filters.get("remote_type"):
        rt = filters["remote_type"]
        cols = ["remote"] if "remote" in table.columns else ["details"]
        table = table[any_match(table, cols, re.escape(rt), regex=True, flags=re.I)]
    for k in ("city","state","country","zipcode"):
        v = (filters.get(k) or "").strip()
        if v:
            table = table[any_match(table, ["title","location","details"], re.escape(v), regex=True, flags=re.I)]
    if filters.get("education_level"):
        table = table[any_match(table, ["details"], re.escape(str(filters["education_level"])), regex=True)]
    if filters.get("experience_level"):
        table = table[any_match(table, ["details"], re.escape(str(filters["experience_level"])), regex=True)]
    if filters.get("salary_min") and "salary" in table.columns:
        try:
            minv = int(str(filters["salary_min"]).replace("$","").replace(",",""))
            amt = table["salary"].fillna("").str.replace(",","").str.extract(r"(\d+)")[0].astype(float)
            table = table[amt >= minv]
        except Exception:
            pass

# Ensure columns exist
cols = ["title","company","location","posted_date","salary","education","remote","host","link","source","details"]
for c in cols:
    if c not in table.columns:
        table[c] = None

desc = describe_filters(filters)
if len(table) == 0:
    header = f"Sorry, I couldn’t find any internships on the CSUSB page" + (f" for **{desc}**." if desc else ".")
else:
    header = f"Here are **{len(table)}** internships from the CSUSB page" + (f" for **{desc}**." if desc else ".")

answer_md = header + "\n\n" + links_md(table)

with st.chat_message("assistant"):
    st.markdown(answer_md)
    if not table.empty:
        st.dataframe(
            table[cols],
            use_container_width=True,
            hide_index=True,
            column_config={"link": st.column_config.LinkColumn("link", help="Open posting")},
        )

st.session_state.messages.append({"role": "assistant", "content": answer_md})
