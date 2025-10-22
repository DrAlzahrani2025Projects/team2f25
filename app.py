import os, time, math, re, json
from resume_parser import extract_resume_text, llm_resume_extract, save_resume, answer_from_resume

from collections import deque
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


from scraper import (
    scrape_csusb_listings,
    CSUSB_CSE_URL,
    quick_company_links_playwright,
)
from query_to_filter import parse_query_to_filter, classify_intent

# === NEW: résumé helpers ===
from resume_parser import (
    extract_resume_text,
    llm_resume_extract,
    save_resume,
    answer_from_resume,
)

APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"

# ---------- Page setup ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")

# --- CSS injector ---
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
    f"🎯 Deep search enabled • Source: {CSUSB_CSE_URL}\n\n"
    "Try: **nasa internships**, **google remote intern**, **only java developer**, **show all internships**"
)

# --- Mode toggle ---
mode = st.sidebar.radio("Mode", ["Auto", "General chat", "Internships"], index=0)
deep_mode = st.sidebar.checkbox("Deep Search", value=False, help="⚠️ Slow: Scrapes company career pages for detailed postings (recommended: OFF for first try)")
st.sidebar.markdown("---")
st.sidebar.caption("⚙️ **Settings**")
max_pages = st.sidebar.slider("Max company pages", 5, 50, 10, 5, help="Number of company sites to deep-scrape")

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
def fetch_csusb_df(max_pages: int = 80, deep: bool = True) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = scrape_csusb_listings(deep=deep, max_pages=max_pages)
    df.to_parquet(PARQUET_PATH, index=False)
    return df

# ---------- Conversational memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hey! I'm your CSUSB Internship Finder. I can search the CSUSB CSE page and deep-dive into company career sites. What are you looking for?"
    }]

# === NEW: résumé state + auto-load from disk ===
st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("resume_data", {})
try:
    p_json = DATA_DIR / "resume.json"
    if p_json.exists() and not st.session_state["resume_data"]:
        st.session_state["resume_data"] = json.loads(p_json.read_text(encoding="utf-8"))
        st.session_state["resume_text"] = (DATA_DIR / "resume.txt").read_text(encoding="utf-8") if (DATA_DIR / "resume.txt").exists() else ""
except Exception:
    pass

# --- Message renderer ---
def render_msg(role: str, content: str):
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(content)

# --- Render history ---
for m in st.session_state.messages:
    render_msg(m["role"], m["content"])

def history_text(last_n: int = 8) -> str:
    msgs = st.session_state.messages[-last_n:]
    return "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in msgs])

# ---------- LLM general reply ----------
def llm_general_reply(user_text: str) -> str:
    """
    General small talk / help handled by local Ollama model via LangChain.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.3,
            streaming=False,
            model_kwargs={"num_ctx": 1536, "num_predict": 180}
        )
        
        sys = "You are a helpful and friendly assistant for CSUSB internship search. Keep replies concise and encouraging."
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys),
            ("human", "History:\n{history}\n\nUser: {q}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"history": history_text(), "q": user_text})
        return response.content.strip()
        
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return "I'm here to help! What would you like to know?"

# ---------- Utility helpers ----------
def nontrivial_filters(f: dict) -> int:
    n = 0
    if f.get("show_all"): n += 1
    if f.get("company_name") and len(str(f["company_name"]).strip()) > 2: n += 1
    if any(len(str(s).strip()) > 1 for s in (f.get("skills") or [])): n += 1
    JOBISH = {"intern","developer","engineer","analyst","qa","data","software"}
    if any(t in JOBISH for t in [str(t).lower() for t in (f.get("title_keywords") or [])]):
        n += 1
    return n

def describe_filters(f: dict) -> str:
    parts = []
    if f.get("company_name"): parts.append(f"**{f['company_name']}**")
    if f.get("title_keywords"): parts.append(" ".join(f["title_keywords"]))
    if f.get("skills"): parts.append(f"({', '.join(f['skills'])})")
    if f.get("city") or f.get("state"):
        parts.append(" ".join(filter(None, [f.get("city"), f.get("state")])))
    return " • ".join(parts)

def links_md(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    out = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        title = (getattr(r, "title", "") or "Internship").strip()
        company = (getattr(r, "company", "") or "").strip()
        location = (getattr(r, "location", "") or "").strip()
        label = f"**{title}**"
        if company:
            label += f" at {company}"
        if location:
            label += f" • {location}"
        link = (getattr(r, "link", "") or "").strip()
        out.append(f"{i}. {label}  \n   🔗 [View & Apply]({link})")
    return "\n\n".join(out)

def any_match(df_: pd.DataFrame, cols: List[str], pattern: str, regex=True, flags=re.I):
    mask = None
    for c in cols:
        if c not in df_.columns: continue
        m = df_[c].fillna("").str.contains(pattern, regex=regex, flags=flags, na=False)
        mask = m if mask is None else (mask | m)
    return mask if mask is not None else pd.Series([False] * len(df_), index=df_.index)

# Floating "+" popover anchored to the chat input (no middle duplicate, no flicker)
user_msg = st.chat_input("Type your question…")


st.markdown("""
<style>
  [data-testid="stPopover"] > button {
    width:10px; height:10px; border-radius:9999px;
    padding:0; margin:0; box-shadow:0 6px 18px rgba(0,0,0,.15);
  }
</style>
""", unsafe_allow_html=True)

with st.popover("➕", help="Upload résumé (PDF/DOCX/TXT)"):
    st.write("Upload a résumé file. Extraction runs automatically.")
    up = st.file_uploader("Upload résumé", type=["pdf","docx","txt"], label_visibility="collapsed")
    if up is not None:
        with st.spinner("Extracting résumé…"):
            text = extract_resume_text(up)
            data = llm_resume_extract(text)
            save_resume(data, text)
            st.session_state.resume_text = text
            st.session_state.resume_data = data
        st.success('Résumé saved. Ask “show experience”, “list my skills”, or “what’s my LinkedIn?”.')

# import streamlit.components.v1 as components

components.html("""
<script>
(() => {
  const doc = parent.document;

  function movePlusIntoChat() {
    const chat   = doc.querySelector('[data-testid="stChatInput"]');
    const btn    = doc.querySelector('[data-testid="stPopoverButton"]');         // the ➕ button
    const pop    = btn ? btn.closest('[data-testid="stPopover"]') : null;       // wrapper that makes the blue bar
    if (!chat || !btn || !pop) return;

    // Move the ➕ button INSIDE the chat input container once
    if (!chat.contains(btn)) {
      chat.appendChild(btn);
    }

    // Kill the full-width bar look on the wrapper
    pop.style.background = 'transparent';
    pop.style.boxShadow  = 'none';
    pop.style.border     = '0';
    pop.style.width      = 'auto';
    pop.style.maxWidth   = 'none';
    pop.style.margin     = '0';
    pop.style.position   = 'static';
    pop.style.display    = 'contents'; // wrapper becomes invisible but functionality stays
  }

  // Run and keep it in place as Streamlit rerenders
  movePlusIntoChat();
  new MutationObserver(movePlusIntoChat).observe(parent.document.body, { childList:true, subtree:true });
  addEventListener('resize', movePlusIntoChat, { passive:true });
})();
</script>
""", height=0)

# ---- Only stop if no input AFTER rendering the + button ----
if not user_msg:
    st.stop()
if not user_msg:
    st.stop()
if not allow_query():
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_msg})
render_msg("user", user_msg)

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
    # If the user has uploaded a resume, allow shorthand like "show experience"
    if st.session_state.get("resume_data"):
        if any(k in q for k in RESUME_KEYS):
            # Prefer cases referring to the user's data
            return (" my " in f" {q} ") or (" me " in f" {q} ") or True
    return False

if is_resume_question(txt_lo):
    data = st.session_state.get("resume_data") or {}
    if data:
        reply = answer_from_resume(user_msg, data)
    else:
        reply = "I don’t have a résumé saved yet. Click the **➕** button beside the chat box to upload one."
    render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

    # --- after reading user_msg, before parse_query_to_filter/classify_intent ---
txt_lo = user_msg.lower()

RESUME_KEYS = {
    "resume","résumé","cv","experience","experiences","skills","projects",
    "education","school","degree","certifications","gpa",
    "linkedin","github","portfolio","website","email","phone","name","summary"
}

def is_resume_question(q: str) -> bool:
    if any(k in q for k in ("resume", "résumé", "cv")):
        return True
    if st.session_state.get("resume_data"):  # allow shorthand like "show experience"
        return any(k in q for k in RESUME_KEYS)
    return False


# ---------- Routing (unchanged) ----------
filters = parse_query_to_filter(user_msg)
intent = classify_intent(user_msg)
has_intern_kw = "intern" in txt_lo
has_filters = nontrivial_filters(filters) > 0
st.sidebar.caption(f"🔍 Intent: {intent} | Filters: {has_filters}")

if mode == "General chat":
    route = "general"
elif mode == "Internships":
    route = "internships"
else:
    if intent == "internship_search" or has_intern_kw or has_filters:
        route = "internships"
    else:
        route = "general"

# ---------- GENERAL CHAT ----------
if route == "general":
    with st.spinner("Thinking..."):
        reply = llm_general_reply(user_msg)
    render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# ---------- INTERNSHIP SEARCH ----------
need_refresh = cache_age_hours() > 6 or any(w in txt_lo for w in ["refresh", "reload", "latest"])
df = load_cached_df()

if df.empty or need_refresh:
    if need_refresh:
        fetch_csusb_df.clear()
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    with status_placeholder:
        st.info(f"🔄 {'Deep-scraping' if deep_mode else 'Fetching'} internships from CSUSB CSE page...")
    progress_bar.progress(10)
    df = fetch_csusb_df(max_pages=max_pages, deep=deep_mode)
    progress_bar.progress(100)
    status_placeholder.empty()
    progress_bar.empty()

table = df.copy()

# Apply filters
if not filters.get("show_all"):
    comp = (filters.get("company_name") or "").strip()
    if comp:
        pat = "(" + "|".join(map(re.escape, [comp] + ([p for p in comp.split() if len(p) > 2] if " " in comp else []))) + ")"
        mask = any_match(table, ["company","title","details","host"], pat, regex=True)
        initial_matches = table[mask]
        if len(initial_matches) > 0:
            table = initial_matches
            st.sidebar.caption(f"✓ Found {len(table)} in cached data")
        else:
            st.sidebar.caption(f"⚠️ Not in cache, searching {comp} directly...")
            with st.spinner(f"🔍 Searching {comp} career pages (this may take 30-60 seconds)..."):
                progress = st.progress(0)
                status = st.empty()
                status.info(f"Step 1/3: Looking for {comp} on CSUSB page...")
                progress.progress(20)
                fb = quick_company_links_playwright(comp, deep=deep_mode)
                if isinstance(fb, pd.DataFrame) and len(fb) > 0:
                    status.success(f"Step 3/3: Found {len(fb)} internships!")
                    progress.progress(100)
                    time.sleep(0.5)
                    status.empty(); progress.empty()
                    table = fb
                else:
                    status.warning(f"Could not find internships for {comp}")
                    progress.progress(100)
                    time.sleep(1)
                    status.empty(); progress.empty()
                    table = pd.DataFrame()

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
            table = table[any_match(table, ["title","details"], pat, regex=True)]
        else:
            pat = r"(" + "|".join(map(re.escape, title_tokens)) + r")"
            table = table[any_match(table, ["title","details"], pat, regex=True)]

    for s in (filters.get("skills") or []):
        s = str(s).strip()
        if s:
            table = table[any_match(table, ["title","details"], re.escape(s), regex=True)]

    explicit_loc = bool(re.search(r"\b(remote|onsite|hybrid|usa|united states|[A-Z]{2}|\d{5})\b", user_msg, re.I))
    for k in ("city","state","country","zipcode"):
        v = (filters.get(k) or "").strip()
        if v and explicit_loc:
            table = table[any_match(table, ["title","location","details"], re.escape(v), regex=True)]
        else:
            filters[k] = ""

    if filters.get("remote_type"):
        rt = filters["remote_type"]
        cols = ["remote"] if "remote" in table.columns else ["details"]
        table = table[any_match(table, cols, re.escape(rt), regex=True)]
    
    for k in ("city","state","country","zipcode"):
        v = (filters.get(k) or "").strip()
        if v:
            table = table[any_match(table, ["title","location","details"], re.escape(v), regex=True, flags=re.I)]

    if filters.get("education_level"):
        table = table[any_match(table, ["details","education"], re.escape(str(filters["education_level"])), regex=True)]
    if filters.get("experience_level"):
        table = table[any_match(table, ["details"], re.escape(str(filters["experience_level"])), regex=True)]
    if filters.get("salary_min") and "salary" in table.columns:
        try:
            minv = int(str(filters["salary_min"]).replace("$","").replace(",",""))
            amt = table["salary"].fillna("").str.replace(",","").str.extract(r"(\d+)")[0].astype(float)
            table = table[amt >= minv]
        except:
            pass

# Ensure columns exist
cols = ["title","company","location","posted_date","salary","education","remote","host","link","source","details"]
for c in cols:
    if c not in table.columns:
        table[c] = None

desc = describe_filters(filters)

# Generate response
if len(table) == 0:
    header = f"😔 Sorry, I couldn't find any internships" + (f" for **{desc}**" if desc else "") + " on the CSUSB page."
    if deep_mode:
        header += "\n\n💡 **Tip**: Try a broader search or check if the company name is spelled correctly."
    answer_md = header
else:
    header = f"🎉 Found **{len(table)}** internship" + ("s" if len(table) > 1 else "")
    if desc:
        header += f" for {desc}"
    header += "!"
    if deep_mode:
        header += "\n\n✨ *Results include deep-scraped data from company career pages*"
    answer_md = header + "\n\n" + links_md(table)

render_msg("assistant", answer_md)

# Show detailed table
if not table.empty:
    st.markdown("### 📊 Detailed Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Internships", len(table))
    with col2:
        unique_companies = table["company"].dropna().nunique()
        st.metric("Companies", unique_companies)
    with col3:
        remote_count = table[table["remote"].fillna("").str.contains("remote", case=False, na=False)].shape[0] if "remote" in table.columns else 0
        st.metric("Remote Positions", remote_count)
    display_cols = [c for c in cols if c in table.columns and table[c].notna().any()]
    st.dataframe(
        table[display_cols],
        width='stretch',
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn("Apply Link", help="Click to open posting"),
            "title": st.column_config.TextColumn("Title", width="medium"),
            "company": st.column_config.TextColumn("Company", width="medium"),
            "location": st.column_config.TextColumn("Location", width="small"),
            "posted_date": st.column_config.DateColumn("Posted", width="small"),
        },
    )
    csv = table.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"csusb_internships_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

st.session_state.messages.append({"role": "assistant", "content": answer_md})
