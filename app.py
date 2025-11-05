import os, time, math, re, json
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import httpx

from llm_provider import get_openai_client, get_model

from scraper import scrape_csusb_listings, CSUSB_CSE_URL
from query_to_filter import classify_intent

# === Résumé helpers ===
from resume_parser import (
    extract_resume_text,
    llm_resume_extract,
    save_resume,
    answer_from_resume,
)

APP_TITLE = "CSUSB Internship Finder Agent"
DATA_DIR = Path("data")
PARQUET_PATH = DATA_DIR / "internships.parquet"
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
def some_api_call(query: str):
    with httpx.Client() as client:
        r = client.post(
            f"{BACKEND_URL}/navigate",
            json={
                "start_url": "https://www.csusb.edu/cse/internships-careers",
                "query": query,
                "max_hops": 40
            },
            timeout=60
        )
        r.raise_for_status()
        return r.json()


# ---------- Page setup ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")

# ---------- Health check (optional, non-blocking) ----------
def backend_ok() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{BACKEND_URL}/health")
            return r.status_code == 200
    except Exception:
        return False

if not backend_ok():
    st.warning(f"Backend not reachable at {BACKEND_URL}. Some features may be limited.")

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
    f"🎯 LLM-guided career page navigator • Source: {CSUSB_CSE_URL}\n\n"
    "Try: **nasa internships**, **google remote intern**, **microsoft software engineer**, **show all internships**"
)

_flash = st.session_state.get("resume_flash")
if _flash:
    st.success(_flash)
    st.session_state["resume_flash"] = ""

# --- Mode toggle ---
mode = st.sidebar.radio("Mode", ["Auto", "General chat", "Internships"], index=0)
deep_mode = st.sidebar.checkbox(
    "Deep Search",
    value=False,
    help="⚠️ Slow: Scrapes company career pages for detailed postings (recommended: OFF for first try)",
)

st.sidebar.markdown("---")
st.sidebar.caption("⚙️ **Settings**")
max_hops = st.sidebar.slider(
    "Max navigation hops per company",
    3, 10, 5, 1,
    help="How many page clicks to follow before giving up"
)

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

# ---------- Conversational memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hey! I'm your CSUSB Internship Finder. I'll intelligently navigate company career pages to find internships matching your criteria. What are you looking for?"
    }]

# === Résumé state + auto-load from disk ===
st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("resume_data", {})
try:
    p_json = DATA_DIR / "resume.json"
    if p_json.exists() and not st.session_state["resume_data"]:
        st.session_state["resume_data"] = json.loads(p_json.read_text(encoding="utf-8"))
        st.session_state["resume_text"] = (DATA_DIR / "resume.txt").read_text(encoding="utf-8") if (DATA_DIR / "resume.txt").exists() else ""
except Exception:
    pass

# === Track preference collection state ===
st.session_state.setdefault("collecting_prefs", False)
st.session_state.setdefault("current_pref_step", 0)
st.session_state.setdefault("user_preferences", {})
st.session_state.setdefault("initial_query", "")

# Preference questions - one at a time
PREF_QUESTIONS = [
    {"key": "interests", "question": "What are your interests or fields you'd like to work in? (e.g., software development, data science, product management, marketing)"},
    {"key": "roles", "question": "What specific job roles are you looking for? (e.g., Software Engineer Intern, Data Analyst Intern, Product Manager Intern)"},
    {"key": "location", "question": "Do you have a location preference? (e.g., Remote, On-site, specific city/region)"},
    {"key": "skills", "question": "What are your key technical or professional skills? (e.g., Python, JavaScript, problem-solving, communication)"},
]

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

# ---------- Backend helpers for LLM ----------
def backend_complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.2,
    max_tokens: int = 400
) -> str:
    """
    Call the backend chat completion endpoint on CHAT_API_URL.
    """
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                f"{CHAT_API_URL}/chat/complete",
                json={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            j = r.json()
            return (j.get("response") or "").strip()
    except Exception as e:
        st.error(f"LLM error: {e}")
        return ""

# ---------- LLM general reply ----------


# ---------- LLM general reply (OpenAI-direct, no HTTP to :8000) ----------


def llm_general_reply(user_text: str) -> str:
    try:
        body = {
            "prompt": user_text,
            "system_prompt": "You are a helpful and friendly assistant for CSUSB internship search.",
            "temperature": 0.3,
            "max_tokens": 220,
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{CHAT_API_URL}/chat/complete", json=body)  # Use CHAT_API_URL and /chat/complete
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
    except Exception as e:
        return f"Sorry, chat API error: {e}"

# ---------- Extract preference from user response ----------
def extract_preference_from_response(user_response: str, pref_key: str) -> List[str]:
    """
    Extract structured preference terms from a freeform response.
    Uses the backend LLM; returns a list of strings.
    """
    sys_extract = (
        f"You are an internship preference extractor. Extract key terms from the user's response about {pref_key}.\n\n"
        "Return ONLY a JSON array of strings. Example: [\"item1\", \"item2\", \"item3\"]\n"
        "Be faithful to what they said. Don't invent terms."
    )
    raw = backend_complete(
        prompt=f"User response: {user_response}\n\nReturn JSON array now.",
        system_prompt=sys_extract,
        temperature=0.1,
        max_tokens=150,
    )
    # Parse JSON array
    try:
        # Direct parse
        return json.loads(raw) if isinstance(json.loads(raw), list) else []
    except Exception:
        m = re.search(r"\[[\s\S]*\]", raw or "")
        if not m:
            return []
        try:
            arr = json.loads(m.group(0))
            return arr if isinstance(arr, list) else []
        except Exception:
            return []

# ---------- Backend navigation request ----------
def navigate_career_page(company_url: str, query: str) -> dict:
    """
    Send navigation request to BACKEND_URL.
    """
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{BACKEND_URL}/navigate",
                json={
                    "start_url": company_url,
                    "query": query,
                    "max_hops": 40  # Change if you have a slider variable
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException as e:
        st.warning(f"Navigation timeout for {company_url}")
        return {
            "success": False,
            "error": f"Timeout: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except httpx.HTTPStatusError as e:
        st.warning(f"Backend error for {company_url}: {e}")
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except httpx.ConnectError as e:
        st.error(f"⚠️ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return {
            "success": False,
            "error": f"Connection failed: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }

# ---------- LLM-directed internship search ----------
def llm_internship_search_directed(
    user_text: str,
    csusb_links_df: pd.DataFrame,
    user_prefs: dict | None = None
) -> Tuple[str, pd.DataFrame]:
    """
    1. LLM receives CSUSB career page links (via backend /chat/complete)
    2. LLM decides which to navigate (up to 5), factoring user preferences
    3. Backend navigates each one using /navigate
    4. LLM summarizes and presents best results
    """
    if csusb_links_df.empty:
        return "Sorry, no company career pages available at the moment.", pd.DataFrame()

    # Step 1: Build list of available companies
    links_list = csusb_links_df[['company', 'link']].drop_duplicates().to_dict('records')
    available_companies: List[str] = []
    for item in links_list:
        company = item.get('company') or 'Unknown'
        link = item.get('link') or ''
        if link and link.startswith('http'):
            available_companies.append(f"{company}: {link}")

    if not available_companies:
        return "No valid company links found in the database.", pd.DataFrame()

    links_text = "\n".join(available_companies)
    st.sidebar.info(f"📋 Available companies: {len(available_companies)}")

    # Include user preferences in the prompt
    prefs_text = ""
    if user_prefs:
        interests = user_prefs.get('interests', [])
        roles = user_prefs.get('roles', [])
        location = user_prefs.get('location', 'Any')
        skills = user_prefs.get('skills', [])

        interests_str = ', '.join(interests) if isinstance(interests, list) else str(interests)
        roles_str = ', '.join(roles) if isinstance(roles, list) else str(roles)
        skills_str = ', '.join(skills) if isinstance(skills, list) else str(skills)

        prefs_text = (
            f"\nUser Preferences:\n"
            f"- Interests: {interests_str}\n- Roles: {roles_str}\n"
            f"- Location: {location}\n- Skills: {skills_str}"
        )

    sys_step1 = (
        "You are an internship search assistant. You have a list of companies with career page URLs.\n\n"
        "Your task: Match the user's query to companies in the available list, prioritizing their stated preferences.\n\n"
        "Rules:\n"
        "1. Return ONLY valid JSON (no markdown, no explanations)\n"
        "2. Use EXACT company names and URLs from the list provided\n"
        "3. If a company name in the query doesn't match any in the list, DON'T make it up\n"
        "4. If no matches found, return empty array\n"
        "5. Prioritize companies matching user preferences\n"
        "6. Select up to 5 companies to navigate\n\n"
        "Return JSON with two fields: "
        "\"companies_to_navigate\" (array of objects with \"company\" and \"url\" fields) "
        "and \"reasoning\" (string)."
        + prefs_text
    )

    raw_selection = backend_complete(
        prompt=f"Available companies and URLs:\n{links_text}\n\nUser query: {user_text}\n\nFind matching companies. Return ONLY valid JSON as response, nothing else.",
        system_prompt=sys_step1,
        temperature=0.2,
        max_tokens=500,
    )

    # Parse selection JSON
    try:
        parsed = json.loads(raw_selection)
    except Exception:
        m = re.search(r"[\{\[][\s\S]*[\}\]]", raw_selection or "")
        parsed = json.loads(m.group(0)) if m else {"companies_to_navigate": [], "reasoning": "Parse error"}

    if isinstance(parsed, list):
        scrape_decision = {"companies_to_navigate": parsed, "reasoning": "Companies matched"}
    else:
        scrape_decision = parsed

    companies_to_navigate = (scrape_decision or {}).get("companies_to_navigate", [])[:5]
    if not companies_to_navigate:
        reasoning = (scrape_decision or {}).get("reasoning", "No matching companies found")
        available_list = ", ".join([
            item.get('company') or 'Unknown'
            for item in links_list
            if item.get('company')
        ][:10])
        return (
            f"I couldn't find matching companies. {reasoning}\n\n**Available companies on CSUSB:**\n{available_list}",
            pd.DataFrame()
        )

    # Step 2: Navigate each company via backend
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    all_navigation_results: List[Dict[str, Any]] = []

    for idx, company_info in enumerate(companies_to_navigate, 1):
        company_name = company_info.get("company", "Unknown")
        company_url = company_info.get("url", "")

        if not company_url or not company_url.startswith("http"):
            continue

        with status_placeholder:
            st.info(f"🔍 Navigating {company_name} ({idx}/{len(companies_to_navigate)})...")

        progress_bar.progress(int((idx - 1) / len(companies_to_navigate) * 100))

        nav_result = navigate_career_page(company_url, query)

        found_links = nav_result.get("found_links", [])
        visited_urls = nav_result.get("visited_urls", [])
        final_url = nav_result.get("final_url") or (visited_urls[-1] if visited_urls else company_url)

        all_navigation_results.append({
            "company": company_name,
            "final_url": final_url,
            "visited_urls": visited_urls,
            "found_links": found_links,
            "success": nav_result.get("success", False)
        })

        time.sleep(1)

    status_placeholder.empty()
    progress_bar.empty()

    # If nothing had links, at least return career pages
    results_with_links = [r for r in all_navigation_results if r.get("found_links")]
    if not results_with_links:
        career_pages = []
        df_results = []
        for r in all_navigation_results:
            company = r.get("company", "Unknown")
            url = r.get("final_url", "")
            if url:
                career_pages.append(f"- [{company}]({url})")
                df_results.append({
                    "title": f"{company} Career Page",
                    "company": company,
                    "url": url,
                    "link": url
                })
        if career_pages:
            pages_text = "\n".join(career_pages)
            result_df = pd.DataFrame(df_results) if df_results else pd.DataFrame()
            return (
                f"I found the career pages but couldn't automatically extract job listings. Here are the pages you can visit:\n\n{pages_text}",
                result_df
            )
        return (
            f"I navigated {len(companies_to_navigate)} companies but couldn't find job listing pages. The companies may have dynamic job sites.",
            pd.DataFrame()
        )

    # Step 3: LLM analyzes and presents results
    results_text = json.dumps(all_navigation_results, indent=2)
    sys_step2 = (
        "You are a helpful internship search assistant. You have results from navigating company career pages.\n\n"
        "For each company, you have:\n"
        "- The career page they reached (final_url)\n"
        "- Intermediate pages visited\n"
        "- Links found on the final page\n\n"
        "Analyze and provide:\n"
        "1. Brief summary of what was found\n"
        "2. Highlight the most relevant internship opportunities\n"
        "3. Provide direct links when available\n\n"
        "Be conversational, encouraging, and concise. Focus on actionable next steps."
    )
    result_text = backend_complete(
        prompt=f"Navigation results:\n{results_text}\n\nUser query: {user_text}\n\nAnalyze and recommend:",
        system_prompt=sys_step2,
        temperature=0.3,
        max_tokens=700,
    )
    if not result_text:
        result_text = "I analyzed the navigation results. Here are the most relevant actions based on what we found."

    total_links = sum(len(r.get("found_links", [])) for r in all_navigation_results)
    successful_navs = sum(1 for r in all_navigation_results if r.get("success"))
    stats_text = f"\n\n📊 **Navigation Summary:** Explored {len(all_navigation_results)} companies, found {total_links} links across {successful_navs} successful navigations."
    result_text += stats_text

    # Convert to dataframe for display
    df_results: List[Dict[str, str]] = []
    for nav_result in all_navigation_results:
        company = nav_result.get("company") or "Unknown"
        found_links = nav_result.get("found_links") or []
        for link in found_links[:10]:
            if not isinstance(link, dict):
                continue
            link_url = link.get("url") or ""
            link_text = link.get("text") or ""
            if link_url and link_url.startswith('http'):
                df_results.append({
                    "title": link_text if link_text else "Link",
                    "company": company,
                    "url": link_url,
                    "link": link_url
                })
    result_df = pd.DataFrame(df_results) if df_results else pd.DataFrame()

    return result_text, result_df

# ==================== RESUME UPLOAD SECTION ====================

# Session state setup
st.session_state.setdefault("resume_uploader_key", "resume_uploader_0")
st.session_state.setdefault("resume_flash", "")
st.session_state.setdefault("show_resume_uploader", False)

# Sidebar resume uploader
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📎 Resume Upload")

    up = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
        key=st.session_state["resume_uploader_key"],
        help="Upload PDF, DOCX, or TXT file",
    )

    if up is not None:
        with st.spinner("Extracting resume..."):
            text = extract_resume_text(up)
            data = llm_resume_extract(text)
            save_resume(data, text)
            st.session_state.resume_text = text
            st.session_state.resume_data = data

        st.success("✅ Resume saved!")

        import time as _t
        st.session_state["resume_uploader_key"] = f"resume_uploader_{int(_t.time()*1000)}"
        st.rerun()

    # Show resume info if loaded
    if st.session_state.get("resume_data"):
        with st.expander("📄 Resume Info"):
            data = st.session_state["resume_data"]
            if data.get("name"):
                st.write(f"**Name:** {data['name']}")
            if data.get("email"):
                st.write(f"**Email:** {data['email']}")
            if data.get("skills"):
                st.write(f"**Skills:** {', '.join(data['skills'][:5])}")

# Chat input
user_msg = st.chat_input("Type your question…")

if not user_msg:
    st.stop()
if not allow_query():
    st.stop()

st.session_state.messages.append({"role": "user", "content": user_msg})
render_msg("user", user_msg)

query = user_msg.strip()
if not query or query.lower() in ["hi", "hello", "hey"]:
    query = "show all internships"



# ---------- Routing ----------
txt_lo = user_msg.lower()
intent = classify_intent(user_msg)

# Detect résumé questions BEFORE routing
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
            return True
    return False

if is_resume_question(txt_lo):
    data = st.session_state.get("resume_data") or {}
    if data:
        reply = answer_from_resume(user_msg, data)
    else:
        reply = "I don't have a résumé saved yet. Use the **Resume Upload** panel in the sidebar to add one."
    render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# Mode + Intent routing
has_intern_kw = "intern" in txt_lo
st.sidebar.caption(f"🎯 Intent: {intent}")

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
        reply = llm_general_reply(user_msg)
    render_msg("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.stop()

# ---------- INTERNSHIP SEARCH: PREFERENCE COLLECTION (ONE QUESTION AT A TIME) ----------
# Start preference collection if not already doing so
if not st.session_state.get("collecting_prefs"):
    st.session_state.collecting_prefs = True
    st.session_state.current_pref_step = 0
    st.session_state.initial_query = query
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
initial_query = st.session_state.get("initial_query", query)


with st.spinner("Searching for internships matching your preferences..."):
   answer_md, results_df = llm_internship_search_directed(query, csusb_df, user_prefs)

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
