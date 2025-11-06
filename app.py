import os, time, math, re, json
from resume_parser import extract_resume_text, llm_resume_extract, save_resume, answer_from_resume

from collections import deque
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
# from streamlit import session_state as ss

import httpx

from scraper import scrape_csusb_listings, CSUSB_CSE_URL
from query_to_filter import classify_intent

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
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------- Page setup ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")

# --- Ollama warm-up (once) ---
@st.cache_resource(show_spinner=False)
def ensure_ollama_ready():
    """Ping local Ollama and pull the model if missing (one-time)."""
    import os, json, urllib.request
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        names = [m.get("name","") for m in (data.get("models", []) or [])]
        if model not in names:
            st.info(f"Pulling model `{model}`… (one-time)")
            body = json.dumps({"name": model}).encode("utf-8")
            req = urllib.request.Request(f"{host}/api/pull", data=body, headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=300).read()
    except Exception:
        st.warning("Ollama isn't reachable; general chat may be slow or fail.")

ensure_ollama_ready()




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
deep_mode = st.sidebar.checkbox("Deep Search", value=False, help="⚠️ Slow: Scrapes company career pages for detailed postings (recommended: OFF for first try)")










st.sidebar.markdown("---")
st.sidebar.caption("⚙️ **Settings**")
max_hops = st.sidebar.slider("Max navigation hops per company", 3, 10, 5, 1, help="How many page clicks to follow before giving up")


















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

# === NEW: track preference collection state ===
st.session_state.setdefault("collecting_prefs", False)
st.session_state.setdefault("current_pref_step", 0)
st.session_state.setdefault("user_preferences", {})
st.session_state.setdefault("initial_query", "")

# Preference questions - one at a time
PREF_QUESTIONS = [
    {"key": "interests", "question": "What are your interests or fields you'd like to work in? (e.g., software development, data science, product management, marketing)"},
    {"key": "roles", "question": "What specific job roles are you looking for? (e.g., Software Engineer Intern, Data Analyst Intern, Product Manager Intern)"},
    {"key": "location", "question": "Do you have a location preference? (e.g., Remote, On-site, specific city/region)"},
    {"key": "skills", "question": "What are your key technical or professional skills? (e.g., Python, JavaScript, problem-solving, communication)"}
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

# ---------- LLM general reply ----------
def llm_general_reply(user_text: str) -> str:
    """Call Ollama LLM for general chat."""
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

# ---------- NEW: Extract preference from user response ----------
def extract_preference_from_response(user_response: str, pref_key: str) -> list:
    """Extract structured preference from user's freeform response."""
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.1,
            streaming=False,
            model_kwargs={"num_ctx": 1024, "num_predict": 100}
        )
        
        sys_extract = f"""You are an internship preference extractor. Extract key terms from the user's response about {pref_key}.

Return ONLY a JSON array of strings. Example: ["item1", "item2", "item3"]

Be faithful to what they said. Don't invent terms."""

        prompt_extract = ChatPromptTemplate.from_messages([
            ("system", sys_extract),
            ("human", f"User response: {user_response}\n\nReturn JSON array now.")
        ])
        
        response = (prompt_extract | llm).invoke({})
        raw = response.content.strip()
        
        # Extract JSON array
        json_match = re.search(r'\[[\s\S]*\]', raw)
        if json_match:
            extracted = json.loads(json_match.group(0))
            return extracted if isinstance(extracted, list) else []
        return []
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

# ---------- Backend navigation request ----------
def navigate_career_page(company_url: str, query: str) -> dict:
    """
    Send navigation request to backend.
    Backend will use LLM to navigate the career page until it finds job listings.

    """
    print(f"\n{'='*60}")
    print(f"Calling backend to navigate: {company_url}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Query: {query}")
    print(f"Max hops: {max_hops}")
    print(f"{'='*60}")
    
    try:
        with httpx.Client(timeout=120.0) as client:
            print(f"Sending POST request to {BACKEND_URL}/navigate")
            
            response = client.post(
                f"{BACKEND_URL}/navigate",
                json={
                    "start_url": company_url,
                    "query": query,
                    "max_hops": max_hops
                }
            )
            
            print(f"Response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            print(f"Response data: {json.dumps(result, indent=2)[:500]}")
            
            return result
            
    except httpx.TimeoutException as e:
        print(f"✗ Timeout error: {e}")
        st.warning(f"Navigation timeout for {company_url}")
        return {
            "success": False,
            "error": f"Timeout: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP error: {e}")
        st.warning(f"Backend error for {company_url}: {e}")
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except httpx.ConnectError as e:
        print(f"✗ Connection error: {e}")
        st.error(f"⚠️ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return {
            "success": False,
            "error": f"Connection failed: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except Exception as e:
        print(f"✗ Navigation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }

# ---------- LLM-directed internship search ----------
def llm_internship_search_directed(user_text: str, csusb_links_df: pd.DataFrame, user_prefs: dict = None) -> tuple[str, pd.DataFrame]:


    """
    1. LLM receives CSUSB career page links
    2. LLM decides which to navigate (up to 5), filtered by user preferences
    3. Backend navigates each one using LLM guidance
    4. LLM filters and presents best results
    """
    if csusb_links_df.empty:
        return "Sorry, no company career pages available at the moment.", pd.DataFrame()
    
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        model_name = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.2,
            streaming=False,
            model_kwargs={"num_ctx": 3072, "num_predict": 300}
        )
        
        # Step 1: Get unique companies and build list
        links_list = csusb_links_df[['company', 'link']].drop_duplicates().to_dict('records')
        
        # Filter out None values and build list
        available_companies = []
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
            
            # Handle both list and string formats
            interests_str = ', '.join(interests) if isinstance(interests, list) else str(interests)
            roles_str = ', '.join(roles) if isinstance(roles, list) else str(roles)
            skills_str = ', '.join(skills) if isinstance(skills, list) else str(skills)
            
            prefs_text = f"\nUser Preferences:\n- Interests: {interests_str}\n- Roles: {roles_str}\n- Location: {location}\n- Skills: {skills_str}"
        
        sys_step1 = """You are an internship search assistant. You have a list of companies with career page URLs.

Your task: Match the user's query to companies in the available list, prioritizing their stated preferences.

Rules:
1. Return ONLY valid JSON (no markdown, no explanations)
2. Use EXACT company names and URLs from the list provided
3. If a company name in the query doesn't match any in the list, DON'T make it up
4. If no matches found, return empty array
5. Prioritize companies matching user preferences
6. Select up to 5 companies to navigate

Return JSON with two fields: "companies_to_navigate" (array of objects with "company" and "url" fields) and "reasoning" (string).""" + prefs_text

        prompt1 = ChatPromptTemplate.from_messages([
            ("system", sys_step1),
            ("human", "Available companies and URLs:\n{links}\n\nUser query: {query}\n\nFind matching companies. Return ONLY valid JSON as response, nothing else.")
        ])
        
        chain1 = prompt1 | llm
        response1 = chain1.invoke({
            "links": links_text,
            "query": user_text
        })
        
        # Debug output
        st.sidebar.text_area("LLM Response (Debug)", response1.content[:300], height=100)
        
        try:
            raw_response = response1.content.strip()
            
            # Try to parse as direct JSON first
            try:
                parsed = json.loads(raw_response)
                # If it's an array, wrap it in the expected structure
                if isinstance(parsed, list):
                    scrape_decision = {"companies_to_navigate": parsed, "reasoning": "Companies matched"}
                else:
                    scrape_decision = parsed
            except json.JSONDecodeError:
                # Try to find JSON object or array in the response
                json_match = re.search(r'[\{\[][\s\S]*[\}\]]', raw_response)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list):
                        scrape_decision = {"companies_to_navigate": parsed, "reasoning": "Companies matched"}
                    else:
                        scrape_decision = parsed
                else:
                    scrape_decision = {"companies_to_navigate": [], "reasoning": "Could not extract JSON from response"}
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"LLM response was: {response1.content[:500]}")
            scrape_decision = {"companies_to_navigate": [], "reasoning": f"Parse error: {str(e)}"}
        
        companies_to_navigate = scrape_decision.get("companies_to_navigate", [])[:5]
        
        if not companies_to_navigate:
            reasoning = scrape_decision.get("reasoning", "No matching companies found")
            available_list = ", ".join([
                item.get('company') or 'Unknown' 
                for item in links_list 
                if item.get('company')
            ][:10])
            return f"I couldn't find matching companies. {reasoning}\n\n**Available companies on CSUSB:**\n{available_list}", pd.DataFrame()
        
        # Step 2: Navigate each company
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        all_navigation_results = []
        
        for idx, company_info in enumerate(companies_to_navigate, 1):
            company_name = company_info.get("company", "Unknown")
            company_url = company_info.get("url", "")
            
            if not company_url or not company_url.startswith("http"):
                continue
            
            with status_placeholder:
                st.info(f"🔍 Navigating {company_name} ({idx}/{len(companies_to_navigate)})...")
            
            progress_bar.progress(int((idx - 1) / len(companies_to_navigate) * 100))
            
            # Call backend to navigate
            nav_result = navigate_career_page(company_url, user_text)
            
            # Store result even if not "successful"
            found_links = nav_result.get("found_links", [])
            visited_urls = nav_result.get("visited_urls", [])
            
            # Use the last visited URL as final_url if none provided
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
        
        # Count how many had links
        results_with_links = [r for r in all_navigation_results if r.get("found_links")]
        
        if not results_with_links:
            # Still provide the career page URLs
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
                return f"I found the career pages but couldn't automatically extract job listings. Here are the pages you can visit:\n\n{pages_text}", result_df
            
            return f"I navigated {len(companies_to_navigate)} companies but couldn't find job listing pages. The companies may have dynamic job sites.", pd.DataFrame()
        
        # Step 3: LLM analyzes and presents results
        results_text = json.dumps(all_navigation_results, indent=2)
        
        sys_step2 = """You are a helpful internship search assistant. You have results from navigating company career pages.

For each company, you have:
- The career page they reached (final_url)
- Intermediate pages visited
- Links found on the final page

Analyze and provide:
1. Brief summary of what was found
2. Highlight the most relevant internship opportunities
3. Provide direct links when available

Be conversational, encouraging, and concise. Focus on actionable next steps."""

        prompt2 = ChatPromptTemplate.from_messages([
            ("system", sys_step2),
            ("human", "Navigation results:\n{results}\n\nUser query: {query}\n\nAnalyze and recommend:")
        ])
        
        chain2 = prompt2 | llm
        response2 = chain2.invoke({
            "results": results_text,
            "query": user_text
        })
        
        result_text = response2.content.strip()
        
        # Add stats
        total_links = sum(len(r.get("found_links", [])) for r in all_navigation_results)
        successful_navs = sum(1 for r in all_navigation_results if r.get("success"))
        
        stats_text = f"\n\n📊 **Navigation Summary:** Explored {len(all_navigation_results)} companies, found {total_links} links across {successful_navs} successful navigations."
        result_text += stats_text
        
        # Convert to dataframe for display
        df_results = []
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
        
    except Exception as e:
        st.error(f"Search Error: {str(e)}")
        print(f"Error in llm_internship_search_directed: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error during the search. Please try again.", pd.DataFrame()

# Floating "+" popover anchored to the chat input (no middle duplicate, no flicker)
# Replace the entire resume upload section with this:

# ==================== RESUME UPLOAD SECTION ====================

# Session state setup
st.session_state.setdefault("resume_uploader_key", "resume_uploader_0")
st.session_state.setdefault("resume_flash", "")
st.session_state.setdefault("show_resume_uploader", False)

# CSS for floating resume button
st.markdown("""
<style>
/* Floating + button near chat input */
.resume-upload-btn {
    position: fixed;
    bottom: 24px;
    left: calc(50% - 340px);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(180deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    font-size: 24px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s ease;
}

.resume-upload-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
}

/* Adjust chat input padding */
[data-testid="stChatInput"] {
    padding-left: 50px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar resume uploader (cleaner approach)
with st.sidebar:



    st.markdown("---")
    st.markdown("### 📎 Resume Upload")
    
    up = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
        key=st.session_state["resume_uploader_key"],
        help="Upload PDF, DOCX, or TXT file"
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

# Stop if no input
if not user_msg:
    st.stop()
if not allow_query():
    st.stop()

# Rest of your code continues...

# Continue with rest of your app logic...
# Rest of your code continues...
st.session_state.messages.append({"role": "user", "content": user_msg})
render_msg("user", user_msg)

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
    answer_md, results_df = llm_internship_search_directed(initial_query, csusb_df, user_prefs)

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
