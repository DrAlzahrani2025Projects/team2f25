import os, json
from resume_parser import extract_resume_text, llm_resume_extract, save_resume

from collections import deque
from pathlib import Path

import streamlit as st
# from streamlit import session_state as ss



CSUSB_CSE_URL = os.getenv("CSUSB_CSE_URL", "https://www.csusb.edu/cse/internships-careers")


# --- CSS injector ---
def inject_css(path: str = "styles.css"):
    p = Path(path)
    if p.exists():
        mtime = p.stat().st_mtime
        key = "css_mtime"
        if st.session_state.get(key) != mtime:
            st.session_state[key] = mtime
            st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def setup_ui(app_title):
    global mode, deep_mode, max_hops
    st.set_page_config(page_title=app_title, page_icon="💼", layout="wide")

    inject_css()

    st.title(app_title)
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

def setup_session_states(data_dir):
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
        p_json = data_dir / "resume.json"
        if p_json.exists() and not st.session_state["resume_data"]:
            st.session_state["resume_data"] = json.loads(p_json.read_text(encoding="utf-8"))
            st.session_state["resume_text"] = (data_dir / "resume.txt").read_text(encoding="utf-8") if (data_dir / "resume.txt").exists() else ""
    except Exception:
        pass

    # === NEW: track preference collection state ===
    st.session_state.setdefault("collecting_prefs", False)
    st.session_state.setdefault("current_pref_step", 0)
    st.session_state.setdefault("user_preferences", {})
    st.session_state.setdefault("initial_query", "")
    st.session_state.setdefault("resume_uploader_key", "resume_uploader_0")
    st.session_state.setdefault("resume_flash", "")
    st.session_state.setdefault("show_resume_uploader", False)

# --- Message renderer ---
def render_msg(role: str, content: str):
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(content)

def init_resume_ui():
    # Floating "+" popover anchored to the chat input (no middle duplicate, no flicker)
    # Replace the entire resume upload section with this:

    # ==================== RESUME UPLOAD SECTION ====================

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

# --- Render history ---
def render_message_history():
    for m in range(len(st.session_state.messages)):
        if m == 0:
            render_msg("llm", st.session_state.messages[m]["content"])
        elif m % 2 == 0:
            render_msg("llm", st.session_state.messages[m].content)
        else: 
            render_msg("user", st.session_state.messages[m].content)

def get_max_hops():
    return max_hops

def get_mode():
    return mode

def is_deep_mode():
    return deep_mode