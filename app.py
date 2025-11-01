import os, re, json, time
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document

# ========= CONFIG =========
APP_TITLE = "LLM Internship Assistant — Phase 1 + Phase 2"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama (local LLM) config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

# Interview questions max
MAX_Q = 10

st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")

# Load styles if present
if Path("styles.css").exists():
    st.markdown(Path("styles.css").read_text(), unsafe_allow_html=True)

st.title(APP_TITLE)

# ========= IMPORT SCRAPER =========
try:
    from scraper import (
        scrape_csusb_listings,
        quick_company_links_playwright,
        CSUSB_CSE_URL
    )
except Exception as e:
    st.error(f"Deep scraper not found: {e}")
    st.stop()

# ========= OLLAMA HELPERS =========
@st.cache_resource
def have_llm() -> bool:
    """Check if the Ollama API is reachable and the model appears in /api/tags."""
    import urllib.request, json as j
    try:
        with urllib.request.urlopen(OLLAMA_HOST.rstrip("/") + "/api/tags", timeout=2) as r:
            d = j.loads(r.read().decode() or "{}")
            names = [m.get("name", "") for m in d.get("models", [])]
            return any(MODEL_NAME in n for n in names)
    except Exception:
        return False

def llm_chat(messages, temp=0.2, npred=160, timeout_s=12) -> str:
    """Call Ollama /api/chat directly via httpx. Return string content (or empty)."""
    try:
        import httpx
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "num_predict": npred,
                "temperature": temp
            }
        }
        with httpx.Client(timeout=timeout_s) as c:
            r = c.post(OLLAMA_HOST.rstrip("/") + "/api/chat", json=payload)
            r.raise_for_status()
            return ((r.json().get("message") or {}).get("content") or "").strip()
    except Exception:
        return ""

# ========= RESUME UTILITIES =========
def _read_pdf(b: bytes) -> str:
    out = []
    try:
        pdf = PdfReader(BytesIO(b))
        for p in pdf.pages[:10]:
            try:
                out.append(p.extract_text() or "")
            except Exception:
                pass
    except Exception:
        pass
    return "\n".join(out)

def _read_docx(b: bytes) -> str:
    try:
        doc = Document(BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def resume_text(uploaded_file) -> str:
    """Convert uploaded resume (pdf/docx/txt) to plain text."""
    name = (uploaded_file.name or "").lower()
    b = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        return _read_pdf(b)
    if name.endswith(".docx"):
        return _read_docx(b)
    try:
        return b.decode("utf-8", "ignore")
    except Exception:
        return b.decode("latin-1", "ignore")

def resume_json_llm(text: str) -> Dict:
    """Parse resume text to structured JSON using the local LLM (if available)."""
    if not text.strip() or not have_llm():
        return {}
    sys = (
        "You are a resume parser. Return ONLY compact JSON with keys: "
        '{"name":"","email":"","phone":"","skills":[],"education":[],"experience":[]}. '
        "Do not include commentary."
    )
    out = llm_chat(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": text[:15000]},
        ],
        npred=300,
        timeout_s=20,
    )
    m = re.search(r"\{[\s\S]*\}", out)
    try:
        return json.loads(m.group(0) if m else out)
    except Exception:
        return {}

# ========= INTERVIEW HELPERS =========
def next_q(hist: List[Dict], rem: int) -> str:
    """
    Ask one adaptive question at a time (LLM-only).
    The model should cover roles, companies, skills, location, work mode,
    timeline, visa, industries, and a final note — at most 10 total questions.
    """
    sys = (
        "Ask one short, precise question to learn a student's internship goals. "
        "You have at most 10 questions total. Cover roles, companies, skills, location, work mode, "
        "timeline, visa, industries, and any final note. Return only the question text."
    )
    conv = "\n".join([f"Q:{h['q']}\nA:{h['a']}" for h in hist])
    out = llm_chat(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Questions left:{rem}\n{conv}\nNext question:"},
        ],
        npred=80,
    )
    for l in out.splitlines():
        l = l.strip(" -•:")
        if l:
            return l[:220]
    return "What internship roles interest you?"

def _split_list(s: str) -> List[str]:
    return [i.strip() for i in re.split(r"[,;/\n]+", s or "") if i.strip()]

def normalize_answer(q: str, a: str, profile: Dict):
    """
    Lightweight mapping of answer → profile fields based on the question text.
    The LLM asks the question; we key off certain words to store answers.
    """
    s = a.strip()
    ql = (q or "").lower()

    if "role" in ql:
        profile["roles"] = _split_list(s)
    elif "comp" in ql:  # company/companies
        profile["companies"] = _split_list(s)
    elif "loc" in ql or "where" in ql:
        profile["locations"] = _split_list(s)
    elif "skill" in ql:
        profile["skills"] = _split_list(s)
    elif "mode" in ql or "remote" in ql or "hybrid" in ql or "on-site" in ql or "onsite" in ql:
        profile["work_mode"] = s
    elif "when" in ql or "timeline" in ql or "start" in ql:
        profile["timeline"] = s
    elif "visa" in ql or "auth" in ql:
        profile["auth"] = s
    elif "industr" in ql:
        profile["industries"] = _split_list(s)
    else:
        profile["notes"] = (profile.get("notes", "") + " " + s).strip()

def profile_summary(p: Dict) -> str:
    def bullets(lst):
        return "".join([f"\n- {i}" for i in (lst or [])]) if lst else "—"
    return "\n\n".join([
        f"**Name:** {p.get('name','—')}",
        f"**Roles:**{bullets(p.get('roles'))}",
        f"**Companies:**{bullets(p.get('companies'))}",
        f"**Skills:**{bullets(p.get('skills'))}",
        f"**Locations:**{bullets(p.get('locations'))}",
        f"**Timeline:** {p.get('timeline','—')}",
        f"**Work Mode:** {p.get('work_mode','—')}",
        f"**Authorization:** {p.get('auth','—')}",
        f"**Industries:**{bullets(p.get('industries'))}",
        f"**Notes:** {p.get('notes','—')}",
    ])

# ========= STATE =========
st.session_state.setdefault("hist", [])
st.session_state.setdefault("i", 0)
st.session_state.setdefault("done", False)
st.session_state.setdefault(
    "profile",
    {
        "name": "",
        "roles": [],
        "companies": [],
        "skills": [],
        "locations": [],
        "work_mode": "",
        "timeline": "",
        "auth": "",
        "industries": [],
        "notes": "",
    },
)

# ========= SIDEBAR =========
with st.sidebar:
    st.subheader("Progress")
    st.progress(min(st.session_state.i, MAX_Q) / MAX_Q)
    st.caption(f"{st.session_state.i}/{MAX_Q} questions")
    st.markdown("---")
    st.subheader("Upload Résumé")
    up = st.file_uploader("PDF/DOCX/TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if up is not None:
        with st.spinner("Parsing résumé..."):
            t = resume_text(up)
            j = resume_json_llm(t)
            # use what we can
            if j.get("name"):
                st.session_state.profile["name"] = j["name"]
            if j.get("skills") and not st.session_state.profile.get("skills"):
                st.session_state.profile["skills"] = j["skills"]
        st.success("Résumé uploaded ✅")
        st.rerun()

# ========= MAIN =========
if st.session_state.done:
    st.success("✅ Phase 1 complete")
    st.info(profile_summary(st.session_state.profile))

    st.markdown("---")
    st.subheader("Phase 2 – Deep Internship Search")
    st.caption(f"Source: {CSUSB_CSE_URL}")

    deep_csusb = st.toggle("Deep scrape CSUSB links", True)
    deep_company = st.toggle("Deep scrape companies from your answers", True)
    max_pages = st.slider("Max pages/company", 10, 100, 40, 10)

    if st.button("Run Deep Search"):
        with st.spinner("Scraping..."):
            df_list = []

            # CSUSB page (shallow + optional deep follow)
            try:
                df_csusb = scrape_csusb_listings(deep=deep_csusb, max_pages=max_pages)
                if isinstance(df_csusb, pd.DataFrame) and not df_csusb.empty:
                    df_list.append(df_csusb)
            except Exception as e:
                st.warning(f"CSUSB scrape error: {e}")

            # Company-focused deep scraping from interview answers
            if deep_company:
                for c in st.session_state.profile.get("companies", []):
                    try:
                        d = quick_company_links_playwright(c, deep=True)
                        if isinstance(d, pd.DataFrame) and not d.empty:
                            df_list.append(d)
                    except Exception as e:
                        st.warning(f"{c}: {e}")

            df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

        if df.empty:
            st.info("No results found.")
        else:
            # Normalize typical columns if missing
            for col in ["title", "company", "location", "posted_date", "link", "source", "host"]:
                if col not in df.columns:
                    df[col] = None
            st.write(f"**{len(df)} internships found**")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "internships.csv",
                "text/csv",
            )

    if st.button("🔁 Restart Interview"):
        st.session_state.clear()
        st.rerun()

else:
    remaining = MAX_Q - st.session_state.i
    if remaining <= 0:
        st.session_state.done = True
        st.rerun()

    # Ask next adaptive question (LLM)
    q = next_q(st.session_state.hist, remaining)
    st.subheader(f"Question {st.session_state.i + 1} of {MAX_Q}")
    st.write(q)
    ans = st.text_input("Your answer", key=f"ans{st.session_state.i}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Skip"):
            st.session_state.hist.append({"q": q, "a": "(skipped)"})
            st.session_state.i += 1
            st.rerun()
    with c2:
        if st.button("Next"):
            if not ans.strip():
                st.warning("Type an answer or click Skip.")
            else:
                st.session_state.hist.append({"q": q, "a": ans})
                normalize_answer(q, ans, st.session_state.profile)
                st.session_state.i += 1
                st.rerun()

    st.markdown("---")
    st.caption("Profile so far:")
    st.info(profile_summary(st.session_state.profile))
