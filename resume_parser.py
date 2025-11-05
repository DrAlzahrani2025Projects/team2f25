# resume_parser.py
import json, os, re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader
from docx import Document

# OpenAI client helpers
from llm_provider import get_openai_client, get_model

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- File -> text ----------
def _read_pdf(b: bytes) -> str:
    out: List[str] = []
    reader = PdfReader(BytesIO(b))
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)

def _read_docx(b: bytes) -> str:
    buf = BytesIO(b)
    doc = Document(buf)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_resume_text(uploaded_file) -> str:
    """
    Accepts a Streamlit UploadedFile-like object and returns best-effort UTF-8 text.
    Supports PDF, DOCX, and plain text.
    """
    name = (uploaded_file.name or "").lower()
    b = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        return _read_pdf(b)
    if name.endswith(".docx"):
        return _read_docx(b)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

# ---------- OpenAI helper ----------
def _openai_chat(system: str, user: str, max_tokens: int = 600, temperature: float = 0.1) -> str:
    """
    Calls OpenAI Chat Completions and returns assistant content (string).
    """
    client = get_openai_client()
    model = get_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- LLM extraction ----------
def llm_resume_extract(resume_text: str) -> Dict[str, Any]:
    """
    Uses a single {resume_text} variable and returns structured JSON only.
    (OpenAI-based; no LangChain/Ollama.)
    """
    if not (resume_text or "").strip():
        return {}

    system = (
        "You extract structured résumé data and output compact JSON only. "
        "Follow this strict schema (omit null/empty fields). "
        "{"
        '  "name": "string",'
        '  "email": "string",'
        '  "phone": "string",'
        '  "links": {"linkedin": "url", "github": "url", "portfolio": "url", "other": ["url", ...]},'
        '  "summary": "1-2 sentences",'
        '  "skills": ["token", ...],'
        '  "education": [{"school":"", "degree":"", "field":"", "start":"","end":"","gpa":""}],'
        '  "experience": [{"company":"","title":"","start":"","end":"","location":"","bullets":["..."]}],'
        '  "projects": [{"name":"","tech":["..."],"summary":""}],'
        '  "certifications": ["..."]'
        "}"
        " Return strictly minified JSON. Do not include any commentary."
    )

    # Truncate to keep prompt size reasonable
    text = (resume_text or "").strip()
    if len(text) > 12000:
        text = text[:12000]

    user = f"RESUME TEXT:\n{text}\n\nReturn JSON now."
    out = _openai_chat(system, user, max_tokens=600, temperature=0.1)

    # Extract the first JSON object from the reply
    m = re.search(r"\{[\s\S]*\}", out or "")
    json_str = m.group(0) if m else (out or "{}")

    data: Dict[str, Any] = {}
    try:
        data = json.loads(json_str)
    except Exception:
        data = {}

    # Fallbacks for basic fields if the model returned little
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    phone = re.search(r"(\+?\d{1,2}\s*)?(\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})", text or "")
    linkedin = re.search(r"(https?://)?(www\.)?linkedin\.com/[A-Za-z0-9_/\-]+", text or "", re.I)
    github = re.search(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9_\-]+", text or "", re.I)

    data.setdefault("email", email.group(0) if email else "")
    data.setdefault("phone", phone.group(0) if phone else "")
    links = data.get("links") or {}
    if linkedin and not links.get("linkedin"):
        links["linkedin"] = linkedin.group(0)
    if github and not links.get("github"):
        links["github"] = github.group(0)
    data["links"] = links

    # crude name heuristic: first non-empty line, no '@', not a header word
    if not data.get("name"):
        for line in (text.splitlines()[:8]):
            l = line.strip()
            if l and "@" not in l and len(l) <= 60 and not re.search(r"(objective|summary|resume|curriculum vitae)", l, re.I):
                data["name"] = l
                break

    return data

def save_resume(data: Dict[str, Any], resume_text: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "resume.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA_DIR / "resume.txt").write_text(resume_text, encoding="utf-8")

# ---------- Answering from stored résumé ----------
def answer_from_resume(question: str, data: Dict[str, Any]) -> str:
    q = (question or "").lower()

    def bullets(items: List[str]) -> str:
        return "\n".join([f"- {i}" for i in items])

    if "name" in q:
        v = data.get("name") or "Not found"
        return f"**Name:** {v}"
    if "email" in q:
        v = data.get("email") or "Not found"
        return f"**Email:** {v}"
    if "phone" in q or "mobile" in q:
        v = data.get("phone") or "Not found"
        return f"**Phone:** {v}"
    if "linkedin" in q:
        v = (data.get("links") or {}).get("linkedin") or "Not found"
        return f"**LinkedIn:** {v}"
    if "github" in q:
        v = (data.get("links") or {}).get("github") or "Not found"
        return f"**GitHub:** {v}"
    if "portfolio" in q or "website" in q:
        v = (data.get("links") or {}).get("portfolio") or "Not found"
        return f"**Portfolio:** {v}"
    if "skill" in q:
        skills = data.get("skills") or []
        return "**Skills**\n\n" + (bullets(skills) if skills else "_None captured_")
    if "education" in q or "school" in q or "degree" in q:
        edu = data.get("education") or []
        if not edu: 
            return "_No education entries captured_"
        lines = []
        for e in edu:
            parts = [e.get("degree"), e.get("field"), e.get("school")]
            when = " - ".join([e.get("start") or "", e.get("end") or ""]).strip(" -")
            if when: parts.append(f"({when})")
            if e.get("gpa"): parts.append(f"GPA {e['gpa']}")
            lines.append(" • ".join([p for p in parts if p]))
        return "**Education**\n\n" + bullets(lines)
    if "project" in q:
        projs = data.get("projects") or []
        if not projs: 
            return "_No projects captured_"
        lines = []
        for p in projs[:5]:
            tech = ", ".join(p.get("tech") or [])
            s = f"{p.get('name','Project')} — {p.get('summary','')}"
            if tech: 
                s += f"  \n   _Tech_: {tech}"
            lines.append(s)
        return "**Projects**\n\n" + "\n\n".join([f"- {x}" for x in lines])
    if "experience" in q or "work" in q or "employment" in q:
        ex = data.get("experience") or []
        if not ex: 
            return "_No experience captured_"
        lines = []
        for e in ex[:5]:
            when = " - ".join([e.get("start") or "", e.get("end") or ""]).strip(" -")
            head = " • ".join([v for v in [e.get("title"), e.get("company"), when, e.get("location")] if v])
            bullets_ = "\n".join([f"   - {b}" for b in (e.get("bullets") or [])[:4]])
            lines.append(head + ("\n" + bullets_ if bullets_ else ""))
        return "**Experience**\n\n" + "\n\n".join(lines)

    # default summary
    name = data.get("name", "Candidate")
    skills = ", ".join(data.get("skills")[:10] or [])
    summary = data.get("summary") or ""
    return f"**Résumé on file for {name}.**\n\n{summary}\n\n**Key skills:** {skills if skills else '_n/a_'}"
