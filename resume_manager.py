# resume_manager.py
import json, os, re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from pypdf import PdfReader
from docx import Document

# Use our OpenAI provider helpers
from llm_provider import get_openai_client, get_model

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# File -> text utilities
# ---------------------------
def _read_pdf(b: bytes) -> str:
    out = []
    reader = PdfReader(BytesIO(b))
    for p in reader.pages:
        try:
            out.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)

def _read_docx(b: bytes) -> str:
    buf = BytesIO(b)
    doc = Document(buf)
    return "\n".join([p.text for p in doc.paragraphs])

def read_file_to_text(uploaded) -> str:
    """
    Accepts a Streamlit UploadedFile-like object.
    Returns best-effort UTF-8 text for PDF/DOCX/TXT.
    """
    name = (uploaded.name or "").lower()
    b = uploaded.getvalue()
    if name.endswith(".pdf"):
        return _read_pdf(b)
    if name.endswith(".docx"):
        return _read_docx(b)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

# ---------------------------
# OpenAI chat helper
# ---------------------------
def _openai_chat(system: str, user: str, max_tokens: int = 400, temperature: float = 0.1) -> str:
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

# ---------------------------
# LLM: résumé -> JSON
# ---------------------------
def llm_structured_resume(resume_text: str) -> Dict[str, Any]:
    """
    Single LLM call that returns compact JSON only.
    Literal braces are escaped in the prompt here by doubling {{ }} if needed in other templating,
    but since we directly pass strings to OpenAI, standard braces are fine.
    """
    if not (resume_text or "").strip():
        return {}

    system = (
        "You are a precise résumé parser. Return ONLY compact JSON (no prose). "
        "Schema (omit empty fields): "
        "{"
        '  "name": "string",'
        '  "email": "string",'
        '  "phone": "string",'
        '  "links": {"linkedin":"url","github":"url","portfolio":"url","other":["url",...]},'
        '  "summary": "1-2 sentences",'
        '  "skills": ["token",...],'
        '  "education": [{"school":"","degree":"","field":"","start":"","end":"","gpa":""}],'
        '  "experience": [{"company":"","title":"","start":"","end":"","location":"","bullets":["..."]}],'
        '  "projects": [{"name":"","tech":["..."],"summary":""}],'
        '  "certifications": ["..."]'
        "}"
        " Rules: Be faithful to input text. Do not invent data. Lowercase skill tokens."
    )
    user = f"RESUME TEXT:\n{resume_text[:20000]}\n\nReturn JSON now."
    raw = _openai_chat(system, user, max_tokens=600, temperature=0.1)

    # Extract JSON payload
    m = re.search(r"\{[\s\S]*\}", raw or "")
    json_str = m.group(0) if m else (raw or "{}")
    try:
        return json.loads(json_str)
    except Exception:
        return {}

# ---------------------------
# LLM: router (is the user asking about résumé?)
# ---------------------------
def llm_is_resume_question(user_text: str) -> bool:
    system = (
        "Return JSON only. Decide if the user is asking about the résumé on file "
        'with yes/no: {"resume_q": true|false}. '
        "Treat queries like 'my name in resume', 'list my skills', 'show projects', "
        "'what is my linkedin', 'education from my cv' as true."
    )
    user = user_text
    out = _openai_chat(system, user, max_tokens=30, temperature=0.0)
    m = re.search(r"\{[\s\S]*\}", out or "")
    try:
        d = json.loads(m.group(0) if m else (out or "{}"))
        return bool(d.get("resume_q"))
    except Exception:
        return False

# ---------------------------
# LLM: grounded résumé QA
# ---------------------------
def llm_answer_from_resume(user_text: str, resume_text: str, resume_json: Optional[Dict[str, Any]] = None) -> str:
    system = (
        "You are a concise assistant that answers ONLY using the provided résumé content. "
        "If the answer is not present, say you cannot find it in the résumé. "
        "Prefer exact values from JSON; otherwise quote short snippets from TEXT. "
        "Never fabricate."
    )
    j = json.dumps(resume_json or {}, ensure_ascii=False)
    user = (
        "QUESTION:\n"
        f"{user_text}\n\n"
        "RESUME JSON (may be partial):\n"
        f"{j}\n\n"
        "RESUME TEXT:\n"
        f"{resume_text[:20000]}\n\n"
        "Answer succinctly. If listing skills/education/experience, use short bullet points."
    )
    return _openai_chat(system, user, max_tokens=350, temperature=0.1)

# ---------------------------
# persistence
# ---------------------------
def save_resume(resume_text: str, resume_json: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "resume.txt").write_text(resume_text or "", encoding="utf-8")
    (DATA_DIR / "resume.json").write_text(
        json.dumps(resume_json or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
