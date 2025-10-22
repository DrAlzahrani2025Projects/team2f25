import os, re, json
from typing import Dict, Any

# ---------- Configuration ----------
USE_OLLAMA = True
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

# ---------- Keyword Sets ----------
GENERIC_STOP = {
    # general/function words
    "i","you","your","yours","me","my","mine","we","our","ours","they","them","their","theirs",
    "this","that","these","those","it","its","is","am","are","was","were","be","being","been",
    "do","does","did","a","an","the","and","or","but","if","then","else","than","not","no","yes",
    "please","hi","hello","hey","how","what","who","where","when","why","which","name","age",
    "u","yo","sup","thanks","thank","thankyou",

    # job-search generic words (don’t treat them as keywords)
    "intern","interns","internship","internships","job","jobs","career","careers",
    "opening","openings","position","positions","apply","application","role","roles",
    "only","strict","exact","just","show","list","give","find",
    "in","at","for","from","to","csusb","cse","website","site","listed"
}

TECH_SKILLS = {
    "java","python","c++","c#","javascript","typescript","go","rust","kotlin","swift","r","matlab","sql",
    "react","angular","vue","node","express","django","flask","fastapi","spring","spring boot",".net","asp.net",
    "pandas","numpy","pytorch","tensorflow","scikit-learn","spark","hadoop","tableau","power bi",
    "selenium","cypress","playwright","pytest","junit","postman",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","bash","git","jira",
    "mysql","postgresql","mongodb","redis"
}

GREETINGS = {
    "hi","hello","hey","how are you","good morning","good afternoon","good evening",
    "what is your name","your name","who are you","help","thanks","thank you",
    "your age","how old are you"
}


# ---------- LLM helper ----------
def _llm_json(sys_msg: str, user: str, num_ctx=2048, num_predict=160, temp=0.1) -> Dict[str, Any]:
    """
    Call local Ollama via langchain_ollama. Return {} on any failure so the app keeps running.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        tmpl = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{q}")])
        llm = ChatOllama(
            base_url=OLLAMA_HOST, model=MODEL_NAME,
            temperature=temp, streaming=False,
            model_kwargs={"num_ctx": num_ctx, "num_predict": num_predict},
        )
        out = (tmpl | llm).invoke({"q": user}).content
        m = re.search(r"\{[\s\S]*\}", out)
        return json.loads(m.group(0) if m else out)
    except Exception:
        return {}

# ---------- Local tokenization (used only for skills/keywords fallback) ----------
def _extract_skills_and_keywords(s: str) -> tuple[list[str], list[str]]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\.\+#\-]{1,}", s)]
    skills, keywords = [], []
    for t in tokens:
        if t in GENERIC_STOP:
            continue
        if t in TECH_SKILLS:
            skills.append(t)
        else:
            keywords.append(t)
    return skills[:6], keywords[:6]

# ---------- Query Parsing ----------
def parse_query_to_filter(q: str) -> Dict[str, Any]:
    if not q:
        return {}

    s = q.strip()

    # Strict schema: LLM is the single source of truth for company, intent & location.
    sys = (
        "You extract job-search filters from a short user query.\n"
        "Return ONLY compact JSON with these keys (omit keys that don't apply):\n"
        "{\n"
        '  "intent": "internship_search|general_question",\n'
        '  "company_name": "string",\n'
        '  "title_keywords": ["token", ...],\n'
        '  "skills": ["token", ...],\n'
        '  "show_all": true|false,\n'
        '  "role_match": "broad|strict",\n'
        '  "city": "string",\n'
        '  "state": "string",\n'
        '  "country": "string",\n'
        '  "zipcode": "string",\n'
        '  "remote_type": "remote|hybrid|onsite"\n'
        "}\n"
        "Rules:\n"
        "- Never infer or guess any location fields; include them ONLY if the user explicitly typed them.\n"
        "- Lower-case tokens; keep arrays ≤ 6; avoid nulls; do not invent values."
    )

    llm_data: Dict[str, Any] = {}
    if USE_OLLAMA:
        llm_data = _llm_json(sys, s)
        if not isinstance(llm_data, dict):
            llm_data = {}

    # Fallbacks if LLM is unavailable or returns nothing
    # role_match + show_all from simple regex cues
    role_match = "strict" if re.search(r"\b(strict|exact|only|just)\b", s, re.I) else "broad"
    show_all = bool(
        re.search(r"\b(?:show|list|give|fetch|display)\s+(?:all|every)\s+internship[s]?\b", s, re.I) or
        re.search(r"\bcsusb\s+(?:listed|list)\s+internship[s]?\b", s, re.I)
    )
    skills, keywords = _extract_skills_and_keywords(s)

    # Merge fallbacks only if missing
    llm_data.setdefault("role_match", role_match)
    llm_data.setdefault("show_all", show_all)
    llm_data.setdefault("title_keywords", keywords)
    llm_data.setdefault("skills", skills)

    # Normalize arrays and intent
    llm_data["title_keywords"] = [t.strip().lower() for t in llm_data.get("title_keywords", [])][:6]
    llm_data["skills"] = [t.strip().lower() for t in llm_data.get("skills", [])][:6]
    if not llm_data.get("intent"):
        llm_data["intent"] = "internship_search" if re.search(r"\bintern", s, re.I) else "general_question"

    # Belt & suspenders: drop any location fields if the user did NOT type a location cue.
    explicit_loc = re.search(
        r"\b(remote|onsite|hybrid|usa|united states|uk|england|canada|india|london|new york|ny|ca|tx|\d{5})\b",
        s, re.I
    )
    if not explicit_loc:
        for k in ("city", "state", "country", "zipcode"):
            llm_data.pop(k, None)

    return llm_data


def classify_intent(q: str) -> str:
    """
    Prefer the LLM’s intent. Fall back to lightweight regex so the app still works
    if the LLM is unavailable. No relative imports; GREETINGS is local.
    """
    s = (q or "").lower().strip()

    # --- NEW: résumé / personal-info guard -------------------------------
    # Any query clearly about the user's résumé should be treated as general,
    # so the app doesn't route it into the internship search flow.
    if (
        re.search(r"\b(resume|résumé|cv)\b", s)
        or re.search(r"\b(skills?|experience|projects?|education|linkedin|github|email|phone|portfolio)\b", s)
    ):
        return "general_question"
    # ---------------------------------------------------------------------

    # small-talk fast path
    if s in GREETINGS or any(g in s for g in GREETINGS):
        return "general_question"

    # Ask the tiny LLM directly (deterministic)
    if USE_OLLAMA:
        d = _llm_json(
            'Return JSON exactly like {"intent":"internship_search"} or {"intent":"general_question"}. '
            'If the user typed only a company name, treat it as internship_search.',
            s, num_ctx=512, num_predict=30, temp=0.0
        )
        if isinstance(d, dict) and d.get("intent") in {"internship_search", "general_question"}:
            return d["intent"]

    # Fallbacks (model off / failure)
    if re.search(r"\bintern(ship|ships)?\b", s):
        return "internship_search"
    if re.search(r"\b(find|show|list|apply|search|available|display|get)\b.*\b(intern|job|role|position|career|opening)\b", s):
        return "internship_search"
    # Treat a short single token (likely a company/domain) as internship intent
    if re.fullmatch(r"[a-z0-9\.\- ]{2,30}", s) and not re.search(r"\b(hi|hello|help|thanks)\b", s):
        return "internship_search"

    return "general_question"
