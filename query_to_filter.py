import os, re, json
from typing import Dict, Any

# ============================================================
# CONFIGURATION
# ============================================================
# Toggle whether to use Ollama for natural-language parsing.
USE_OLLAMA = True
# The Ollama API endpoint (by default local service).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# Default model name to use for parsing/filter extraction.
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

# ============================================================
# BASIC KEYWORD SETS
# ============================================================

# Words to ignore when extracting tokens.
GENERIC_STOP = {
    # General function / pronouns / filler words
    "i","you","your","yours","me","my","mine","we","our","ours","they","them","their","theirs",
    "this","that","these","those","it","its","is","am","are","was","were","be","being","been",
    "do","does","did","a","an","the","and","or","but","if","then","else","than","not","no","yes",
    "please","hi","hello","hey","how","what","who","where","when","why","which","name","age",
    "u","yo","sup","thanks","thank","thankyou",

    # Common job-search filler terms (not useful as keywords)
    "intern","interns","internship","internships","job","jobs","career","careers",
    "opening","openings","position","positions","apply","application","role","roles",
    "only","strict","exact","just","show","list","give","find",
    "in","at","for","from","to","csusb","cse","website","site","listed"
}

# Common technical skills to recognize directly (used for fallback extraction)
TECH_SKILLS = {
    "java","python","c++","c#","javascript","typescript","go","rust","kotlin","swift","r","matlab","sql",
    "react","angular","vue","node","express","django","flask","fastapi","spring","spring boot",".net","asp.net",
    "pandas","numpy","pytorch","tensorflow","scikit-learn","spark","hadoop","tableau","power bi",
    "selenium","cypress","playwright","pytest","junit","postman",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","bash","git","jira",
    "mysql","postgresql","mongodb","redis"
}

# Simple greetings used to detect small-talk vs internship queries.
GREETINGS = {
    "hi","hello","hey","how are you","good morning","good afternoon","good evening",
    "what is your name","your name","who are you","help","thanks","thank you",
    "your age","how old are you"
}

# ============================================================
# LLM HELPER FUNCTION
# ============================================================

def _llm_json(sys_msg: str, user: str, num_ctx=2048, num_predict=160, temp=0.1) -> Dict[str, Any]:
    """
    Calls a local Ollama model via LangChain to extract structured JSON.
    Returns {} on any failure so the rest of the app can continue.

    Parameters:
        sys_msg:  The system prompt that defines what to extract.
        user:     The user query text.
        num_ctx:  Context window.
        num_predict: Max tokens to predict.
        temp:     Sampling temperature (low = deterministic).
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate

        # Compose a prompt template with both system and user messages
        tmpl = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{q}")])

        # Initialize the Ollama LLM
        llm = ChatOllama(
            base_url=OLLAMA_HOST, model=MODEL_NAME,
            temperature=temp, streaming=False,
            model_kwargs={"num_ctx": num_ctx, "num_predict": num_predict},
        )

        # Invoke the chain with the user text
        out = (tmpl | llm).invoke({"q": user}).content

        # Extract JSON from the response (regex for { ... } structure)
        m = re.search(r"\{[\s\S]*\}", out)
        return json.loads(m.group(0) if m else out)
    except Exception:
        # Fallback to empty dict if LLM fails
        return {}

# ============================================================
# TOKENIZATION (used for non-LLM fallback)
# ============================================================

def _extract_skills_and_keywords(s: str) -> tuple[list[str], list[str]]:
    """
    Extract basic 'skills' and 'keywords' by tokenizing locally.
    This is used when LLM is off or unavailable.
    """
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

# ============================================================
# MAIN: QUERY PARSER
# ============================================================

def parse_query_to_filter(q: str) -> Dict[str, Any]:
    """
    Main function: turns a free-text user query into a structured filter dictionary
    using either LLM extraction or rule-based fallback.
    """
    if not q:
        return {}

    s = q.strip()

    # Define system prompt to instruct the LLM what JSON to produce
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

    # ---- Step 1: Try LLM extraction ----
    llm_data: Dict[str, Any] = {}
    if USE_OLLAMA:
        llm_data = _llm_json(sys, s)
        if not isinstance(llm_data, dict):
            llm_data = {}

    # ---- Step 2: Add fallback data if LLM gave nothing ----
    # Check for "strict"/"only"/"exact" modifiers
    role_match = "strict" if re.search(r"\b(strict|exact|only|just)\b", s, re.I) else "broad"
    # Detect "show all" pattern
    show_all = bool(
        re.search(r"\b(?:show|list|give|fetch|display)\s+(?:all|every)\s+internship[s]?\b", s, re.I) or
        re.search(r"\bcsusb\s+(?:listed|list)\s+internship[s]?\b", s, re.I)
    )
    # Simple token extraction for fallback
    skills, keywords = _extract_skills_and_keywords(s)

    # Merge fallback info into LLM result
    llm_data.setdefault("role_match", role_match)
    llm_data.setdefault("show_all", show_all)
    llm_data.setdefault("title_keywords", keywords)
    llm_data.setdefault("skills", skills)

    # Normalize arrays and ensure intent
    llm_data["title_keywords"] = [t.strip().lower() for t in llm_data.get("title_keywords", [])][:6]
    llm_data["skills"] = [t.strip().lower() for t in llm_data.get("skills", [])][:6]
    if not llm_data.get("intent"):
        # Default: if contains "intern", assume internship search
        llm_data["intent"] = "internship_search" if re.search(r"\bintern", s, re.I) else "general_question"

    # If the query doesn’t contain a location cue, drop any location fields.
    explicit_loc = re.search(
        r"\b(remote|onsite|hybrid|usa|united states|uk|england|canada|india|london|new york|ny|ca|tx|\d{5})\b",
        s, re.I
    )
    if not explicit_loc:
        for k in ("city", "state", "country", "zipcode"):
            llm_data.pop(k, None)

    return llm_data

# ============================================================
# INTENT CLASSIFIER
# ============================================================

def classify_intent(q: str) -> str:
    """
    Determines whether the user's input is a general chat
    or an internship-search intent.
    """
    s = (q or "").lower().strip()

    # --- Step 1: Guard résumé-related questions ---
    # These should always route to general chat (not internship search).
    if (
        re.search(r"\b(resume|résumé|cv)\b", s)
        or re.search(r"\b(skills?|experience|projects?|education|linkedin|github|email|phone|portfolio)\b", s)
    ):
        return "general_question"

    # --- Step 2: Small-talk / greetings ---
    if s in GREETINGS or any(g in s for g in GREETINGS):
        return "general_question"

    # --- Step 3: Ask LLM for deterministic intent (if available) ---
    if USE_OLLAMA:
        d = _llm_json(
            'Return JSON exactly like {"intent":"internship_search"} or {"intent":"general_question"}. '
            'If the user typed only a company name, treat it as internship_search.',
            s, num_ctx=512, num_predict=30, temp=0.0
        )
        if isinstance(d, dict) and d.get("intent") in {"internship_search", "general_question"}:
            return d["intent"]

    # --- Step 4: Fallback rule-based detection ---
    if re.search(r"\bintern(ship|ships)?\b", s):
        return "internship_search"
    if re.search(r"\b(find|show|list|apply|search|available|display|get)\b.*\b(intern|job|role|position|career|opening)\b", s):
        return "internship_search"
    # A single short token (like "Amazon") is likely an internship intent.
    if re.fullmatch(r"[a-z0-9\.\- ]{2,30}", s) and not re.search(r"\b(hi|hello|help|thanks)\b", s):
        return "internship_search"

    # Default fallback
    return "general_question"
