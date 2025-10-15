import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional

USE_OLLAMA = True
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

GENERIC_STOP = {
    "intern","interns","internship","internships","job","jobs","career","careers",
    "opening","openings","position","positions","apply","application","role","roles",
    "only","strict","exact","just","show","list","give","find","me","the","a","an",
    "in","at","for","from","to","please","csusb","cse","website","site","all"
}

KNOWN_COMPANIES = {
    # add common names so we catch them fast
    "amazon","nasa","google","microsoft","oracle","pfizer","pwc","lanl","kpmg",
    "goldman sachs","ibm","boeing","northrop","northrop grumman","lockheed",
    "lockheed martin","raytheon","virgin galactic","doe","naval","navsea",
    "merck","disney","edwards","jpmorgan","jp morgan","pwc","deloitte"
}

TECH_SKILLS = {
    "java","python","c++","c#","javascript","typescript","go","rust","kotlin","swift","r","matlab","sql",
    "react","angular","vue","node","express","django","flask","fastapi","spring","springboot","spring-boot",".net","asp.net",
    "pandas","numpy","pytorch","tensorflow","scikit-learn","spark","hadoop","tableau","powerbi","power-bi",
    "selenium","cypress","playwright","pytest","junit","postman",
    "aws","azure","gcp","docker","kubernetes","terraform","linux","bash","git","jira",
    "mysql","postgresql","mongodb","redis"
}

def _llm_json(sys_msg: str, user: str, num_ctx=4096, num_predict=256, temp=0.1) -> Dict[str, Any]:
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        tmpl = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{q}")])
        llm = ChatOllama(
            base_url=OLLAMA_HOST,
            model=MODEL_NAME,
            temperature=temp,
            streaming=False,
            model_kwargs={"num_ctx": num_ctx, "num_predict": num_predict},
        )
        out = (tmpl | llm).invoke({"q": user}).content
        m = re.search(r"\{[\s\S]*\}", out)
        return json.loads(m.group(0) if m else out)
    except Exception:
        return {}

def _extract_company_heuristic(s: str) -> Optional[str]:
    s_low = s.lower()

    # 1) “at/for/from/to <company>”
    m = re.search(r"\b(?:at|for|from|to)\s+([a-z][a-z\.\-&\s]{2,30})", s_low)
    if m:
        cand = m.group(1).strip(" .-")
        cand = re.sub(r"\.(com|gov|edu|org).*", "", cand)
        return cand

    # 2) known list
    for c in KNOWN_COMPANIES:
        if c in s_low:
            return c

    # 3) domain pattern
    m = re.search(r"\b([a-z0-9\-]+)\.(?:gov|com|edu|org)\b", s_low)
    if m:
        return m.group(1)

    # 4) fallback guess: single distinctive token that isn't a skill/stop-word
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\.\-]{1,}", s_low)]
    cand_tokens = []
    for t in tokens:
        t_clean = t.strip(".-")
        if len(t_clean) < 3:
            continue
        if t_clean in GENERIC_STOP:
            continue
        if t_clean in TECH_SKILLS:
            continue
        cand_tokens.append(t_clean)
    # choose longest token (heuristic)
    if cand_tokens:
        cand_tokens.sort(key=len, reverse=True)
        return cand_tokens[0]

    return None

def _extract_skills_and_keywords(s: str, company_name: Optional[str]) -> Tuple[List[str], List[str]]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\.\+#\-]{1,}", s)]
    comp_parts = set()
    if company_name:
        comp_parts = {w for w in re.split(r"[\s\-]+", company_name.lower()) if len(w) > 2}
    skills, keywords = [], []
    for t in tokens:
        if t in GENERIC_STOP:
            continue
        if company_name and (t == company_name or t in comp_parts):
            continue
        if t in TECH_SKILLS:
            skills.append(t)
            continue
        keywords.append(t)
    return skills[:6], keywords[:6]

def parse_query_to_filter(q: str) -> Dict[str, Any]:
    if not q:
        return {}

    s = q.strip()
    show_all  = bool(re.search(r"\b(all\s+internships|show\s+all\s+internships|list\s+all\s+internships)\b", s, re.I))
    role_match = "strict" if re.search(r"\b(strict|exact|only|just|exactly)\b", s, re.I) else "broad"
    company_name = _extract_company_heuristic(s)

    skills, keywords = _extract_skills_and_keywords(s, company_name)

    data = {
        "title_keywords": keywords,
        "skills": skills,
        "company_name": company_name,
        "role_match": role_match,
        "show_all": show_all,
    }

    if USE_OLLAMA:
        sys = (
            "Extract job filters and return JSON with keys (omit nulls): "
            "title_keywords, skills, city, state, country, zipcode, remote_type, employment_type, "
            "experience_level, education_level, salary_min, salary_max, company_name, role_match, show_all. "
            "Use lower-case tokens; arrays <= 6; do not invent."
        )
        llm_data = _llm_json(sys, s)
        if isinstance(llm_data, dict):
            # keep our deterministic fields if present
            llm_data.update(data)
            return llm_data

    return data

def classify_intent(q: str) -> str:
    s = (q or "").lower()
    if re.search(r"\b(intern|role|opening|position|career|apply)\b", s):
        return "internship_search"
    if _extract_company_heuristic(s):
        return "internship_search"
    try:
        d = _llm_json("Return exactly 'internship_search' or 'general_question'.", s, 512, 20, 0.0)
        txt = (json.dumps(d).lower() if isinstance(d, dict) else str(d).lower())
        if "internship" in txt:
            return "internship_search"
        if "general" in txt:
            return "general_question"
    except Exception:
        pass
    return "internship_search"
