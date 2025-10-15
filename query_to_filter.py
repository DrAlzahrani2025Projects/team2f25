import os, re, json
from typing import Dict, Any, List

USE_OLLAMA = True
OLLAMA_HOST = os.getenv("OLLAMA_HOST","http://127.0.0.1:11434")
MODEL_NAME  = os.getenv("MODEL_NAME","qwen2.5:0.5b")

GENERIC_STOP = {
    "intern","interns","internship","internships","job","jobs","career","careers",
    "opening","openings","position","positions","apply","application","role","roles",
    "only","strict","exact","just","show","list","give","all","find","me","the","a","an",
    "in","at","for","from","to","please","csusb","cse","website","site"
}
KNOWN_COMPANIES = {
    "nasa","google","microsoft","oracle","pfizer","pwc","lanl","kpmg","goldman sachs","ibm",
    "boeing","northrop","northrop grumman","lockheed","lockheed martin","raytheon","virgin galactic",
    "doe","naval","navsea","merck","disney","edwards","jpmorgan","jp morgan"
}

def _llm_json(sys_msg: str, user: str, num_ctx=4096, num_predict=512, temp=0.1) -> Dict[str,Any]:
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        tmpl = ChatPromptTemplate.from_messages([("system", sys_msg), ("human", "{q}")])
        llm = ChatOllama(base_url=OLLAMA_HOST, model=MODEL_NAME, temperature=temp, streaming=False,
                         model_kwargs={"num_ctx": num_ctx, "num_predict": num_predict})
        out = (tmpl | llm).invoke({"q": user}).content
        m = re.search(r"\{[\s\S]*\}", out)
        return json.loads(m.group(0) if m else out)
    except Exception:
        return {}

def _extract_company_heuristic(s: str) -> str | None:
    s_low = s.lower()
    if m := re.search(r"\b(?:at|for|from|to)\s+([a-z][a-z\.\-&\s]{2,30})", s_low):
        cand = m.group(1).strip(" .-")
        cand = re.sub(r"\.(com|gov|edu|org).*","", cand)
        return cand
    for c in KNOWN_COMPANIES:
        if c in s_low: return c
    if m := re.search(r"\b([a-z0-9\-]+)\.(?:gov|com|edu|org)\b", s_low):
        return m.group(1)
    return None

def _clean_title_keywords(tokens: List[str], company_name: str | None) -> List[str]:
    out=[]
    comp_parts=set()
    if company_name:
        comp_parts={w for w in re.split(r"[\s\-]+", company_name.lower()) if len(w)>2}
    for t in tokens or []:
        t=str(t).strip().lower()
        if not t or t in GENERIC_STOP: 
            continue
        if company_name and (t==company_name or t in comp_parts):
            continue  # don't treat company token as role keyword
        out.append(t)
    return out[:6]

def parse_query_to_filter(q: str) -> Dict[str,Any]:
    if not q: return {}
    s=q.strip()
    show_all = bool(re.search(r"\b(all\s+internships|show\s+all|list\s+all)\b", s, re.I))
    role_match = "strict" if re.search(r"\b(strict|exact|only|just)\b", s, re.I) else "broad"
    company_name = _extract_company_heuristic(s)

    if USE_OLLAMA:
        sys=("Extract job filters; return JSON keys (omit nulls): "
             "title_keywords, skills, city, state, country, zipcode, remote_type, employment_type, "
             "experience_level, education_level, salary_min, salary_max, company_name, role_match, show_all. "
             "Lower-case tokens; arrays ≤6; do not invent.")
        data=_llm_json(sys, s)
        if isinstance(data,dict):
            data["role_match"] = data.get("role_match") or role_match
            data["show_all"] = data.get("show_all") or show_all
            data["company_name"] = data.get("company_name") or company_name
            data["title_keywords"] = _clean_title_keywords(data.get("title_keywords") or [], data["company_name"])
            return data

    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9#+\-]{2,}", s)]
    return {
        "title_keywords": _clean_title_keywords(toks, company_name),
        "company_name": company_name,
        "role_match": role_match,
        "show_all": show_all,
    }

def classify_intent(q: str) -> str:
    s=(q or "").lower()
    if re.search(r"\b(intern|role|opening|position|career|apply)\b", s): return "internship_search"
    if _extract_company_heuristic(s): return "internship_search"
    try:
        d=_llm_json("Return exactly 'internship_search' or 'general_question'.", s, 512, 20, 0.0)
        txt=(json.dumps(d).lower() if isinstance(d,dict) else str(d).lower())
        if "internship" in txt: return "internship_search"
        if "general" in txt: return "general_question"
    except Exception: pass
    return "internship_search"
