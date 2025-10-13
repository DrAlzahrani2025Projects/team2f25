"""
query_to_filter.py — Convert NL query into structured filters.

If OPENAI_API_KEY is set, uses LangChain + OpenAI to parse.
Otherwise falls back to simple heuristics.
"""

from __future__ import annotations
import os
import re
from typing import Dict, List

# Optional LLM mode
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))

def _heuristic_parse(q: str) -> Dict:
    q = (q or "").strip()
    if not q:
        return {}

    # role (basic)
    role = None
    role_match = re.search(r"(software|data|ml|ai|cyber|security|devops|backend|frontend|full[- ]stack)[^,]*", q, re.I)
    if role_match:
        role = role_match.group(0)

    # location (City or State)
    loc = None
    loc_match = re.search(r"(San Bernardino|Riverside|Los Angeles|San Diego|Irvine|California|CA)", q, re.I)
    if loc_match:
        loc = loc_match.group(1)

    # skills (comma or 'and' separated)
    skills: List[str] = []
    for token in re.split(r"[,\sand]+", q, flags=re.I):
        token = token.strip().lower()
        if token in {"python", "java", "c++", "c", "sql", "javascript", "typescript", "react", "aws", "azure", "gcp"}:
            skills.append(token)
    skills = list(dict.fromkeys(skills))  # dedupe

    # keywords
    kw: List[str] = []
    for k in ["intern", "internship", "software", "data", "ml", "ai", "security", "cloud", "devops", "backend", "frontend"]:
        if re.search(rf"\b{k}\b", q, re.I):
            kw.append(k)

    return {
        "role": role,
        "location": loc,
        "skills": skills,
        "keywords": kw,
    }

def parse_query_to_filter(query: str) -> Dict:
    if not USE_LLM:
        return _heuristic_parse(query)

    # LLM path
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract internship search filters as JSON keys: role, location, skills (list), keywords (list). Keep it concise."),
            ("human", "User query: {query}")
        ])
        parser = JsonOutputParser()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        chain = prompt | llm | parser
        out = chain.invoke({"query": query})
        # Ensure keys exist
        out.setdefault("role", None)
        out.setdefault("location", None)
        out.setdefault("skills", [])
        out.setdefault("keywords", [])
        return out
    except Exception:
        # Fall back gracefully if LLM path errors
        return _heuristic_parse(query)
