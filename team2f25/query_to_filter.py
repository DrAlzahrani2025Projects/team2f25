"""
Minimal NL -> structured filter mapping.
- Uses LangChain+OpenAI if OPENAI_API_KEY is present
- Otherwise uses simple heuristics (fast + dependency-light)
Output: {role, location, skills, keywords}
"""

import os
import json
import re
from typing import Dict, List

KNOWN_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "react",
    "node", "sql", "aws", "azure", "gcp", "ml", "machine learning", "data",
    "docker", "linux", "kubernetes"
]

ROLES = [
    "software", "software engineering", "data science", "cybersecurity",
    "devops", "frontend", "backend", "full stack"
]

def _heuristic_parse(query: str) -> Dict:
    q = (query or "").lower().strip()

    # location: text after ' in ' up to comma or end
    location = None
    m = re.search(r"\bin\s+([a-z][a-z\s\.-]+?)(?:[,;]|$)", q)
    if m:
        location = m.group(1).strip().title()

    # role
    role = None
    for r in ROLES:
        if r in q:
            role = r
            break

    # skills
    skills = sorted({s for s in KNOWN_SKILLS if s in q})

    # remaining keywords (very lightweight)
    words = re.findall(r"[a-zA-Z#\+]+", q)
    keywords = [w for w in words if w not in skills][:10]

    return {
        "role": role,
        "location": location,
        "skills": skills,
        "keywords": keywords
    }

def parse_query_to_filter(query: str) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _heuristic_parse(query)

    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract internship search filters. Return ONLY JSON with keys: role, location, skills, keywords."),
            ("human", "Query: {query}")
        ])
        resp = (prompt | llm).invoke({"query": query})
        text = resp.content if hasattr(resp, "content") else str(resp)
        # Try to parse JSON; fall back to heuristic if it fails
        import json
        data = json.loads(text)
        base = _heuristic_parse(query)
        return {
            "role": data.get("role") or base["role"],
            "location": data.get("location") or base["location"],
            "skills": data.get("skills") or base["skills"],
            "keywords": data.get("keywords") or base["keywords"],
        }
    except Exception:
        return _heuristic_parse(query)
