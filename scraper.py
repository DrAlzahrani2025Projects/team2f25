from __future__ import annotations
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime
import re, html, tldextract, pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

JUNK_HOSTS = {"youtube.com", "youtu.be"}
JUNK_KEYWORDS = {
    "proposal form", "evaluation form", "student evaluation", "supervisor evaluation",
    "report form", "handbook", "resume", "cv", "smartscholarship", "scholarship",
    "faculty & staff", "career center", "advising"
}
ALLOW_HOST_HINTS = {
    "myworkdayjobs", "workday", "greenhouse", "lever", "taleo", "icims", "smartrecruiters",
    "jobs", "careers", "career"
}

RE_SALARY = re.compile(r"(\$[\d,]+(?:\s*[-–]\s*\$[\d,]+)?(?:\s*/\s*(?:hour|hr|month|year|yr))?)", re.I)
RE_EXP_YEARS = re.compile(r"(?:at\s+least|min(?:imum)?)?\s*(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\s+years?", re.I)
RE_DEGREE = re.compile(r"\b(high school|associate|bachelor'?s?|master'?s?|ph\.?d|doctoral)\b", re.I)
RE_LOCATION_LINE = re.compile(r"(?:location|where)\s*[:\-]\s*([^\n\r|]+)", re.I)

def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _infer_company(u: str) -> Optional[str]:
    try:
        host = urlparse(u).netloc
        ext = tldextract.extract(host)
        core = ext.domain or host.split(".")[-2]
        return core.capitalize() if core else None
    except Exception:
        return None

def _is_candidate_link(text: str, url: str) -> bool:
    low = f"{text} {url}".lower()
    if any(k in low for k in JUNK_KEYWORDS): 
        return False
    host = urlparse(url).netloc.lower()
    if any(h in host for h in JUNK_HOSTS):
        return False
    if "intern" in low or "co-op" in low:
        return True
    return any(h in host or h in low for h in ALLOW_HOST_HINTS)

def _extract_details(html_src: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(html_src, "lxml")
    h1 = _clean(soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
    title = h1 or _clean(soup.title.get_text(" ", strip=True) if soup.title else "")
    text = _clean(html.unescape(soup.get_text("\n", strip=True)))
    d: Dict[str, Optional[str]] = {
        "details": text[:6000] if text else None,
        "title_from_page": title[:300] if title else None,
    }
    if m := RE_SALARY.search(text): d["salary"] = m.group(1)
    if m := RE_EXP_YEARS.search(text):
        d["exp_min"] = m.group(1);  d["exp_max"] = m.group(2) if m.group(2) else None
    if m := RE_DEGREE.search(text): d["education"] = m.group(1).title()
    if m := RE_LOCATION_LINE.search(text): d["location"] = _clean(m.group(1))
    if re.search(r"\bremote\b", text, re.I): d["remote"] = "remote"
    elif re.search(r"\bhybrid\b", text, re.I): d["remote"] = "hybrid"
    elif re.search(r"\bon[-\s]?site\b", text, re.I): d["remote"] = "onsite"
    return d

def _collect_links(page_html: str, base: str) -> List[Dict]:
    soup = BeautifulSoup(page_html, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" "))
        href = a["href"]
        if not text: 
            continue
        abs_url = urljoin(base, href)
        host = urlparse(abs_url).netloc.lower()
        key = (text.lower(), abs_url)
        if key in seen:
            continue
        if not _is_candidate_link(text, abs_url):
            continue
        company = _infer_company(abs_url)
        rows.append({
            "title": text,
            "company": company,
            "location": None,
            "posted_date": None,
            "tags": None,
            "link": abs_url,
            "host": host,
            "source": base,
        })
        seen.add(key)
    return rows

def scrape_csusb_listings(url: str = CSUSB_CSE_URL, timeout_ms: int = 60_000, deep: bool = True, max_pages: int = 80) -> pd.DataFrame:
    rows: List[Dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1366, "height": 768})
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try: page.wait_for_load_state("networkidle", timeout=3000)
            except Exception: pass
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(600)
        except PWTimeout:
            pass

        html_src = page.content()
        base_rows = _collect_links(html_src, base=url)

        if not deep:
            rows = base_rows
        else:
            for r in base_rows[:max_pages]:
                try:
                    page.goto(r["link"], wait_until="domcontentloaded", timeout=timeout_ms)
                    page.wait_for_timeout(600)
                    det = _extract_details(page.content())
                    if det.get("title_from_page") and re.search(r"\bintern", det["title_from_page"], re.I):
                        r["title"] = det["title_from_page"]
                    page_text = (det.get("details") or "") + " " + r["title"]
                    if not re.search(r"\bintern|co[-\s]?op\b", page_text, re.I):
                        continue
                    r.update({
                        "requirements": None,
                        "deadline": None,
                        "salary": det.get("salary"),
                        "education": det.get("education"),
                        "location": det.get("location") or r.get("location"),
                        "remote": det.get("remote"),
                        "details": det.get("details"),
                    })
                    rows.append(r)
                except Exception:
                    continue

        browser.close()

    today = datetime.utcnow().date().isoformat()
    for r in rows:
        r["posted_date"] = r.get("posted_date") or today

    cols = [
        "title","company","location","posted_date","tags","link","host","source",
        "deadline","requirements","salary","education","remote","details"
    ]
    df = pd.DataFrame(rows, columns=cols)
    return df
