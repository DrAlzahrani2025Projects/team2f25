from __future__ import annotations
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime
from pathlib import Path
import os, re, html
import pandas as pd
from bs4 import BeautifulSoup # pyright: ignore[reportMissingImports]
from playwright.sync_api import sync_playwright # pyright: ignore[reportMissingImports]

# ---------- constants ----------
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

# We still expose these envs so your app doesn’t break, but we won’t use deep scraping.
MAX_PAGES  = int(os.getenv("MAX_PAGES", "30"))
TIMEOUT_MS = int(os.getenv("TIMEOUT_MS", "10000"))  # 10s timeout for the one CSUSB page

# Heuristics to keep only internship-ish anchors from the CSUSB page
JUNK_HOSTS = {"youtube.com", "youtu.be"}
JUNK_KEYWORDS = {
    "proposal form","evaluation form","student evaluation","supervisor evaluation",
    "report form","handbook","resume","cv","smartscholarship","scholarship",
    "faculty & staff","career center","advising"
}
ALLOW_HOST_HINTS = {
    "myworkdayjobs","workday","greenhouse","lever","taleo","icims","smartrecruiters",
    "jobs","careers","career"
}

# ---------- small helpers ----------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _infer_company(abs_url: str) -> Optional[str]:
    try:
        host = urlparse(abs_url).netloc.lower()
        # domain part before TLD
        parts = host.split(".")
        if len(parts) >= 2:
            core = parts[-2]
        else:
            core = host
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
    # keep common career-site patterns too (often internship pages live there)
    return any(h in host or h in low for h in ALLOW_HOST_HINTS)

def _collect_links(page_html: str, base: str) -> List[Dict]:
    soup = BeautifulSoup(page_html, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text:
            continue
        href = a["href"]
        abs_url = urljoin(base, href)
        host = urlparse(abs_url).netloc.lower()

        key = (text.lower(), abs_url)
        if key in seen:
            continue
        if not _is_candidate_link(text, abs_url):
            continue

        rows.append({
            "title": text,
            "company": _infer_company(abs_url),
            "location": None,
            "posted_date": datetime.utcnow().date().isoformat(),
            "tags": None,
            "link": abs_url,
            "host": host,
            "source": base,
            "deadline": None,
            "requirements": None,
            "salary": None,
            "education": None,
            "remote": None,
            "details": None,
        })
        seen.add(key)
    return rows

# ---------- MAIN (CSUSB-only; no deep scraping) ----------
def scrape_csusb_listings(
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS,
    deep: bool = False,           # kept for compatibility; ignored
    max_pages: int = MAX_PAGES,   # kept for compatibility; ignored
) -> pd.DataFrame:
    """
    FAST scraper that visits only the CSUSB CSE 'Internships & Careers' page
    and returns all internship-like links found there. No outbound page visits.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})

        # Block heavy assets (saves a bit even on CSUSB page)
        def _should_block(u: str, rtype: str) -> bool:
            if rtype in {"image", "media", "font", "stylesheet"}:
                return True
            return any(b in u for b in ["analytics", "doubleclick", "tracking", "facebook"])
        ctx.route("**/*", lambda route, req:
            route.abort() if _should_block(req.url, req.resource_type) else route.continue_()
        )

        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # tiny idle wait (don’t block demo)
        try:
            page.wait_for_load_state("networkidle", timeout=1500)
        except Exception:
            pass

        html_src = page.content()
        browser.close()

    rows = _collect_links(html_src, base=url)

    cols = [
        "title","company","location","posted_date","tags","link","host","source",
        "deadline","requirements","salary","education","remote","details"
    ]
    df = pd.DataFrame(rows, columns=cols)
    # make sure all expected columns exist even if empty
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

# ---------- fast company-only fallback (still CSUSB-only) ----------
def quick_company_links_playwright(
    company_token: str,
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS
) -> pd.DataFrame:
    """
    Return only anchors on the CSUSB page whose text or host contains company_token.
    Still no outbound visits.
    """
    token = (company_token or "").strip().lower()
    if not token:
        return pd.DataFrame(columns=[
            "title","company","location","posted_date","tags","link","host","source",
            "deadline","requirements","salary","education","remote","details"
        ])

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        soup = BeautifulSoup(page.content(), "lxml")
        browser.close()

    main = soup.find("main") or soup
    rows, seen = [], set()
    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text:
            continue
        href = a["href"]
        abs_url = urljoin(url, href)
        host = urlparse(abs_url).netloc.lower()
        key = (text.lower(), abs_url)
        if key in seen:
            continue
        if token in text.lower() or token in host:
            rows.append({
                "title": text,
                "company": _infer_company(abs_url),
                "location": None,
                "posted_date": datetime.utcnow().date().isoformat(),
                "tags": None,
                "link": abs_url,
                "host": host,
                "source": url,
                "deadline": None,
                "requirements": None,
                "salary": None,
                "education": None,
                "remote": None,
                "details": None,
            })
            seen.add(key)

    cols = [
        "title","company","location","posted_date","tags","link","host","source",
        "deadline","requirements","salary","education","remote","details"
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df
