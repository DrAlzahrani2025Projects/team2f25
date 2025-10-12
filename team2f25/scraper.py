"""
scraper.py — CSUSB CSE internships scraper with light enrichment.

What it does
------------
- Loads https://www.csusb.edu/cse/internships-careers with Playwright (Chromium)
- Extracts meaningful anchors from the main content area
- Normalizes absolute URLs
- Infers `company` from the link's domain (e.g., careers.microsoft.com -> Microsoft)
- Guesses `location` from anchor text when possible (very lightweight)
- Returns a pandas DataFrame with columns:
  ["title", "company", "location", "posted_date", "tags", "link", "source"]

Notes
-----
- The page mostly lists outbound links to external job portals; company/location
  are rarely present in-page. Location inference here is intentionally light.
- For richer details, a future “deep scrape” step could follow each outbound link.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse
import re

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"

# A normal desktop UA helps some sites serve complete markup.
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# Keep links that look like internships/careers; exclude admin/docs.
KEEP_KEYWORDS = ["intern", "internship", "career", "careers", "job", "jobs", "opportunit"]
EXCLUDE_KEYWORDS = ["form", "proposal", "evaluation", "pdf"]

# Light-weight location hints (can be expanded later)
CITY_HINTS = [
    "San Bernardino", "Riverside", "Los Angeles", "San Diego", "Irvine",
    "Redlands", "Victorville", "Ontario", "Rancho Cucamonga", "CA", "California",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _absolute(href: str) -> str:
    """Turn a relative/anchor href into an absolute URL under csusb.edu."""
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"https://www.csusb.edu{href}"
    return f"https://www.csusb.edu/{href.lstrip('./')}"


def _looks_relevant(text: str, href: str) -> bool:
    """Return True for links that look like internship/career content."""
    hay = f"{text} {href}".lower()
    if any(x in hay for x in EXCLUDE_KEYWORDS):
        return False
    return any(k in hay for k in KEEP_KEYWORDS)


def _infer_company_from_url(url: str) -> str | None:
    """
    Infer company/organization from domain.

    Examples:
      careers.microsoft.com -> Microsoft
      jobs.apple.com       -> Apple
      www.nasa.gov         -> Nasa
    """
    try:
        host = urlparse(url).netloc
        if not host:
            return None
        parts = [p for p in host.split(".") if p not in ("www", "careers", "jobs", "job", "work", "boards")]
        brand = parts[-2] if len(parts) >= 2 else parts[0]
        brand = brand.replace("-", " ").replace("_", " ").strip()
        return brand.title() if brand else None
    except Exception:
        return None


def _guess_location_from_text(text: str) -> str | None:
    """Very light location guess from anchor text."""
    for hint in CITY_HINTS:
        if hint.lower() in text.lower():
            return hint
    m = re.search(r"\b([A-Za-z ]+),\s*([A-Z]{2})\b", text)
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"
    return None


def _extract_from_html(html: str) -> List[Dict]:
    """Parse HTML and extract relevant rows with minimal enrichment."""
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("main") or soup.select_one("[role='main']") or soup.select_one("#main-content") or soup

    rows: List[Dict] = []
    for a in main.find_all("a"):
        href = (a.get("href") or "").strip()
        text = (a.get_text(strip=True) or "").strip()
        if not href or not text:
            continue
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        if not _looks_relevant(text, href):
            continue

        url = _absolute(href)
        company = _infer_company_from_url(url) if "csusb.edu" not in url else None
        location = _guess_location_from_text(text)

        rows.append({
            "title": text,
            "company": company,
            "location": location,
            "posted_date": None,   # filled below
            "tags": None,
            "link": url,
            "source": "csusb_cse_internships",
        })

    # De-duplicate by (title, link)
    seen = set()
    out: List[Dict] = []
    for r in rows:
        key = (r["title"], r["link"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_csusb_listings(url: str = CSUSB_CSE_URL, timeout_ms: int = 60_000) -> pd.DataFrame:
    """
    Fetch the CSUSB CSE internships page with Playwright and return a DataFrame.

    Columns:
      - title, company, location, posted_date, tags, link, source
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1366, "height": 768})
        page = ctx.new_page()

        try:
            # Prefer full network settle for more consistent DOM
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(800)
        except PWTimeout:
            # Fallback to DOMContentLoaded if networkidle times out
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(800)

        html = page.content()
        browser.close()

    rows = _extract_from_html(html)
    today = datetime.utcnow().date().isoformat()
    for r in rows:
        if not r["posted_date"]:
            r["posted_date"] = today

    df = pd.DataFrame(rows, columns=[
        "title", "company", "location", "posted_date", "tags", "link", "source"
    ])
    return df

