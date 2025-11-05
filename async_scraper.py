# async_scraper.py
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Dict
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import ClientTimeout, ClientConnectorError
from bs4 import BeautifulSoup


# ---------- Config ----------
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

# Domain / link filters
JUNK_HOSTS = {
    "csusb.edu",
    "facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
}
JUNK_SCHEMES = ("mailto:", "tel:", "javascript:")

# Hints that a URL/text is about careers/interns
CAREER_HOST_HINTS = (
    "workday",
    "myworkdayjobs",
    "greenhouse",
    "lever",
    "icims",
    "smartrecruiters",
    "taleo",
    "successfactors",
    "avature",
    "brassring",
    "eightfold",
    "jobs.",
    ".jobs",
    "careers.",
    "/careers",
    "/career",
    "/jobs",
    "/job",
    "/intern",
    "/internship",
)

# Internship keyword detector
INTERNSHIP_TERMS = re.compile(
    r"\b(intern|internship|co-?op|student|early[-\s]?career|graduate|apprentice|placement|summer analyst)\b",
    re.I,
)


# ---------- Models ----------
@dataclass
class Job:
    title: str
    company: Optional[str]
    location: Optional[str]
    url: str
    posted_date: Optional[str]
    source_page: str        # the company page we scraped
    host: str
    details: Optional[str]

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------- Helpers ----------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""


def _is_external_career_link(url: str, text: str) -> bool:
    """
    External (non-csusb) + has career-ish hints either in URL or anchor text/parent text.
    """
    if not url or any(url.lower().startswith(s) for s in JUNK_SCHEMES):
        return False
    h = _host(url)
    if not h or any(bad in h for bad in JUNK_HOSTS):
        return False
    low_url, low_txt = url.lower(), text.lower()
    if any(k in low_url for k in CAREER_HOST_HINTS):
        return True
    if re.search(r"\b(career|careers|jobs?|intern|students?)\b", low_txt):
        return True
    return False


def _looks_like_internship(text: str) -> bool:
    return bool(INTERNSHIP_TERMS.search(text or ""))


def _maybe_location(block_text: str) -> Optional[str]:
    m = re.search(
        r"\b(Remote|Hybrid|On[- ]?site|Virtual|[A-Z][a-z]+,\s*[A-Z]{2}|United States|USA|Canada|India|UK)\b",
        block_text,
        re.I,
    )
    return _clean(m.group(0)) if m else None


# ---------- Async HTTP ----------
async def _fetch_html(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, timeout=ClientTimeout(total=20)) as r:
            if r.status >= 400:
                return ""
            return await r.text()
    except (asyncio.TimeoutError, ClientConnectorError):
        return ""
    except Exception:
        return ""


# ---------- Step 1: Collect external links from CSUSB page ----------
async def collect_company_links(csusb_url: str = CSUSB_CSE_URL) -> List[str]:
    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:
        html = await _fetch_html(session, csusb_url)

    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup

    links: List[str] = []
    seen = set()

    for a in main.find_all("a", href=True):
        raw_text = _clean(a.get_text(" ", strip=True))
        if not raw_text:
            continue

        href = a["href"].strip()
        abs_url = urljoin(csusb_url, href)
        if _host(abs_url).endswith("csusb.edu"):
            # keep only external
            continue

        # allow parent text as a hint
        parent_text = _clean((a.parent or a).get_text(" ", strip=True))
        combined_text = f"{raw_text} {parent_text}"

        if not _is_external_career_link(abs_url, combined_text):
            continue

        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(abs_url)

    return links


# ---------- Step 2: Parse jobs from a single page ----------
def parse_jobs_from_html(page_url: str, html: str) -> List[Job]:
    """
    Very robust generic parser:
    - look at anchors that mention intern terms in title, href, or nearby text
    - infer company from domain
    - infer location from surrounding text
    """
    soup = BeautifulSoup(html or "", "lxml")
    out: List[Job] = []

    anchors = soup.find_all("a", href=True, limit=800)
    for a in anchors:
        text = _clean(a.get_text(" ", strip=True))
        href = a["href"].strip()
        full = urljoin(page_url, href)

        # local context to improve title + details
        parent = a.parent or a
        block_text = _clean(parent.get_text(" ", strip=True))

        # detect “internship-ness”
        if not (_looks_like_internship(text) or _looks_like_internship(href) or _looks_like_internship(block_text)):
            continue

        # title heuristic
        title = text if len(text) >= 5 else (block_text[:120] or "Internship")

        # company heuristic from host
        host = _host(full)
        company_guess = host.split(".")[0].capitalize() if host else None

        # location heuristic
        location = _maybe_location(block_text)

        out.append(
            Job(
                title=_clean(title),
                company=company_guess,
                location=location,
                url=full,
                posted_date=datetime.utcnow().date().isoformat(),
                source_page=page_url,
                host=host,
                details=(block_text[:300] or None),
            )
        )

    # deduplicate by URL
    uniq: Dict[str, Job] = {}
    for j in out:
        uniq[j.url] = j
    return list(uniq.values())


# ---------- Step 3: Visit all company links concurrently ----------
async def fetch_many_html(urls: Iterable[str], concurrency: int = 12) -> List[Tuple[str, str]]:
    sem = asyncio.Semaphore(concurrency)
    out: List[Tuple[str, str]] = []

    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:

        async def one(u: str):
            async with sem:
                html = await _fetch_html(session, u)
                if html:
                    out.append((u, html))

        await asyncio.gather(*(one(u) for u in urls))

    return out


async def scrape_csusb_internships(
    csusb_url: str = CSUSB_CSE_URL,
    max_companies: int = 60,
    concurrency: int = 12,
) -> List[Job]:
    # 1) Collect external company/career links
    links = await collect_company_links(csusb_url)
    if not links:
        return []

    if len(links) > max_companies:
        links = links[:max_companies]

    # 2) Concurrently fetch those pages
    fetched = await fetch_many_html(links, concurrency=concurrency)

    # 3) Parse internship postings from each page
    jobs: List[Job] = []
    for page_url, html in fetched:
        jobs.extend(parse_jobs_from_html(page_url, html))

    # keep only those that truly look like internships
    jobs = [j for j in jobs if _looks_like_internship(j.title) or _looks_like_internship(j.details or "")]
    # final dedupe by URL
    uniq: Dict[str, Job] = {}
    for j in jobs:
        uniq[j.url] = j
    return list(uniq.values())
