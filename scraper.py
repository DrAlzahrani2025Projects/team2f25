from __future__ import annotations
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse
from datetime import datetime
from pathlib import Path
import os, re, html, time
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ---------- constants ----------
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

MAX_PAGES  = int(os.getenv("MAX_PAGES", "30"))
TIMEOUT_MS = int(os.getenv("TIMEOUT_MS", "10000"))

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

DEEP_SCRAPE_HOSTS = {
    "myworkdayjobs.com", "workday.com", "greenhouse.io", "lever.co", 
    "taleo.net", "icims.com", "smartrecruiters.com", "careers.microsoft.com",
    "jobs.merck.com", "oracle.com", "pfizer.com", "jobs.apple.com",
    "careers.google.com", "nasa.gov", "edwards.com", "lanl.gov",
    "jobs.us.pwc.com", "ey.com", "deloitte.com", "goldmansachs.com",
    "gecareers.com", "disneycareers.com", "jpmorgan.com", "boeing.com",
    "lockheedmartinjobs.com", "northropgrumman.com", "virgingalactic.com"
}

# ---------- small helpers ----------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _infer_company(abs_url: str) -> Optional[str]:
    try:
        host = urlparse(abs_url).netloc.lower()
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

def _is_job_listing_page(html: str, url: str) -> bool:
    """
    Check if page contains actual searchable job postings.
    Must have multiple job result indicators + search/filter UI.
    """
    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text().lower()
    
    # MUST have search results or job board indicators
    has_search_results = any(phrase in page_text for phrase in [
        "search results", "job results", "showing", "results found", "matching jobs",
        "open positions", "current openings", "apply now"
    ])
    
    if not has_search_results:
        return False
    
    # Count actual job postings (stricter criteria)
    job_count = 0
    
    # Look for job containers with multiple required fields
    job_containers = soup.find_all(["article", "li", "div", "tr"], class_=re.compile(r"job.*(?:posting|card|item|result|listing)", re.I))
    job_count += len(job_containers)
    
    # Look for location + title combinations (sign of actual postings)
    locations = soup.find_all(class_=re.compile(r"location|city|region", re.I))
    if locations:
        job_count += len(locations)
    
    # Look for application/apply buttons (sign of real postings)
    apply_buttons = soup.find_all(["button", "a"], string=re.compile(r"apply|apply now|view|details", re.I))
    job_count += len(apply_buttons) * 0.3
    
    # Need at least 5 job indicators
    return job_count >= 5

def _find_job_search_page(page, base_url: str, timeout_ms: int = 15000, max_depth: int = 2) -> Optional[str]:
    """
    Navigate through pages to find actual job listings.
    Handles landing pages, job boards, and search pages.
    """
    visited = {base_url}
    current_url = base_url
    depth = 0
    
    while depth < max_depth:
        try:
            page.goto(current_url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=1500)
            except Exception:
                pass
            
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.3)
            
            html = page.content()
            
            # Check if this is a job listing page
            if _is_job_listing_page(html, current_url):
                print(f"Found job listing page at depth {depth}: {current_url}")
                return current_url
            
            # Try to find next page link
            soup = BeautifulSoup(html, "lxml")
            
            # Priority: look for "jobs", "search", "browse", "positions" buttons/links
            next_link = None
            priority_keywords = ["search jobs", "browse jobs", "all jobs", "view jobs", "open positions", "current openings", "internships"]
            
            for a in soup.find_all("a", href=True):
                text = _clean(a.get_text(" ", strip=True)).lower()
                href = a.get("href", "").strip()
                
                if not text or not href:
                    continue
                
                # Skip bad destinations
                if any(bad in text for bad in ["contact", "help", "download", "pdf"]):
                    continue
                if any(bad in href.lower() for bad in ["youtube", "instagram", "facebook", "linkedin.com"]):
                    continue
                
                abs_url = urljoin(current_url, href)
                
                # Only same-domain
                if urlparse(abs_url).netloc.lower() != urlparse(base_url).netloc.lower():
                    continue
                
                if abs_url in visited:
                    continue
                
                # Prioritize by keyword match
                if any(kw in text for kw in priority_keywords):
                    next_link = abs_url
                    break
                
                # Look for URL patterns
                if any(pattern in abs_url.lower() for pattern in ["/job", "/position", "/opening", "/career", "/listing"]):
                    if not next_link:
                        next_link = abs_url
            
            if next_link:
                print(f"Following link from {current_url} to {next_link}")
                visited.add(next_link)
                current_url = next_link
                depth += 1
            else:
                print(f"No next link found at {current_url}, stopping")
                break
        
        except Exception as e:
            print(f"Error in navigation: {e}")
            break
    
    return current_url

def _extract_job_postings(html: str, url: str) -> List[Dict]:
    """Extract individual job postings from a job listing page."""
    rows = []
    soup = BeautifulSoup(html, "lxml")
    seen_links = set()
    
    # Strategy 1: Look for job posting containers
    job_containers = soup.find_all(["article", "li", "div"], class_=re.compile(r"job.*posting|posting|job.*card|job.*item|job.*result", re.I))
    
    if not job_containers:
        # Strategy 2: Look for all links that look like job postings
        all_links = soup.find_all("a", href=True)
        
        for link_elem in all_links:
            title = _clean(link_elem.get_text(strip=True))
            href = link_elem.get("href", "").strip()
            
            if not title or len(title) < 3 or not href:
                continue
            
            # Filter out nav links
            if any(skip in title.lower() for skip in ["sign in", "log in", "back", "next", "prev", "home"]):
                continue
            
            # Skip very short titles
            if len(title) < 8 and title.lower() in ["apply", "view more", "details"]:
                continue
            
            abs_url = urljoin(url, href)
            
            if abs_url in seen_links:
                continue
            
            # Only same domain
            if urlparse(abs_url).netloc.lower() != urlparse(url).netloc.lower():
                continue
            
            # Check if looks like a job posting
            title_lower = title.lower()
            if not any(role in title_lower for role in ["intern", "developer", "engineer", "analyst", "associate", "specialist", "manager"]):
                continue
            
            seen_links.add(abs_url)
            
            # Extract location from parent
            location = None
            parent = link_elem.parent
            for _ in range(3):  # Check up to 3 levels up
                if parent:
                    loc_elem = parent.find(class_=re.compile(r"location|city", re.I))
                    if loc_elem:
                        location = _clean(loc_elem.get_text(strip=True))
                        break
                    parent = parent.parent
            
            parent_text = link_elem.parent.get_text(strip=True)[:2000] if link_elem.parent else title
            
            rows.append({
                "title": title,
                "company": _infer_company(url),
                "location": location,
                "posted_date": datetime.utcnow().date().isoformat(),
                "tags": None,
                "link": abs_url,
                "host": urlparse(url).netloc.lower(),
                "source": url,
                "deadline": None,
                "requirements": None,
                "salary": None,
                "education": None,
                "remote": None,
                "details": parent_text,
            })
        
        return rows
    
    # If containers found, extract from them
    for container in job_containers[:50]:
        title_elem = container.find(["a", "h2", "h3", "h4", "span"])
        if not title_elem:
            continue
        
        title = _clean(title_elem.get_text(strip=True))
        if not title or len(title) < 3:
            continue
        
        link_elem = container.find("a", href=True)
        if not link_elem:
            continue
        
        abs_url = urljoin(url, link_elem["href"])
        
        if abs_url in seen_links:
            continue
        
        seen_links.add(abs_url)
        
        location = None
        loc_elem = container.find(class_=re.compile(r"location|city", re.I))
        if loc_elem:
            location = _clean(loc_elem.get_text(strip=True))
        
        rows.append({
            "title": title,
            "company": _infer_company(url),
            "location": location,
            "posted_date": datetime.utcnow().date().isoformat(),
            "tags": None,
            "link": abs_url,
            "host": urlparse(url).netloc.lower(),
            "source": url,
            "deadline": None,
            "requirements": None,
            "salary": None,
            "education": None,
            "remote": None,
            "details": container.get_text(strip=True)[:2000],
        })
    
    print(f"Extracted {len(rows)} job postings from {url}")
    return rows

def _scrape_career_portal(url: str, timeout_ms: int = 20000) -> List[Dict]:
    """
    Scrape a career portal with multi-level navigation.
    Navigates to actual job listing pages, then extracts postings.
    """
    rows = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
            ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})
            
            def _should_block(u: str, rtype: str) -> bool:
                if rtype in {"image", "media", "font"}:
                    return True
                return any(b in u for b in ["analytics", "doubleclick", "tracking"])
            ctx.route("**/*", lambda route, req:
                route.abort() if _should_block(req.url, req.resource_type) else route.continue_()
            )
            
            page = ctx.new_page()
            
            # Navigate to job listing page
            jobs_page_url = _find_job_search_page(page, url, timeout_ms)
            
            # Extract job postings
            html = page.content()
            rows = _extract_job_postings(html, jobs_page_url or url)
            
            browser.close()
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    
    return rows

# ---------- MAIN ----------
def scrape_csusb_listings(
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS,
    deep: bool = True,
    max_pages: int = MAX_PAGES,
) -> pd.DataFrame:
    """
    Scrapes CSUSB page and performs deep multi-level scraping of career portals.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})

        def _should_block(u: str, rtype: str) -> bool:
            if rtype in {"image", "media", "font", "stylesheet"}:
                return True
            return any(b in u for b in ["analytics", "doubleclick", "tracking", "facebook"])
        ctx.route("**/*", lambda route, req:
            route.abort() if _should_block(req.url, req.resource_type) else route.continue_()
        )

        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=1500)
        except Exception:
            pass

        html_src = page.content()
        browser.close()

    rows = _collect_links(html_src, base=url)
    
    # Deep scrape career portals
    if deep:
        career_links = [
            r["link"] for r in rows 
            if any(host in r.get("host", "") for host in DEEP_SCRAPE_HOSTS)
        ]
        
        career_links = list(set(career_links))[:max_pages]
        print(f"Found {len(career_links)} career portals to scrape")
        
        # added truncation for demo
        career_links = career_links[:10]
        for i, career_url in enumerate(career_links, 1):
            print(f"\n[{i}/{len(career_links)}] Scraping: {career_url}")
            career_rows = _scrape_career_portal(career_url, timeout_ms)
            rows.extend(career_rows)
            time.sleep(0.5)

    cols = [
        "title","company","location","posted_date","tags","link","host","source",
        "deadline","requirements","salary","education","remote","details"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates(subset=["link"], keep="first")
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

# ---------- Fallback ----------
def quick_company_links_playwright(
    company_token: str,
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS
) -> pd.DataFrame:
    """
    Return only anchors on the CSUSB page whose text or host contains company_token.
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