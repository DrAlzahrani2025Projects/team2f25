from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Iterable
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright
import asyncio
import os
import re
import time
import json
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------- constants ----------
CSUSB_CSE_URL = "https://www.csusb.edu/cse/internships-careers"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))
TIMEOUT_MS = int(os.getenv("TIMEOUT_MS", "15000"))
DEEP_TIMEOUT_MS = int(os.getenv("DEEP_TIMEOUT_MS", "20000"))

JUNK_HOSTS = {"youtube.com", "youtu.be", "facebook.com", "twitter.com", "linkedin.com"}
JUNK_KEYWORDS = {
    "proposal form", "evaluation form", "student evaluation", "supervisor evaluation",
    "report form", "handbook", "resume", "cv", "smartscholarship", "scholarship",
    "faculty & staff", "career center", "advising", "contact us", "about us"
}
ALLOW_HOST_HINTS = {
    "myworkdayjobs", "workday", "greenhouse", "lever", "taleo", "icims", "smartrecruiters",
    "jobs", "careers", "career", "intern"
}

INTERNSHIP_INDICATORS = [
    "intern", "internship", "co-op", "coop", "summer program",
    "student position", "campus hire", "university", "graduate program",
    "summer analyst", "early insight", "industrial placement", "apprentice", "apprenticeship"
]
INTERNSHIP_TERMS = re.compile(
    r"\b(intern|internship|co-?op|summer\s+analyst|early\s+insight|"
    r"industrial\s+placement|apprentice(ship)?|placement)\b",
    re.I
)

ATS_HINTS = [
    "workdayjobs.com", "myworkdayjobs.com", "greenhouse.io", "lever.co",
    "eightfold.ai", "taleo.net", "successfactors", "smartrecruiters.com",
    "avature.net", "icims.com", "brassring.com"
]

# ---------- LLM Integration ----------
def _extract_with_llm(html_snippet: str, url: str) -> Optional[Dict]:
    """Use Ollama LLM to extract structured internship data from HTML"""
    try:
        from langchain_ollama import ChatOllama
        from langchain.prompts import ChatPromptTemplate

        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            model=os.getenv("MODEL_NAME", "qwen2.5:0.5b"),
            temperature=0.05,
            streaming=False,
            model_kwargs={"num_ctx": 3072, "num_predict": 350},
        )

        system_prompt = """You are a data extraction assistant specializing in job postings. Extract internship information from HTML content.
Return ONLY valid JSON with these fields (omit null/empty fields):
{{{{
  "title": "exact job title as shown",
  "company": "company name",
  "location": "city, state OR remote",
  "posted_date": "YYYY-MM-DD or relative like '2 days ago'",
  "deadline": "application deadline if mentioned",
  "requirements": "key requirements (under 100 chars)",
  "salary": "salary/stipend range if mentioned",
  "education": "Bachelor's, Master's, etc.",
  "remote": "Remote, Hybrid, or On-site",
  "details": "brief 1-2 sentence description"
}}}}

If no clear internship data found, return {{{{}}}}
Be precise. Extract only what's explicitly shown."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "HTML:\n{html}\n\nURL: {url}\n\nExtract internship data:")
        ])

        truncated = html_snippet[:2200]
        response = (prompt | llm).invoke({"html": truncated, "url": url})

        content = response.content.strip()
        json_match = re.search(r"\{[\s\S]*?\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            if data and isinstance(data, dict):
                if data.get("title") and len(str(data.get("title")).strip()) > 5:
                    for key in list(data.keys()):
                        if data[key] in [None, "", "null", "None"]:
                            del data[key]
                        elif isinstance(data[key], str):
                            data[key] = data[key].strip()
                    return data
                    
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    
    return None

# ---------- helpers ----------
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc
    except Exception:
        return ""

def _normalize(u: str, base: str) -> str:
    try:
        return urljoin(base, u)
    except Exception:
        return u

def _infer_company(abs_url: str, text: str = "") -> Optional[str]:
    """Infer company name from URL or link text"""
    try:
        host = urlparse(abs_url).netloc.lower().replace("www.", "")
        parts = host.split(".")

        if "myworkdayjobs" in host and len(parts) > 2:
            return parts[0].capitalize()

        if len(parts) >= 2:
            core = parts[-2]
            core = re.sub(r"(jobs?|careers?|hire|recruiting)$", "", core)
            if len(core) > 2:
                return core.capitalize()

        if text:
            match = re.search(r"^([A-Z][a-zA-Z\s&\.]+?)(?:\s*[-—–]\s*(?:Careers?|Jobs?|Internships?))?$", text)
            if match:
                return match.group(1).strip()
    except Exception:
        pass
    return None

async def _deep_scrape_many(urls: list[str], concurrency: int = 8) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async with async_playwright() as pw:
        async def bound(u):
            async with sem:
                return await _deep_scrape_one_async(pw, u)
        tasks = [asyncio.create_task(bound(u)) for u in urls]
        out = await asyncio.gather(*tasks, return_exceptions=True)
    rows = []
    for r in out:
        if isinstance(r, list): rows.extend(r)
    return rows

def deep_scrape_concurrent(company_urls: list[str], concurrency: int = 8) -> pd.DataFrame:
    if not company_urls: return pd.DataFrame()
    rows = asyncio.run(_deep_scrape_many(company_urls, concurrency=concurrency))
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["link"])
    cols = ["title","company","location","posted_date","tags","link","host","source",
            "deadline","requirements","salary","education","remote","details"]
    for c in cols:
        if c not in df.columns: df[c] = None
    return df[cols]

def _is_candidate_link(text: str, url: str) -> bool:
    """Determine if a link is likely an internship/career posting"""
    low = f"{text} {url}".lower()

    if any(k in low for k in JUNK_KEYWORDS):
        return False

    host = urlparse(url).netloc.lower()
    if any(h in host for h in JUNK_HOSTS):
        return False

    if any(ind in low for ind in INTERNSHIP_INDICATORS):
        return True

    return any(h in host or h in low for h in ALLOW_HOST_HINTS)

def _collect_links(page_html: str, base: str) -> List[Dict]:
    """Collect internship candidate links from a page"""
    soup = BeautifulSoup(page_html, "lxml")
    main = soup.find("main") or soup
    rows, seen = [], set()

    for a in main.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        if not text or len(text) < 3:
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
            "company": _infer_company(abs_url, text),
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

# ---------- Deep-scrape a single page ----------
def _deep_scrape_page(url: str, timeout_ms: int = DEEP_TIMEOUT_MS) -> List[Dict]:
    """Deep scrape a single company career page to find internship postings."""
    results: List[Dict] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            ctx = browser.new_context(user_agent=UA, viewport={"width": 1920, "height": 1080})

            def _should_block(u: str, rtype: str) -> bool:
                if rtype in {"image", "media", "font"}:
                    return True
                return any(b in u for b in ["analytics", "doubleclick", "tracking", "facebook", "twitter"])

            ctx.route("**/*", lambda route, req:
                route.abort() if _should_block(req.url, req.resource_type) else route.continue_()
            )

            page = ctx.new_page()

            try:
                print(f"    Loading: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                time.sleep(2)

                # Try to apply internship filters
                filter_found = False
                filter_selectors = [
                    "button:has-text('Intern')", "a:has-text('Intern')",
                    "button:has-text('Student')", "a:has-text('Student')",
                    "[data-automation-id*='intern' i]",
                    "[aria-label*='intern' i]",
                    "input[value*='intern' i]",
                    ".filter:has-text('Intern')",
                    "[class*='intern' i]",
                    "input[type='checkbox'][value*='intern' i]",
                    "input[type='radio'][value*='intern' i]",
                ]
                
                for selector in filter_selectors:
                    try:
                        elements = page.query_selector_all(selector)
                        for elem in elements[:3]:
                            try:
                                elem.click(timeout=2000)
                                time.sleep(1.5)
                                filter_found = True
                                print(f"    ✓ Applied filter: {selector}")
                                break
                            except Exception:
                                continue
                        if filter_found:
                            break
                    except Exception:
                        continue

                # Try URL parameter filtering
                if not filter_found and "?" not in url:
                    filter_params = [
                        "?category=intern",
                        "?type=internship",
                        "?jobType=intern",
                        "?search=intern",
                    ]
                    for param in filter_params:
                        try:
                            test_url = url + param
                            page.goto(test_url, wait_until="domcontentloaded", timeout=timeout_ms)
                            time.sleep(1)
                            print(f"    ✓ Tried URL filter: {param}")
                            break
                        except Exception:
                            continue

                # Wait for job listings
                wait_selectors = [
                    "a[href*='job']", "a[href*='position']", ".job-card",
                    "[class*='job']", "[data-automation-id*='job']",
                    ".listing", ".opportunity"
                ]
                for selector in wait_selectors:
                    try:
                        page.wait_for_selector(selector, timeout=3000)
                        break
                    except Exception:
                        continue

                time.sleep(1)
                html_content = page.content()
                soup = BeautifulSoup(html_content, "lxml")

                # Find job listing containers
                job_elements = []
                
                # Structured containers
                container_selectors = [
                    {"class": re.compile(r"job.*card|position.*card|listing.*item|opportunity|opening", re.I)},
                    {"class": re.compile(r"search.*result|career.*item|role.*card|posting", re.I)},
                    {"data-automation-id": re.compile(r"joblist|searchresult|listing|job", re.I)},
                    {"id": re.compile(r"job.*list|position.*list|career.*list", re.I)},
                    {"class": re.compile(r"result|item", re.I)},
                ]
                
                for selector in container_selectors:
                    found = soup.find_all(["div", "li", "article", "section", "tr"], selector, limit=100)
                    for elem in found:
                        text = elem.get_text(" ", strip=True).lower()
                        has_link = elem.find("a", href=True) is not None
                        if has_link and (any(kw in text for kw in INTERNSHIP_INDICATORS) or len(text) > 30):
                            job_elements.append(elem)

                # Table rows
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")[1:]
                    for row in rows[:100]:
                        text = row.get_text(" ", strip=True).lower()
                        has_link = row.find("a", href=True) is not None
                        if has_link and any(kw in text for kw in INTERNSHIP_INDICATORS):
                            job_elements.append(row)

                # Direct links with internship keywords
                all_links = soup.find_all("a", href=True, limit=200)
                for link in all_links:
                    text = link.get_text(" ", strip=True).lower()
                    href = link.get("href", "").lower()
                    parent_text = ""
                    
                    parent = link.parent
                    if parent:
                        parent_text = parent.get_text(" ", strip=True).lower()
                    
                    if any(kw in text or kw in href or kw in parent_text for kw in INTERNSHIP_INDICATORS):
                        if parent and len(parent_text) > len(text) and len(parent_text) < 500:
                            job_elements.append(parent)
                        else:
                            job_elements.append(link)

                # Deduplicate
                seen_elements = set()
                unique_elements = []
                for elem in job_elements:
                    elem_id = id(elem)
                    if elem_id not in seen_elements:
                        seen_elements.add(elem_id)
                        unique_elements.append(elem)
                
                job_elements = unique_elements

                print(f"    Found {len(job_elements)} potential job elements")

                # Process elements
                processed_urls = set()
                company_name = _infer_company(url)
                successful_extractions = 0
                
                for idx, element in enumerate(job_elements[:50], 1):
                    try:
                        elem_text = element.get_text(" ", strip=True)
                        elem_text_lower = elem_text.lower()

                        has_intern_mention = any(ind in elem_text_lower for ind in INTERNSHIP_INDICATORS)
                        has_intern_in_href = False
                        
                        link = None
                        if element.name == "a":
                            link = element.get("href")
                            if link:
                                has_intern_in_href = any(ind in link.lower() for ind in INTERNSHIP_INDICATORS)
                        else:
                            link_elem = element.find("a", href=True)
                            if link_elem:
                                link = link_elem.get("href")
                                if link:
                                    has_intern_in_href = any(ind in link.lower() for ind in INTERNSHIP_INDICATORS)

                        is_internship = has_intern_mention or has_intern_in_href

                        if not link:
                            if is_internship and len(elem_text) > 20:
                                title = elem_text[:150] if len(elem_text) > 150 else elem_text
                                results.append({
                                    "title": _clean(title),
                                    "company": company_name,
                                    "location": None,
                                    "link": url,
                                    "posted_date": datetime.utcnow().date().isoformat(),
                                    "salary": None,
                                    "education": None,
                                    "remote": None,
                                    "details": elem_text[:300],
                                    "requirements": None,
                                    "deadline": None,
                                    "host": urlparse(url).netloc,
                                    "source": url,
                                    "tags": None,
                                })
                                successful_extractions += 1
                                print(f"      [{successful_extractions}] No link: {title[:60]}...")
                            continue

                        abs_link = urljoin(url, link)

                        if abs_link in processed_urls:
                            continue
                        processed_urls.add(abs_link)

                        if not is_internship:
                            continue

                        # Extract title
                        title = None
                        
                        title_tags = element.find_all(["h1", "h2", "h3", "h4", "h5", "span", "a", "div"], limit=15)
                        for tag in title_tags:
                            text = tag.get_text(" ", strip=True)
                            if text and 5 < len(text) < 200:
                                if not title or any(kw in text.lower() for kw in ["intern", "co-op", "student", "graduate", "entry"]):
                                    title = text
                                    if any(kw in text.lower() for kw in ["intern", "co-op"]):
                                        break
                        
                        if not title:
                            for tag in title_tags:
                                title_attr = tag.get("aria-label") or tag.get("title")
                                if title_attr and len(title_attr) > 5:
                                    title = title_attr
                                    break
                        
                        if not title:
                            title = elem_text[:100] if elem_text else "Internship Position"
                        
                        # Filter junk titles
                        title_lower = title.lower()
                        junk_titles = {
                            "apply now", "learn more", "read more", "click here", "explore",
                            "english", "español", "français", "português", "日本語", "繁體中文",
                            "opens in new", "sign in", "log in", "search", "filter",
                            "linkedin", "facebook", "twitter", "instagram",
                            "current students", "university partners", "leadership",
                            "academic calendar", "email", "phone", "contact",
                        }
                        
                        if (len(title) < 5 or 
                            any(junk in title_lower for junk in junk_titles) or
                            title_lower in junk_titles or
                            re.match(r'^\d+$', title) or
                            re.match(r'^\w{2,3}$', title)):
                            continue
                        
                        # Extract location
                        location = None
                        location_patterns = [
                            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2})\b",
                            r"\b(Remote|Hybrid|On-?site|Virtual)\b",
                            r"\b([A-Z][a-z]+,\s*[A-Z][a-z]+)\b",
                            r"\b(United States|USA|US|UK|Canada|India)\b",
                        ]
                        for pattern in location_patterns:
                            match = re.search(pattern, elem_text, re.IGNORECASE)
                            if match:
                                location = match.group(0)
                                break

                        # Extract salary
                        salary = None
                        salary_pattern = r"\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*-\s*\$?\d+(?:,\d{3})*(?:\.\d{2})?)?(?:/(?:hour|hr|year|yr|month|mo))?"
                        salary_match = re.search(salary_pattern, elem_text)
                        if salary_match:
                            salary = salary_match.group(0)

                        # Remote info
                        remote_info = None
                        if re.search(r"\b(remote|work from home|wfh)\b", elem_text_lower):
                            remote_info = "Remote"
                        elif re.search(r"\bhybrid\b", elem_text_lower):
                            remote_info = "Hybrid"
                        elif re.search(r"\b(on-?site|office)\b", elem_text_lower):
                            remote_info = "On-site"

                        base_result = {
                            "title": _clean(title),
                            "company": company_name,
                            "location": location,
                            "link": abs_link,
                            "posted_date": datetime.utcnow().date().isoformat(),
                            "salary": salary,
                            "education": None,
                            "remote": remote_info,
                            "details": elem_text[:300] if elem_text else "",
                            "requirements": None,
                            "deadline": None,
                            "host": urlparse(abs_link).netloc,
                            "source": url,
                            "tags": None,
                        }

                        # Try LLM enhancement
                        if len(elem_text) > 50:
                            try:
                                html_snippet = str(element)[:2500]
                                extracted = _extract_with_llm(html_snippet, abs_link)

                                if extracted and extracted.get("title"):
                                    base_result.update({
                                        "title": extracted.get("title") or base_result["title"],
                                        "location": extracted.get("location") or base_result["location"],
                                        "salary": extracted.get("salary") or base_result["salary"],
                                        "education": extracted.get("education"),
                                        "remote": extracted.get("remote") or base_result["remote"],
                                        "details": extracted.get("details") or base_result["details"],
                                        "requirements": extracted.get("requirements"),
                                        "deadline": extracted.get("deadline"),
                                    })
                            except Exception:
                                pass

                        results.append(base_result)
                        successful_extractions += 1
                        
                        title_preview = base_result["title"][:60] + ("..." if len(base_result["title"]) > 60 else "")
                        print(f"      [{successful_extractions}] {title_preview}")

                    except Exception as e:
                        print(f"    Error processing element {idx}: {str(e)[:50]}")
                        continue

                print(f"    ✓ Extracted {successful_extractions} internships from this page")

            except PlaywrightTimeout:
                print(f"    ✗ Timeout loading {url}")
            except Exception as e:
                print(f"    ✗ Error scraping {url}: {str(e)[:100]}")
            finally:
                browser.close()

    except Exception as e:
        print(f"    ✗ Playwright error for {url}: {str(e)[:100]}")

    return results

# --- NEW: concurrent deep scraping (Playwright async) ---
async def _deep_scrape_one_async(pw, url: str, timeout_ms: int = DEEP_TIMEOUT_MS):
    from bs4 import BeautifulSoup
    results = []
    browser = await pw.chromium.launch(
        headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"]
    )
    ctx = await browser.new_context(user_agent=UA)
    # Block heavy assets
    await ctx.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in {"image", "media", "font", "stylesheet"}
        else route.continue_(),
    )
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(1200)
        # Optional quick internship selectors
        try:
            await page.wait_for_selector(
                "a:has-text('Intern'), [class*='intern' i], [data-automation-id*='intern' i]",
                timeout=2000,
            )
        except Exception:
            pass
        html = await page.content()
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True, limit=200):
            txt = (a.get_text(" ", strip=True) or "").lower()
            href = a["href"].strip()
            full = urljoin(url, href)
            if any(k in (txt + " " + href) for k in ["intern", "internship", "co-op", "student"]):
                title = a.get_text(" ", strip=True)[:140] or "Internship"
                results.append({
                    "title": title,
                    "company": _infer_company(full, title),
                    "location": None,
                    "posted_date": datetime.utcnow().date().isoformat(),
                    "tags": None,
                    "link": full,
                    "host": urlparse(full).netloc,
                    "source": url,
                    "deadline": None,
                    "requirements": None,
                    "salary": None,
                    "education": None,
                    "remote": None,
                    "details": "",
                })
    except Exception:
        pass
    finally:
        await ctx.close()
        await browser.close()
    return results


async def _deep_scrape_many(urls: list[str], concurrency: int = 8) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async with async_playwright() as pw:
        async def bound(u):
            async with sem:
                return await _deep_scrape_one_async(pw, u)
        tasks = [asyncio.create_task(bound(u)) for u in urls]
        out = await asyncio.gather(*tasks, return_exceptions=True)
    rows = []
    for r in out:
        if isinstance(r, list):
            rows.extend(r)
    return rows


def deep_scrape_concurrent(company_urls: list[str], concurrency: int = 8) -> pd.DataFrame:
    if not company_urls:
        return pd.DataFrame()
    rows = asyncio.run(_deep_scrape_many(company_urls, concurrency=concurrency))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["link"])
    cols = [
        "title","company","location","posted_date","tags","link","host","source",
        "deadline","requirements","salary","education","remote","details",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


# ---------- MAIN SCRAPER ----------
def scrape_csusb_listings(
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS,
    deep: bool = True,
    max_pages: int = MAX_PAGES,
) -> pd.DataFrame:
    """Scrape CSUSB CSE internship page and optionally deep-scrape linked career sites."""
    all_results: List[Dict] = []

    print(f"Scraping CSUSB page: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})

        def _should_block(u: str, rtype: str) -> bool:
            if rtype in {"image", "media", "font", "stylesheet"}:
                return True
            return any(b in u for b in ["analytics", "doubleclick", "tracking"])

        ctx.route("**/*", lambda route, req:
            route.abort() if _should_block(req.url, req.resource_type) else route.continue_()
        )

        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        try:
            page.wait_for_load_state("networkidle", timeout=2000)
        except Exception:
            pass

        html_src = page.content()
        browser.close()

    csusb_links = _collect_links(html_src, base=url)
    all_results.extend(csusb_links)
    print(f"Found {len(csusb_links)} career links on CSUSB page")

    if deep and csusb_links:
        company_urls = list({item["link"] for item in csusb_links})
        if len(company_urls) > max_pages:
            company_urls = company_urls[:max_pages]

        # NEW: concurrent pass (fast)
        df_fast = deep_scrape_concurrent(company_urls, concurrency=8)
        if not df_fast.empty:
            all_results.extend(df_fast.to_dict("records"))
        
        print(f"Deep scraping {len(company_urls)} company career pages...")
        print(f"This will take approximately {len(company_urls) * 2} seconds...")

        deep_success_count = 0
        total_internships_found = 0
        
        for idx, company_url in enumerate(company_urls, 1):
            print(f"  [{idx}/{len(company_urls)}] Scraping: {company_url}")
            deep_results = _deep_scrape_page(company_url)

            if deep_results:
                num_found = len(deep_results)
                print(f"    ✓ Found {num_found} internships")
                all_results.extend(deep_results)
                deep_success_count += 1
                total_internships_found += num_found
            else:
                print(f"    ✗ No detailed internships found")

            time.sleep(0.5)

        print(f"\nDeep scraping summary:")
        print(f"  - Scraped: {len(company_urls)} company sites")
        print(f"  - Successful: {deep_success_count}/{len(company_urls)} sites")
        print(f"  - Total internships found: {total_internships_found}")

    cols = [
        "title", "company", "location", "posted_date", "tags", "link", "host", "source",
        "deadline", "requirements", "salary", "education", "remote", "details",
    ]

    df = pd.DataFrame(all_results)

    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df.drop_duplicates(subset=["link"], keep="first")

    try:
        df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
        df = df.sort_values("posted_date", ascending=False)
        df["posted_date"] = df["posted_date"].dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    print(f"\nTotal unique internships collected: {len(df)}")
    return df[cols]

# ---------- Company-specific finder ----------
def quick_company_links_playwright(
    company_token: str,
    url: str = CSUSB_CSE_URL,
    timeout_ms: int = TIMEOUT_MS,
    deep: bool = True
) -> pd.DataFrame:
    """Find and optionally deep-scrape links for a specific company."""
    token = (company_token or "").strip().lower()
    if not token:
        return pd.DataFrame()

    print(f"\n🔍 Searching for {company_token} internships...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        soup = BeautifulSoup(page.content(), "lxml")
        browser.close()

    main = soup.find("main") or soup
    rows, seen = [], set()
    company_urls: List[str] = []

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
            company_urls.append(abs_url)
            rows.append({
                "title": text,
                "company": _infer_company(abs_url, text) or company_token,
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

    if not company_urls:
        print(f"  No links found on CSUSB page, trying direct company URLs...")
        
        company_clean = re.sub(r'[^a-z0-9]', '', token.lower())
        potential_urls = [
            f"https://careers.{company_clean}.com",
            f"https://www.{company_clean}.com/careers",
            f"https://www.{company_clean}.com/careers/students",
            f"https://jobs.{company_clean}.com",
            f"https://{company_clean}.com/careers",
            f"https://careers.{token.replace(' ', '')}.com/students",
            f"https://{token.replace(' ', '')}.wd1.myworkdayjobs.com",
            f"https://{company_clean}.wd5.myworkdayjobs.com",
        ]
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
            ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 720})
            page = ctx.new_page()
            
            for test_url in potential_urls[:5]:
                try:
                    print(f"  Trying: {test_url}")
                    response = page.goto(test_url, wait_until="domcontentloaded", timeout=10000)
                    if response and response.status < 400:
                        print(f"  ✓ Found: {test_url}")
                        company_urls.append(test_url)
                        break
                except Exception:
                    continue
            
            browser.close()

    if not company_urls:
        print(f"  ✗ Could not find career page for {company_token}")
        return pd.DataFrame()

    print(f"  Found {len(company_urls)} career page(s)")

    if deep and company_urls:
        print(f"  Deep scraping {len(company_urls)} page(s)...")
        for idx, company_url in enumerate(company_urls[:5], 1):
            print(f"  [{idx}/{len(company_urls[:5])}] Scraping: {company_url}")
            deep_results = _deep_scrape_page(company_url)
            
            for result in deep_results:
                if isinstance(result, dict):
                    result["company"] = result.get("company") or company_token
                    rows.append(result)

    if not rows:
        print(f"  ✗ No internships found for {company_token}")
        return pd.DataFrame()

    cols = [
        "title", "company", "location", "posted_date", "tags", "link", "host", "source",
        "deadline", "requirements", "salary", "education", "remote", "details",
    ]

    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df.drop_duplicates(subset=["link"], keep="first")
    
    print(f"  ✓ Total internships found: {len(df)}")
    return df[cols]

# ---------- Deep company crawler ----------
@dataclass
class ScrapeResult:
    title: str
    company: str
    location: str
    link: str
    posted_date: Optional[str]
    remote: Optional[str]
    host: str
    source: str
    details: str

def _looks_like_intern_title(txt: str) -> bool:
    return bool(INTERNSHIP_TERMS.search(txt or ""))

def deep_scrape_company_careers(
    company: str,
    seed_links: Optional[Iterable[str]] = None,
    max_pages: int = 60,
    wait_ms: int = 400,
    timeout_ms: int = 20000,
) -> pd.DataFrame:
    """Headless crawl starting from the company's known careers page(s)."""
    company = (company or "").strip()
    if not company:
        return pd.DataFrame()

    seeds = set([*(seed_links or [])])
    try:
        df_seed = quick_company_links_playwright(company, deep=False)
        for u in (df_seed["link"].tolist() if isinstance(df_seed, pd.DataFrame) and "link" in df_seed.columns else []):
            seeds.add(u)
    except Exception:
        pass

    if not seeds:
        guesses = [
            f"https://careers.{company.replace(' ', '')}.com",
            f"https://www.{company.replace(' ', '')}.com/careers",
            f"https://www.{company.replace(' ', '')}.com/careers/students",
            f"https://www.{company.replace(' ', '')}.com/careers/early-careers",
        ]
        seeds.update(guesses)

    results: list[ScrapeResult] = []
    seen_urls: set[str] = set()
    queued: list[str] = [u for u in seeds if u.startswith("http")]
    start_domains = {_domain(u) for u in queued if _domain(u)}

    brand = re.sub(r"[^a-z]", "", company.lower())

    def same_brand(domain: str) -> bool:
        d = (domain or "").lower()
        return bool(brand and (brand in d or d.replace("chase", "") in brand or brand.replace("chase", "") in d))

    def should_visit(u: str) -> bool:
        d = _domain(u)
        if not d:
            return False
        if d in start_domains or same_brand(d):
            return True
        return any(h in d for h in ATS_HINTS)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])
        ctx = browser.new_context(ignore_https_errors=True, user_agent=UA)
        page = ctx.new_page()

        try:
            while queued and len(seen_urls) < max_pages:
                url = queued.pop(0)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                try:
                    page.goto(url, timeout=timeout_ms)

                    actual = page.url
                    ad = _domain(actual)
                    if ad and ad not in start_domains:
                        start_domains.add(ad)

                    page.wait_for_timeout(wait_ms)

                    host = _domain(url)
                    anchors = page.query_selector_all("a")
                    candidates: list[tuple[str, str]] = []

                    for a in anchors:
                        try:
                            txt = (a.inner_text() or "").strip()
                            href = (a.get_attribute("href") or "").strip()
                        except Exception:
                            continue
                        if not href:
                            continue
                        full = _normalize(href, url)
                        if _looks_like_intern_title(txt) or _looks_like_intern_title(full):
                            candidates.append((txt, full))

                        if should_visit(full) and full not in seen_urls and len(queued) + len(seen_urls) < max_pages:
                            queued.append(full)

                    for txt, link in candidates:
                        title = txt or "Internship"
                        results.append(ScrapeResult(
                            title=title,
                            company=company,
                            location="",
                            link=link,
                            posted_date=None,
                            remote=None,
                            host=host,
                            source=url,
                            details="",
                        ))

                except Exception:
                    continue
        finally:
            ctx.close()
            browser.close()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results]).drop_duplicates(subset=["link"])
    cols = ["title", "company", "location", "posted_date", "remote", "host", "link", "source", "details"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]