"""
playwright_fetcher.py
Handles fetching and parsing HTML from dynamic JavaScript-heavy websites using Playwright.
- Async class API for FastAPI backend hops (uses async_playwright)
- Synchronous helper fetch_page_html(...) for tool-calling code (uses sync_playwright)
"""
from __future__ import annotations

import asyncio
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"


class PlaywrightFetcher:
    """
    Async browser fetcher that handles JavaScript-heavy websites.
    Automatically waits for content to load before extracting.
    """

    def __init__(self, timeout_ms: int = 15000, wait_ms: int = 2000):
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self.visited_urls: set[str] = set()

    async def fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML from URL using Playwright (async). Handles JavaScript rendering.
        Returns None if fetch fails or if URL was already visited.
        """
        if url in self.visited_urls:
            print(f"  ⚠️ URL already visited: {url}")
            return None

        self.visited_urls.add(url)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                context = await browser.new_context(user_agent=UA)
                page = await context.new_page()
                try:
                    print(f"  📄 Fetching: {url}")
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    await page.wait_for_timeout(self.wait_ms)  # allow JS to render

                    # Best-effort wait for common job-listing selectors (non-fatal)
                    try:
                        await page.wait_for_selector(
                            "[class*='job'], [class*='position'], [class*='listing'], a[href*='job'], a[href*='career']",
                            timeout=3000,
                        )
                    except Exception:
                        pass

                    html = await page.content()
                    print(f"  ✅ Fetched successfully ({len(html)} bytes)")
                    return html
                finally:
                    await context.close()
                    await browser.close()
        except Exception as e:
            print(f"  ❌ Error fetching {url}: {str(e)[:200]}")
            return None

    def extract_text_and_links(self, html: str, base_url: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Extract readable text and links from HTML.
        Returns: (text_preview, links)
        where links is a list of dicts: {"text": str, "url": str, "domain": str}
        """
        try:
            soup = BeautifulSoup(html or "", "lxml")

            # Remove non-content
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            # Main text
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())[:3000]  # cap preview to 3k chars
            print(f"  📝 Extracted text length: {len(text)} chars")

            # Extract links (aggressive but filtered)
            links: List[Dict[str, str]] = []
            seen_urls: set[str] = set()

            a_tags = soup.find_all("a", href=True)
            print(f"  🔗 Found {len(a_tags)} total <a> tags")

            skip_patterns = [
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "youtube.com",
                "linkedin.com/company",
                "mailto:",
                "tel:",
            ]

            for a in a_tags:
                try:
                    href = (a.get("href") or "").strip()
                    if not href or href.startswith("#") or href.lower().startswith("javascript:"):
                        continue

                    # Link text or fallback to aria-label/title
                    txt = a.get_text(strip=True)
                    if not txt:
                        txt = a.get("aria-label", "") or a.get("title", "") or "Link"
                    txt = txt[:100]

                    # Absolute URL
                    try:
                        absolute_url = urljoin(base_url, href)
                    except Exception:
                        continue

                    if not absolute_url.startswith(("http://", "https://")):
                        continue

                    lower_url = absolute_url.lower()
                    if any(sp in lower_url for sp in skip_patterns):
                        continue

                    if absolute_url in seen_urls:
                        continue
                    seen_urls.add(absolute_url)

                    links.append(
                        {
                            "text": txt,
                            "url": absolute_url,
                            "domain": urlparse(absolute_url).netloc,
                        }
                    )
                except Exception:
                    # Skip malformed anchors
                    continue

            print(f"  ✅ Extracted {len(links)} valid links after filtering")
            if links:
                print("  📋 Sample links:")
                for i, lk in enumerate(links[:5], 1):
                    print(f"    {i}. {lk['text'][:50]} -> {lk['url'][:80]}")
            else:
                print("  ⚠️ WARNING: No links extracted!")

            return text, links

        except Exception as e:
            print(f"  ❌ Error parsing HTML: {e}")
            import traceback

            traceback.print_exc()
            return "", []


# -----------------------------------------------------------------------------
# Synchronous helper for tool-calling code (used by llm_orchestrator.py)
# -----------------------------------------------------------------------------
def fetch_page_html(url: str, timeout_ms: int = 15000, wait_ms: int = 2000) -> Optional[str]:
    """
    Synchronous HTML fetcher using sync_playwright.
    Safe to call from environments where an event loop may already be running.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            context = browser.new_context(user_agent=UA)
            page = context.new_page()
            try:
                print(f"  📄 (sync) Fetching: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(wait_ms)

                # Best-effort wait for common job-listing selectors (non-fatal)
                try:
                    page.wait_for_selector(
                        "[class*='job'], [class*='position'], [class*='listing'], a[href*='job'], a[href*='career']",
                        timeout=3000,
                    )
                except Exception:
                    pass

                html = page.content()
                print(f"  ✅ (sync) Fetched successfully ({len(html)} bytes)")
                return html
            finally:
                context.close()
                browser.close()
    except Exception as e:
        print(f"  ❌ (sync) Error fetching {url}: {str(e)[:200]}")
        return None
