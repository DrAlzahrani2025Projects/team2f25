"""
playwright_fetcher.py
Handles fetching and parsing HTML from dynamic JavaScript-heavy websites using Playwright.
"""
import asyncio
import time
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"


class PlaywrightFetcher:
    """
    Async browser fetcher that handles JavaScript-heavy websites.
    Automatically waits for content to load before extracting.
    """
    
    def __init__(self, timeout_ms: int = 15000, wait_ms: int = 2000):
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self.visited_urls = set()
    
    async def fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML from URL using Playwright. Handles JavaScript rendering.
        Returns None if fetch fails.
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
                        "--disable-blink-features=AutomationControlled"
                    ]
                )
                
                context = await browser.new_context(user_agent=UA)
                page = await context.new_page()
                
                try:
                    print(f"  📄 Fetching: {url}")
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    await page.wait_for_timeout(self.wait_ms)  # Wait for JS to render
                    
                    # Try to wait for common job listing selectors
                    try:
                        await asyncio.wait_for(
                            page.wait_for_selector(
                                "[class*='job'], [class*='position'], [class*='listing'], a[href*='job'], a[href*='career']",
                                timeout=3000
                            ),
                            timeout=4
                        )
                    except:
                        pass  # Selector not found, continue anyway
                    
                    html = await page.content()
                    print(f"  ✅ Fetched successfully ({len(html)} bytes)")
                    return html
                    
                finally:
                    await context.close()
                    await browser.close()
                    
        except Exception as e:
            print(f"  ❌ Error fetching {url}: {str(e)[:100]}")
            return None
    
    def extract_text_and_links(self, html: str, base_url: str) -> Tuple[str, List[Dict]]:
        """Extract readable text and links from HTML."""
        try:
            soup = BeautifulSoup(html, "lxml")
            
            # Remove script and style
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            # Get main text
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())[:3000]  # Limit to 3000 chars
            
            print(f"  📝 Extracted text length: {len(text)} chars")
            
            # Extract links - be more aggressive
            links = []
            seen_urls = set()
            
            all_links = soup.find_all("a", href=True)
            print(f"  🔗 Found {len(all_links)} total <a> tags")
            
            for link in all_links:
                try:
                    href = link.get("href", "").strip()
                    if not href:
                        continue
                    
                    # Skip anchors and javascript
                    if href.startswith("#") or href.startswith("javascript:"):
                        continue
                    
                    # Get link text
                    text_content = link.get_text(strip=True)
                    if not text_content:
                        # Try aria-label or title
                        text_content = link.get("aria-label", "") or link.get("title", "")
                    
                    text_content = text_content[:100] if text_content else "Link"
                    
                    # Make absolute URL
                    try:
                        absolute_url = urljoin(base_url, href)
                    except Exception:
                        continue
                    
                    # Basic validation
                    if not absolute_url.startswith(('http://', 'https://')):
                        continue
                    
                    # Skip social media and obvious non-job links
                    skip_patterns = [
                        "facebook.com", "twitter.com", "instagram.com", 
                        "youtube.com", "linkedin.com/company",
                        "mailto:", "tel:"
                    ]
                    
                    if any(skip in absolute_url.lower() for skip in skip_patterns):
                        continue
                    
                    # Deduplicate
                    if absolute_url in seen_urls:
                        continue
                    seen_urls.add(absolute_url)
                    
                    links.append({
                        "text": text_content,
                        "url": absolute_url,
                        "domain": urlparse(absolute_url).netloc
                    })
                    
                except Exception as e:
                    # Skip problematic links
                    continue
            
            print(f"  ✅ Extracted {len(links)} valid links after filtering")
            
            # Debug: print first few links
            if links:
                print(f"  📋 Sample links:")
                for i, link in enumerate(links[:5], 1):
                    print(f"    {i}. {link['text'][:50]} -> {link['url'][:80]}")
            else:
                print(f"  ⚠️ WARNING: No links extracted!")
            
            return text, links
            
        except Exception as e:
            print(f"  ❌ Error parsing HTML: {e}")
            import traceback
            traceback.print_exc()
            return "", []


# For sync wrapper if needed
def fetch_html_sync(url: str, timeout_ms: int = 15000, wait_ms: int = 2000) -> Optional[str]:
    """Synchronous wrapper for fetching HTML."""
    try:
        return asyncio.run(_fetch_html_async(url, timeout_ms, wait_ms))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


async def _fetch_html_async(url: str, timeout_ms: int, wait_ms: int) -> Optional[str]:
    """Internal async fetch function."""
    fetcher = PlaywrightFetcher(timeout_ms, wait_ms)
    return await fetcher.fetch_html(url)