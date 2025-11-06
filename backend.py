import streamlit as st
import os, json
import httpx



BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ---------- Backend navigation request ----------
def navigate_career_page(company_url: str, query: str, max_hops: int) -> dict:
    """
    Send navigation request to backend.
    Backend will use LLM to navigate the career page until it finds job listings.

    """
    print(f"\n{'='*60}")
    print(f"Calling backend to navigate: {company_url}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Query: {query}")
    print(f"Max hops: {max_hops}")
    print(f"{'='*60}")
    
    try:
        with httpx.Client(timeout=120.0) as client:
            print(f"Sending POST request to {BACKEND_URL}/navigate")
            
            response = client.post(
                f"{BACKEND_URL}/navigate",
                json={
                    "start_url": company_url,
                    "query": query,
                    "max_hops": max_hops
                }
            )
            
            print(f"Response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            print(f"Response data: {json.dumps(result, indent=2)[:500]}")
            
            return result
            
    except httpx.TimeoutException as e:
        print(f"✗ Timeout error: {e}")
        st.warning(f"Navigation timeout for {company_url}")
        return {
            "success": False,
            "error": f"Timeout: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP error: {e}")
        st.warning(f"Backend error for {company_url}: {e}")
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except httpx.ConnectError as e:
        print(f"✗ Connection error: {e}")
        st.error(f"⚠️ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return {
            "success": False,
            "error": f"Connection failed: {str(e)}",
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
    except Exception as e:
        print(f"✗ Navigation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "visited_urls": [company_url],
            "found_links": [],
            "final_url": company_url
        }
