# llm_orchestrator.py
import json
from typing import List, Dict, Any, Optional, Tuple

# OpenAI provider utilities (expected to wrap the official openai v1 client)
from llm_provider import get_provider, get_model, get_openai_client

# Your existing utilities
# - get_page_html(url) must return the full rendered HTML (Playwright-backed)
# - extract_links(html) must return a list of dicts: [{"text": ..., "href": ...}, ...]
from playwright_fetcher import fetch_page_html as get_page_html
from scraper import extract_links


# ---- Tool implementations your model can call ----
def _tool_fetch_html(url: str) -> Dict[str, Any]:
    """
    Fetch raw (rendered) HTML from a URL. Uses Playwright under the hood.
    """
    html = get_page_html(url)
    return {"url": url, "html": html or ""}


def _tool_find_links(
    html: str,
    must_contain: Optional[str] = None,
    exclude_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Parse links from HTML and optionally filter them.
    - must_contain: only keep links whose href contains this substring
    - exclude_domains: drop links whose href contains any of these substrings
    """
    exclude_domains = exclude_domains or []
    links = extract_links(html or "")  # expected: [{"text":..., "href":...}, ...]

    out: List[Dict[str, str]] = []
    for lk in links:
        href = (lk.get("href") or "").strip()
        if not href:
            continue
        if any(dom for dom in exclude_domains if dom and dom in href):
            continue
        if must_contain and must_contain not in href:
            continue
        out.append({"text": lk.get("text", "").strip(), "href": href})

    return {"links": out}


# ---- Expose tools to the model (OpenAI tools / function calling schema) ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_html",
            "description": "Fetch raw HTML of a URL using Playwright (handles JS rendering).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch"}
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_links",
            "description": "Extract links from given HTML, optionally filtering results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "HTML to parse for links"},
                    "must_contain": {
                        "type": "string",
                        "description": "Keep only links whose href contains this substring",
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Drop links whose href contains any of these substrings",
                    },
                },
                "required": ["html"],
                "additionalProperties": False,
            },
        },
    },
]


def _dispatch_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the appropriate local tool implementation.
    """
    try:
        if name == "fetch_html":
            return _tool_fetch_html(**args)
        if name == "find_links":
            return _tool_find_links(**args)
        return {"error": f"unknown tool {name}"}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}


# ---- Orchestrator ----
def run_task(task: str) -> Dict[str, Any]:
    """
    Give the model a task like:
    - 'From https://www.csusb.edu/cse/internships-careers collect internal internship links,
       then follow KPMG link and return direct Apply URLs for software internships only.
       Return JSON with fields: csusb_internal, kpmg_apply_links.'

    This uses OpenAI tool calling to iteratively fetch pages and extract links,
    then returns a final JSON object (no extra prose).
    """
    if get_provider() != "openai":
        raise RuntimeError("Set LLM_PROVIDER=openai to use ChatGPT for this orchestrator.")

    client = get_openai_client()
    model = get_model()

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a precise research assistant.\n"
                "Plan steps, call tools when needed, then return a final JSON object.\n"
                "Always end with valid JSON only, no extra prose.\n"
                "Prefer internal links when the user says 'internal'."
            ),
        },
        {"role": "user", "content": task},
    ]

    # Multi-step tool loop (bounded to avoid infinite cycles)
    MAX_TOOL_STEPS = 8
    for _ in range(MAX_TOOL_STEPS):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        choice = resp.choices[0]
        msg = choice.message

        # If the model wants to call tools
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Append the assistant tool-call message once
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute each tool, append tool results
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                result = _dispatch_tool(name, args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            # Continue loop to let the model observe tool outputs
            continue

        # No tool calls -> final answer expected (JSON only)
        final_text = msg.content or "{}"
        try:
            return json.loads(final_text)
        except Exception:
            # If the model didn’t return valid JSON, wrap the raw text
            return {"raw": final_text}

    return {"error": "tool_loop_exceeded"}
