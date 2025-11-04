# llm_orchestrator.py
import json
from typing import List, Dict, Any
from llm_provider import get_provider, get_model, get_openai_client

# ⬇️ Import your existing utilities. If names differ, adjust the imports.
from playwright_fetcher import fetch_page_html as get_page_html    # must return full HTML for a URL
from scraper import extract_links                 # must return [{"text":..., "href":...}, ...]

# ---- Tool implementations your model can call ----
def _tool_fetch_html(url: str) -> Dict[str, Any]:
    html = get_page_html(url)
    return {"url": url, "html": html}

def _tool_find_links(html: str, must_contain: str = None,
                     exclude_domains: List[str] | None = None) -> Dict[str, Any]:
    exclude_domains = exclude_domains or []
    links = extract_links(html)  # [{"text":..., "href":...}]
    out = []
    for lk in links:
        href = (lk.get("href") or "").strip()
        if not href:
            continue
        if any(dom in href for dom in exclude_domains):
            continue
        if must_contain and must_contain not in href:
            continue
        out.append({"text": lk.get("text", ""), "href": href})
    return {"links": out}

# ---- Expose tools to the model ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_html",
            "description": "Fetch raw HTML of a URL using Playwright (handles JS rendering).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
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
                    "html": {"type": "string"},
                    "must_contain": {"type": "string"},
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["html"],
            },
        },
    },
]

# ---- Orchestrator ----
def run_task(task: str) -> Dict[str, Any]:
    """
    Give the model a task like:
    - 'From https://www.csusb.edu/cse/internships-careers collect internal internship links,
       then follow KPMG link and return direct Apply URLs for software internships only.
       Return JSON with fields: csusb_internal, kpmg_apply_links.'
    """
    if get_provider() != "openai":
        raise RuntimeError("Set LLM_PROVIDER=openai to use ChatGPT for this orchestrator.")

    client = get_openai_client()
    model = get_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise research assistant. "
                "Plan steps, call tools when needed, then return a final JSON object. "
                "Always end with valid JSON only, no extra prose. "
                "Prefer internal links when the user says 'internal'."
            ),
        },
        {"role": "user", "content": task},
    ]

    # multi-step tool loop
    for _ in range(8):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message

        # If the model wants to call a tool
        if msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                if name == "fetch_html":
                    result = _tool_fetch_html(**args)
                elif name == "find_links":
                    result = _tool_find_links(**args)
                else:
                    result = {"error": f"unknown tool {name}"}

                # return tool result to the model
                messages.append(msg)  # assistant step with tool call
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
            continue

        # No tool call -> final answer
        final_text = msg.content or "{}"
        try:
            return json.loads(final_text)
        except Exception:
            # if the model didn’t return JSON, wrap it
            return {"raw": final_text}

    return {"error": "tool_loop_exceeded"}
