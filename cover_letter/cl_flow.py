# cover_letter/cl_flow.py
# LLM-driven, proactive orchestration for the cover-letter conversation.
# The LLM plans each step by emitting compact JSON "actions" (no prose),
# and we execute them inside one continuous Streamlit chat.
#
# Supported actions:
#   {"action":"ask","field":"<full_name|email|phone|city|highlights|extras|role_interest>","question":"..."}
#   {"action":"set","field":"...","value":"..."}
#   {"action":"set_url","url":"https://..."}
#   {"action":"fetch_company","company":"google"}   # app.py will fetch and then call ask_next_question()
#   {"action":"answer","text":"..."}                # answer general cover-letter questions
#   {"action":"generate"}                           # generate the cover letter now
#
# PROACTIVITY:
# - If resume is present and a single result is visible, we can auto-pick its URL.
# - If user gives a company/title while we're asking for a link, we pick from results or queue a fetch.
# - If the user asks questions about cover letters any time, LLM answers (action "answer"), then continues.

from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional
import json
import os
import time
import re
import streamlit as st

from .cl_state import (
    init_cover_state, get_profile, set_profile_field,
    set_target_url, next_unanswered_key
)
from .cl_generator import make_cover_letter
from langchain_ollama import ChatOllama
# ---------------- Utilities ----------------

def _results_preview(df) -> List[Dict[str, str]]:
    """Compact, LLM-friendly view of current results (first 12 rows)."""
    out: List[Dict[str, str]] = []
    try:
        if df is None or df.empty:
            return out
        cols = [c for c in ["title", "company", "link", "url"] if c in df.columns]
        for _, row in df.head(12).iterrows():
            item = {k: str(row.get(k, "")) for k in cols}
            if "url" in item and not item.get("link"):
                item["link"] = item.pop("url")
            out.append(item)
    except Exception:
        pass
    return out

def _llm() -> Optional["ChatOllama"]:
    try:
        from langchain_ollama import ChatOllama
    except Exception:
        return None
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
    try:
        return ChatOllama(base_url=base, model=model, temperature=0.2, streaming=False)
    except Exception:
        return None

def _default_render(role: str, content: str) -> None:
    """Fallback renderer using Streamlit chat bubbles."""
    with st.chat_message(role):
        st.write(content)

# ---------------- Planner (LLM decides next step) ----------------

def _plan_next_step(user_msg: str) -> Dict[str, Any]:
    """
    Ask the LLM to pick a single next action, given the current context.
    If LLM is unavailable or returns bad JSON, use a robust fallback policy.
    """
    profile = get_profile()
    resume_text = st.session_state.get("resume_text", "")
    resume_json = st.session_state.get("resume_json", {})
    target_url = st.session_state.get("cover_target_url", "")
    results = _results_preview(st.session_state.get("last_results_df"))
    collecting = bool(st.session_state.get("collecting_cover_profile"))

    # Short-circuit: if we're mid-collection and missing basics, bias toward those
    def _fallback() -> Dict[str, Any]:
        if not target_url:
            return {"action": "ask", "field": "role_interest",
                    "question": "Please paste the job link, or tell me a company/title to target."}
        for k in ["full_name", "email", "phone", "city"]:
            if not (profile.get(k) or "").strip():
                return {"action": "ask", "field": k, "question": f"Please share your {k.replace('_', ' ')}."}
        return {"action": "generate"}

    planner = _llm()
    if planner is None:
        return _fallback()

    from langchain_core.prompts import ChatPromptTemplate
    sys = (
        "You are a proactive assistant that manages a cover-letter workflow. "
        "You must return ONLY a single compact JSON object with an action (no prose). "
        "Available actions:\n"
        '  {"action":"ask","field":"<full_name|email|phone|city|highlights|extras|role_interest>","question":"..."}\n'
        '  {"action":"set","field":"...","value":"..."}\n'
        '  {"action":"set_url","url":"https://..."}\n'
        '  {"action":"fetch_company","company":"google"}\n'
        '  {"action":"answer","text":"..."}\n'
        '  {"action":"generate"}\n'
        "Policy:\n"
        "- If the user asks a general question related to cover letters (format, tone, length, tips), use action 'answer'.\n"
        "- If no resume is present, ask the user to upload it via sidebar and wait for 'done'.\n"
        "- Prefer to complete missing basics (full_name, email, phone, city) succinctly.\n"
        "- For the target role (role_interest):\n"
        "    * If the user pasted a URL, use set_url.\n"
        "    * Else, if a company/title matches an item in 'results', pick that link.\n"
        "    * Else, return fetch_company with the company/title.\n"
        "- When you have a URL and basic contact info (and any optional highlights/extras), return generate.\n"
        "- Ask one thing at a time. Keep questions short and specific."
    )
    blob = {
        "last_user_msg": user_msg or "",
        "collecting": collecting,
        "profile": profile,
        "resume_present": bool((resume_text or "").strip()),
        "resume_json_keys": list(resume_json.keys()) if isinstance(resume_json, dict) else [],
        "results": results,
        "target_url": target_url,
    }
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys),
        ("human", "{blob}")
    ])
    try:
        out = (prompt | planner).invoke({"blob": json.dumps(blob, ensure_ascii=False)})
        txt = (getattr(out, "content", "") or "").strip()
        m = re.search(r"\{[\s\S]*\}", txt)
        js = json.loads(m.group(0) if m else txt)
        if not isinstance(js, dict):
            raise ValueError("Planner did not return a JSON object")
        if js.get("action") not in {"ask", "set", "set_url", "fetch_company", "answer", "generate"}:
            raise ValueError("Planner returned unknown action")
        return js
    except Exception:
        return _fallback()

# ---------------- Public API used by app.py ----------------

def offer_cover_letter(render: Callable[[str, str], None] = _default_render) -> None:
    init_cover_state()
    if st.session_state.get("want_cover_letter") is None:
        st.session_state["want_cover_letter"] = True
        render("assistant", "Want me to draft a tailored cover letter? I’ll ask a couple of quick questions, use your resume, and generate a download.")

def start_collection(render: Callable[[str, str], None] = _default_render) -> None:
    init_cover_state()
    st.session_state["collecting_cover_profile"] = True

    # Proactive nudge once if resume missing
    if not (st.session_state.get("resume_text") or "").strip() and not st.session_state.get("asked_for_resume"):
        st.session_state["asked_for_resume"] = True
        render("assistant", "Please upload your resume (PDF/DOCX/TXT) in the left sidebar, then say “done”.")
        return

    # If exactly one result is shown, auto-pick its link to speed up
    df = st.session_state.get("last_results_df")
    if df is not None and len(df) == 1:
        url_col = "link" if "link" in df.columns else ("url" if "url" in df.columns else None)
        if url_col:
            set_target_url(str(df.iloc[0][url_col]))

    _drive_once("", render)

def ask_next_question(render: Callable[[str, str], None] = _default_render) -> None:
    """Called after app fetches company results; we continue the LLM plan."""
    _drive_once("", render)

def handle_user_message(message_text: str, render: Callable[[str, str], None] = _default_render) -> bool:
    """
    Handle messages for the cover-letter workflow FIRST.
    Returns True if we consumed the message; False to let the rest of the app handle it.
    """
    init_cover_state()
    msg = (message_text or "").strip()
    low = msg.lower()

    # Allow "select row N" to quickly pick a link
    if low.startswith("select row ") or low.startswith("row "):
        try:
            idx = int(low.split()[-1])
            df = st.session_state.get("last_results_df")
            if df is not None and 0 <= idx < len(df):
                url_col = "link" if "link" in df.columns else ("url" if "url" in df.columns else None)
                if url_col:
                    set_target_url(str(df.iloc[idx][url_col]))
                    st.session_state["collecting_cover_profile"] = True
                    _drive_once("", render)
                    return True
        except Exception:
            pass

    # Accept the initial “yes/ok” to start
    if st.session_state.get("want_cover_letter") and not st.session_state.get("collecting_cover_profile"):
        if any(w in low for w in ["yes", "yep", "sure", "ok", "okay", "please", "start", "begin", "create", "make one", "draft"]):
            st.session_state["collecting_cover_profile"] = True
            _drive_once(msg, render)
            return True
        if low.startswith("http://") or low.startswith("https://"):
            set_target_url(msg)
            st.session_state["collecting_cover_profile"] = True
            _drive_once("", render)
            return True

    # If collecting: treat "done" as resume uploaded; otherwise pass to planner
    if st.session_state.get("collecting_cover_profile"):
        if low in {"done", "uploaded", "i uploaded", "resume uploaded"}:
            if (st.session_state.get("resume_text") or "").strip():
                _drive_once("", render)
            else:
                render("assistant", "I still don’t see a resume. Please upload it in the left sidebar and then say “done”.")
            return True
        _drive_once(msg, render)
        return True

    return False

# ---------------- Driver loop ----------------

def _drive_once(user_msg: str, render: Callable[[str, str], None]) -> None:
    """
    Perform one LLM planning/execution step given the latest user message and current state.
    The step is idempotent and never loops infinitely, preventing UI "greying".
    """
    # If resume is missing and we've already asked, gently remind and pause
    if not (st.session_state.get("resume_text") or "").strip() and st.session_state.get("asked_for_resume"):
        render("assistant", "Once your resume is uploaded in the sidebar, just type “done”.")
        return

    # Determine if we are currently asking for the role/link
    current_key = next_unanswered_key()
    asking_for_link = (current_key == "role_interest")

    # Get the plan for this turn
    step = _plan_next_step(user_msg)

    act = step.get("action")
    if act == "answer":
        # Answer general cover-letter questions, then proceed with the flow
        txt = (step.get("text") or "").strip() or "Here’s what I recommend."
        render("assistant", txt)
        # Immediately plan the next actionable step
        step = _plan_next_step("")

        # Fall through to execute the next step (avoid double return)
        act = step.get("action")

    if act == "ask":
        q = (step.get("question") or "").strip() or "Please share that detail."
        render("assistant", q)
        return

    if act == "set":
        field = (step.get("field") or "").strip()
        value = (step.get("value") or "").strip()
        if field and value:
            set_profile_field(field, value)
        # Immediately plan the next step (prevents “same reply” feeling)
        _drive_once("", render)
        return

    if act == "set_url":
        url = (step.get("url") or "").strip()
        if url:
            set_target_url(url)
        _drive_once("", render)
        return

    if act == "fetch_company":
        company = (step.get("company") or "").strip()
        # Guard against junk: only fetch if we are currently asking for the link AND input is meaningful
        if asking_for_link and len(company) >= 3:
            st.session_state["pending_company_query"] = company
            render("assistant", f"Got it — I’ll pull roles for **{company}** and then continue.")
        else:
            render("assistant", "Please paste the job link or share a company/title (at least 3 characters).")
        return

    if act == "generate":
        _generate_and_show_letter(render)
        return

    # Fallback — ask for the role link if nothing else made sense
    render("assistant", "Please paste the job link, or tell me a company/title to target.")

# ---------------- Finalization ----------------

def _generate_and_show_letter(render: Callable[[str, str], None]) -> None:
    profile: Dict[str, str] = get_profile()
    target_url = profile.get("role_interest") or st.session_state.get("cover_target_url") or ""
    resume_text = st.session_state.get("resume_text", "")

    letter = make_cover_letter(profile=profile, resume_text=resume_text, target_url=target_url)

    record = {
        "ts": int(time.time()),
        "target": target_url,
        "text": letter,
        "profile": dict(profile),
    }
    st.session_state.setdefault("generated_cover_letters", []).append(record)

    render("assistant", "Here’s your tailored cover letter:\n\n" + letter)
    _show_download(record)
    st.session_state["collecting_cover_profile"] = False

def _show_download(record: Dict[str, str]) -> None:
    try:
        import io
        buf = io.BytesIO(record["text"].encode("utf-8"))
        fname = f"cover_letter_{record['ts']}.txt"
        st.download_button(
            label="Download Cover Letter (.txt)",
            data=buf,
            file_name=fname,
            mime="text/plain",
            use_container_width=True,
        )
    except Exception:
        pass
