import os
import time
import requests
import streamlit as st

# ---- Page Config ----
st.set_page_config(page_title="Team2f25 Chat")

# ---- Config ----
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LLM_MODEL", "llama3")   # e.g., "llama3", "qwen2:7b", etc.
TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT", "2.5"))  # keep < 3s budget

# ---- Simple per-minute rate limit (required) ----
# If >10 questions/minute, show the required message
def allowed_to_ask() -> bool:
    now = time.time()
    window_start = now - 60
    if "ask_times" not in st.session_state:
        st.session_state.ask_times = []
    st.session_state.ask_times = [t for t in st.session_state.ask_times if t >= window_start]
    if len(st.session_state.ask_times) >= 10:
        st.error("You’ve reached the limit of 10 questions per minute because the server has limited resources. Please try again in 3 minutes.")
        return False
    st.session_state.ask_times.append(now)
    return True

# ---- Ollama API call ----
def ask_ollama(prompt: str) -> str:
    """
    Calls Ollama's /api/generate endpoint (non-streaming) and returns the text.
    Works with any local/remote Ollama server.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except requests.exceptions.Timeout:
        return "(timeout) Model took too long to respond."
    except Exception as e:
        return f"(error) {e}"

# ---- Streamlit UI ----
st.title("Team2f25 Chat (Ollama)")

question = st.text_input("Ask a question:")
col1, col2 = st.columns(2)
with col1:
    model = st.text_input("Model", value=MODEL, help="e.g., llama3, qwen2:7b")
with col2:
    base = st.text_input("Ollama URL", value=OLLAMA_BASE_URL, help="e.g., http://localhost:11434")

if model != MODEL:
    MODEL = model
if base != OLLAMA_BASE_URL:
    OLLAMA_BASE_URL = base

if st.button("Send"):
    if allowed_to_ask():
        if question.strip():
            with st.spinner("Thinking..."):
                answer = ask_ollama(question.strip())
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.warning("Please enter a question.")
