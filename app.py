import os
import time
import requests
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

MODEL = os.getenv("MODEL_NAME", "qwen2:0.5b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Streamlit page setup
st.set_page_config(page_title="Chat with Qwen", page_icon="💬", layout="centered")
css_path = "styles.css"
try:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass
st.title(f"💬 Chat with Qwen ({MODEL})")

# Helper functions to check Ollama and model readiness
def ollama_ready() -> bool:
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5).ok
    except Exception:
        return False

def model_ready(model: str = MODEL) -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5).json()
        return any(m.get("name") == model for m in r.get("models", []))
    except Exception:
        return False
# Auto-refresh until Ollama and the model are ready
POLL_S = 2  # auto-refresh interval (seconds)

# Auto-refresh until ready
if not ollama_ready():
    st.warning("🖥️ Ollama is starting… This page will refresh automatically.")
    time.sleep(POLL_S)
    st.rerun()

if not model_ready():
    st.warning(f"🚧 Model **{MODEL}** is loading… This page will refresh automatically.")
    time.sleep(POLL_S)
    st.rerun()

st.success("✅ Ollama and model are ready!")


# Keep it tiny: just a list of messages
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content=f"Hi! I'm {MODEL}. Ask me anything.")]

# Render all messages
for m in st.session_state.messages:
    with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
        st.markdown(m.content)

# User prompt
prompt = st.chat_input("Type a message…")
if prompt:
    # append user
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # call LLM (no chains, no memory — dead simple)
    llm = ChatOllama(
        base_url=OLLAMA_HOST,
        model=MODEL,
        temperature=0.3,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ai: AIMessage = llm.invoke(st.session_state.messages)
            st.markdown(ai.content)

    # append assistant
    st.session_state.messages.append(ai)
    st.rerun()
