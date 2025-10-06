import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

MODEL = os.getenv("MODEL_NAME", "qwen2:0.5b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

st.set_page_config(page_title="Basic Chat", page_icon="💬", layout="centered")
st.title("💬 Basic Chat (qwen2:0.5b)")

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
