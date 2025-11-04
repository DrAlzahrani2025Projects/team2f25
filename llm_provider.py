# llm_provider.py
import os

def get_provider():
    return os.getenv("LLM_PROVIDER", "openai").lower()

def get_model():
    return os.getenv("LLM_MODEL", "gpt-5.1")  # change if you prefer

def get_openai_client():
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from env
