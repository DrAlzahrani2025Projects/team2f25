# llm_provider.py
import os

def get_provider() -> str:
    """
    Returns the selected LLM provider. We default to OpenAI in this project.
    """
    return os.getenv("LLM_PROVIDER", "openai").lower()

def get_model() -> str:
    """
    Returns the chat model name. Preference order:
    1) OPENAI_MODEL
    2) LLM_MODEL
    3) sensible default ("gpt-4o-mini")
    """
    return (
        os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4o-mini"
    )

def get_openai_client():
    """
    Returns an OpenAI client instance (SDK v1).
    Requires OPENAI_API_KEY to be set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY in your environment."
        )

    from openai import OpenAI
    # The OpenAI() constructor reads OPENAI_API_KEY from the environment.
    return OpenAI()
