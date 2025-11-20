# llm.py
"""
Centralized LLM connection manager for the CSUSB Internship Finder.
"""

import os
import streamlit as st
from typing import Optional
from langchain_ollama import ChatOllama
from functools import lru_cache


class LLMConfig:
    """Configuration class for LLM settings."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        num_predict: int = 400,
        streaming: bool = False
    ):
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model_name = model_name or os.getenv("MODEL_NAME", "qwen2.5:0.5b")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.streaming = streaming
        self.num_predict = num_predict
    
    def __hash__(self):
        """Make config hashable for caching."""
        return hash((
            self.base_url,
            self.model_name,
            self.temperature,
            self.num_ctx,
            self.num_predict
        ))
    
    def __eq__(self, other):
        """Enable equality comparison for caching."""
        if not isinstance(other, LLMConfig):
            return False
        return (
            self.base_url == other.base_url and
            self.model_name == other.model_name and
            self.temperature == other.temperature and
            self.num_ctx == other.num_ctx and
            self.num_predict == other.num_predict
        )


@lru_cache(maxsize=10)
def get_llm(config: LLMConfig) -> ChatOllama:
    """
    Get a cached LLM instance based on configuration.
    
    Args:
        config: LLMConfig instance with connection parameters
        
    Returns:
        ChatOllama instance
        
    Note:
        Uses @lru_cache for connection reuse. Same config = same connection.
    """
    return ChatOllama(
        base_url=config.base_url,
        model=config.model_name,
        temperature=config.temperature,
        streaming=False,
        model_kwargs={
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict
        }
    )


def create_llm(
    temperature: float = 0.1,
    num_ctx: int = 4096,
    num_predict: int = 400,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> ChatOllama:
    """
    Convenience function to create an LLM with custom parameters.
    
    Args:
        temperature: Model temperature (0.0-1.0)
        num_ctx: Context window size
        num_predict: Maximum tokens to predict
        base_url: Ollama server URL (defaults to env OLLAMA_HOST)
        model_name: Model name (defaults to env MODEL_NAME)
        
    Returns:
        ChatOllama instance (cached if config matches previous call)
    """
    config = LLMConfig(
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict
    )
    return get_llm(config)


def get_default_llm() -> ChatOllama:
    """
    Get default LLM instance with standard configuration.
    
    Returns:
        ChatOllama instance with default settings
    """
    return create_llm()


def clear_llm_cache():
    """Clear the LLM connection cache. Useful for testing or troubleshooting."""
    get_llm.cache_clear()


# Pre-defined configurations for common use cases (cached as singletons)
_CREATIVE_CONFIG = LLMConfig(temperature=0.2, num_ctx=4096, num_predict=1000)
_CLASSIFICATION_CONFIG = LLMConfig(temperature=0.0, num_ctx=512, num_predict=30)
_RESUME_CONFIG = LLMConfig(temperature=0.0, num_ctx=512, num_predict=20)
_RESUME_EXTRACTOR_CONFIG = LLMConfig(temperature=0.0, num_ctx= 4096, num_predict= 350)
_PLANNER_CONFIG = LLMConfig(temperature=0.2, num_ctx=512, streaming=False)
_DEFAULT_CONFIG = LLMConfig(temperature=0.1, num_ctx=4096, num_predict=400)

@st.cache_resource
def get_resume_extractor_llm() -> ChatOllama:
    """LLM optimized for structured output (JSON parsing, etc.)"""
    return get_llm(_RESUME_EXTRACTOR_CONFIG)

@st.cache_resource
def get_creative_llm() -> ChatOllama:
    """LLM optimized for creative content (cover letters, etc.)"""
    return get_llm(_CREATIVE_CONFIG)

@st.cache_resource
def get_resume_llm() -> ChatOllama:
    """LLM optimized for creative content (cover letters, etc.)"""
    return get_llm(_RESUME_CONFIG)

@st.cache_resource
def get_planner_llm() -> ChatOllama:
    """LLM optimized for creative content (cover letters, etc.)"""
    return get_llm(_PLANNER_CONFIG)

@st.cache_resource
def get_classification_llm() -> ChatOllama:
    """LLM optimized for classification tasks (intent detection, etc.)"""
    return get_llm(_CLASSIFICATION_CONFIG)

@st.cache_resource
def get_default_llm() -> ChatOllama:
    """
    Get default LLM instance with standard configuration.
    
    Returns:
        ChatOllama instance with default settings
    """
    return get_llm(_DEFAULT_CONFIG)
