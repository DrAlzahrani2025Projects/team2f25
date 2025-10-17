# main.py
"""
FastAPI microservice for LLM inference with proper async support.
Fixed with error handling, rate limiting, and security improvements.
"""
import asyncio
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import AsyncCallbackHandler

from constants import (
    OLLAMA_HOST,
    DEFAULT_MODEL,
    SYSTEM_PROMPT_CHAT,
    MAX_TOKENS_GENERATE,
    NUM_CTX,
    LOG_FORMAT,
    LOG_LEVEL,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL, logging.INFO)
)
logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITING
# ============================================================================
limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Team2F25 LLM Microservice",
    description="FastAPI microservice for LLM inference using LangChain + Ollama",
    version="1.0.0"
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# CORS CONFIGURATION
# ============================================================================
# In production, specify exact origins
ALLOWED_ORIGINS = [
    "http://localhost:5002",
    "http://127.0.0.1:5002",
    "https://sec.cse.csusb.edu",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ============================================================================
# STREAMING CALLBACK HANDLER
# ============================================================================
class StreamingHandler(AsyncCallbackHandler):
    """Async callback handler for streaming LLM responses."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = False
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token."""
        await self.queue.put(token)
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        """Called when LLM finishes generation."""
        self.done = True
        await self.queue.put(None)
    
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        logger.error(f"LLM error during streaming: {error}")
        self.done = True
        await self.queue.put(None)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================
async def ensure_model():
    """Ensure the selected model is available in Ollama."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Check available models
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if model exists
            model_exists = any(DEFAULT_MODEL in name for name in model_names)
            
            if not model_exists:
                logger.info(f"Model {DEFAULT_MODEL} not found. Pulling...")
                
                # Pull the model
                pull_response = await client.post(
                    f"{OLLAMA_HOST}/api/pull",
                    json={"name": DEFAULT_MODEL},
                    timeout=httpx.Timeout(300.0)  # 5 min timeout
                )
                pull_response.raise_for_status()
                
                logger.info(f"Model {DEFAULT_MODEL} pulled successfully")
            else:
                logger.info(f"Model {DEFAULT_MODEL} is ready")
                
    except Exception as e:
        logger.error(f"Error ensuring model availability: {e}")


@app.on_event("startup")
async def startup_event():
    """Run model check on startup."""
    logger.info("Starting up FastAPI service...")
    asyncio.create_task(ensure_model())


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================
@app.get("/healthz", tags=["Health"])
async def healthz():
    """
    Health check endpoint for container orchestration.
    Verifies Ollama service is reachable and model is available.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if our model is available
            model_available = any(DEFAULT_MODEL in name for name in model_names)
            
            return {
                "status": "healthy" if model_available else "degraded",
                "ollama_host": OLLAMA_HOST,
                "model": DEFAULT_MODEL,
                "model_available": model_available,
                "available_models": model_names[:5]  # Show first 5
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            {
                "status": "unhealthy",
                "error": str(e),
                "ollama_host": OLLAMA_HOST
            },
            status_code=503
        )


# ============================================================================
# CHAT ENDPOINT (STREAMING)
# ============================================================================
@app.post("/chat", tags=["Chat"])
@limiter.limit("20/minute")  # Rate limit: 20 requests per minute
async def chat_stream(request: Request, body: dict = Body(...)):
    """
    Streaming chat endpoint.
    
    Request body:
        {
            "prompt": "Your question here",
            "system_prompt": "Optional custom system prompt",
            "temperature": 0.2,
            "max_tokens": 256
        }
    
    Returns:
        Streaming text response
    """
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    
    if len(user_prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 2000 chars)")
    
    # Allow custom parameters
    system_prompt = body.get("system_prompt", SYSTEM_PROMPT_CHAT)
    temperature = body.get("temperature", 0.2)
    max_tokens = body.get("max_tokens", MAX_TOKENS_GENERATE)
    
    # Validate parameters
    temperature = max(0.0, min(1.0, float(temperature)))
    max_tokens = max(10, min(512, int(max_tokens)))
    
    logger.info(f"Chat request: prompt_len={len(user_prompt)}, temp={temperature}")
    
    try:
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # Create callback handler for streaming
        handler = StreamingHandler()
        
        # Create LLM with streaming
        llm = ChatOllama(
            base_url=OLLAMA_HOST,
            model=DEFAULT_MODEL,
            temperature=temperature,
            streaming=True,
            callbacks=[handler],
            model_kwargs={
                "num_ctx": NUM_CTX,
                "num_predict": max_tokens
            },
        )
        
        # Create chain
        chain = prompt | llm
        
        # Run chain in background
        async def run_chain():
            try:
                await chain.ainvoke({"question": user_prompt})
            except Exception as e:
                logger.error(f"Chain error: {e}")
                await handler.queue.put(f"\n[Error: {str(e)}]")
                handler.done = True
        
        # Token streaming generator
        async def token_generator():
            task = asyncio.create_task(run_chain())
            
            try:
                while not handler.done:
                    try:
                        token = await asyncio.wait_for(handler.queue.get(), timeout=30.0)
                        if token is None:
                            break
                        yield token.encode("utf-8")
                    except asyncio.TimeoutError:
                        logger.warning("Stream timeout")
                        yield b"\n[Timeout]"
                        break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"\n[Stream error: {str(e)}]".encode("utf-8")
            finally:
                await task
        
        return StreamingResponse(
            token_generator(),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )
    
    except Exception as e:
        logger.error(f"Chat stream failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ============================================================================
# CHAT ENDPOINT (NON-STREAMING)
# ============================================================================
@app.post("/chat/complete", tags=["Chat"])
@limiter.limit("20/minute")
async def chat_complete(request: Request, body: dict = Body(...)):
    """
    Non-streaming chat endpoint.
    
    Request body:
        {
            "prompt": "Your question here",
            "system_prompt": "Optional custom system prompt",
            "temperature": 0.2,
            "max_tokens": 256
        }
    
    Returns:
        {
            "response": "Complete response text",
            "model": "qwen2.5:0.5b",
            "tokens": 42
        }
    """
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    
    if len(user_prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 2000 chars)")
    
    system_prompt = body.get("system_prompt", SYSTEM_PROMPT_CHAT)
    temperature = max(0.0, min(1.0, float(body.get("temperature", 0.2))))
    max_tokens = max(10, min(512, int(body.get("max_tokens", MAX_TOKENS_GENERATE))))
    
    logger.info(f"Chat complete request: prompt_len={len(user_prompt)}")
    
    try:
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        # Create LLM without streaming
        llm = ChatOllama(
            base_url=OLLAMA_HOST,
            model=DEFAULT_MODEL,
            temperature=temperature,
            streaming=False,
            model_kwargs={
                "num_ctx": NUM_CTX,
                "num_predict": max_tokens
            },
        )
        
        # Create and run chain
        chain = prompt | llm
        response = await chain.ainvoke({"question": user_prompt})
        
        response_text = response.content or ""
        
        return {
            "response": response_text,
            "model": DEFAULT_MODEL,
            "tokens": len(response_text.split())  # Rough estimate
        }
        
    except Exception as e:
        logger.error(f"Chat complete failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# ============================================================================
# MODEL INFO ENDPOINT
# ============================================================================
@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the current model."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{OLLAMA_HOST}/api/show",
                params={"name": DEFAULT_MODEL}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Model info fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not fetch model info: {str(e)}")


# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "service": "Team2F25 LLM Microservice",
        "version": "1.0.0",
        "model": DEFAULT_MODEL,
        "ollama_host": OLLAMA_HOST,
        "endpoints": {
            "health": "/healthz",
            "chat_stream": "/chat (POST)",
            "chat_complete": "/chat/complete (POST)",
            "model_info": "/model/info"
        },
        "rate_limits": {
            "chat": "20 requests/minute"
        }
    }


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload