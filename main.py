# main.py
"""
FastAPI microservice for LLM inference with proper async support.
OpenAI-based implementation with error handling, rate limiting, and CORS.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import asyncio
import logging
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from llm_provider import get_openai_client, get_model

# =============================================================================
# CONFIG / CONSTANTS
# =============================================================================
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(levelname)s:%(name)s:%(message)s")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SYSTEM_PROMPT_CHAT = os.getenv(
    "SYSTEM_PROMPT_CHAT",
    "You are a helpful, concise assistant for CSUSB internship search. "
    "Be accurate, encouraging, and avoid unnecessary verbosity."
)
MAX_TOKENS_GENERATE = int(os.getenv("MAX_TOKENS_GENERATE", "512"))
OPENAI_MODEL = get_model()  # prefers OPENAI_MODEL / LLM_MODEL / default

# Validate API key early
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("llm-service")

# =============================================================================
# RATE LIMITING
# =============================================================================
limiter = Limiter(key_func=get_remote_address)

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Team2F25 LLM Microservice (OpenAI)",
    description="FastAPI microservice for LLM inference using OpenAI",
    version="2.0.0",
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================================================================
# CORS CONFIGURATION
# =============================================================================
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

# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================
@app.get("/healthz", tags=["Health"])
async def healthz():
    """
    Health check endpoint for container orchestration.
    We simply verify we can construct an OpenAI client and return the configured model.
    """
    try:
        # Constructing the client is enough; we avoid making a billable request here.
        _ = get_openai_client()
        return {"status": "healthy", "model": OPENAI_MODEL}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

# =============================================================================
# INTERNAL: OpenAI helpers (streaming + non-streaming)
# =============================================================================
def _build_messages(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

async def _openai_stream_generator(
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[bytes, None]:
    """
    Asynchronous generator that yields streamed bytes from OpenAI chat completions.
    Runs the streaming loop in a worker thread to avoid blocking the event loop.
    """
    client = get_openai_client()
    q: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def _run_streaming():
        try:
            with client.chat.completions.stream(
                model=OPENAI_MODEL,
                messages=_build_messages(system_prompt, user_prompt),
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                for event in stream:
                    # Events may include chunks, deltas, and end signals depending on SDK version
                    if hasattr(event, "delta") and event.delta and event.delta.content:
                        q.put_nowait(event.delta.content)
                    elif hasattr(event, "choices") and event.choices:
                        # Fallback: older style chunks
                        for ch in event.choices:
                            # ch.delta.content for delta-based
                            delta = getattr(ch, "delta", None)
                            if delta and getattr(delta, "content", None):
                                q.put_nowait(delta.content)
                # Stream closed
        except Exception as e:
            q.put_nowait(f"\n[Error: {e}]")
        finally:
            q.put_nowait(None)

    # Kick off the thread
    await asyncio.to_thread(_run_streaming)

    # Drain queue asynchronously
    while True:
        token = await q.get()
        if token is None:
            break
        yield token.encode("utf-8")

def _openai_complete_text(
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Non-streaming chat completion. Returns the assistant text.
    """
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=_build_messages(system_prompt, user_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

# =============================================================================
# CHAT ENDPOINT (STREAMING)
# =============================================================================
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
        Streaming text/plain response
    """
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    if len(user_prompt) > 8000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 8000 chars)")

    system_prompt = body.get("system_prompt", SYSTEM_PROMPT_CHAT)
    temperature = max(0.0, min(1.0, float(body.get("temperature", 0.2))))
    max_tokens = max(10, min(MAX_TOKENS_GENERATE, int(body.get("max_tokens", 256))))

    logger.info(f"[stream] prompt_len={len(user_prompt)} temp={temperature} max_tokens={max_tokens}")

    async def generator():
        try:
            async for chunk in _openai_stream_generator(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n[Stream error: {e}]".encode("utf-8")

    return StreamingResponse(
        generator(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )

# =============================================================================
# CHAT ENDPOINT (NON-STREAMING)
# =============================================================================
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
            "model": "<model>",
            "tokens": <approx token/word count>
        }
    """
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    if len(user_prompt) > 8000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 8000 chars)")

    system_prompt = body.get("system_prompt", SYSTEM_PROMPT_CHAT)
    temperature = max(0.0, min(1.0, float(body.get("temperature", 0.2))))
    max_tokens = max(10, min(MAX_TOKENS_GENERATE, int(body.get("max_tokens", 256))))

    logger.info(f"[complete] prompt_len={len(user_prompt)} temp={temperature} max_tokens={max_tokens}")

    try:
        text = await asyncio.to_thread(
            _openai_complete_text,
            user_prompt,
            system_prompt,
            temperature,
            max_tokens,
        )
        return {
            "response": text,
            "model": OPENAI_MODEL,
            "tokens": len(text.split()),  # rough estimate
        }
    except Exception as e:
        logger.error(f"Chat complete failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# =============================================================================
# MODEL INFO ENDPOINT
# =============================================================================
@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return basic info about the configured model."""
    return {"model": OPENAI_MODEL}

# =============================================================================
# ROOT ENDPOINT
# =============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "service": "Team2F25 LLM Microservice (OpenAI)",
        "version": "2.0.0",
        "model": OPENAI_MODEL,
        "endpoints": {
            "health": "/healthz",
            "chat_stream": "/chat (POST)",
            "chat_complete": "/chat/complete (POST)",
            "model_info": "/model/info",
        },
        "rate_limits": {"chat": "20 requests/minute"},
    }
