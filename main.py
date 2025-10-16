import os, json, asyncio, httpx
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# LangChain
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler  # <- safer import

app = FastAPI(title="Local Qwen2 0.5B • LangChain (Ollama)")

# --- Config ---
OLLAMA_HOST   = (os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434") or "").rstrip("/")
MODEL_NAME    = os.getenv("MODEL_NAME", "qwen2.5:0.5b")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Answer concisely (1–3 sentences).")
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "256"))
NUM_CTX       = int(os.getenv("NUM_CTX", "2048"))

# --- CORS (optional but helpful if UI runs elsewhere, e.g., Streamlit on 5002) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
        return {"status": "ok", "model": MODEL_NAME}
    except Exception as e:
        return JSONResponse({"status": "degraded", "error": str(e)}, status_code=503)

async def ensure_model():
    """Ensure the selected model is present in Ollama; pull if missing."""
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.get(f"{OLLAMA_HOST}/api/tags")
        r.raise_for_status()
        tags = r.json().get("models", [])
        have = any(str(m.get("name", "")).startswith(MODEL_NAME) for m in tags)
        if not have:
            pr = await c.post(
                f"{OLLAMA_HOST}/api/pull",
                json={"name": MODEL_NAME},
                timeout=httpx.Timeout(None)  # disable timeout for long pulls
            )
            pr.raise_for_status()

@app.on_event("startup")
async def _startup():
    # Don’t block startup; pull in background
    asyncio.create_task(ensure_model())

@app.post("/chat")
async def chat(body: dict = Body(...)):
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        return JSONResponse({"error": "empty prompt"}, status_code=400)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    cb = AsyncIteratorCallbackHandler()

    llm = ChatOllama(
        base_url=OLLAMA_HOST,
        model=MODEL_NAME,
        temperature=0.2,
        streaming=True,
        callbacks=[cb],
        model_kwargs={"num_ctx": NUM_CTX, "num_predict": MAX_TOKENS},
    )
    chain = prompt | llm

    async def run_chain():
        try:
            await chain.ainvoke({"question": user_prompt})
        finally:
            await cb.aiter.aclose()

    async def token_stream():
        task = asyncio.create_task(run_chain())
        try:
            async for token in cb.aiter:
                if token:
                    yield token.encode("utf-8")
        except Exception as e:
            # surface error at end of stream
            yield f"\n\n[stream error] {e}\n".encode("utf-8")
        finally:
            await task

    return StreamingResponse(
        token_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )
