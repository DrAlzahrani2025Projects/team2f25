import os, json, asyncio, httpx
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, JSONResponse

# LangChain
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import AsyncIteratorCallbackHandler


app = FastAPI(title="Local Qwen2 0.5B • LangChain (Ollama)")

# Config
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME    = os.getenv("MODEL_NAME", "qwen2:0.5b")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Answer concisely (1–3 sentences).")
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "256"))
NUM_CTX       = int(os.getenv("NUM_CTX", "2048"))

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
        have = any(m.get("name", "").startswith(MODEL_NAME) for m in tags)
        if not have:
            # httpx accepts Timeout(None) instead of timeout=None in some versions
            pr = await c.post(f"{OLLAMA_HOST}/api/pull", json={"name": MODEL_NAME}, timeout=httpx.Timeout(None))
            pr.raise_for_status()

@app.post("/chat")
async def chat(body: dict = Body(...)):
    user_prompt = (body.get("prompt") or "").strip()
    if not user_prompt:
        return JSONResponse({"error": "empty prompt"}, status_code=400)

    await ensure_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    cb = AsyncIteratorCallbackHandler()

    llm = ChatOllama(
        base_url=OLLAMA_HOST,
        model=MODEL_NAME,
        temperature=0.2,
        streaming=True,
        callbacks=[cb],
        # Ollama-specific params go here:
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
        async for token in cb.aiter:
            if token:
                # stream plain text; switch to "text/event-stream" if you prefer SSE
                yield token.encode("utf-8")
        await task

    return StreamingResponse(token_stream(), media_type="text/plain")
