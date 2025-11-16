"""
main.py — Minimal FastAPI server for CSUSB CSE links only.

What it does:
- GET /healthz        -> simple health info (and cached link count if available)
- GET /csusb/links    -> returns cached links from scrape_csusb_listings(deep=False)

Removed:
- All LLM/Ollama/langchain-related endpoints (/chat, /chat/complete, /model/info)
- Rate limiting, streaming, and any navigator/LLM logic
"""

from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime

# Local scraper (must exist in the same project directory)
from scraper import scrape_csusb_listings


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CSUSB CSE Links API",
    description="Returns links extracted from CSUSB CSE pages (no deep search).",
    version="1.0.0",
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; connect-src 'self' ws: wss:; font-src 'self' data:; "
    "media-src 'self'; manifest-src 'self'; base-uri 'self'; form-action 'self'; "
    "object-src 'none'; frame-ancestors 'none'; upgrade-insecure-requests"
)


    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
# (Only at Nginx for HTTPS) # response.headers["Strict-Transport-Security"] = ...

    return response


# In production, constrain origins as needed
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:5002",
    "http://127.0.0.1:5002",
    "https://sec.cse.csusb.edu",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Simple in-memory cache for the dataframe
# -----------------------------------------------------------------------------
CACHE_TTL = int(os.getenv("CSUSB_CACHE_TTL", "3600"))  # seconds
_cache: Dict[str, Any] = {"at": 0.0, "df": None}


def _scrape_df() -> pd.DataFrame:
    """
    Synchronous scrape for CSUSB CSE links only.
    scraper.scrape_csusb_listings() should already be configured to avoid deep crawling.
    """
    df = scrape_csusb_listings(deep=False, max_pages=1)
    # Ensure expected columns exist
    expected = ["link", "title", "company", "host", "source", "posted_date"]
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df[expected].drop_duplicates(subset=["link"], keep="first")


async def _get_df(force: bool = False) -> pd.DataFrame:
    """
    Get cached DataFrame; refresh if stale or force=True.
    """
    now = time.time()
    if not force and _cache["df"] is not None and (now - _cache["at"] < CACHE_TTL):
        return _cache["df"]

    # Run the sync scrape in a worker thread (so we don't block the event loop)
    df = await asyncio.to_thread(_scrape_df)
    _cache["df"] = df
    _cache["at"] = time.time()
    return df


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

def _iso8601(ts: float) -> str:
    try:
        return datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        return ""
@app.get("/healthz", tags=["Health"])

async def healthz():
    """
    Basic health with cache info.
    """
    try:
        df = _cache["df"]
        count = int(len(df)) if df is not None else 0
        # Calculate age in minutes instead of raw seconds
        age_sec = time.time() - float(_cache["at"]) if _cache["at"] else None
        age_min = age_sec / 60 if age_sec is not None else None
        # Generate ISO8601 timestamp for cache time
        cached_at_iso = _iso8601(_cache["at"]) if _cache["at"] else ""
        # TTL in minutes
        ttl_min = CACHE_TTL / 60

        return {
            "status": "ok",
            "cached_count": count,
            "cached_at": cached_at_iso,
            "cache_age_minutes": age_min,
            "cache_ttl_minutes": ttl_min,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health error: {e}")



def _iso8601(ts: float) -> str:
    try:
        return datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        return ""

@app.get("/csusb/links", tags=["Links"])
async def csusb_links(
    refresh: bool = Query(False, description="If true, bypass cache and rescrape now."),
):
    """
    Return CSUSB CSE links as JSON (cached).
    """
    try:
        df = await _get_df(force=refresh)
        items = df.to_dict(orient="records")
        cached_at_iso = _iso8601(_cache["at"]) if _cache["at"] else ""
        ttl_min = CACHE_TTL / 60
        return {
            "count": len(items),
            "cached_at": cached_at_iso,
            "cache_ttl_minutes": ttl_min,
            "items": items,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scrape error: {e}")



@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "CSUSB CSE Links API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/healthz",
            "links": "/csusb/links",
        },
        "notes": "This API only returns links from CSUSB CSE pages (no deep external navigation).",
    }

