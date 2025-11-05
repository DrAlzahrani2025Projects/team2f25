# Dockerfile
FROM python:3.11-slim

# -------------------------------------------------------------------
# Base env
# -------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -------------------------------------------------------------------
# System deps (minimal) + CA certs + fonts for headless Chromium
# NOTE: We do NOT install Ollama; this image is OpenAI-only.
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash \
    fonts-liberation libglib2.0-0 libnss3 libnspr4 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 \
    libxkbcommon0 libxcomposite1 libxrandr2 libxdamage1 \
    libxfixes3 libdrm2 libgbm1 libasound2 libxshmfence1 \
    libpango-1.0-0 libcairo2 libx11-6 libxext6 libx11-xcb1 libxcb1 \
    dos2unix \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------------------------------------------------
# Streamlit theme config (nice defaults)
# -------------------------------------------------------------------
RUN mkdir -p .streamlit && \
    echo '[theme]' > .streamlit/config.toml && \
    echo 'base="light"' >> .streamlit/config.toml && \
    echo 'primaryColor="#2563eb"' >> .streamlit/config.toml && \
    echo 'backgroundColor="#f8fafc"' >> .streamlit/config.toml && \
    echo 'secondaryBackgroundColor="#ffffff"' >> .streamlit/config.toml && \
    echo 'textColor="#111827"' >> .streamlit/config.toml && \
    echo 'font="sans serif"' >> .streamlit/config.toml

# Placeholder styles (your real styles.css will overwrite below)
RUN echo '/* Default styles */' > styles.css && \
    echo '.stApp { background: #ffffff; }' >> styles.css

# -------------------------------------------------------------------
# Python deps
# -------------------------------------------------------------------
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------
# Playwright (Python) + Chromium
#   - The CLI is provided by the Python package.
#   - install-deps pulls remaining OS libs if any missing.
# -------------------------------------------------------------------
RUN python -m playwright install chromium && \
    python -m playwright install-deps chromium

# -------------------------------------------------------------------
# App code
# (Include your converted OpenAI files)
# -------------------------------------------------------------------
COPY app.py main.py scraper.py query_to_filter.py backend_navigator.py playwright_fetcher.py async_scraper.py resume_manager.py llm_orchestrator.py llm_provider.py resume_parser.py entrypoint.sh ./ 
COPY styles.css ./

# Normalize line endings for shell scripts and make executable
RUN dos2unix /app/*.sh || true && chmod +x /app/entrypoint.sh

# -------------------------------------------------------------------
# Environment (OpenAI-first)
#   - Do NOT bake OPENAI_API_KEY into the image; pass at runtime.
#   - Adjust defaults as needed via `-e` or compose env.
# -------------------------------------------------------------------
ENV STREAMLIT_SERVER_PORT=5002 \
    STREAMLIT_SERVER_BASE_URL_PATH=team2f25 \
    BACKEND_PORT=8000 \
    OPENAI_MODEL=gpt-4o-mini

# If using uvicorn backend on 8000 and Streamlit on 5002:
EXPOSE 5002 8000 8001

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
ENTRYPOINT ["./entrypoint.sh"]
