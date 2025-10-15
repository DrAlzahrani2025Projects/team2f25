FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates fonts-liberation libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libxkbcommon0 libxcomposite1 libxrandr2 libxdamage1 libxfixes3 libdrm2 libgbm1 libasound2 \
    libxshmfence1 libpango-1.0-0 libx11-6 libxext6 libx11-xcb1 libxcb1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://ollama.com/install.sh | sh || true

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m playwright install --with-deps chromium

COPY app.py scraper.py query_to_filter.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/data

ENV MODEL_NAME=qwen2.5:0.5b \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    STREAMLIT_SERVER_PORT=5002 \
    STREAMLIT_SERVER_BASE_URL_PATH=team2f25

EXPOSE 5002 11434
ENTRYPOINT ["/app/entrypoint.sh"]
