
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash \
    fonts-liberation libglib2.0-0 libnss3 libnspr4 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 \
    libxkbcommon0 libxcomposite1 libxrandr2 libxdamage1 \
    libxfixes3 libdrm2 libgbm1 libasound2 libxshmfence1 \
    libpango-1.0-0 libcairo2 libx11-6 libxext6 \
    libx11-xcb1 libxcb1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Create .streamlit directory and config
RUN mkdir -p .streamlit && \
    echo '[theme]' > .streamlit/config.toml && \
    echo 'base="light"' >> .streamlit/config.toml && \
    echo 'primaryColor="#2563eb"' >> .streamlit/config.toml && \
    echo 'backgroundColor="#f8fafc"' >> .streamlit/config.toml && \
    echo 'secondaryBackgroundColor="#ffffff"' >> .streamlit/config.toml && \
    echo 'textColor="#111827"' >> .streamlit/config.toml && \
    echo 'font="sans serif"' >> .streamlit/config.toml

# Create default styles.css
RUN echo '/* Default styles */' > styles.css && \
    echo '.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }' >> styles.css

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright
RUN playwright install chromium && \
    playwright install-deps chromium

# Copy application files
COPY app.py scraper.py query_to_filter.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/data

# Environment variables
ENV MODEL_NAME=qwen2.5:0.5b \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    STREAMLIT_SERVER_PORT=5002 \
    STREAMLIT_SERVER_BASE_URL_PATH=team2f25

EXPOSE 5002 11434
ENTRYPOINT ["./entrypoint.sh"]
