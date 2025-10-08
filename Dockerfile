FROM python:3.11-slim

ENV BASE_PATH=team2f25
# System deps + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install Python deps first for better Docker caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py entrypoint.sh ./
COPY styles.css .
COPY .streamlit/ .streamlit/
RUN chmod +x /app/entrypoint.sh

# Defaults
ENV MODEL_NAME=qwen2:0.5b \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    UI_PORT=5002

EXPOSE 5002 11434
ENTRYPOINT ["/app/entrypoint.sh"]
