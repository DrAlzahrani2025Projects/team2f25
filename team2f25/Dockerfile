# Single-container: Python + Streamlit + Playwright (Chromium)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System basics (slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && python -m playwright install --with-deps chromium

# App code
COPY .streamlit /app/.streamlit
COPY app.py /app/app.py
COPY scraper.py /app/scraper.py
COPY query_to_filter.py /app/query_to_filter.py
COPY data /app/data

# Streamlit settings (team2f25 / port 5002)
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=5002
ENV STREAMLIT_SERVER_BASE_URL_PATH=team2f25

EXPOSE 5002
CMD ["streamlit", "run", "app.py"]
