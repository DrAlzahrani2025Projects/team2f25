FROM python:3.11-slim

# Install curl to install Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama and pre-pull a small model
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# normalize line endings just in case
RUN sed -i 's/\r$//' /entrypoint.sh

# Expose ports
EXPOSE 5002 11434

ENV LLM_MODEL=qwen2:0.5b
ENV LLM_TIMEOUT=60
ENV OLLAMA_KEEP_ALIVE=10m
ENV OLLAMA_NUM_PARALLEL=2

CMD ["/entrypoint.sh"]
