FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional: useful for numpy/pandas/matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (if available) to leverage Docker layer caching
COPY requirements.txt ./ 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir streamlit

# Copy the rest of the project files
COPY . .

# Streamlit will run on 2502
EXPOSE 2502

# Use exec form for CMD to handle signals properly
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=2502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false", \
     "--server.baseUrlPath=/team2f25"]
