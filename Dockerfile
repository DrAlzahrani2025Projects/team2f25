FROM python:3.11-slim
WORKDIR /app

COPY streamlit_app.py .

# Install Streamlit directly (add others if needed)
RUN pip install --no-cache-dir streamlit

COPY . .

EXPOSE 2502

CMD streamlit run streamlit_app.py \
    --server.port=2502 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false \
    --server.baseUrlPath=/team2f25