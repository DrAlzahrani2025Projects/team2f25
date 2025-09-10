FROM python:3.11-slim
WORKDIR /app

# install Flask without requirements.txt
RUN pip install --no-cache-dir flask

COPY . .
EXPOSE 2502
CMD ["python", "app.py"]
