FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir flask

ENV PORT=2502
EXPOSE 2502

CMD ["python", "app.py"]
