FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir flask

ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]
