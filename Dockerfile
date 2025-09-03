FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN pip install flask

EXPOSE 5000
CMD ["python", "app.py"]
