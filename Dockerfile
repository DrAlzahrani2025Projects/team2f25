FROM python:3.11-slim

ARG APPUSER=app
RUN useradd -m ${APPUSER}

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chown -R ${APPUSER}:${APPUSER} /app
USER ${APPUSER}

EXPOSE 2502

CMD ["streamlit", "run", "app.py","--server.port=2502","--server.address=0.0.0.0","--server.enableCORS=false","--server.baseUrlPath=/team2f25"]
