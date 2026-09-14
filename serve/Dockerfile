FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY serve/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY serve/app.py serve/loadtest.py /app/
# The model is baked in so a pod never depends on a volume or a download to become ready.
COPY models/ /models/

RUN useradd --uid 10001 --no-create-home app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "20"]
