FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 trader \
    && mkdir -p /app/data /app/logs /app/reports

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=trader:trader . .
RUN chown -R trader:trader /app

USER trader

# The dashboard listens inside the container; compose publishes it on 127.0.0.1 only.
EXPOSE 8765

HEALTHCHECK --interval=5m --timeout=10s --start-period=3m --retries=2 \
    CMD ["python", "-m", "core.healthcheck"]

CMD ["python", "main.py"]
