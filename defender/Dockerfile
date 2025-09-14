FROM python:3.11-slim

# Security & size hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy runtime requirements first for layer caching
COPY defender/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy server code
COPY defender/app /app/defender/app

# Copy model artifacts into image
COPY artifacts /app/artifacts

# Env wiring (EMBER by default)
ENV MODEL_KIND=ember \
    ARTIFACTS_DIR=/app/artifacts \
    MODEL_PATH=/app/artifacts/ember_model_calibrated.joblib \
    FEATURE_SPEC=/app/artifacts/ember_feature_spec.json \
    THRESHOLD_JSON=/app/artifacts/threshold.json \
    PYTHONPATH=/app

# Expose port and start gunicorn
EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "defender.app.server:app", "--workers", "2", "--threads", "1", "--timeout", "10"]
