# Runtime image 
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies
COPY defender/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code needed to serve
COPY avlab ./avlab
COPY defender/app ./defender/app

ENV MODEL_PATH=/app/artifacts/baseline.joblib
EXPOSE 8080

# Gunicorn entrypoint
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "defender.app.server:app"]
