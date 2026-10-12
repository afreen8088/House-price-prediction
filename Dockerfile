# =========================================================================
# HOMESTEAD House Price Prediction MLOps Production Dockerfile
# Lean, multi-layer container running FastAPI & Uvicorn as a non-root user
# =========================================================================

FROM python:3.11-slim AS runtime

# Set environment flags for optimal container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    MODEL_PATH="/app/model.joblib" \
    METRICS_PATH="/app/metrics.json" \
    CORS_ORIGINS="*" \
    PORT=8000

# Install minimal OS dependencies if needed for compilation/C-libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -m -u 1001 -s /bin/bash appuser

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy required application modules, web interface, and persisted model artifacts
COPY app/ ./app/
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY model.joblib ./model.joblib
COPY metrics.json ./metrics.json

# Adjust permissions for non-root runtime
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose production port
EXPOSE 8000

# Healthcheck probe against FastAPI readiness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch ASGI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
