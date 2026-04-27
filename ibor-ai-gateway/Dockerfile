# ─────────────────────────────────────────────────────────────────────────────
# IBOR AI Gateway (FastAPI) Dockerfile
# Simple pip-based install (Docker is already isolated)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source code first
COPY . .

# Install dependencies using pip directly from pyproject.toml
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI application
CMD ["sh", "-c", "uvicorn ai_gateway.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
