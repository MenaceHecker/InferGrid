# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies only, no app code
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.4.18

# Copy only the dependency manifest first (maximises layer cache)
COPY pyproject.toml .

# Install runtime deps into an isolated prefix — excludes dev extras
RUN uv pip install --system --no-cache --no-dev -e .

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal image, no build tools
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY app/ ./app/

# Models directory — mount a volume here in production
RUN mkdir -p /app/models && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Use exec form so signals propagate correctly to uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]