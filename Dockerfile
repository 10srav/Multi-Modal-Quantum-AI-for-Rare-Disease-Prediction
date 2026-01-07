# Multi-stage build for HGPS Multi-Modal AI System
# Production-optimized Docker image

# ============================================================================
# Stage 1: Builder - Install dependencies and build
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Stage 2: Production - Minimal runtime image
# ============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="HGPS AI Team"
LABEL version="1.0.0"
LABEL description="Multi-Modal Quantum AI for Rare Disease Prediction"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser requirements.txt ./

# Create directories for models, data, and logs
RUN mkdir -p /app/models /app/data /app/logs /app/results \
    && chown -R appuser:appuser /app

# Copy pre-trained models if available
COPY --chown=appuser:appuser models/ ./models/ 2>/dev/null || true

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose port
EXPOSE ${API_PORT}

# Run the application with uvicorn
CMD ["python", "-m", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "120", \
     "--access-log"]

# ============================================================================
# Stage 3: Development - Includes dev tools
# ============================================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    isort \
    mypy \
    ipython

# Copy test files
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser pytest.ini ./

USER appuser

# Development command
CMD ["python", "-m", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]
