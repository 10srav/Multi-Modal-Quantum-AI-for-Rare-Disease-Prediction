# Multi-Modal Quantum AI for Rare Disease Prediction
# Docker configuration for API and Dashboard

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Create directories for data and models
RUN mkdir -p data models results figures

# Expose ports
EXPOSE 8000 8501

# Default command - run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
