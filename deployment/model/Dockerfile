# Production Dockerfile for ChainGuard Model Server
FROM python:3.12-slim


WORKDIR /app

# Install system dependencies required for C/C++ extensions & PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pinned requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Expose model HTTP REST API port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Export models if needed and launch REST API server
CMD ["python3", "serve_models.py", "--serve", "--port", "8080"]
