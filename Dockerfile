# Use Python 3.11 slim image for smaller size and better security
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Notebook Analyzer"
LABEL description="Web interface for analyzing Jupyter notebooks GPU requirements"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY app_vercel.py .
COPY templates/ templates/
COPY notebook-analyzer.py .

# Create necessary directories and set permissions
RUN mkdir -p /tmp/uploads /app/logs && \
    chown -R appuser:appuser /app /tmp/uploads && \
    chmod 755 /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set default environment variables
ENV HOST=0.0.0.0 \
    PORT=5000 \
    DEBUG=false

# Run the application
CMD ["python", "app.py"] 