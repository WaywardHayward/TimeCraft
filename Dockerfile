# TimeCraft Dockerfile
# Lightweight Python-based build for time series generation REST API

FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV DATA_ROOT=/app/data
ENV PYTHONPATH=/app:/app/BRIDGE:/app/diffusion
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/checkpoints

# Copy requirements first for better caching
COPY BRIDGE/requirements.txt ./bridge_requirements.txt

# Create a basic requirements file for REST API functionality
RUN echo "fastapi" > requirements.txt && \
    echo "uvicorn[standard]" >> requirements.txt && \
    echo "python-multipart" >> requirements.txt && \
    echo "pydantic>=1.10.5" >> requirements.txt && \
    echo "requests>=2.28.2" >> requirements.txt && \
    echo "pandas" >> requirements.txt && \
    echo "numpy" >> requirements.txt && \
    echo "scikit-learn" >> requirements.txt

# Install Python dependencies for REST API and BRIDGE components
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r bridge_requirements.txt

# Copy application files
COPY --chown=root:root . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["python", "api_server.py"]