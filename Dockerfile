# TimeCraft API Server Dockerfile
# Builds a container for the demo API server

FROM python:3.8-slim

WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir \
    fastapi==0.68.0 \
    uvicorn==0.15.0 \
    numpy==1.19.2 \
    pydantic==1.10.5

# Copy API server code
COPY api_server.py .

# Expose API port
EXPOSE 8080

# Start the API server
CMD ["python", "api_server.py"]
