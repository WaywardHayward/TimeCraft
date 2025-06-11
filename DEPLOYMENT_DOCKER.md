# TimeCraft Docker Deployment Guide

This guide provides comprehensive instructions for deploying the TimeCraft application as a Docker container for both local development and production environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Docker Build and Run](#docker-build-and-run)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Volume Configuration](#volume-configuration)
- [OpenAI and Azure OpenAI Setup](#openai-and-azure-openai-setup)
- [Optional ML Dependencies](#optional-ml-dependencies)
- [Health Checks and Testing](#health-checks-and-testing)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## Quick Start

### 1. Clone and Build

```bash
git clone https://github.com/WaywardHayward/TimeCraft.git
cd TimeCraft
docker build -t timecraft-api .
```

### 2. Basic Run (No OpenAI)

```bash
docker run -d -p 8080:8080 --name timecraft-api timecraft-api
```

### 3. Test the Deployment

```bash
curl http://localhost:8080/health
```

## Environment Variables

The following table describes all environment variables used by TimeCraft:

| Variable | Required | Default | Description | Example |
|----------|----------|---------|-------------|---------|
| `DATA_ROOT` | No | `/app/data` | Root directory for data storage | `/app/data` |
| `PYTHONPATH` | No | `/app:/app/BRIDGE:/app/diffusion` | Python module search paths | `/app:/app/BRIDGE:/app/diffusion` |
| `PYTHONUNBUFFERED` | No | `1` | Disable Python output buffering | `1` |
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key for LLM features | `sk-...` or Azure key |
| `OPENAI_API_BASE` | No** | - | OpenAI API base URL (for Azure) | `https://your-resource.openai.azure.com/` |
| `OPENAI_API_VERSION` | No** | - | OpenAI API version (for Azure) | `2024-02-15-preview` |
| `OPENAI_API_TYPE` | No** | `openai` | OpenAI API type | `azure` or `openai` |

**Notes:**
- *Required for LLM-powered features (text generation, refinement)
- **Required when using Azure OpenAI

### How to Obtain API Keys

#### Standard OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)

#### Azure OpenAI
1. Create an Azure OpenAI resource in the [Azure Portal](https://portal.azure.com/)
2. Navigate to your OpenAI resource
3. Go to "Keys and Endpoint" section
4. Copy the key and endpoint URL
5. Note the API version from the resource details

## Docker Build and Run

### Standard Build

```bash
# Build the image
docker build -t timecraft-api .

# Run with minimal configuration
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  timecraft-api
```

### Build with OpenAI Configuration

```bash
# Run with Standard OpenAI
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your_openai_api_key_here \
  timecraft-api

# Run with Azure OpenAI
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your_azure_openai_key \
  -e OPENAI_API_BASE=https://your-resource.openai.azure.com/ \
  -e OPENAI_API_VERSION=2024-02-15-preview \
  -e OPENAI_API_TYPE=azure \
  timecraft-api
```

### Build with Volume Mounts

```bash
# Create host directories
mkdir -p ./data ./logs

# Run with volume mounts
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e OPENAI_API_KEY=your_api_key \
  timecraft-api
```

## Docker Compose Deployment

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  timecraft-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      # Core configuration
      - DATA_ROOT=/app/data
      - PYTHONPATH=/app:/app/BRIDGE:/app/diffusion
      
      # OpenAI Configuration - set these in .env file or environment
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - OPENAI_API_BASE=${OPENAI_API_BASE:-}
      - OPENAI_API_VERSION=${OPENAI_API_VERSION:-}
      - OPENAI_API_TYPE=${OPENAI_API_TYPE:-openai}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Deploy with Docker Compose

```bash
# Create environment file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
# For Azure OpenAI, uncomment and set:
# OPENAI_API_BASE=https://your-resource.openai.azure.com/
# OPENAI_API_VERSION=2024-02-15-preview
# OPENAI_API_TYPE=azure
EOF

# Deploy
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Volume Configuration

### Host Data and Log Directories

```bash
# Create directories on host
mkdir -p /opt/timecraft/data
mkdir -p /opt/timecraft/logs
mkdir -p /opt/timecraft/checkpoints

# Set proper permissions
sudo chown -R 1000:1000 /opt/timecraft

# Run with production volume mounts
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  -v /opt/timecraft/data:/app/data \
  -v /opt/timecraft/logs:/app/logs \
  -v /opt/timecraft/checkpoints:/app/checkpoints \
  -e OPENAI_API_KEY=your_api_key \
  timecraft-api
```

### Docker Compose with Production Volumes

```yaml
version: '3.8'

services:
  timecraft-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATA_ROOT=/app/data
      - PYTHONPATH=/app:/app/BRIDGE:/app/diffusion
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_API_BASE=${OPENAI_API_BASE:-}
      - OPENAI_API_VERSION=${OPENAI_API_VERSION:-}
      - OPENAI_API_TYPE=${OPENAI_API_TYPE:-openai}
    volumes:
      - /opt/timecraft/data:/app/data
      - /opt/timecraft/logs:/app/logs
      - /opt/timecraft/checkpoints:/app/checkpoints
    restart: unless-stopped
    user: "1000:1000"
```

## OpenAI and Azure OpenAI Setup

### Standard OpenAI Configuration

```bash
# Environment variables
export OPENAI_API_KEY=sk-your_openai_api_key_here

# Docker run
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  timecraft-api
```

### Azure OpenAI Configuration

```bash
# Environment variables
export OPENAI_API_KEY=your_azure_openai_key
export OPENAI_API_BASE=https://your-resource.openai.azure.com/
export OPENAI_API_VERSION=2024-02-15-preview
export OPENAI_API_TYPE=azure

# Docker run
docker run -d \
  --name timecraft-api \
  -p 8080:8080 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OPENAI_API_BASE=$OPENAI_API_BASE \
  -e OPENAI_API_VERSION=$OPENAI_API_VERSION \
  -e OPENAI_API_TYPE=$OPENAI_API_TYPE \
  timecraft-api
```

### Testing OpenAI Integration

```bash
# Test text refinement with OpenAI
curl -X POST "http://localhost:8080/refine-text" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_text": "This time series shows sales data with weekly patterns.",
    "team_iterations": 2,
    "global_iterations": 1
  }'

# Test time series generation from text
curl -X POST "http://localhost:8080/generate-timeseries-from-text" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "text_description": "Generate a weekly sales pattern with weekend peaks",
    "model_name": "gpt-4"
  }'
```

## Optional ML Dependencies

The base Docker image includes lightweight dependencies. For full ML functionality, you can extend the Dockerfile:

### Dockerfile Extension for ML

Create a `Dockerfile.ml` with enhanced ML capabilities:

```dockerfile
FROM timecraft-api:latest

USER root

# Install PyTorch and additional ML dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    pytorch-lightning \
    transformers \
    datasets \
    accelerate \
    einops \
    diffusers \
    xformers

# Install additional scientific computing libraries
RUN pip install --no-cache-dir \
    scipy \
    matplotlib \
    seaborn \
    plotly \
    jupyter

USER appuser

# Override entrypoint if needed
ENTRYPOINT ["python", "api_server.py"]
```

### Build and Deploy ML Version

```bash
# Build ML-enhanced image
docker build -f Dockerfile.ml -t timecraft-api:ml .

# Run ML version
docker run -d \
  --name timecraft-api-ml \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e OPENAI_API_KEY=your_api_key \
  timecraft-api:ml
```

### Docker Compose for ML Version

```yaml
version: '3.8'

services:
  timecraft-api:
    build:
      context: .
      dockerfile: Dockerfile.ml
    ports:
      - "8080:8080"
    environment:
      - DATA_ROOT=/app/data
      - PYTHONPATH=/app:/app/BRIDGE:/app/diffusion
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## Health Checks and Testing

### API Health Check

The TimeCraft API includes comprehensive health monitoring:

```bash
# Basic health check
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy", "message": "TimeCraft API is running"}

# Detailed status check
curl http://localhost:8080/status

# System information
curl http://localhost:8080/
```

### Docker Health Check

The container includes built-in health checks:

```bash
# Check container health status
docker ps

# View health check logs
docker inspect timecraft-api | jq '.[0].State.Health'

# Manual health check
docker exec timecraft-api curl -f http://localhost:8080/health
```

### Demo and Test Commands

#### CSV Analysis Test

```bash
# Create sample CSV
cat > sample_data.csv << EOF
timestamp,value
2024-01-01,100
2024-01-02,105
2024-01-03,98
2024-01-04,110
2024-01-05,95
EOF

# Test CSV upload and analysis
curl -X POST "http://localhost:8080/analyze-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data.csv"
```

#### Time Series Description Generation

```bash
# Test description generation
curl -X POST "http://localhost:8080/generate-description" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_data.csv" \
  -F "dataset_name=test_data" \
  -F "prediction_length=24"
```

#### Interactive API Documentation

Access the interactive Swagger UI documentation:
- URL: http://localhost:8080/swagger
- Alternative: http://localhost:8080/docs

## Production Deployment

### Cloud Platform Deployment

#### Docker on AWS EC2

```bash
# Launch EC2 instance and install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy TimeCraft
git clone https://github.com/WaywardHayward/TimeCraft.git
cd TimeCraft

# Set environment variables
export OPENAI_API_KEY=your_production_api_key

# Deploy with production settings
docker run -d \
  --name timecraft-api \
  --restart unless-stopped \
  -p 80:8080 \
  -v /opt/timecraft/data:/app/data \
  -v /opt/timecraft/logs:/app/logs \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  timecraft-api
```

#### Google Cloud Run

```bash
# Build and tag for GCR
docker build -t gcr.io/your-project/timecraft-api .
docker push gcr.io/your-project/timecraft-api

# Deploy to Cloud Run
gcloud run deploy timecraft-api \
  --image gcr.io/your-project/timecraft-api \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --set-env-vars OPENAI_API_KEY=your_api_key \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry your-registry --image timecraft-api .

# Deploy to ACI
az container create \
  --resource-group your-rg \
  --name timecraft-api \
  --image your-registry.azurecr.io/timecraft-api \
  --ports 8080 \
  --environment-variables \
    OPENAI_API_KEY=your_api_key \
    OPENAI_API_BASE=https://your-resource.openai.azure.com/ \
    OPENAI_API_VERSION=2024-02-15-preview \
    OPENAI_API_TYPE=azure
```

### Production Configuration

#### Reverse Proxy with Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### SSL/TLS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use

```bash
# Check what's using port 8080
sudo lsof -i :8080

# Use different port
docker run -d -p 8081:8080 --name timecraft-api timecraft-api
```

#### 2. Container Won't Start

```bash
# Check container logs
docker logs timecraft-api

# Run interactively for debugging
docker run -it --rm -p 8080:8080 timecraft-api

# Check resource usage
docker stats timecraft-api
```

#### 3. OpenAI API Issues

```bash
# Verify environment variables
docker exec timecraft-api env | grep OPENAI

# Test OpenAI connectivity
docker exec timecraft-api python -c "
import os
print('API Key set:', bool(os.environ.get('OPENAI_API_KEY')))
print('API Base:', os.environ.get('OPENAI_API_BASE', 'default'))
"
```

#### 4. Volume Permission Issues

```bash
# Fix permissions on host
sudo chown -R 1000:1000 ./data ./logs

# Check container user
docker exec timecraft-api id

# Run with specific user
docker run -d --user 1000:1000 -p 8080:8080 timecraft-api
```

#### 5. Memory Issues

```bash
# Check memory usage
docker stats timecraft-api

# Limit memory usage
docker run -d -m 2g -p 8080:8080 timecraft-api

# For ML workloads, increase memory
docker run -d -m 8g -p 8080:8080 timecraft-api:ml
```

### Health Check Debugging

```bash
# Test health endpoint manually
curl -v http://localhost:8080/health

# Check container health history
docker inspect timecraft-api | jq '.[0].State.Health.Log'

# Disable health check if problematic
docker run -d --no-healthcheck -p 8080:8080 timecraft-api
```

### Logging and Monitoring

```bash
# View real-time logs
docker logs -f timecraft-api

# Export logs to file
docker logs timecraft-api > timecraft.log 2>&1

# Log rotation
docker run -d \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  -p 8080:8080 timecraft-api
```

## Security Considerations

### Environment Variable Security

```bash
# Use .env files instead of command line
echo "OPENAI_API_KEY=your_key" > .env
chmod 600 .env

# Use Docker secrets for production
echo "your_api_key" | docker secret create openai_key -
```

### Network Security

```bash
# Bind to localhost only
docker run -d -p 127.0.0.1:8080:8080 timecraft-api

# Use custom network
docker network create timecraft-net
docker run -d --network timecraft-net --name timecraft-api timecraft-api
```

### Container Security

The TimeCraft Docker image follows security best practices:

- ✅ Non-root user (`appuser`, UID 1000)
- ✅ Minimal base image (`python:3.8-slim`)
- ✅ No unnecessary packages or tools
- ✅ Health checks enabled
- ✅ Proper file permissions
- ✅ Environment variable validation

### Production Security Checklist

- [ ] Use HTTPS/TLS in production
- [ ] Store API keys in secure secret management
- [ ] Enable container security scanning
- [ ] Use specific image tags (not `latest`)
- [ ] Implement rate limiting
- [ ] Monitor for vulnerabilities
- [ ] Regular security updates
- [ ] Audit logs enabled

---

## Support and Documentation

- **API Documentation**: http://localhost:8080/swagger
- **Health Check**: http://localhost:8080/health
- **System Status**: http://localhost:8080/status
- **Repository Issues**: https://github.com/WaywardHayward/TimeCraft/issues

For additional support or questions about deployment, please refer to the main repository documentation or create an issue.