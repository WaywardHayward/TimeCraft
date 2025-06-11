# TimeCraft Docker Setup

This directory contains the Docker configuration for running TimeCraft as a REST API service.

## Quick Start

### Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t timecraft-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8080:8080 --name timecraft-api timecraft-api
   ```

3. **Test the API:**
   ```bash
   curl http://localhost:8080/health
   ```

### Using Docker Compose

1. **Start the service:**
   ```bash
   docker-compose up -d
   ```

2. **Stop the service:**
   ```bash
   docker-compose down
   ```

## API Endpoints

The TimeCraft REST API exposes the following endpoints:

### Health and Status
- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `GET /status` - Detailed system status
- `GET /docs` - Interactive API documentation (Swagger UI)

### Core Functionality
- `POST /generate-description` - Generate textual descriptions for time series data
- `POST /refine-text` - Refine textual descriptions using multi-agent approach
- `POST /analyze-csv` - Analyze uploaded CSV files
- `GET /models` - List available models

## Usage Examples

### Upload and Analyze CSV
```bash
curl -X POST "http://localhost:8080/analyze-csv" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-timeseries-data.csv"
```

### Generate Time Series Descriptions
```bash
curl -X POST "http://localhost:8080/generate-description" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-timeseries-data.csv" \
  -F "dataset_name=my_dataset" \
  -F "prediction_length=168"
```

### Text Refinement
```bash
curl -X POST "http://localhost:8080/refine-text" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_text": "This time series shows an increasing trend with seasonal variations.",
    "team_iterations": 3,
    "global_iterations": 2
  }'
```

## Configuration

### Environment Variables
- `DATA_ROOT` - Data directory path (default: `/app/data`)
- `PYTHONPATH` - Python module search path
- `OPENAI_API_KEY` - OpenAI API key for LLM features (optional)

### Volumes
- `/app/data` - Data storage directory
- `/app/logs` - Log files directory

## Features

### Current Features
- ✅ REST API with FastAPI
- ✅ Health checks and status monitoring
- ✅ CSV file upload and analysis
- ✅ Interactive API documentation
- ✅ Basic time series processing
- ✅ Demo mode for testing

### Full TimeCraft Features (requires ML dependencies)
- 🔄 Time series to text description generation
- 🔄 Multi-agent text refinement
- 🔄 Cross-domain time series generation
- 🔄 Text-based control for generation
- 🔄 Target-aware generation with influence guidance

## Docker Image Details

### Base Image
- `python:3.8-slim` - Lightweight Python runtime

### Installed Dependencies
- FastAPI - Modern web framework for APIs
- Uvicorn - ASGI server
- Pandas - Data manipulation library
- NumPy - Numerical computing
- Scikit-learn - Machine learning library
- Pydantic - Data validation

### Security
- Non-root user (`appuser`) for container execution
- Minimal attack surface with slim base image
- Health checks for monitoring

### Performance
- Optimized Docker layers for better caching
- Efficient package installation
- Lightweight dependencies for faster startup

## Development

### Local Development
```bash
# Install dependencies
pip install fastapi uvicorn python-multipart pandas numpy scikit-learn

# Run the server
python api_server.py
```

### Adding ML Dependencies
To enable full TimeCraft functionality, add PyTorch and other ML dependencies to the Dockerfile:
```dockerfile
RUN pip install torch torchvision torchaudio pytorch-lightning transformers
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   docker run -d -p 8081:8080 --name timecraft-api timecraft-api
   ```

2. **Check container logs:**
   ```bash
   docker logs timecraft-api
   ```

3. **Interactive debugging:**
   ```bash
   docker run -it --rm -p 8080:8080 timecraft-api
   ```

### Health Check
The container includes a health check that verifies the API is responding:
```bash
docker ps  # Look for "healthy" status
```

## License

This project is licensed under the MIT License. See the main repository for details.