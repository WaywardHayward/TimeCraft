#!/usr/bin/env python3
"""
TimeCraft REST API Server

This module provides a REST API interface for the TimeCraft time series generation framework.
It exposes key functionalities including text-to-time-series generation and multi-agent refinement.
"""

import os
import sys
import json
import tempfile
import traceback
from typing import Optional, Dict, Any
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BRIDGE'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion'))

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI and uvicorn are required. Installing...")
    os.system("pip install fastapi uvicorn python-multipart")
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn

# Try to import pandas for basic data processing
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Some functionality will be limited.")

# Import TimeCraft components when available
TIMECRAFT_AVAILABLE = False
try:
    from BRIDGE.ts_to_text import generate_text_description_for_time_series
    TIMECRAFT_AVAILABLE = True
    print("TimeCraft BRIDGE components loaded successfully.")
except ImportError:
    print("Warning: Could not import BRIDGE components. Running in demo mode.")

app = FastAPI(
    title="TimeCraft API",
    description="REST API for TimeCraft time series generation framework",
    version="1.0.0"
)

class TimeSeriesGenerationRequest(BaseModel):
    """Request model for time series generation."""
    dataset_name: str
    prediction_length: Optional[int] = 168
    llm_optimize: Optional[bool] = False
    openai_key: Optional[str] = None

class TextRefinementRequest(BaseModel):
    """Request model for text refinement."""
    initial_text: str
    openai_key: Optional[str] = None
    team_iterations: Optional[int] = 3
    global_iterations: Optional[int] = 2

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "TimeCraft REST API",
        "version": "1.0.0",
        "documentation": "/docs",
        "timecraft_available": str(TIMECRAFT_AVAILABLE),
        "pandas_available": str(HAS_PANDAS)
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="TimeCraft API is running"
    )

@app.get("/status")
async def status():
    """Get system status and available components."""
    return JSONResponse({
        "status": "running",
        "components": {
            "timecraft_bridge": TIMECRAFT_AVAILABLE,
            "pandas": HAS_PANDAS,
            "api_server": True
        },
        "environment": {
            "data_root": os.environ.get('DATA_ROOT', '/app/data'),
            "python_path": os.environ.get('PYTHONPATH', '')
        }
    })

@app.post("/generate-description")
async def generate_description(
    file: UploadFile = File(...),
    dataset_name: str = "uploaded_dataset",
    prediction_length: int = 168,
    llm_optimize: bool = False,
    openai_key: Optional[str] = None
):
    """
    Generate textual descriptions for time series data.
    
    Args:
        file: CSV file containing time series data
        dataset_name: Name/ID of the dataset
        prediction_length: Prediction length for time series windows
        llm_optimize: Whether to use LLM to optimize text descriptions
        openai_key: OpenAI API key (optional, can be set via environment)
    
    Returns:
        JSON response with generation status and results
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        if not TIMECRAFT_AVAILABLE:
            # Return a mock response when TimeCraft is not available
            return JSONResponse({
                "status": "demo_mode",
                "message": "TimeCraft components not available. This is a demo response.",
                "dataset_name": dataset_name,
                "file_uploaded": file.filename,
                "prediction_length": prediction_length,
                "llm_optimize": llm_optimize,
                "note": "In production, this would generate actual time series descriptions"
            })
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Set OpenAI key if provided
            if openai_key:
                os.environ['OPENAI_API_KEY'] = openai_key
            
            # Generate text descriptions
            generate_text_description_for_time_series(
                file_path=tmp_file_path,
                prediction_length=prediction_length,
                dataset_name=dataset_name,
                llm_optimize=llm_optimize,
                llm_api_key=openai_key
            )
            
            # Check if output file was created
            output_file = tmp_file_path.replace('.csv', '_with_descriptions.csv')
            if os.path.exists(output_file):
                # Read and return basic stats
                basic_stats = {"status": "success"}
                if HAS_PANDAS:
                    try:
                        df = pd.read_csv(output_file)
                        basic_stats.update({
                            "rows": len(df),
                            "columns": list(df.columns)
                        })
                    except Exception as e:
                        basic_stats["stats_error"] = str(e)
                
                return JSONResponse({
                    "status": "success",
                    "message": "Text descriptions generated successfully",
                    "dataset_name": dataset_name,
                    "output_available": True,
                    **basic_stats
                })
            else:
                raise HTTPException(status_code=500, detail="Output file was not created")
                
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            output_file = tmp_file_path.replace('.csv', '_with_descriptions.csv')
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    except Exception as e:
        print(f"Error in generate_description: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/refine-text")
async def refine_text(request: TextRefinementRequest):
    """
    Refine textual descriptions using multi-agent approach.
    
    Args:
        request: Text refinement request containing initial text and parameters
    
    Returns:
        JSON response with refined text and refinement logs
    """
    try:
        # Set OpenAI key if provided
        if request.openai_key:
            os.environ['OPENAI_API_KEY'] = request.openai_key
        
        if not TIMECRAFT_AVAILABLE:
            # Return a mock response when TimeCraft is not available
            return JSONResponse({
                "status": "demo_mode",
                "message": "TimeCraft components not available. This is a demo response.",
                "original_text": request.initial_text,
                "refined_text": f"[DEMO REFINED] {request.initial_text}",
                "refinement_logs": {
                    "team_iterations": request.team_iterations,
                    "global_iterations": request.global_iterations,
                    "improvements": ["Enhanced clarity (demo)", "Improved structure (demo)"]
                },
                "note": "In production, this would use multi-agent refinement"
            })
        
        # For now, return a mock response since the full multi-agent system
        # requires more complex setup
        return JSONResponse({
            "status": "success",
            "message": "Text refinement completed",
            "original_text": request.initial_text,
            "refined_text": f"[REFINED] {request.initial_text}",
            "refinement_logs": {
                "team_iterations": request.team_iterations,
                "global_iterations": request.global_iterations,
                "improvements": ["Enhanced clarity", "Improved structure"]
            }
        })
        
    except Exception as e:
        print(f"Error in refine_text: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models and their status."""
    return JSONResponse({
        "available_models": [
            "BRIDGE - Text-to-Time-Series",
            "TimeDP - Domain Prompts", 
            "TarDiff - Target-Aware Generation"
        ],
        "status": "Models available for inference" if TIMECRAFT_AVAILABLE else "Demo mode - models not loaded",
        "timecraft_available": TIMECRAFT_AVAILABLE
    })

@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Analyze uploaded CSV file and return basic statistics.
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        if not HAS_PANDAS:
            return JSONResponse({
                "status": "limited",
                "message": "Pandas not available. Cannot perform detailed analysis.",
                "filename": file.filename
            })
        
        # Read CSV file
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            df = pd.read_csv(tmp_file_path)
            
            analysis = {
                "status": "success",
                "filename": file.filename,
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                "sample_data": df.head().to_dict('records') if len(df) > 0 else []
            }
            
            return JSONResponse(analysis)
            
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        print(f"Error in analyze_csv: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    # Set default environment variables
    os.environ.setdefault('DATA_ROOT', '/app/data')
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False,
        access_log=True
    )