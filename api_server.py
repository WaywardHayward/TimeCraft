#!/usr/bin/env python3
"""
TimeCraft REST API Server (Demo Version)

This module provides a REST API for generating synthetic time series data based on
text descriptions. It's designed to demonstrate the capabilities of time series
generation from natural language descriptions.

Key Features:
- Text to time series generation
- Multiple pattern support (sine waves, linear trends, random walks)
- Realistic sensor data simulation
- CORS-enabled for web integration
"""

import os
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class AggregateTimeSeriesRequest(BaseModel):
    """
    Request model for generating multiple time series.
    
    Attributes:
        text_description (str): Natural language description of the scenario
        num_tags (int): Number of time series to generate (default: 5)
        sequence_length (int): Number of points in each series (default: 168)
    """
    text_description: str
    num_tags: int = 5
    sequence_length: int = 168

def generate_mock_timeseries(length: int = 168, pattern_type: str = "default", 
                           base_value: float = 1.0) -> List[float]:
    """
    Generate synthetic time series data with realistic patterns.
    
    Args:
        length (int): Number of points in the series
        pattern_type (str): Type of pattern to generate ('sine', 'linear', or 'default')
        base_value (float): Base value for the series
    
    Returns:
        List[float]: Generated time series data
    """
    time_points = np.linspace(0, length * 0.1, length)
    
    if pattern_type == "sine":
        # Generate a sine wave with noise for cyclic patterns (e.g., temperature)
        data = base_value + np.sin(time_points) * (base_value * 0.1) + np.random.normal(0, base_value * 0.05, length)
    elif pattern_type == "linear":
        # Generate a linear trend with noise for accumulating values (e.g., pressure)
        trend = np.linspace(0, base_value * 0.2, length)
        data = base_value + trend + np.random.normal(0, base_value * 0.02, length)
    else:
        # Generate random walk for general sensor data
        noise = np.random.normal(0, base_value * 0.05, length)
        data = base_value + np.cumsum(noise)
    
    return data.tolist()

def generate_tag_names(description: str, num_tags: int) -> List[str]:
    """
    Generate appropriate tag names based on scenario description.
    
    Args:
        description (str): Text description of the scenario
        num_tags (int): Number of tags to generate
    
    Returns:
        List[str]: List of generated tag names
    """
    keywords_to_tags = {
        "temperature": ["Temperature_Sensor_1", "Temperature_Sensor_2", "Ambient_Temperature"],
        "pressure": ["Pressure_Gauge_1", "Pressure_Gauge_2", "System_Pressure"],
        "vibration": ["Vibration_X", "Vibration_Y", "Vibration_Z"],
        "factory": ["Production_Rate", "Machine_Efficiency", "Power_Consumption"],
        "sensor": ["Sensor_A", "Sensor_B", "Sensor_C"]
    }
    
    tags = []
    lower_desc = description.lower()
    
    # Match keywords to generate relevant tag names
    for keyword, tag_list in keywords_to_tags.items():
        if keyword in lower_desc:
            tags.extend(tag_list)
            if len(tags) >= num_tags:
                break
    
    # Fill remaining slots with generic tags
    while len(tags) < num_tags:
        tags.append(f"Tag_{len(tags) + 1}")
    
    return tags[:num_tags]

# Create FastAPI app
app = FastAPI(
    title="TimeCraft API",
    description="REST API for TimeCraft time series generation (Demo Mode)",
    version="1.0.0"
)

# Configure CORS for web interface integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "TimeCraft REST API",
        "version": "1.0.0",
        "mode": "demo"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring API status."""
    return {
        "status": "healthy",
        "message": "TimeCraft API is running (Demo Mode)"
    }

@app.post("/generate-aggregate-timeseries")
async def generate_aggregate_timeseries(request: AggregateTimeSeriesRequest):
    """
    Generate multiple time series based on a text description.
    
    This endpoint generates synthetic time series data that matches the scenario
    described in the text. It uses different patterns based on the type of
    sensor or measurement described.
    
    Args:
        request (AggregateTimeSeriesRequest): Request containing description and parameters
    
    Returns:
        dict: Generated time series data and metadata
    """
    try:
        # Generate appropriate tag names based on the description
        tag_names = generate_tag_names(request.text_description, request.num_tags)
        
        # Generate mock data for each tag
        timeseries_data = {}
        for tag in tag_names:
            if "temperature" in tag.lower():
                base_value = 25.0  # baseline temperature (°C)
                timeseries_data[tag] = generate_mock_timeseries(request.sequence_length, "sine", base_value)
            elif "pressure" in tag.lower():
                base_value = 100.0  # baseline pressure (kPa)
                timeseries_data[tag] = generate_mock_timeseries(request.sequence_length, "linear", base_value)
            else:
                timeseries_data[tag] = generate_mock_timeseries(request.sequence_length)

        return {
            "status": "success",
            "text_description": request.text_description,
            "tags": tag_names,
            "sequence_length": request.sequence_length,
            "aggregate_timeseries": timeseries_data,
            "timestamp": "2025-06-18T00:00:00Z"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
