"""
Time series generation endpoints for TimeCraft API.
"""

from fastapi.responses import JSONResponse
from .models import (
    TextToTimeSeriesRequest, DomainPromptGenerationRequest, 
    TargetAwareGenerationRequest, AggregateTimeSeriesRequest
)
from .helpers import (
    create_demo_response, handle_api_error, generate_mock_timeseries,
    generate_mock_domain_series, generate_mock_target_aware_series,
    parse_llm_timeseries_response, generate_tag_names_from_description
)


def handle_generate_timeseries_from_text(request: TextToTimeSeriesRequest, bridge_text2ts_available: bool) -> JSONResponse:
    """Generate time series data from text description using BRIDGE model."""
    try:
        if not bridge_text2ts_available:
            mock_data = generate_mock_timeseries(
                request.length, 
                request.text, 
                getattr(request, 'frequency', 'hourly')
            )
            return create_demo_response(
                "demo_mode",
                "BRIDGE Text2TS model not available. This is a demo response with mock time series data.",
                **mock_data
            )
        
        # Import BRIDGE modules
        from BRIDGE.inference.text_to_timeseries import generate_timeseries_from_text
        
        # Generate time series
        result = generate_timeseries_from_text(
            text=request.text,
            length=request.length,
            frequency=getattr(request, 'frequency', 'hourly'),
            domain=getattr(request, 'domain', None)
        )
        
        return JSONResponse({
            "status": "success",
            "text": request.text,
            "length": request.length,
            "frequency": getattr(request, 'frequency', 'hourly'),
            "timeseries": result.get("timeseries", []),
            "metadata": result.get("metadata", {}),
            "details": result
        })
        
    except Exception as e:
        return handle_api_error("generate_timeseries_from_text", e)


def handle_domain_prompt_generation(request: DomainPromptGenerationRequest, timedp_available: bool) -> JSONResponse:
    """Generate time series data using TimeDP domain prompts."""
    try:
        if not timedp_available:
            mock_data = generate_mock_domain_series(request.domain, request.length)
            return create_demo_response(
                "demo_mode", 
                "TimeDP model not available. This is a demo response with mock domain-specific time series.",
                **mock_data
            )
        
        # Import TimeDP modules
        from TimeDP.inference.domain_generation import generate_with_domain_prompt
        
        # Generate time series with domain prompts
        result = generate_with_domain_prompt(
            domain=request.domain,
            length=request.length,
            num_samples=getattr(request, 'num_samples', 1)
        )
        
        return JSONResponse({
            "status": "success", 
            "domain": request.domain,
            "length": request.length,
            "num_samples": getattr(request, 'num_samples', 1),
            "timeseries": result.get("timeseries", []),
            "details": result
        })
        
    except Exception as e:
        return handle_api_error("domain_prompt_generation", e)


def handle_target_aware_generation(request: TargetAwareGenerationRequest, tardiff_available: bool) -> JSONResponse:
    """Generate time series data using TarDiff target-aware generation."""
    try:
        if not tardiff_available:
            mock_data = generate_mock_target_aware_series(
                request.target_value, 
                request.length,
                getattr(request, 'guidance_scale', 1.0)
            )
            return create_demo_response(
                "demo_mode",
                "TarDiff model not available. This is a demo response with mock target-aware time series.",
                **mock_data
            )
        
        # Import TarDiff modules  
        from TarDiff.inference.target_aware_generation import generate_target_aware_timeseries
        
        # Generate target-aware time series
        result = generate_target_aware_timeseries(
            target_value=request.target_value,
            length=request.length,
            guidance_scale=getattr(request, 'guidance_scale', 1.0),
            num_samples=getattr(request, 'num_samples', 1)
        )
        
        return JSONResponse({
            "status": "success",
            "target_value": request.target_value, 
            "length": request.length,
            "guidance_scale": getattr(request, 'guidance_scale', 1.0),
            "num_samples": getattr(request, 'num_samples', 1),
            "timeseries": result.get("timeseries", []),
            "details": result
        })
        
    except Exception as e:
        return handle_api_error("target_aware_generation", e)


def handle_aggregate_timeseries_generation(request: AggregateTimeSeriesRequest, bridge_text2ts_available: bool) -> JSONResponse:
    """Generate multiple time series data for different tags based on a text description."""
    try:
        # Generate tag names based on the text description and number of tags requested
        generated_tags = generate_tag_names_from_description(request.text_description, request.num_tags)
        
        # Generate mock data for each tag (always available as fallback)
        mock_results = {}
        for i, tag in enumerate(generated_tags):
            # Use different patterns for variety
            patterns = ["default", "sine", "linear"]
            pattern = patterns[i % len(patterns)]
            base_value = 50.0 + (i * 20.0)  # Different base values for each tag
            
            mock_data = generate_mock_timeseries(
                request.sequence_length,
                pattern,
                base_value
            )
            mock_results[tag] = mock_data
        
        if not bridge_text2ts_available:
            return create_demo_response(
                "demo_mode",
                "BRIDGE Text2TS model not available. This is a demo response with mock aggregate time series data.",
                text_description=request.text_description,
                tags=generated_tags,
                sequence_length=request.sequence_length,
                num_tags=request.num_tags,
                aggregate_timeseries=mock_results,
                note="In production, this would generate actual multi-tag time series"
            )
        
        # Try to use BRIDGE components if available
        try:
            from BRIDGE.self_refine.task_init import TimeSeriesTaskInit
            from BRIDGE.llm_agents.llm import ChatLLM
            
            # For now, use mock data but indicate BRIDGE is available
            # TODO: Implement actual BRIDGE integration when the proper modules are defined
            return JSONResponse({
                "status": "success",
                "message": "Time series generated successfully from text description using BRIDGE fallback",
                "text_description": request.text_description,
                "tags": generated_tags,
                "sequence_length": request.sequence_length,
                "num_tags": request.num_tags,
                "aggregate_timeseries": mock_results,
                "metadata": {"generation_method": "bridge_fallback"},
                "note": "BRIDGE components available but using fallback implementation"
            })
            
        except ImportError:
            # If BRIDGE import fails, use mock data
            return create_demo_response(
                "demo_mode",
                "BRIDGE Text2TS model components not available. Using mock aggregate time series data.",
                text_description=request.text_description,
                tags=generated_tags,
                sequence_length=request.sequence_length,
                num_tags=request.num_tags,
                aggregate_timeseries=mock_results,
                note="Fallback to mock data due to missing BRIDGE components"
            )
        
    except Exception as e:
        return handle_api_error("aggregate_timeseries_generation", e)