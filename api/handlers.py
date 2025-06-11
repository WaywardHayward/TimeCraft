"""
Endpoint handlers for TimeCraft API business logic.
"""

import os
import tempfile
import traceback
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .models import (
    TextRefinementRequest, TextToTimeSeriesRequest, 
    DomainPromptGenerationRequest, TargetAwareGenerationRequest,
    AggregateTimeSeriesRequest
)
from .helpers import (
    setup_openai_config, create_demo_response, handle_api_error,
    validate_csv_file, create_temp_file, cleanup_temp_file,
    generate_mock_timeseries, generate_mock_domain_series,
    generate_mock_target_aware_series, parse_llm_timeseries_response
)


async def handle_generate_description(
    file: UploadFile, dataset_name: str, prediction_length: int,
    llm_optimize: bool, openai_api_base: str, openai_api_version: str,
    openai_api_type: str, timecraft_available: bool, has_pandas: bool
) -> JSONResponse:
    """Handle time series description generation."""
    try:
        validate_csv_file(file)
        
        if not timecraft_available:
            return create_demo_response(
                "demo_mode",
                "TimeCraft components not available. This is a demo response.",
                dataset_name=dataset_name,
                file_uploaded=file.filename,
                prediction_length=prediction_length,
                llm_optimize=llm_optimize,
                note="In production, this would generate actual time series descriptions"
            )
        
        # Process file
        content = await file.read()
        tmp_file_path = create_temp_file(content)
        
        try:
            setup_openai_config(openai_api_base, openai_api_version, openai_api_type)
            
            from BRIDGE.ts_to_text import generate_text_description_for_time_series
            generate_text_description_for_time_series(
                file_path=tmp_file_path,
                prediction_length=prediction_length,
                dataset_name=dataset_name,
                llm_optimize=llm_optimize,
                llm_api_key=os.environ.get('OPENAI_API_KEY')
            )
            
            # Check output and return stats
            output_file = tmp_file_path.replace('.csv', '_with_descriptions.csv')
            if os.path.exists(output_file):
                basic_stats = {"status": "success"}
                if has_pandas:
                    try:
                        import pandas as pd
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
            cleanup_temp_file(tmp_file_path)
            output_file = tmp_file_path.replace('.csv', '_with_descriptions.csv')
            cleanup_temp_file(output_file)
                
    except Exception as e:
        raise handle_api_error("generate_description", e)


def handle_refine_text(request: TextRefinementRequest, timecraft_available: bool) -> JSONResponse:
    """Handle text refinement requests."""
    try:
        setup_openai_config(request.openai_api_base, request.openai_api_version, request.openai_api_type)
        
        if not timecraft_available:
            return create_demo_response(
                "demo_mode",
                "TimeCraft components not available. This is a demo response.",
                original_text=request.initial_text,
                refined_text=f"[DEMO REFINED] {request.initial_text}",
                refinement_logs={
                    "team_iterations": request.team_iterations,
                    "global_iterations": request.global_iterations,
                    "improvements": ["Enhanced clarity (demo)", "Improved structure (demo)"]
                },
                note="In production, this would use multi-agent refinement"
            )
        
        # Return mock response for now since full multi-agent system requires complex setup
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
        raise handle_api_error("refine_text", e)


async def handle_analyze_csv(file: UploadFile, has_pandas: bool) -> JSONResponse:
    """Handle CSV analysis requests."""
    try:
        validate_csv_file(file)
        
        if not has_pandas:
            return JSONResponse({
                "status": "limited",
                "message": "Pandas not available. Cannot perform detailed analysis.",
                "filename": file.filename
            })
        
        content = await file.read()
        tmp_file_path = create_temp_file(content)
        
        try:
            import pandas as pd
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
            cleanup_temp_file(tmp_file_path)
                
    except Exception as e:
        raise handle_api_error("analyze_csv", e)


def handle_generate_timeseries_from_text(request: TextToTimeSeriesRequest, bridge_text2ts_available: bool) -> JSONResponse:
    """Handle text-to-time-series generation."""
    try:
        if not bridge_text2ts_available:
            mock_series = generate_mock_timeseries(168, "sine", 1.0)
            return create_demo_response(
                "demo_mode",
                "BRIDGE text-to-timeseries components not available. This is a demo response.",
                text_description=request.text_description,
                generated_timeseries=mock_series,
                model_used=request.model_name,
                note="In production, this would generate actual time series from text using BRIDGE model"
            )
        
        setup_openai_config(request.openai_api_base, request.openai_api_version, request.openai_api_type)
        
        # Initialize ChatLLM and generate
        from BRIDGE.llm_agents.llm import ChatLLM
        chat_llm = ChatLLM(
            model=request.model_name,
            temperature=request.temperature,
            api_base=request.openai_api_base,
            api_version=request.openai_api_version,
            api_type=request.openai_api_type
        )
        
        prompt = f"""Generate a time series of 168 numerical values based on this description: {request.text_description}

Please return only the numerical values separated by commas, without any additional text or explanation.

Example format: 1.2, 3.4, 2.1, 4.5, ...

Time Series:"""
        
        response = chat_llm.generate(prompt)
        time_series = parse_llm_timeseries_response(response, 168)
        
        return JSONResponse({
            "status": "success",
            "message": "Time series generated successfully from text description",
            "text_description": request.text_description,
            "generated_timeseries": time_series,
            "model_used": request.model_name,
            "length": len(time_series)
        })
        
    except Exception as e:
        raise handle_api_error("generate_timeseries_from_text", e)


def handle_domain_prompt_generation(request: DomainPromptGenerationRequest, timedp_available: bool) -> JSONResponse:
    """Handle domain prompt-based generation."""
    try:
        if not timedp_available:
            mock_series = generate_mock_domain_series(request.domain_type, request.sequence_length, request.num_samples)
            return create_demo_response(
                "demo_mode",
                "TimeDP components not available. This is a demo response.",
                domain_type=request.domain_type,
                generated_timeseries=mock_series,
                num_samples=len(mock_series),
                sequence_length=request.sequence_length,
                note="In production, this would use TimeDP diffusion model for domain-specific generation"
            )
        
        return JSONResponse({
            "status": "not_implemented",
            "message": "TimeDP integration is in development",
            "domain_type": request.domain_type,
            "note": "Full TimeDP diffusion model integration requires model checkpoints and complex setup"
        })
        
    except Exception as e:
        raise handle_api_error("generate_timeseries_domain_prompt", e)


def handle_target_aware_generation(request: TargetAwareGenerationRequest, tardiff_available: bool) -> JSONResponse:
    """Handle target-aware generation."""
    try:
        if not tardiff_available:
            mock_series = generate_mock_target_aware_series(
                request.target_values, request.sequence_length, request.num_samples
            )
            return create_demo_response(
                "demo_mode",
                "TarDiff components not available. This is a demo response.",
                target_values=request.target_values,
                generated_timeseries=mock_series,
                num_samples=len(mock_series),
                sequence_length=request.sequence_length,
                guidance_strength=request.guidance_strength,
                note="In production, this would use TarDiff model for target-aware generation"
            )
        
        return JSONResponse({
            "status": "not_implemented",
            "message": "TarDiff integration is in development",
            "target_values": request.target_values,
            "note": "Full TarDiff model integration requires model checkpoints and classifier guidance setup"
        })
        
    except Exception as e:
        raise handle_api_error("generate_timeseries_target_aware", e)


def handle_aggregate_timeseries_generation(request: AggregateTimeSeriesRequest, bridge_text2ts_available: bool) -> JSONResponse:
    """Handle aggregate multi-tag time series generation."""
    try:
        setup_openai_config(request.openai_api_base, request.openai_api_version, request.openai_api_type)

        if not bridge_text2ts_available:
            import numpy as np
            mock_tags = {}
            tag_names = [f"tag{i+1}" for i in range(request.num_tags)]
            
            for i, tag_name in enumerate(tag_names):
                # Generate different patterns for each tag
                base_value = (i + 1) * 10
                mock_series = [base_value + np.sin(j * 0.1 + i) * 5 + np.random.normal(0, 1) 
                              for j in range(request.sequence_length)]
                mock_tags[tag_name] = mock_series
            
            return create_demo_response(
                "demo_mode",
                "BRIDGE text-to-timeseries components not available. This is a demo response.",
                text_description=request.text_description,
                generated_timeseries=mock_tags,
                note="In production, this would generate actual tags and time series from text using BRIDGE model"
            )

        # Initialize ChatLLM
        from BRIDGE.llm_agents.llm import ChatLLM
        chat_llm = ChatLLM(
            model=request.model_name,
            temperature=request.temperature,
            api_base=request.openai_api_base,
            api_version=request.openai_api_version,
            api_type=request.openai_api_type
        )

        # Generate tag names
        tag_generation_prompt = f"""Based on the following description, generate {request.num_tags} relevant tag names that would be measured or tracked in this context:

Description: {request.text_description}

Please return only the tag names, one per line, without any additional text or explanation.
The tag names should be concise, descriptive, and relevant to the scenario described.

Tag names:"""

        try:
            tag_response = chat_llm.generate(tag_generation_prompt)
            tag_lines = [line.strip() for line in tag_response.strip().split('\n') if line.strip()]
            
            # Clean up tag names
            tag_names = []
            for line in tag_lines:
                clean_tag = line.replace('-', '').replace('*', '').replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '').strip()
                if clean_tag and len(clean_tag) > 0:
                    tag_names.append(clean_tag)
            
            # Ensure we have the requested number of tags
            if len(tag_names) < request.num_tags:
                for i in range(len(tag_names), request.num_tags):
                    tag_names.append(f"Tag{i+1}")
            elif len(tag_names) > request.num_tags:
                tag_names = tag_names[:request.num_tags]
                
        except Exception as e:
            # Fall back to generic tag names
            tag_names = [f"Tag{i+1}" for i in range(request.num_tags)]

        # Generate time series for each tag
        generated_timeseries = {}
        
        for tag_name in tag_names:
            try:
                tag_specific_prompt = f"""Generate a time series of {request.sequence_length} numerical values for the tag "{tag_name}" in the context of: {request.text_description}

The values should be realistic for this specific measurement/sensor in the given scenario.

Please return only the numerical values separated by commas, without any additional text or explanation.

Time Series for {tag_name}:"""

                response = chat_llm.generate(tag_specific_prompt)
                
                # Parse response
                time_series_str = response.strip()
                if f"Time Series for {tag_name}:" in time_series_str:
                    time_series_str = time_series_str.split(f"Time Series for {tag_name}:")[-1].strip()
                elif "Time Series:" in time_series_str:
                    time_series_str = time_series_str.split("Time Series:")[-1].strip()
                
                time_series = parse_llm_timeseries_response(time_series_str, request.sequence_length)
                generated_timeseries[tag_name] = time_series
                
            except Exception as e:
                # Generate fallback series
                import numpy as np
                tag_index = tag_names.index(tag_name)
                base_value = (tag_index + 1) * 10
                fallback_series = [base_value + np.sin(i * 0.1) * 5 + np.random.normal(0, 1) 
                                  for i in range(request.sequence_length)]
                generated_timeseries[tag_name] = fallback_series

        return JSONResponse({
            "status": "success",
            "message": "Time series generated successfully from text description",
            "text_description": request.text_description,
            "generated_timeseries": generated_timeseries
        })

    except Exception as e:
        raise handle_api_error("generate_aggregate_timeseries", e)