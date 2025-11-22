#!/usr/bin/env python3
"""
Mixture-of-Agents Tool Module

This module implements the Mixture-of-Agents (MoA) methodology that leverages
the collective strengths of multiple LLMs through a layered architecture to
achieve state-of-the-art performance on complex reasoning tasks.

Based on the research paper: "Mixture-of-Agents Enhances Large Language Model Capabilities"
by Junlin Wang et al. (arXiv:2406.04692v1)

Key Features:
- Multi-layer LLM collaboration for enhanced reasoning
- Parallel processing of reference models for efficiency
- Intelligent aggregation and synthesis of diverse responses
- Specialized for extremely difficult problems requiring intense reasoning
- Optimized for coding, mathematics, and complex analytical tasks

Available Tool:
- mixture_of_agents_tool: Process complex queries using multiple frontier models

Architecture:
1. Reference models generate diverse initial responses in parallel
2. Aggregator model synthesizes responses into a high-quality output
3. Multiple layers can be used for iterative refinement (future enhancement)

Models Used:
- Reference Models: claude-opus-4-20250514, gemini-2.5-pro, o4-mini, deepseek-r1
- Aggregator Model: claude-opus-4-20250514 (highest capability for synthesis)

Configuration:
    To customize the MoA setup, modify the configuration constants at the top of this file:
    - REFERENCE_MODELS: List of models for generating diverse initial responses
    - AGGREGATOR_MODEL: Model used to synthesize the final response
    - REFERENCE_TEMPERATURE/AGGREGATOR_TEMPERATURE: Sampling temperatures
    - MIN_SUCCESSFUL_REFERENCES: Minimum successful models needed to proceed

Usage:
    from mixture_of_agents_tool import mixture_of_agents_tool
    import asyncio
    
    # Process a complex query
    result = await mixture_of_agents_tool(
        user_prompt="Solve this complex mathematical proof..."
    )
"""

import json
import os
import asyncio
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI

# Initialize Nous Research API client for MoA processing
nous_client = AsyncOpenAI(
    api_key=os.getenv("NOUS_API_KEY"),
    base_url="https://inference-api.nousresearch.com/v1"
)

# Configuration for MoA processing
# Reference models - these generate diverse initial responses in parallel
REFERENCE_MODELS = [
    "claude-opus-4-20250514",
    "gemini-2.5-pro", 
    "gpt-5",
    "deepseek-r1"
]

# Aggregator model - synthesizes reference responses into final output
AGGREGATOR_MODEL = "claude-opus-4-20250514"  # Use highest capability model for aggregation

# Temperature settings optimized for MoA performance
REFERENCE_TEMPERATURE = 0.6  # Balanced creativity for diverse perspectives
AGGREGATOR_TEMPERATURE = 0.4  # Focused synthesis for consistency

# Failure handling configuration
MIN_SUCCESSFUL_REFERENCES = 1  # Minimum successful reference models needed to proceed

# System prompt for the aggregator model (from the research paper)
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

# Debug mode configuration
DEBUG_MODE = os.getenv("MOA_TOOLS_DEBUG", "false").lower() == "true"
DEBUG_SESSION_ID = str(uuid.uuid4())
DEBUG_LOG_PATH = Path("./logs")
DEBUG_DATA = {
    "session_id": DEBUG_SESSION_ID,
    "start_time": datetime.datetime.now().isoformat(),
    "debug_enabled": DEBUG_MODE,
    "tool_calls": []
} if DEBUG_MODE else None

# Create logs directory if debug mode is enabled
if DEBUG_MODE:
    DEBUG_LOG_PATH.mkdir(exist_ok=True)
    print(f"🐛 MoA debug mode enabled - Session ID: {DEBUG_SESSION_ID}")


def _log_debug_call(tool_name: str, call_data: Dict[str, Any]) -> None:
    """
    Log a debug call entry to the global debug data structure.
    
    Args:
        tool_name (str): Name of the tool being called
        call_data (Dict[str, Any]): Data about the call including parameters and results
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return
    
    call_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool_name": tool_name,
        **call_data
    }
    
    DEBUG_DATA["tool_calls"].append(call_entry)


def _save_debug_log() -> None:
    """
    Save the current debug data to a JSON file in the logs directory.
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return
    
    try:
        debug_filename = f"moa_tools_debug_{DEBUG_SESSION_ID}.json"
        debug_filepath = DEBUG_LOG_PATH / debug_filename
        
        # Update end time
        DEBUG_DATA["end_time"] = datetime.datetime.now().isoformat()
        DEBUG_DATA["total_calls"] = len(DEBUG_DATA["tool_calls"])
        
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            json.dump(DEBUG_DATA, f, indent=2, ensure_ascii=False)
        
        print(f"🐛 MoA debug log saved: {debug_filepath}")
        
    except Exception as e:
        print(f"❌ Error saving MoA debug log: {str(e)}")


def _construct_aggregator_prompt(system_prompt: str, responses: List[str]) -> str:
    """
    Construct the final system prompt for the aggregator including all model responses.
    
    Args:
        system_prompt (str): Base system prompt for aggregation
        responses (List[str]): List of responses from reference models
        
    Returns:
        str: Complete system prompt with enumerated responses
    """
    response_text = "\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
    return f"{system_prompt}\n\n{response_text}"


async def _run_reference_model_safe(
    model: str,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 6
) -> tuple[str, str, bool]:
    """
    Run a single reference model with retry logic and graceful failure handling.
    
    Args:
        model (str): Model identifier to use
        user_prompt (str): The user's query
        temperature (float): Sampling temperature for response generation
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        tuple[str, str, bool]: (model_name, response_content_or_error, success_flag)
    """
    for attempt in range(max_retries):
        try:
            print(f"🤖 Querying {model} (attempt {attempt + 1}/{max_retries})")
            
            # Build parameters for the API call
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}]
            }
            
            # GPT models (especially gpt-4o-mini) don't support custom temperature values
            # Only include temperature for non-GPT models
            if not model.lower().startswith('gpt-'):
                api_params["temperature"] = temperature
            
            response = await nous_client.chat.completions.create(**api_params)
            
            content = response.choices[0].message.content.strip()
            print(f"✅ {model} responded ({len(content)} characters)")
            return model, content, True
            
        except Exception as e:
            error_str = str(e)
            # Log more detailed error information for debugging
            if "invalid" in error_str.lower():
                print(f"⚠️  {model} invalid request error (attempt {attempt + 1}): {error_str}")
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                print(f"⚠️  {model} rate limit error (attempt {attempt + 1}): {error_str}")
            else:
                print(f"⚠️  {model} unknown error (attempt {attempt + 1}): {error_str}")
                
            if attempt < max_retries - 1:
                # Exponential backoff for rate limiting: 2s, 4s, 8s, 16s, 32s, 60s
                sleep_time = min(2 ** (attempt + 1), 60)
                print(f"   Retrying in {sleep_time}s...")
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{model} failed after {max_retries} attempts: {error_str}"
                print(f"❌ {error_msg}")
                return model, error_msg, False


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: int = None
) -> str:
    """
    Run the aggregator model to synthesize the final response.
    
    Args:
        system_prompt (str): System prompt with all reference responses
        user_prompt (str): Original user query
        temperature (float): Focused temperature for consistent aggregation
        max_tokens (int): Maximum tokens in final response
        
    Returns:
        str: Synthesized final response
    """
    print(f"🧠 Running aggregator model: {AGGREGATOR_MODEL}")
    
    # Build parameters for the API call
    api_params = {
        "model": AGGREGATOR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    # GPT models (especially gpt-4o-mini) don't support custom temperature values
    # Only include temperature for non-GPT models
    if not AGGREGATOR_MODEL.lower().startswith('gpt-'):
        api_params["temperature"] = temperature
    
    response = await nous_client.chat.completions.create(**api_params)
    
    content = response.choices[0].message.content.strip()
    print(f"✅ Aggregation complete ({len(content)} characters)")
    return content


async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[str]] = None,
    aggregator_model: Optional[str] = None
) -> str:
    """
    Process a complex query using the Mixture-of-Agents methodology.
    
    This tool leverages multiple frontier language models to collaboratively solve
    extremely difficult problems requiring intense reasoning. It's particularly
    effective for:
    - Complex mathematical proofs and calculations
    - Advanced coding problems and algorithm design
    - Multi-step analytical reasoning tasks
    - Problems requiring diverse domain expertise
    - Tasks where single models show limitations
    
    The MoA approach uses a fixed 2-layer architecture:
    1. Layer 1: Multiple reference models generate diverse responses in parallel (temp=0.6)
    2. Layer 2: Aggregator model synthesizes the best elements into final response (temp=0.4)
    
    Args:
        user_prompt (str): The complex query or problem to solve
        reference_models (Optional[List[str]]): Custom reference models to use
        aggregator_model (Optional[str]): Custom aggregator model to use
    
    Returns:
        str: JSON string containing the MoA results with the following structure:
             {
                 "success": bool,
                 "response": str,
                 "models_used": {
                     "reference_models": List[str],
                     "aggregator_model": str
                 },
                 "processing_time": float
             }
    
    Raises:
        Exception: If MoA processing fails or API key is not set
    """
    start_time = datetime.datetime.now()
    
    debug_call_data = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": reference_models or REFERENCE_MODELS,
            "aggregator_model": aggregator_model or AGGREGATOR_MODEL,
            "reference_temperature": REFERENCE_TEMPERATURE,
            "aggregator_temperature": AGGREGATOR_TEMPERATURE,
            "min_successful_references": MIN_SUCCESSFUL_REFERENCES
        },
        "error": None,
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "models_used": {}
    }
    
    try:
        print(f"🚀 Starting Mixture-of-Agents processing...")
        print(f"📝 Query: {user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}")
        
        # Validate API key availability
        if not os.getenv("NOUS_API_KEY"):
            raise ValueError("NOUS_API_KEY environment variable not set")
        
        # Use provided models or defaults
        ref_models = reference_models or REFERENCE_MODELS
        agg_model = aggregator_model or AGGREGATOR_MODEL
        
        print(f"🔄 Using {len(ref_models)} reference models in 2-layer MoA architecture")
        
        # Layer 1: Generate diverse responses from reference models (with failure handling)
        print("📡 Layer 1: Generating reference responses...")
        model_results = await asyncio.gather(*[
            _run_reference_model_safe(model, user_prompt, REFERENCE_TEMPERATURE)
            for model in ref_models
        ])
        
        # Separate successful and failed responses
        successful_responses = []
        failed_models = []
        
        for model_name, content, success in model_results:
            if success:
                successful_responses.append(content)
            else:
                failed_models.append(model_name)
        
        successful_count = len(successful_responses)
        failed_count = len(failed_models)
        
        print(f"📊 Reference model results: {successful_count} successful, {failed_count} failed")
        
        if failed_models:
            print(f"⚠️  Failed models: {', '.join(failed_models)}")
        
        # Check if we have enough successful responses to proceed
        if successful_count < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(f"Insufficient successful reference models ({successful_count}/{len(ref_models)}). Need at least {MIN_SUCCESSFUL_REFERENCES} successful responses.")
        
        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models
        
        # Layer 2: Aggregate responses using the aggregator model
        print("🧠 Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, 
            successful_responses
        )
        
        final_response = await _run_aggregator_model(
            aggregator_system_prompt,
            user_prompt,
            AGGREGATOR_TEMPERATURE
        )
        
        # Calculate processing time
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"✅ MoA processing completed in {processing_time:.2f} seconds")
        
        # Prepare successful response (only final aggregated result, minimal fields)
        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": ref_models,
                "aggregator_model": agg_model
            }
        }
        
        debug_call_data["success"] = True
        debug_call_data["final_response_length"] = len(final_response)
        debug_call_data["processing_time_seconds"] = processing_time
        debug_call_data["models_used"] = result["models_used"]
        
        # Log debug information
        _log_debug_call("mixture_of_agents_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in MoA processing: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Calculate processing time even for errors
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare error response (minimal fields)
        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": reference_models or REFERENCE_MODELS,
                "aggregator_model": aggregator_model or AGGREGATOR_MODEL
            },
            "error": error_msg
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _log_debug_call("mixture_of_agents_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps(result, indent=2, ensure_ascii=False)


def check_nous_api_key() -> bool:
    """
    Check if the Nous Research API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("NOUS_API_KEY"))


def check_moa_requirements() -> bool:
    """
    Check if all requirements for MoA tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    return check_nous_api_key()


def get_debug_session_info() -> Dict[str, Any]:
    """
    Get information about the current debug session.
    
    Returns:
        Dict[str, Any]: Dictionary containing debug session information
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return {
            "enabled": False,
            "session_id": None,
            "log_path": None,
            "total_calls": 0
        }
    
    return {
        "enabled": True,
        "session_id": DEBUG_SESSION_ID,
        "log_path": str(DEBUG_LOG_PATH / f"moa_tools_debug_{DEBUG_SESSION_ID}.json"),
        "total_calls": len(DEBUG_DATA["tool_calls"])
    }


def get_available_models() -> Dict[str, List[str]]:
    """
    Get information about available models for MoA processing.
    
    Returns:
        Dict[str, List[str]]: Dictionary with reference and aggregator models
    """
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_models": [AGGREGATOR_MODEL],
        "supported_models": REFERENCE_MODELS + [AGGREGATOR_MODEL]
    }


def get_moa_configuration() -> Dict[str, Any]:
    """
    Get the current MoA configuration settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration parameters
    """
    return {
        "reference_models": REFERENCE_MODELS,
        "aggregator_model": AGGREGATOR_MODEL,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": len(REFERENCE_MODELS),
        "failure_tolerance": f"{len(REFERENCE_MODELS) - MIN_SUCCESSFUL_REFERENCES}/{len(REFERENCE_MODELS)} models can fail"
    }


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🤖 Mixture-of-Agents Tool Module")
    print("=" * 50)
    
    # Check if API key is available
    api_available = check_nous_api_key()
    
    if not api_available:
        print("❌ NOUS_API_KEY environment variable not set")
        print("Please set your API key: export NOUS_API_KEY='your-key-here'")
        print("Get API key at: https://inference-api.nousresearch.com/")
        exit(1)
    else:
        print("✅ Nous Research API key found")
    
    print("🛠️  MoA tools ready for use!")
    
    # Show current configuration
    config = get_moa_configuration()
    print(f"\n⚙️  Current Configuration:")
    print(f"  🤖 Reference models ({len(config['reference_models'])}): {', '.join(config['reference_models'])}")
    print(f"  🧠 Aggregator model: {config['aggregator_model']}")
    print(f"  🌡️  Reference temperature: {config['reference_temperature']}")
    print(f"  🌡️  Aggregator temperature: {config['aggregator_temperature']}")
    print(f"  🛡️  Failure tolerance: {config['failure_tolerance']}")
    print(f"  📊 Minimum successful models: {config['min_successful_references']}")
    
    # Show debug mode status
    if DEBUG_MODE:
        print(f"\n🐛 Debug mode ENABLED - Session ID: {DEBUG_SESSION_ID}")
        print(f"   Debug logs will be saved to: ./logs/moa_tools_debug_{DEBUG_SESSION_ID}.json")
    else:
        print("\n🐛 Debug mode disabled (set MOA_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from mixture_of_agents_tool import mixture_of_agents_tool")
    print("  import asyncio")
    print("")
    print("  async def main():")
    print("      result = await mixture_of_agents_tool(")
    print("          user_prompt='Solve this complex mathematical proof...'")
    print("      )")
    print("      print(result)")
    print("  asyncio.run(main())")
    
    print("\nBest use cases:")
    print("  - Complex mathematical proofs and calculations")
    print("  - Advanced coding problems and algorithm design")
    print("  - Multi-step analytical reasoning tasks")
    print("  - Problems requiring diverse domain expertise")
    print("  - Tasks where single models show limitations")
    
    print("\nPerformance characteristics:")
    print("  - Higher latency due to multiple model calls")
    print("  - Significantly improved quality for complex tasks")
    print("  - Parallel processing for efficiency")
    print(f"  - Optimized temperatures: {REFERENCE_TEMPERATURE} for reference models, {AGGREGATOR_TEMPERATURE} for aggregation")
    print("  - Token-efficient: only returns final aggregated response")
    print("  - Resilient: continues with partial model failures")
    print(f"  - Configurable: easy to modify models and settings at top of file")
    print("  - State-of-the-art results on challenging benchmarks")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export MOA_TOOLS_DEBUG=true")
    print("  # Debug logs capture all MoA processing steps and metrics")
    print("  # Logs saved to: ./logs/moa_tools_debug_UUID.json")
