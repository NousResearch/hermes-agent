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

Models Used (config-driven via config.yaml moa section):
- Reference Models and Aggregator are loaded from config at import time.
- Use `/moac` command to view or change the council lineup.
- No hardcoded defaults — config is the single source of truth.

Configuration:
    MoA models are configured via the `moa` section in config.yaml:
      moa:
        reference_models:
          - provider: opencode-go
            model: kimi-k2.6
          ...
        aggregator:
          provider: opencode-go
          model: kimi-k2.6
    Use `/moac` to view or change the council lineup.

Usage:
    from mixture_of_agents_tool import mixture_of_agents_tool
    import asyncio
    
    # Process a complex query
    result = await mixture_of_agents_tool(
        user_prompt="Solve this complex mathematical proof..."
    )
"""

import json
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional, Sequence, Union
from tools.openrouter_client import get_async_client as _get_openrouter_client, check_api_key as check_openrouter_api_key
from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

# Configuration for MoA processing
# Reference models - these generate diverse initial responses in parallel.
# A model spec is either:
# - a legacy OpenRouter string ("provider/model"), or
# - a provider-routed dict: {"provider": "...", "model": "...", ...}.
ModelSpec = Union[str, Dict[str, Any]]

# ---------------------------------------------------------------------------
# Config-driven model loading — no hardcoded defaults.
# REFERENCE_MODELS and AGGREGATOR_MODEL are loaded from config.yaml at import
# time. Use /moac to view or change the council.
# ---------------------------------------------------------------------------

def _load_moa_config() -> Dict[str, Any]:
    """Load MoA model configuration from config.yaml.

    Returns a dict with 'reference_models' (list of ModelSpec) and
    'aggregator' (ModelSpec). If the moa section is missing or malformed,
    raises RuntimeError — there are no hardcoded fallbacks.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        moa_cfg = cfg.get("moa")
        if not isinstance(moa_cfg, dict):
            raise RuntimeError("No 'moa' section in config.yaml — use /moac to configure the council")
        refs = moa_cfg.get("reference_models")
        if not isinstance(refs, list) or len(refs) == 0:
            raise RuntimeError("moa.reference_models must be a non-empty list in config.yaml")
        agg = moa_cfg.get("aggregator")
        if not isinstance(agg, dict):
            raise RuntimeError("moa.aggregator must be a dict with provider and model in config.yaml")
        # Validate each spec
        for i, spec in enumerate(refs):
            if isinstance(spec, str):
                continue  # legacy OpenRouter string — allowed
            if not isinstance(spec, dict):
                raise RuntimeError(f"moa.reference_models[{i}] must be a dict or string, got {type(spec).__name__}")
            if not spec.get("provider") or not spec.get("model"):
                raise RuntimeError(f"moa.reference_models[{i}] missing provider or model")
        if not agg.get("provider") or not agg.get("model"):
            raise RuntimeError("moa.aggregator missing provider or model")
        return {"reference_models": refs, "aggregator": agg}
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load MoA config from config.yaml: {e}") from e

# Lazy-loaded on first use — defers config read so tests can import the module
# without a full config.yaml.  Use get_reference_models() / get_aggregator_model().
_moa_config: Optional[Dict[str, Any]] = None
REFERENCE_MODELS: List[ModelSpec] = []       # populated on first access
AGGREGATOR_MODEL: Optional[ModelSpec] = None  # populated on first access


def _ensure_moa_config() -> None:
    """Load MoA config on first use.  No-op if already loaded."""
    global _moa_config, REFERENCE_MODELS, AGGREGATOR_MODEL
    if _moa_config is not None:
        return
    _moa_config = _load_moa_config()
    REFERENCE_MODELS = _moa_config["reference_models"]
    AGGREGATOR_MODEL = _moa_config["aggregator"]

# Temperature settings optimized for MoA performance
REFERENCE_TEMPERATURE = 0.6  # Balanced creativity for diverse perspectives
AGGREGATOR_TEMPERATURE = 0.4  # Focused synthesis for consistency

# Failure handling configuration
MIN_SUCCESSFUL_REFERENCES = 1  # Minimum successful reference models needed to proceed

# System prompt for the aggregator model (from the research paper)
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from several frontier models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. Critically evaluate every response: some may be biased, incomplete, overconfident, or wrong. Do not simply replicate the given answers; produce a refined, accurate, and comprehensive answer.

Responses from models:"""

_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")



def _model_provider(model_spec: ModelSpec) -> Optional[str]:
    if isinstance(model_spec, dict):
        provider = str(model_spec.get("provider") or "").strip()
        return provider or None
    return "openrouter"


def _model_name(model_spec: ModelSpec) -> str:
    if isinstance(model_spec, dict):
        return str(model_spec.get("model") or "").strip()
    return str(model_spec).strip()


def _model_label(model_spec: ModelSpec) -> str:
    if isinstance(model_spec, dict):
        provider = _model_provider(model_spec) or "unknown"
        model = _model_name(model_spec)
        return f"{provider}/{model}" if model else provider
    return _model_name(model_spec)


def _model_stance(model_spec: ModelSpec) -> str:
    return ""  # stances removed — all models get neutral treatment


def _model_specs_require_openrouter(model_specs: Sequence[ModelSpec]) -> bool:
    return any(not isinstance(spec, dict) or (_model_provider(spec) == "openrouter") for spec in model_specs)


def _build_messages_for_reference(model_spec: ModelSpec, user_prompt: str) -> List[Dict[str, str]]:
    # All reference models get the same neutral treatment — no adversarial stances.
    return [{"role": "user", "content": user_prompt}]


def _reasoning_extra_body_for_model(model_spec: ModelSpec) -> Dict[str, Any]:
    provider = _model_provider(model_spec)
    model_name = _model_name(model_spec).lower()
    # Kimi rejects OpenRouter-style extra_body.reasoning, and Copilot Gemini
    # frequently returns reasoning-only payloads for tiny smoke prompts. Keep
    # maximum reasoning where it is known to work (Codex/OpenRouter/DeepSeek).
    if provider == "opencode-go" and model_name.startswith("kimi-"):
        return {}
    if provider == "copilot" and model_name.startswith("gemini-"):
        return {}
    return {"reasoning": {"enabled": True, "effort": "xhigh"}}


def _extract_content(response: Any) -> str:
    content = extract_content_or_reasoning(response)
    if content:
        return content
    try:
        return str(response.choices[0].message.content or "")
    except Exception:
        return ""


async def _dispatch_llm_call(
    model_spec: ModelSpec,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Any:
    """Dispatch an LLM call to the correct backend based on model_spec.

    Provider-routed specs (dict with non-openrouter provider) go through
    ``async_call_llm``; OpenRouter strings and openrouter-routed dicts go
    through the OpenRouter client.  Single helper, single decision point.
    """
    provider = _model_provider(model_spec)
    model_name = _model_name(model_spec)

    if isinstance(model_spec, dict) and provider and provider != "openrouter":
        return await async_call_llm(
            task="moa",
            provider=provider,
            model=model_name,
            base_url=str(model_spec.get("base_url") or "") or None,
            api_key=str(model_spec.get("api_key") or "") or None,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(model_spec.get("timeout") or 300),
            extra_body=extra_body or {},
        )

    api_params: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "extra_body": extra_body or {},
    }
    if not model_name.lower().startswith("gpt-"):
        api_params["temperature"] = temperature
    return await _get_openrouter_client().chat.completions.create(**api_params)


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
    model: ModelSpec,
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
            model_label = _model_label(model)
            logger.info("Querying %s (attempt %s/%s)", model_label, attempt + 1, max_retries)

            messages = _build_messages_for_reference(model, user_prompt)
            extra_body = _reasoning_extra_body_for_model(model)
            response = await _dispatch_llm_call(model, messages, temperature, max_tokens, extra_body)

            content = _extract_content(response)
            if not content:
                # Reasoning-only response — let the retry loop handle it
                logger.warning("%s returned empty content (attempt %s/%s), retrying", _model_label(model), attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
            logger.info("%s responded (%s characters)", _model_label(model), len(content))
            return _model_label(model), content, True
            
        except Exception as e:
            error_str = str(e)
            # Keep retry-path logging concise; full tracebacks are reserved for
            # terminal failure paths so long-running MoA retries don't flood logs.
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", _model_label(model), attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", _model_label(model), attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", _model_label(model), attempt + 1, error_str)

            if attempt < max_retries - 1:
                # Exponential backoff for rate limiting: 2s, 4s, 8s, 16s, 32s, 60s
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{_model_label(model)} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return _model_label(model), error_msg, False


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: Optional[int] = None,
    aggregator_model: Optional[ModelSpec] = None
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
    agg_spec = aggregator_model or AGGREGATOR_MODEL
    logger.info("Running aggregator model: %s", _model_label(agg_spec))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    extra_body = _reasoning_extra_body_for_model(agg_spec)
    response = await _dispatch_llm_call(agg_spec, messages, temperature, max_tokens, extra_body)

    content = _extract_content(response)

    # Retry once on empty content (reasoning-only response)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await _dispatch_llm_call(agg_spec, messages, temperature, max_tokens, extra_body)
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%s characters)", len(content))
    return content


async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[ModelSpec]] = None,
    aggregator_model: Optional[ModelSpec] = None
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
    _ensure_moa_config()
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
        logger.info("Starting Mixture-of-Agents processing...")
        logger.info("Query: %s", user_prompt[:100])
        
        # Use provided models or defaults
        ref_models = reference_models or REFERENCE_MODELS
        agg_model = aggregator_model or AGGREGATOR_MODEL

        # Validate OpenRouter API key only when any selected spec needs OpenRouter.
        if _model_specs_require_openrouter([*ref_models, agg_model]) and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set for OpenRouter MoA model specs")
        
        logger.info("Using %s reference models in 2-layer MoA architecture", len(ref_models))
        
        # Layer 1: Generate diverse responses from reference models (with failure handling)
        logger.info("Layer 1: Generating reference responses...")
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
        
        logger.info("Reference model results: %s successful, %s failed", successful_count, failed_count)
        
        if failed_models:
            logger.warning("Failed models: %s", ', '.join(failed_models))
        
        # Check if we have enough successful responses to proceed
        if successful_count < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(f"Insufficient successful reference models ({successful_count}/{len(ref_models)}). Need at least {MIN_SUCCESSFUL_REFERENCES} successful responses.")
        
        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models
        
        # Layer 2: Aggregate responses using the aggregator model
        logger.info("Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, 
            successful_responses
        )
        
        final_response = await _run_aggregator_model(
            aggregator_system_prompt,
            user_prompt,
            AGGREGATOR_TEMPERATURE,
            aggregator_model=agg_model
        )
        
        # Calculate processing time
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info("MoA processing completed in %.2f seconds", processing_time)
        
        # Prepare successful response (only final aggregated result, minimal fields)
        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": [_model_label(m) for m in ref_models],
                "aggregator_model": _model_label(agg_model)
            }
        }
        
        debug_call_data["success"] = True
        debug_call_data["final_response_length"] = len(final_response)
        debug_call_data["processing_time_seconds"] = processing_time
        debug_call_data["models_used"] = result["models_used"]
        
        # Log debug information
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error in MoA processing: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)
        
        # Calculate processing time even for errors
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare error response (minimal fields)
        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": [_model_label(m) for m in (reference_models or REFERENCE_MODELS)],
                "aggregator_model": _model_label(aggregator_model or AGGREGATOR_MODEL)
            },
            "error": error_msg
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)


def check_moa_requirements() -> bool:
    """
    Check if all requirements for MoA tools are met.
    
    Returns:
        bool: True if requirements are met, False otherwise
    """
    _ensure_moa_config()
    return True if not _model_specs_require_openrouter([*REFERENCE_MODELS, AGGREGATOR_MODEL]) else check_openrouter_api_key()



def get_moa_configuration() -> Dict[str, Any]:
    """
    Get the current MoA configuration settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration parameters
    """
    _ensure_moa_config()
    return {
        "reference_models": REFERENCE_MODELS,
        "reference_model_labels": [_model_label(m) for m in REFERENCE_MODELS],
        "aggregator_model": AGGREGATOR_MODEL,
        "aggregator_model_label": _model_label(AGGREGATOR_MODEL),
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
    
    # Check if configured model specs are available.
    api_available = check_moa_requirements()

    if not api_available:
        print("❌ MoA requirements not met")
        print("OpenRouter model specs require OPENROUTER_API_KEY; provider-routed specs require their provider credentials.")
        raise SystemExit(1)
    else:
        print("✅ MoA requirements satisfied")
    
    print("🛠️  MoA tools ready for use!")
    
    # Show current configuration
    config = get_moa_configuration()
    print("\n⚙️  Current Configuration:")
    print(f"  🤖 Reference models ({len(config['reference_model_labels'])}): {', '.join(config['reference_model_labels'])}")
    print(f"  🧠 Aggregator model: {config['aggregator_model_label']}")
    print(f"  🌡️  Reference temperature: {config['reference_temperature']}")
    print(f"  🌡️  Aggregator temperature: {config['aggregator_temperature']}")
    print(f"  🛡️  Failure tolerance: {config['failure_tolerance']}")
    print(f"  📊 Minimum successful models: {config['min_successful_references']}")
    
    # Show debug mode status
    if _debug.active:
        print(f"\n🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: ./logs/moa_tools_debug_{_debug.session_id}.json")
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
    print("  - Configurable: easy to modify models and settings at top of file")
    print("  - State-of-the-art results on challenging benchmarks")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export MOA_TOOLS_DEBUG=true")
    print("  # Debug logs capture all MoA processing steps and metrics")
    print("  # Logs saved to: ./logs/moa_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

MOA_SCHEMA = {
    "name": "mixture_of_agents",
    "description": "Route a hard problem through multiple frontier LLMs collaboratively. Makes 5 API calls (4 reference models + 1 aggregator) with maximum reasoning effort — use sparingly for genuinely difficult problems. Best for: complex math, advanced algorithms, multi-step analytical reasoning, problems benefiting from diverse perspectives.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The complex query or problem to solve using multiple AI models. Should be a challenging problem that benefits from diverse perspectives and collaborative reasoning."
            }
        },
        "required": ["user_prompt"]
    }
}

registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(user_prompt=args.get("user_prompt", "")),
    check_fn=check_moa_requirements,
    # Provider-routed defaults can use profile/auth-backed providers without
    # OpenRouter; check_moa_requirements() handles OpenRouter-only specs.
    requires_env=[],
    is_async=True,
    emoji="🧠",
)
