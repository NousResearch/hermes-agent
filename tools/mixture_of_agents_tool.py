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

Models Used (via OpenRouter):
- Reference Models: claude-opus-4.6, gemini-3-pro-preview, gpt-5.4-pro, deepseek-v3.2
- Aggregator Model: claude-opus-4.6 (highest capability for synthesis)

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
import logging
import os
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from tools.openrouter_client import get_async_client as _get_openrouter_client, check_api_key as check_openrouter_api_key
from agent.auxiliary_client import extract_content_or_reasoning
from hermes_cli.config import load_config
from tools.debug_helpers import DebugSession
import sys

logger = logging.getLogger(__name__)

# Configuration for MoA processing
# Reference models - these generate diverse initial responses in parallel.
# Keep this list aligned with current top-tier OpenRouter frontier options.
REFERENCE_MODELS = [
    "anthropic/claude-opus-4.6",
    "google/gemini-2.5-pro",
    "openai/gpt-5.4-pro",
    "deepseek/deepseek-v3.2",
]

# Aggregator model - synthesizes reference responses into final output.
# Prefer the strongest synthesis model in the current OpenRouter lineup.
AGGREGATOR_MODEL = "anthropic/claude-opus-4.6"

# Temperature settings optimized for MoA performance
REFERENCE_TEMPERATURE = 0.6  # Balanced creativity for diverse perspectives
AGGREGATOR_TEMPERATURE = 0.4  # Focused synthesis for consistency

# Failure handling configuration
MIN_SUCCESSFUL_REFERENCES = 1  # Minimum successful reference models needed to proceed

# System prompt for the aggregator model (from the research paper)
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")


def _get_moa_config() -> Dict[str, Any]:
    """Return the ``moa:`` section of config.yaml.

    Why: the MoA tool may be invoked across many sessions in one long-lived
    gateway process; an extra process-lifetime ``@lru_cache`` here pinned the
    FIRST config forever, so a later session with a different ``HERMES_HOME``
    (or an edited config) silently got stale MoA settings. ``load_config()``
    is itself cached on the config file's ``(path, mtime_ns, size)``, so it
    already amortizes the deep-merge AND refreshes when the file or profile
    changes — there is no work to save by wrapping it again, only a staleness
    bug to introduce.
    What: loads config via the project's standard ``load_config()`` and returns
    its ``moa`` dict (empty dict if the section or loader is unavailable).
    Test: monkeypatch ``load_config`` to return a known ``moa`` section and
    assert the returned dict matches; swap ``load_config`` between two calls and
    assert the second call reflects the new value (no stale cache); on loader
    failure assert an empty dict is returned (no raise).
    """
    try:
        return load_config().get("moa", {}) or {}
    except Exception:
        # Config is best-effort here — a load failure must not break MoA, it
        # just falls back to the module-constant defaults.
        return {}


def _resolve(call_arg: Any, config_key: str, default: Any) -> Any:
    """Resolve one MoA setting by precedence: call-arg > moa: config > default.

    Why: every configurable MoA knob follows the same three-tier precedence;
    centralizing it keeps the resolution consistent and the call site small.
    What: returns *call_arg* when it is not None and not an empty list/string;
    otherwise the ``moa:`` config value for *config_key* when that is non-empty;
    otherwise *default* (the module constant).
    Test: assert call-arg wins over a config value; config wins over default
    when call-arg is None; default is used when both are absent/empty.
    """
    if call_arg is not None and call_arg != "" and call_arg != []:
        return call_arg
    cfg_val = _get_moa_config().get(config_key)
    if cfg_val is not None and cfg_val != "" and cfg_val != []:
        return cfg_val
    return default


def _reasoning_extra_body(reasoning_effort: Any) -> Dict[str, Any]:
    """Build the ``extra_body`` reasoning block, or empty when disabled.

    Why: LiteLLM bridges and local models reject the OpenRouter-style
    ``extra_body.reasoning`` payload, so it must be omittable.
    What: returns ``{"extra_body": {"reasoning": {...}}}`` for a truthy effort
    other than ``"none"``; returns ``{}`` otherwise so the caller sends no
    reasoning block at all.
    Test: assert ``""`` and ``"none"`` yield ``{}``; assert ``"xhigh"`` yields a
    dict whose ``extra_body.reasoning.effort == "xhigh"`` and ``enabled`` True.
    """
    effort = str(reasoning_effort or "").strip().lower()
    if not effort or effort == "none":
        return {}
    return {"extra_body": {"reasoning": {"enabled": True, "effort": effort}}}


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
    max_retries: int = 6,
    reasoning_effort: Any = "xhigh",
    base_url: str = "",
    api_key: str = "",
) -> tuple[str, str, bool]:
    """
    Run a single reference model with retry logic and graceful failure handling.

    Args:
        model (str): Model identifier to use
        user_prompt (str): The user's query
        temperature (float): Sampling temperature for response generation
        max_tokens (int): Maximum tokens in response
        max_retries (int): Maximum number of retry attempts
        reasoning_effort (Any): OpenRouter reasoning effort; falsy or "none"
            omits the extra_body reasoning block (for LiteLLM/local models).
        base_url (str): Optional OpenAI-compatible endpoint override.
        api_key (str): Optional API key paired with base_url.

    Returns:
        tuple[str, str, bool]: (model_name, response_content_or_error, success_flag)
    """
    for attempt in range(max_retries):
        try:
            logger.info("Querying %s (attempt %s/%s)", model, attempt + 1, max_retries)

            # Build parameters for the API call.  The reasoning block is omitted
            # entirely for falsy / "none" effort so LiteLLM and local models that
            # reject extra_body.reasoning still work.
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
                **_reasoning_extra_body(reasoning_effort),
            }

            # GPT models (especially gpt-4o-mini) don't support custom temperature values
            # Only include temperature for non-GPT models
            if not model.lower().startswith('gpt-'):
                api_params["temperature"] = temperature

            response = await _get_openrouter_client(base_url=base_url, api_key=api_key).chat.completions.create(**api_params)
            
            content = extract_content_or_reasoning(response)
            if not content:
                # Reasoning-only / empty response. Retry while attempts remain;
                # on the FINAL attempt fail explicitly instead of falling
                # through to the success path — a blank "success" here makes the
                # aggregator ingest an empty reference (false-success bug).
                logger.warning("%s returned empty content (attempt %s/%s), retrying", model, attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
                return model, f"{model} returned empty content after {max_retries} attempts", False
            logger.info("%s responded (%s characters)", model, len(content))
            return model, content, True
            
        except Exception as e:
            error_str = str(e)
            # Keep retry-path logging concise; full tracebacks are reserved for
            # terminal failure paths so long-running MoA retries don't flood logs.
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", model, attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", model, attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", model, attempt + 1, error_str)

            if attempt < max_retries - 1:
                # Exponential backoff for rate limiting: 2s, 4s, 8s, 16s, 32s, 60s
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{model} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return model, error_msg, False

    # Loop-exhaustion guard: every path inside the loop returns, but the loop
    # body never runs when max_retries <= 0. Without this the function could
    # implicitly return None, violating the -> tuple[str, str, bool] contract
    # (ty: invalid-return-type).
    return model, f"{model} returned no response after {max_retries} attempts", False


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: Optional[int] = None,
    aggregator_model: str = AGGREGATOR_MODEL,
    reasoning_effort: Any = "xhigh",
    base_url: str = "",
    api_key: str = "",
) -> str:
    """
    Run the aggregator model to synthesize the final response.

    Args:
        system_prompt (str): System prompt with all reference responses
        user_prompt (str): Original user query
        temperature (float): Focused temperature for consistent aggregation
        max_tokens (int): Maximum tokens in final response
        aggregator_model (str): Model used to synthesize the final response.
        reasoning_effort (Any): OpenRouter reasoning effort; falsy or "none"
            omits the extra_body reasoning block (for LiteLLM/local models).
        base_url (str): Optional OpenAI-compatible endpoint override.
        api_key (str): Optional API key paired with base_url.

    Returns:
        str: Synthesized final response
    """
    logger.info("Running aggregator model: %s", aggregator_model)

    # Build parameters for the API call.  The reasoning block is omitted
    # entirely for falsy / "none" effort so LiteLLM and local models work.
    api_params = {
        "model": aggregator_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        **_reasoning_extra_body(reasoning_effort),
    }

    # GPT models (especially gpt-4o-mini) don't support custom temperature values
    # Only include temperature for non-GPT models
    if not aggregator_model.lower().startswith('gpt-'):
        api_params["temperature"] = temperature

    response = await _get_openrouter_client(base_url=base_url, api_key=api_key).chat.completions.create(**api_params)

    content = extract_content_or_reasoning(response)

    # Retry once on empty content (reasoning-only response)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await _get_openrouter_client(base_url=base_url, api_key=api_key).chat.completions.create(**api_params)
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%s characters)", len(content))
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
    
    Settings resolve by precedence: explicit call-arg > ``moa:`` section in
    config.yaml > module-constant default. An absent/empty ``moa:`` section
    therefore preserves the original hardcoded behavior exactly.

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

    # Resolve every setting once by precedence (call-arg > moa: config > default).
    # An absent/empty moa: section leaves these equal to the module constants,
    # so behavior is identical to before when nothing is configured.
    ref_models = _resolve(reference_models, "reference_models", REFERENCE_MODELS)
    agg_model = _resolve(aggregator_model, "aggregator_model", AGGREGATOR_MODEL)
    ref_temperature = _resolve(None, "reference_temperature", REFERENCE_TEMPERATURE)
    agg_temperature = _resolve(None, "aggregator_temperature", AGGREGATOR_TEMPERATURE)
    min_successful = _resolve(None, "min_successful_references", MIN_SUCCESSFUL_REFERENCES)
    reasoning_effort = _resolve(None, "reasoning_effort", "xhigh")
    moa_base_url = _resolve(None, "base_url", "")
    moa_api_key = _resolve(None, "api_key", "")

    # Annotated Dict[str, Any] because this accumulator is later assigned
    # str / int / float / list values for keys whose literal-inferred type
    # would otherwise be a narrower union (ty: invalid-assignment).
    debug_call_data: Dict[str, Any] = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": ref_models,
            "aggregator_model": agg_model,
            "reference_temperature": ref_temperature,
            "aggregator_temperature": agg_temperature,
            "min_successful_references": min_successful
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

        # Validate API key availability. Only required on the default OpenRouter
        # path — a custom base_url (LiteLLM / local) carries its own auth (or
        # none), so don't block it on OPENROUTER_API_KEY.
        if not moa_base_url and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        logger.info("Using %s reference models in 2-layer MoA architecture", len(ref_models))

        # Layer 1: Generate diverse responses from reference models (with failure handling)
        logger.info("Layer 1: Generating reference responses...")
        model_results = await asyncio.gather(*[
            _run_reference_model_safe(
                model, user_prompt, ref_temperature,
                reasoning_effort=reasoning_effort,
                base_url=moa_base_url, api_key=moa_api_key,
            )
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
        if successful_count < min_successful:
            raise ValueError(f"Insufficient successful reference models ({successful_count}/{len(ref_models)}). Need at least {min_successful} successful responses.")
        
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
            agg_temperature,
            aggregator_model=agg_model,
            reasoning_effort=reasoning_effort,
            base_url=moa_base_url,
            api_key=moa_api_key,
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
                "reference_models": ref_models,
                "aggregator_model": agg_model
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
                "reference_models": ref_models,
                "aggregator_model": agg_model
            },
            "error": error_msg
        }
        
        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()
        
        return json.dumps(result, indent=2, ensure_ascii=False)


def check_moa_requirements() -> bool:
    """Report whether the MoA tool is usable in the current environment.

    Why: the tool has two auth paths. The DEFAULT path talks to OpenRouter and
    needs ``OPENROUTER_API_KEY``. But a local / LiteLLM deploy sets
    ``moa.base_url`` to its own OpenAI-compatible endpoint, which carries its
    own auth (or none) — gating that case on ``OPENROUTER_API_KEY`` made the
    tool silently unavailable for the PR's headline use case. So a configured
    ``moa.base_url`` is sufficient on its own.
    What: returns True when an OpenRouter key is present OR a ``moa.base_url`` is
    configured (read via the merged ``moa:`` config); False otherwise.
    Test: with no ``OPENROUTER_API_KEY`` in env and ``moa.base_url`` set in
    config, assert this returns True; with neither, assert False.
    """
    # _resolve reads the merged moa: config (config_path/mtime-cached via
    # load_config); cheap enough for registry.get_definitions() probing, which
    # additionally TTL-caches check_fn results. No heavy work at definition time.
    return check_openrouter_api_key() or bool(_resolve(None, "base_url", ""))



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
    api_available = check_openrouter_api_key()
    
    if not api_available:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY='your-key-here'")
        print("Get API key at: https://openrouter.ai/")
        sys.exit(1)
    else:
        print("✅ OpenRouter API key found")
    
    print("🛠️  MoA tools ready for use!")
    
    # Show current configuration
    config = get_moa_configuration()
    print("\n⚙️  Current Configuration:")
    print(f"  🤖 Reference models ({len(config['reference_models'])}): {', '.join(config['reference_models'])}")
    print(f"  🧠 Aggregator model: {config['aggregator_model']}")
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

# MOA_SCHEMA is intentionally left exposing only ``user_prompt`` — surfacing
# reference_models / aggregator_model as model-facing schema params is a
# separate, deferred change. The handler still forwards them so any caller
# that supplies them (e.g. config-driven or programmatic) reaches the function
# instead of being silently dropped.
registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(
        user_prompt=args.get("user_prompt", ""),
        reference_models=args.get("reference_models"),
        aggregator_model=args.get("aggregator_model"),
    ),
    check_fn=check_moa_requirements,
    # requires_env is DISPLAY/diagnostic metadata for the default OpenRouter
    # path only. The actual availability gate is check_fn above, which also
    # accepts a configured moa.base_url (LiteLLM / local) with no OpenRouter
    # key — so OPENROUTER_API_KEY is the default-path requirement, optional when
    # moa.base_url is set.
    requires_env=["OPENROUTER_API_KEY"],
    is_async=True,
    emoji="🧠",
)
