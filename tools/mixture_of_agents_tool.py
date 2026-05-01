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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict
from tools.openrouter_client import get_async_client as _get_openrouter_client, check_api_key as check_openrouter_api_key
from agent.auxiliary_client import extract_content_or_reasoning, resolve_provider_client
from tools.debug_helpers import DebugSession

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


class ModelRoute(TypedDict, total=False):
    """A single MoA route — provider+model plus optional per-route knobs."""

    provider: str
    model: str
    label: str
    temperature: float
    max_tokens: int
    omit_temperature: bool
    extra_body: Dict[str, Any]


_OPENROUTER_DEFAULT_EXTRA_BODY: Dict[str, Any] = {
    "reasoning": {"enabled": True, "effort": "xhigh"}
}

_route_client_cache: Dict[Tuple[str, str], Tuple[Any, str]] = {}


def _default_route_label(provider: str, model: str) -> str:
    if provider == "openrouter":
        return model
    return f"{provider}:{model}"


def _normalize_route(spec: Any, *, default_provider: str = "openrouter") -> ModelRoute:
    """Coerce a route spec into a validated ModelRoute.

    Accepts either a bare string (legacy: assumed OpenRouter slug) or a mapping
    with at least ``provider`` and ``model``.
    """
    if isinstance(spec, str):
        if not spec.strip():
            raise ValueError("MoA route string must not be empty")
        return {
            "provider": default_provider,
            "model": spec,
            "label": _default_route_label(default_provider, spec),
        }
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"MoA route must be a string or mapping, got {type(spec).__name__}: {spec!r}"
        )
    provider = (spec.get("provider") or "").strip()
    model = (spec.get("model") or "").strip()
    if not provider:
        raise ValueError(f"MoA route missing required 'provider': {dict(spec)!r}")
    if not model:
        raise ValueError(f"MoA route missing required 'model': {dict(spec)!r}")
    route: ModelRoute = {"provider": provider, "model": model}
    label = spec.get("label")
    route["label"] = str(label) if label else _default_route_label(provider, model)
    if "temperature" in spec:
        route["temperature"] = float(spec["temperature"])
    if "max_tokens" in spec:
        route["max_tokens"] = int(spec["max_tokens"])
    if "omit_temperature" in spec:
        route["omit_temperature"] = bool(spec["omit_temperature"])
    if "extra_body" in spec and spec["extra_body"] is not None:
        if not isinstance(spec["extra_body"], Mapping):
            raise ValueError(
                f"MoA route 'extra_body' must be a mapping if set: {dict(spec)!r}"
            )
        route["extra_body"] = dict(spec["extra_body"])
    return route


def _resolve_extra_body(route: ModelRoute) -> Optional[Dict[str, Any]]:
    if "extra_body" in route:
        return dict(route["extra_body"]) if route["extra_body"] else None
    if route["provider"] == "openrouter":
        return dict(_OPENROUTER_DEFAULT_EXTRA_BODY)
    return None


def _should_send_temperature(route: ModelRoute) -> bool:
    if route.get("omit_temperature"):
        return False
    # legacy carve-out: gpt-* models on OpenRouter reject custom temperatures.
    # Other providers route through their own normalization.
    if route["provider"] == "openrouter" and route["model"].lower().startswith("gpt-"):
        return False
    return True


def _resolve_client_for_route(route: ModelRoute) -> Tuple[Any, str]:
    """Resolve an async client for a route, caching one client per provider+model."""
    key = (route["provider"], route["model"])
    cached = _route_client_cache.get(key)
    if cached is not None:
        return cached
    if route["provider"] == "openrouter":
        client = _get_openrouter_client()
        resolved_model = route["model"]
    else:
        client, resolved_model = resolve_provider_client(
            route["provider"], model=route["model"], async_mode=True
        )
    if client is None:
        raise RuntimeError(
            f"MoA: provider {route['provider']!r} is not configured "
            f"(model={route['model']!r}). Run `hermes auth login {route['provider']}` "
            f"or set the appropriate API key."
        )
    pair = (client, resolved_model or route["model"])
    _route_client_cache[key] = pair
    return pair


def _reset_route_cache() -> None:
    """Drop cached clients. Called between processes / by tests."""
    _route_client_cache.clear()


def _load_moa_config() -> Dict[str, Any]:
    """Load the optional ``moa:`` section from ``~/.hermes/config.yaml``.

    Returns an empty dict on missing section or load failure — callers fall
    back to module-level defaults.
    """
    try:
        from hermes_cli.config import load_config
    except ImportError:
        return {}
    try:
        cfg = load_config()
    except Exception:
        logger.debug("MoA: failed to load hermes config", exc_info=True)
        return {}
    section = cfg.get("moa") or {}
    if not isinstance(section, Mapping):
        logger.warning("MoA: config 'moa' section must be a mapping, ignoring")
        return {}
    return dict(section)


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
    route_or_model: Any,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 6,
) -> tuple[str, str, bool]:
    """Run a single reference route with retry logic and graceful failure handling.

    Accepts either a bare model slug (legacy OpenRouter behaviour) or a
    full route dict (provider, model, optional per-route knobs).

    Returns:
        ``(label, response_content_or_error, success_flag)``
    """
    try:
        route = _normalize_route(route_or_model)
    except ValueError as exc:
        # malformed route — surface immediately, no retries
        label = str(route_or_model)
        logger.error("%s: %s", label, exc, exc_info=True)
        return label, str(exc), False

    label = route["label"]
    eff_temperature = route.get("temperature", temperature)
    eff_max_tokens = route.get("max_tokens", max_tokens)
    extra_body = _resolve_extra_body(route)

    for attempt in range(max_retries):
        try:
            logger.info("Querying %s (attempt %s/%s)", label, attempt + 1, max_retries)
            try:
                client, _ = _resolve_client_for_route(route)
            except RuntimeError as exc:
                # provider not configured — fail fast, retries won't help
                logger.error("%s", exc, exc_info=True)
                return label, str(exc), False

            api_params: Dict[str, Any] = {
                "model": route["model"],
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": eff_max_tokens,
            }
            if _should_send_temperature(route):
                api_params["temperature"] = eff_temperature
            if extra_body:
                api_params["extra_body"] = extra_body

            response = await client.chat.completions.create(**api_params)

            content = extract_content_or_reasoning(response)
            if not content:
                logger.warning(
                    "%s returned empty content (attempt %s/%s), retrying",
                    label, attempt + 1, max_retries,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
            logger.info("%s responded (%s characters)", label, len(content))
            return label, content, True

        except Exception as e:
            error_str = str(e)
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", label, attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", label, attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", label, attempt + 1, error_str)

            if attempt < max_retries - 1:
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{label} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return label, error_msg, False


async def _run_aggregator_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = AGGREGATOR_TEMPERATURE,
    max_tokens: Optional[int] = None,
    route: Optional[ModelRoute] = None,
) -> str:
    """Run the aggregator route to synthesize the final response.

    When ``route`` is None, falls back to the module-level ``AGGREGATOR_MODEL``
    (legacy OpenRouter slug) for backward compatibility.
    """
    if route is None:
        route = _normalize_route(AGGREGATOR_MODEL)

    label = route["label"]
    eff_temperature = route.get("temperature", temperature)
    eff_max_tokens = route.get("max_tokens", max_tokens)
    extra_body = _resolve_extra_body(route)

    logger.info("Running aggregator model: %s", label)

    client, _ = _resolve_client_for_route(route)

    api_params: Dict[str, Any] = {
        "model": route["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": eff_max_tokens,
    }
    if _should_send_temperature(route):
        api_params["temperature"] = eff_temperature
    if extra_body:
        api_params["extra_body"] = extra_body

    response = await client.chat.completions.create(**api_params)
    content = extract_content_or_reasoning(response)

    # one retry on empty content (reasoning-only response)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await client.chat.completions.create(**api_params)
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
    moa_cfg = _load_moa_config()

    raw_refs = (
        list(reference_models)
        if reference_models is not None
        else moa_cfg.get("references")
        or REFERENCE_MODELS
    )
    raw_agg = (
        aggregator_model
        if aggregator_model is not None
        else moa_cfg.get("aggregator") or AGGREGATOR_MODEL
    )
    ref_temperature = float(moa_cfg.get("reference_temperature", REFERENCE_TEMPERATURE))
    agg_temperature = float(moa_cfg.get("aggregator_temperature", AGGREGATOR_TEMPERATURE))
    min_refs = int(moa_cfg.get("min_successful_references", MIN_SUCCESSFUL_REFERENCES))

    debug_call_data: Dict[str, Any] = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": raw_refs,
            "aggregator_model": raw_agg,
            "reference_temperature": ref_temperature,
            "aggregator_temperature": agg_temperature,
            "min_successful_references": min_refs,
        },
        "error": None,
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "models_used": {},
    }

    try:
        try:
            ref_routes = [_normalize_route(spec) for spec in raw_refs]
            agg_route = _normalize_route(raw_agg)
        except ValueError as exc:
            raise ValueError(f"MoA configuration error: {exc}") from exc

        logger.info("Starting Mixture-of-Agents processing...")
        logger.info("Query: %s", user_prompt[:100])

        # Legacy gate: if every route uses OpenRouter we still require the env var,
        # so existing setups behave exactly as before.
        if all(r["provider"] == "openrouter" for r in [*ref_routes, agg_route]):
            if not os.getenv("OPENROUTER_API_KEY"):
                raise ValueError("OPENROUTER_API_KEY environment variable not set")

        ref_labels = [r["label"] for r in ref_routes]
        logger.info("Using %s reference routes in 2-layer MoA architecture", len(ref_routes))

        logger.info("Layer 1: Generating reference responses...")
        model_results = await asyncio.gather(*[
            _run_reference_model_safe(route, user_prompt, ref_temperature)
            for route in ref_routes
        ])

        successful_responses: List[str] = []
        failed_models: List[str] = []
        for label, content, success in model_results:
            if success:
                successful_responses.append(content)
            else:
                failed_models.append(label)

        successful_count = len(successful_responses)
        failed_count = len(failed_models)

        logger.info("Reference model results: %s successful, %s failed", successful_count, failed_count)
        if failed_models:
            logger.warning("Failed models: %s", ", ".join(failed_models))

        if successful_count < min_refs:
            raise ValueError(
                f"Insufficient successful reference models ({successful_count}/{len(ref_routes)}). "
                f"Need at least {min_refs} successful responses."
            )

        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models

        logger.info("Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT,
            successful_responses,
        )

        final_response = await _run_aggregator_model(
            aggregator_system_prompt,
            user_prompt,
            agg_temperature,
            route=agg_route,
        )

        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        logger.info("MoA processing completed in %.2f seconds", processing_time)

        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": ref_labels,
                "aggregator_model": agg_route["label"],
            },
        }

        debug_call_data["success"] = True
        debug_call_data["final_response_length"] = len(final_response)
        debug_call_data["processing_time_seconds"] = processing_time
        debug_call_data["models_used"] = result["models_used"]

        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        error_msg = f"Error in MoA processing: {str(e)}"
        logger.error("%s", error_msg, exc_info=True)

        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Best-effort labels for the error path: try to normalize, fall back to raw input
        try:
            err_ref_labels = [_normalize_route(spec)["label"] for spec in raw_refs]
        except Exception:
            err_ref_labels = [str(spec) for spec in raw_refs]
        try:
            err_agg_label = _normalize_route(raw_agg)["label"]
        except Exception:
            err_agg_label = str(raw_agg)

        result = {
            "success": False,
            "response": "MoA processing failed. Please try again or use a single model for this query.",
            "models_used": {
                "reference_models": err_ref_labels,
                "aggregator_model": err_agg_label,
            },
            "error": error_msg,
        }

        debug_call_data["error"] = error_msg
        debug_call_data["processing_time_seconds"] = processing_time
        _debug.log_call("mixture_of_agents_tool", debug_call_data)
        _debug.save()

        return json.dumps(result, indent=2, ensure_ascii=False)


def check_moa_requirements() -> bool:
    """Return True if MoA can run with the current configuration.

    With no ``moa:`` section in config, requires ``OPENROUTER_API_KEY`` (legacy).
    With a configured section, requires that at least one provider across the
    references and aggregator is reachable via ``resolve_provider_client``.
    """
    moa_cfg = _load_moa_config()
    raw_refs = moa_cfg.get("references")
    raw_agg = moa_cfg.get("aggregator")
    if not raw_refs and not raw_agg:
        return check_openrouter_api_key()

    try:
        routes: List[ModelRoute] = []
        if raw_refs:
            routes.extend(_normalize_route(spec) for spec in raw_refs)
        if raw_agg:
            routes.append(_normalize_route(raw_agg))
    except ValueError:
        return False

    seen_providers = {r["provider"] for r in routes}
    for provider in seen_providers:
        if provider == "openrouter":
            if check_openrouter_api_key():
                return True
            continue
        try:
            client, _ = resolve_provider_client(provider, async_mode=False)
        except Exception:
            continue
        if client is not None:
            return True
    return False


def get_moa_configuration() -> Dict[str, Any]:
    """Return the effective MoA configuration (config-driven if present)."""
    moa_cfg = _load_moa_config()
    raw_refs = moa_cfg.get("references") or REFERENCE_MODELS
    raw_agg = moa_cfg.get("aggregator") or AGGREGATOR_MODEL
    try:
        ref_routes = [_normalize_route(spec) for spec in raw_refs]
        agg_route = _normalize_route(raw_agg)
    except ValueError:
        ref_routes = [_normalize_route(spec) for spec in REFERENCE_MODELS]
        agg_route = _normalize_route(AGGREGATOR_MODEL)

    ref_temp = float(moa_cfg.get("reference_temperature", REFERENCE_TEMPERATURE))
    agg_temp = float(moa_cfg.get("aggregator_temperature", AGGREGATOR_TEMPERATURE))
    min_refs = int(moa_cfg.get("min_successful_references", MIN_SUCCESSFUL_REFERENCES))
    total = len(ref_routes)

    return {
        "reference_models": [r["label"] for r in ref_routes],
        "aggregator_model": agg_route["label"],
        "reference_temperature": ref_temp,
        "aggregator_temperature": agg_temp,
        "min_successful_references": min_refs,
        "total_reference_models": total,
        "failure_tolerance": f"{max(0, total - min_refs)}/{total} models can fail",
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
        exit(1)
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

registry.register(
    name="mixture_of_agents",
    toolset="moa",
    schema=MOA_SCHEMA,
    handler=lambda args, **kw: mixture_of_agents_tool(user_prompt=args.get("user_prompt", "")),
    check_fn=check_moa_requirements,
    requires_env=["OPENROUTER_API_KEY"],
    is_async=True,
    emoji="🧠",
)
