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

Default Models Used:
- Reference Models: NVIDIA Nemotron 3 Super, Google Gemma 4 31B via OpenRouter free tier
- Aggregator Model: Xiaomi MiMo v2 Pro

Legacy OpenRouter-style model slugs still work when passed explicitly.

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
import asyncio
import datetime
import time
import re
from typing import Dict, Any, List, Optional, Sequence, Tuple

from agent.auxiliary_client import (
    async_call_llm,
    extract_content_or_reasoning,
    resolve_provider_client,
)
from tools.debug_helpers import DebugSession

logger = logging.getLogger(__name__)

ModelRoute = Dict[str, Any]

# Configuration for MoA processing
# Reference models - these generate diverse initial responses in parallel.
DEFAULT_REFERENCE_ROUTES: List[ModelRoute] = [
    {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "label": "nvidia/nemotron-3-super-120b-a12b:free",
    },
    {
        "provider": "openrouter",
        "model": "google/gemma-4-31b-it:free",
        "label": "google/gemma-4-31b-it:free",
    },
]

# Aggregator model - synthesizes reference responses into final output.
DEFAULT_AGGREGATOR_ROUTE: ModelRoute = {
    "provider": "xiaomi",
    "model": "mimo-v2-pro",
    "label": "xiaomi/mimo-v2-pro",
}

# Legacy exports kept for compatibility with existing tests/callers.
REFERENCE_MODELS = [route["label"] for route in DEFAULT_REFERENCE_ROUTES]
AGGREGATOR_MODEL = DEFAULT_AGGREGATOR_ROUTE["label"]

# Temperature settings optimized for MoA performance
REFERENCE_TEMPERATURE = 0.6  # Balanced creativity for diverse perspectives
AGGREGATOR_TEMPERATURE = 0.4  # Focused synthesis for consistency

# Failure handling configuration
MIN_SUCCESSFUL_REFERENCES = 1  # Minimum successful reference models needed to proceed
SELF_DRAFT_SUFFIX = " (self-draft)"
_MOA_FORENSIC_REPAIR_ATTEMPTS = 2
_LEGACY_PAID_REFERENCE_LABELS = {
    "minimax/MiniMax-M2.7-highspeed",
    "deepseek/deepseek-reasoner",
}

# System prompt for the aggregator model (from the research paper)
AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

MOA_FORENSIC_SYSTEM_PROMPT = """You are analyzing a mixture-of-agents run. Return JSON only with this exact top-level structure:
{
  "decision_trace": {
    "model_proposals": {},
    "overlap": [],
    "conflicts": [],
    "final_candidates": [],
    "synthesis_summary": ""
  },
  "aggregator_influence_log": {
    "kept_from_models": {},
    "discarded_or_deprioritized": [],
    "resolution_notes": [],
    "influence_summary": ""
  }
}

Rules:
- JSON only, no markdown fences.
- Use concise strings.
- In model_proposals and kept_from_models, use the exact model labels from the supplied reference outputs as keys.
- Fill arrays/strings only with concrete content from this run.
- Use empty arrays/objects instead of inventing unsupported facts.
- Never echo template placeholders like model_label, concrete proposal, concrete summary, or "...".
- Base the analysis only on the supplied reference outputs and final answer."""

MOA_FORENSIC_REPAIR_PROMPT = """Your previous reply was invalid for one of these reasons:
- it was not valid JSON
- it included extra prose or duplicate JSON objects
- it echoed template placeholders instead of concrete values

Return exactly one valid JSON object only.
No prose. No markdown fences. No duplicate object.
Use the real model keys exactly as supplied."""

_debug = DebugSession("moa_tools", env_var="MOA_TOOLS_DEBUG")


def _load_moa_task_config() -> Dict[str, Any]:
    """Return auxiliary.moa config when available."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}

    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get("moa", {}) if isinstance(aux, dict) else {}
    return task_config if isinstance(task_config, dict) else {}


def _route_label(route: ModelRoute) -> str:
    label = str(route.get("label") or "").strip()
    if label:
        return label
    provider = str(route.get("provider") or "").strip()
    model = str(route.get("model") or "").strip()
    if provider and provider not in ("auto", "openrouter"):
        return f"{provider}/{model}"
    return model


def _infer_route_from_model_name(model_name: str) -> ModelRoute:
    """Infer provider/model pairs for legacy string model specs."""
    raw_name = str(model_name or "").strip()
    if not raw_name:
        raise ValueError("Model route requires a non-empty model name")

    try:
        from hermes_cli.models import detect_provider_for_model

        detected = detect_provider_for_model(raw_name, "")
    except Exception:
        detected = None

    if detected:
        provider, resolved_model = detected
    elif "/" in raw_name:
        # Preserve old OpenRouter-style slugs such as anthropic/claude-*
        provider, resolved_model = "openrouter", raw_name
    else:
        provider, resolved_model = "auto", raw_name

    route: ModelRoute = {
        "provider": provider,
        "model": resolved_model,
    }
    route["label"] = _route_label(route)
    return route


def _normalize_model_route(spec: Any, role: str) -> ModelRoute:
    """Normalize a route spec into {provider, model, base_url, api_key, label}."""
    if isinstance(spec, str):
        return _infer_route_from_model_name(spec)

    if not isinstance(spec, dict):
        raise ValueError(f"{role} must be a string model slug or dict route spec")

    provider = str(spec.get("provider") or "").strip()
    model = str(spec.get("model") or "").strip()
    if not model:
        raise ValueError(f"{role}.model must be set")

    if not provider:
        inferred = _infer_route_from_model_name(model)
        provider = inferred["provider"]
        model = inferred["model"]

    route: ModelRoute = {
        "provider": provider,
        "model": model,
    }

    base_url = str(spec.get("base_url") or "").strip()
    if base_url:
        route["base_url"] = base_url

    api_key = str(spec.get("api_key") or "").strip()
    if api_key:
        route["api_key"] = api_key

    label = str(spec.get("label") or "").strip()
    if label:
        route["label"] = label
    else:
        route["label"] = _route_label(route)

    return route


def _resolve_moa_routes(
    reference_models: Optional[List[Any]] = None,
    aggregator_model: Optional[Any] = None,
) -> tuple[List[ModelRoute], ModelRoute]:
    """Resolve MoA routes from explicit args, config, or built-in defaults."""
    task_config = _load_moa_task_config()

    raw_references = (
        reference_models
        if reference_models is not None
        else task_config.get("reference_models")
    )
    if not isinstance(raw_references, list) or not raw_references:
        raw_references = DEFAULT_REFERENCE_ROUTES
    elif {
        _route_label(_normalize_model_route(spec, "reference_model"))
        for spec in raw_references
    } == _LEGACY_PAID_REFERENCE_LABELS:
        raw_references = DEFAULT_REFERENCE_ROUTES

    raw_aggregator = (
        aggregator_model
        if aggregator_model is not None
        else task_config.get("aggregator_model")
    )
    if raw_aggregator in (None, "", {}):
        raw_aggregator = DEFAULT_AGGREGATOR_ROUTE

    return (
        [
            _normalize_model_route(spec, f"reference_models[{idx}]")
            for idx, spec in enumerate(raw_references)
        ],
        _normalize_model_route(raw_aggregator, "aggregator_model"),
    )


def _route_is_available(route: ModelRoute) -> bool:
    """Return True if Hermes can resolve credentials for this route."""
    try:
        client, _ = resolve_provider_client(
            route.get("provider"),
            route.get("model"),
            explicit_base_url=route.get("base_url"),
            explicit_api_key=route.get("api_key"),
        )
        return client is not None
    except Exception:
        return False


def _build_self_draft_route(aggregator_route: ModelRoute) -> ModelRoute:
    """Return an independent first-pass route for the aggregator model."""
    route = dict(aggregator_route)
    route["label"] = f"{_route_label(aggregator_route)}{SELF_DRAFT_SUFFIX}"
    return route


def _construct_aggregator_prompt(
    system_prompt: str,
    responses: Sequence[Tuple[str, str]],
) -> str:
    """
    Construct the final system prompt for the aggregator including all model responses.
    
    Args:
        system_prompt (str): Base system prompt for aggregation
        responses (Sequence[Tuple[str, str]]): Ordered (model_label, response) pairs
        
    Returns:
        str: Complete system prompt with enumerated responses
    """
    response_text = "\n\n".join(
        f"{idx + 1}. [{model_label}]\n{response}"
        for idx, (model_label, response) in enumerate(responses)
    )
    return f"{system_prompt}\n\n{response_text}"


def _preview_reference_response(content: str, limit: int = 180) -> str:
    normalized = " ".join(str(content or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("No JSON object found in forensic analysis response")

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError("No JSON object found in forensic analysis response")


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _mapping_of_lists(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for key, items in value.items():
        normalized_key = str(key).strip()
        normalized_items = _string_list(items)
        if normalized_key:
            result[normalized_key] = normalized_items
    return result


def _empty_forensic_analysis() -> Dict[str, Any]:
    return {
        "decision_trace": {
            "model_proposals": {},
            "overlap": [],
            "conflicts": [],
            "final_candidates": [],
            "synthesis_summary": "",
        },
        "aggregator_influence_log": {
            "kept_from_models": {},
            "discarded_or_deprioritized": [],
            "resolution_notes": [],
            "influence_summary": "",
        },
    }


def _normalize_forensic_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    decision = payload.get("decision_trace") if isinstance(payload.get("decision_trace"), dict) else {}
    influence = payload.get("aggregator_influence_log") if isinstance(payload.get("aggregator_influence_log"), dict) else {}
    return {
        "decision_trace": {
            "model_proposals": _mapping_of_lists(decision.get("model_proposals")),
            "overlap": _string_list(decision.get("overlap")),
            "conflicts": _string_list(decision.get("conflicts")),
            "final_candidates": _string_list(decision.get("final_candidates")),
            "synthesis_summary": str(decision.get("synthesis_summary") or "").strip(),
        },
        "aggregator_influence_log": {
            "kept_from_models": _mapping_of_lists(influence.get("kept_from_models")),
            "discarded_or_deprioritized": _string_list(influence.get("discarded_or_deprioritized")),
            "resolution_notes": _string_list(influence.get("resolution_notes")),
            "influence_summary": str(influence.get("influence_summary") or "").strip(),
        },
    }


def _forensic_analysis_has_placeholders(payload: Dict[str, Any]) -> bool:
    blocked = {
        "...",
        "model_label",
        "actual_model_name",
        "proposal",
        "concrete proposal",
        "shared idea",
        "concrete shared idea",
        "meaningful disagreement",
        "concrete disagreement",
        "final pick",
        "concrete final pick",
        "short summary",
        "concrete summary",
        "kept point",
        "concrete kept point",
        "discarded point",
        "concrete discarded point",
        "how conflicts were resolved",
        "concrete resolution note",
    }

    def _contains(value: Any) -> bool:
        if isinstance(value, dict):
            return any(_contains(key) or _contains(item) for key, item in value.items())
        if isinstance(value, list):
            return any(_contains(item) for item in value)
        text = str(value or "").strip().lower()
        return bool(text) and text in blocked

    return _contains(payload)


def _forensic_analysis_is_empty(payload: Dict[str, Any]) -> bool:
    decision = payload.get("decision_trace", {})
    influence = payload.get("aggregator_influence_log", {})
    return not any([
        decision.get("model_proposals"),
        decision.get("overlap"),
        decision.get("conflicts"),
        decision.get("final_candidates"),
        decision.get("synthesis_summary"),
        influence.get("kept_from_models"),
        influence.get("discarded_or_deprioritized"),
        influence.get("resolution_notes"),
        influence.get("influence_summary"),
    ])


def _extract_forensic_proposals(text: str, max_items: int = 3) -> List[str]:
    proposals: List[str] = []
    normalized = str(text or "").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        compact = line.lstrip("#>*- ").strip()
        lower = compact.lower()
        if lower.startswith("winner:"):
            compact = compact.split(":", 1)[1].strip()
        elif lower.startswith("bottom line:"):
            compact = compact.split(":", 1)[1].strip()
        elif compact[:2].isdigit() and len(compact) > 3 and compact[2] in (".", ")"):
            compact = compact[3:].strip()
        if compact and compact not in proposals:
            proposals.append(compact)
        if len(proposals) >= max_items:
            break
    return proposals


def _fallback_forensic_analysis(reference_outputs: Dict[str, str], final_response: str) -> Dict[str, Any]:
    model_proposals = {
        model: _extract_forensic_proposals(content)
        for model, content in reference_outputs.items()
        if _extract_forensic_proposals(content)
    }
    final_candidates = _extract_forensic_proposals(final_response)
    overlap = sorted({
        proposal
        for proposals in model_proposals.values()
        for proposal in proposals
        if sum(proposal in other for other in model_proposals.values()) > 1
    })
    kept_from_models = {
        model: [proposal for proposal in proposals if proposal in final_candidates] or proposals[:1]
        for model, proposals in model_proposals.items()
    }
    discarded = sorted({
        proposal
        for proposals in model_proposals.values()
        for proposal in proposals
        if proposal not in final_candidates
    })
    return {
        "decision_trace": {
            "model_proposals": model_proposals,
            "overlap": overlap,
            "conflicts": [] if overlap else ["Reference models emphasized different lead arguments."],
            "final_candidates": final_candidates,
            "synthesis_summary": "Deterministic fallback built from raw reference outputs and the final answer.",
        },
        "aggregator_influence_log": {
            "kept_from_models": kept_from_models,
            "discarded_or_deprioritized": discarded,
            "resolution_notes": [] if not final_candidates else ["MiMo kept the strongest repeated or final-answer-aligned points."],
            "influence_summary": "Fallback influence trace derived from direct model outputs.",
        },
    }


async def _run_reference_model_detailed(
    model: Any,
    user_prompt: str,
    temperature: float = REFERENCE_TEMPERATURE,
    max_tokens: int = 32000,
    max_retries: int = 6,
) -> Dict[str, Any]:
    route = _normalize_model_route(model, "reference_model")
    route_label = _route_label(route)
    started = time.monotonic()

    for attempt in range(max_retries):
        try:
            logger.info(
                "Querying %s via %s (attempt %s/%s)",
                route_label,
                route["provider"],
                attempt + 1,
                max_retries,
            )

            response = await async_call_llm(
                task="moa",
                provider=route.get("provider"),
                model=route.get("model"),
                base_url=route.get("base_url"),
                api_key=route.get("api_key"),
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = extract_content_or_reasoning(response)
            if not content:
                logger.warning(
                    "%s returned empty content (attempt %s/%s), retrying",
                    route_label,
                    attempt + 1,
                    max_retries,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 60))
                    continue
                error_msg = (
                    f"{route_label} returned empty reasoning-only content "
                    f"after {max_retries} attempts"
                )
                logger.error("%s", error_msg)
                return {
                    "model": route_label,
                    "provider": route.get("provider"),
                    "success": False,
                    "content": "",
                    "error": error_msg,
                    "attempts": attempt + 1,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "output_chars": 0,
                }

            logger.info("%s responded (%s characters)", route_label, len(content))
            return {
                "model": route_label,
                "provider": route.get("provider"),
                "success": True,
                "content": content,
                "error": "",
                "attempts": attempt + 1,
                "latency_seconds": round(time.monotonic() - started, 3),
                "output_chars": len(content),
            }

        except Exception as e:
            error_str = str(e)
            if "invalid" in error_str.lower():
                logger.warning("%s invalid request error (attempt %s): %s", route_label, attempt + 1, error_str)
            elif "rate" in error_str.lower() or "limit" in error_str.lower():
                logger.warning("%s rate limit error (attempt %s): %s", route_label, attempt + 1, error_str)
            else:
                logger.warning("%s unknown error (attempt %s): %s", route_label, attempt + 1, error_str)

            if attempt < max_retries - 1:
                sleep_time = min(2 ** (attempt + 1), 60)
                logger.info("Retrying in %ss...", sleep_time)
                await asyncio.sleep(sleep_time)
            else:
                error_msg = f"{route_label} failed after {max_retries} attempts: {error_str}"
                logger.error("%s", error_msg, exc_info=True)
                return {
                    "model": route_label,
                    "provider": route.get("provider"),
                    "success": False,
                    "content": "",
                    "error": error_msg,
                    "attempts": attempt + 1,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "output_chars": 0,
                }


async def _run_reference_model_safe(
    model: Any,
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
    details = await _run_reference_model_detailed(
        model,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    return details["model"], details["content"] or details["error"], bool(details["success"])


async def _run_aggregator_model(
    aggregator_model: Any,
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
    route = _normalize_model_route(aggregator_model, "aggregator_model")
    route_label = _route_label(route)

    logger.info("Running aggregator model: %s via %s", route_label, route["provider"])

    response = await async_call_llm(
        task="moa",
        provider=route.get("provider"),
        model=route.get("model"),
        base_url=route.get("base_url"),
        api_key=route.get("api_key"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = extract_content_or_reasoning(response)

    # Retry once on empty content (reasoning-only response)
    if not content:
        logger.warning("Aggregator returned empty content, retrying once")
        response = await async_call_llm(
            task="moa",
            provider=route.get("provider"),
            model=route.get("model"),
            base_url=route.get("base_url"),
            api_key=route.get("api_key"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = extract_content_or_reasoning(response)

    logger.info("Aggregation complete (%s characters)", len(content))
    return content


async def _run_moa_forensic_analysis(
    aggregator_model: Any,
    user_prompt: str,
    reference_outputs: Dict[str, str],
    final_response: str,
    max_tokens: int = 1400,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    route = _normalize_model_route(aggregator_model, "aggregator_model")
    route_label = _route_label(route)
    started = time.monotonic()
    metrics: Dict[str, Any] = {
        "model": route_label,
        "provider": route.get("provider"),
        "success": False,
        "latency_seconds": 0.0,
        "output_chars": 0,
        "error": "",
    }

    analysis_prompt = (
        f"User prompt:\n{user_prompt}\n\n"
        f"Reference model labels:\n{json.dumps(list(reference_outputs.keys()), ensure_ascii=False)}\n\n"
        f"Reference outputs by model:\n{json.dumps(reference_outputs, ensure_ascii=False, indent=2)}\n\n"
        f"Final aggregated response:\n{final_response}"
    )

    messages = [
        {"role": "system", "content": MOA_FORENSIC_SYSTEM_PROMPT},
        {"role": "user", "content": analysis_prompt},
    ]

    try:
        content = ""
        last_exc: Exception | None = None
        for attempt in range(_MOA_FORENSIC_REPAIR_ATTEMPTS):
            response = await async_call_llm(
                task="moa",
                provider=route.get("provider"),
                model=route.get("model"),
                base_url=route.get("base_url"),
                api_key=route.get("api_key"),
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            content = extract_content_or_reasoning(response) or ""
            try:
                parsed = _normalize_forensic_analysis(_extract_json_object(content))
                if _forensic_analysis_has_placeholders(parsed):
                    raise ValueError("Forensic analysis returned placeholder content")
                if _forensic_analysis_is_empty(parsed):
                    raise ValueError("Forensic analysis returned empty content")
                metrics["latency_seconds"] = round(time.monotonic() - started, 3)
                metrics["output_chars"] = len(content)
                metrics["success"] = True
                return parsed, metrics
            except Exception as exc:
                last_exc = exc
                if attempt == _MOA_FORENSIC_REPAIR_ATTEMPTS - 1:
                    raise
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": MOA_FORENSIC_REPAIR_PROMPT},
                ])
        if last_exc is not None:
            raise last_exc
        raise ValueError("Forensic analysis returned no content")
    except Exception as exc:
        metrics["latency_seconds"] = round(time.monotonic() - started, 3)
        metrics["output_chars"] = len(content or "")
        metrics["error"] = str(exc)
        logger.warning("MoA forensic analysis failed: %s", exc)
        return _fallback_forensic_analysis(reference_outputs, final_response), metrics


async def mixture_of_agents_tool(
    user_prompt: str,
    reference_models: Optional[List[Any]] = None,
    aggregator_model: Optional[Any] = None,
    enable_forensic_analysis: bool = False,
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
                 "failed_models": List[str],
                 "reference_previews": Dict[str, str]
             }
    
    Raises:
        Exception: If MoA processing fails or API key is not set
    """
    start_time = datetime.datetime.now()
    
    ref_routes, agg_route = _resolve_moa_routes(reference_models, aggregator_model)

    debug_call_data = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "reference_models": [_route_label(route) for route in ref_routes],
            "aggregator_model": _route_label(agg_route),
            "reference_temperature": REFERENCE_TEMPERATURE,
            "aggregator_temperature": AGGREGATOR_TEMPERATURE,
            "min_successful_references": MIN_SUCCESSFUL_REFERENCES
        },
        "error": None,
        "success": False,
        "reference_responses_count": 0,
        "failed_models_count": 0,
        "failed_models": [],
        "skipped_reference_models": [],
        "final_response_length": 0,
        "processing_time_seconds": 0,
        "models_used": {}
    }
    failed_models: List[str] = []
    failed_model_errors: Dict[str, str] = {}
    reference_previews: Dict[str, str] = {}
    reference_outputs: Dict[str, str] = {}
    per_model_metrics: Dict[str, Any] = {"reference_models": {}, "aggregator": {}, "forensic_analysis": {}}
    forensic_analysis = _empty_forensic_analysis()
    
    try:
        logger.info("Starting Mixture-of-Agents processing...")
        logger.info("Query: %s", user_prompt[:100])

        available_ref_routes = [route for route in ref_routes if _route_is_available(route)]
        skipped_ref_routes = [
            _route_label(route) for route in ref_routes if not _route_is_available(route)
        ]
        aggregator_available = _route_is_available(agg_route)
        if skipped_ref_routes:
            logger.warning(
                "Skipping unavailable reference models: %s",
                ", ".join(skipped_ref_routes),
            )
        if not available_ref_routes:
            raise ValueError(
                "No available reference models configured for Mixture-of-Agents. "
                "Configure at least one provider key for the current MoA routes."
            )
        if not aggregator_available:
            raise ValueError(
                f"Aggregator model unavailable: {_route_label(agg_route)}. "
                "Configure the provider credentials for the MoA aggregator route."
            )

        self_draft_route = _build_self_draft_route(agg_route)
        external_jobs: List[Tuple[ModelRoute, float]] = [
            (route, REFERENCE_TEMPERATURE) for route in available_ref_routes
        ]

        logger.info(
            "Using %s draft models in MoA-v2 (self-draft + references)",
            len(external_jobs) + 1,
        )

        # Layer 1: Generate MiMo self-draft plus reference responses.
        logger.info("Layer 1: Generating self-draft and reference responses...")
        self_draft_result, *external_results = await asyncio.gather(
            _run_reference_model_detailed(self_draft_route, user_prompt, AGGREGATOR_TEMPERATURE),
            *[
                _run_reference_model_detailed(route, user_prompt, temperature)
                for route, temperature in external_jobs
            ],
        )

        # Separate successful and failed responses
        successful_responses: List[Tuple[str, str]] = []
        external_successes = 0

        for result_row in [self_draft_result, *external_results]:
            model_name = result_row["model"]
            per_model_metrics["reference_models"][model_name] = {
                "provider": result_row.get("provider"),
                "success": bool(result_row.get("success")),
                "attempts": int(result_row.get("attempts") or 0),
                "latency_seconds": result_row.get("latency_seconds"),
                "output_chars": int(result_row.get("output_chars") or 0),
                "error": result_row.get("error") or "",
            }
            if result_row.get("success"):
                content = result_row["content"]
                successful_responses.append((model_name, content))
                reference_outputs[model_name] = content
                reference_previews[model_name] = _preview_reference_response(content)
                if model_name != _route_label(self_draft_route):
                    external_successes += 1
            else:
                failed_models.append(model_name)
                failed_model_errors[model_name] = str(result_row.get("error") or "")
        
        successful_count = len(successful_responses)
        failed_count = len(failed_models)
        
        logger.info("Reference model results: %s successful, %s failed", successful_count, failed_count)
        
        if failed_models:
            logger.warning("Failed models: %s", ', '.join(failed_models))
        
        # MoA-v2 must include at least one successful external reference.
        if external_successes < MIN_SUCCESSFUL_REFERENCES:
            raise ValueError(
                "Insufficient successful external reference models "
                f"({external_successes}/{len(available_ref_routes)}). Need at least "
                f"{MIN_SUCCESSFUL_REFERENCES} successful external response."
            )
        
        debug_call_data["reference_responses_count"] = successful_count
        debug_call_data["failed_models_count"] = failed_count
        debug_call_data["failed_models"] = failed_models
        debug_call_data["skipped_reference_models"] = skipped_ref_routes
        
        # Layer 2: Aggregate responses using the aggregator model
        logger.info("Layer 2: Synthesizing final response...")
        aggregator_system_prompt = _construct_aggregator_prompt(
            AGGREGATOR_SYSTEM_PROMPT, 
            successful_responses
        )
        
        aggregator_started = time.monotonic()
        final_response = await _run_aggregator_model(
            agg_route,
            aggregator_system_prompt,
            user_prompt,
            AGGREGATOR_TEMPERATURE
        )
        per_model_metrics["aggregator"] = {
            "model": _route_label(agg_route),
            "provider": agg_route.get("provider"),
            "success": True,
            "latency_seconds": round(time.monotonic() - aggregator_started, 3),
            "output_chars": len(final_response or ""),
        }

        if enable_forensic_analysis:
            forensic_analysis, forensic_metrics = await _run_moa_forensic_analysis(
                agg_route,
                user_prompt,
                reference_outputs,
                final_response,
            )
        else:
            forensic_analysis = _fallback_forensic_analysis(reference_outputs, final_response)
            forensic_metrics = {
                "model": _route_label(agg_route),
                "provider": agg_route.get("provider"),
                "success": False,
                "latency_seconds": 0.0,
                "output_chars": 0,
                "error": "",
                "skipped": True,
            }
        per_model_metrics["forensic_analysis"] = forensic_metrics
        
        # Calculate processing time
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info("MoA processing completed in %.2f seconds", processing_time)
        
        # Prepare successful response (only final aggregated result, minimal fields)
        result = {
            "success": True,
            "response": final_response,
            "models_used": {
                "reference_models": [model_name for model_name, _ in successful_responses],
                "aggregator_model": _route_label(agg_route)
            },
            "failed_models": failed_models,
            "failed_model_errors": failed_model_errors,
            "reference_previews": reference_previews,
            "reference_outputs": reference_outputs,
            "per_model_metrics": per_model_metrics,
            "decision_trace": forensic_analysis["decision_trace"],
            "aggregator_influence_log": forensic_analysis["aggregator_influence_log"],
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
                "reference_models": [_route_label(route) for route in ref_routes],
                "aggregator_model": _route_label(agg_route)
            },
            "failed_models": failed_models,
            "failed_model_errors": failed_model_errors,
            "reference_previews": reference_previews,
            "reference_outputs": reference_outputs,
            "per_model_metrics": per_model_metrics,
            "decision_trace": forensic_analysis["decision_trace"],
            "aggregator_influence_log": forensic_analysis["aggregator_influence_log"],
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
    ref_routes, agg_route = _resolve_moa_routes()
    return _route_is_available(agg_route) and any(_route_is_available(route) for route in ref_routes)



def get_moa_configuration() -> Dict[str, Any]:
    """
    Get the current MoA configuration settings.
    
    Returns:
        Dict[str, Any]: Dictionary containing all configuration parameters
    """
    ref_routes, agg_route = _resolve_moa_routes()
    return {
        "reference_models": [_route_label(route) for route in ref_routes],
        "reference_routes": ref_routes,
        "aggregator_model": _route_label(agg_route),
        "aggregator_route": agg_route,
        "reference_temperature": REFERENCE_TEMPERATURE,
        "aggregator_temperature": AGGREGATOR_TEMPERATURE,
        "min_successful_references": MIN_SUCCESSFUL_REFERENCES,
        "total_reference_models": len(ref_routes),
        "failure_tolerance": f"{len(ref_routes) - MIN_SUCCESSFUL_REFERENCES}/{len(ref_routes)} models can fail"
    }


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🤖 Mixture-of-Agents Tool Module")
    print("=" * 50)
    
    # Check if API routes are available
    api_available = check_moa_requirements()
    
    if not api_available:
        print("❌ Mixture-of-Agents routes are not fully configured")
        print("Configure the MoA aggregator plus at least one reference provider.")
        exit(1)
    else:
        print("✅ MoA providers available")
    
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
    "description": "Route a hard problem through multiple LLMs collaboratively. By default Hermes uses Xiaomi MiMo v2 Pro as the aggregator with OpenRouter free Nemotron and Gemma reference models. Requires Xiaomi plus OPENROUTER_API_KEY. Use sparingly for genuinely difficult problems: complex math, advanced algorithms, and multi-step analytical reasoning.",
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
    requires_env=["XIAOMI_API_KEY", "OPENROUTER_API_KEY"],
    is_async=True,
    emoji="🧠",
)
