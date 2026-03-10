"""Council Tool -- Adversarial multi-perspective deliberation tools.

Three tools registered in one file (following skills_tool.py pattern):
  - council_query:    Submit a question for 5-persona deliberation
  - council_evaluate: Evaluate content through the council
  - council_gate:     Quick safety review before high-stakes actions

Each tool calls LLM via the OpenAI-compatible API (openai library).
Uses OPENROUTER_API_KEY, OPENAI_API_KEY, or NOUS_API_KEY from env.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from tools.council_personas import (
    CouncilVerdict,
    PersonaResponse,
    DEFAULT_PERSONAS,
    get_persona,
    list_personas,
    load_custom_personas,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

# =============================================================================
# LLM client configuration
# =============================================================================

_DEFAULT_MODEL = "nousresearch/hermes-3-llama-3.1-70b"


def _get_api_config() -> Dict[str, str]:
    """Resolve API key and base URL from environment."""
    if os.getenv("OPENROUTER_API_KEY"):
        return {
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
        }
    if os.getenv("NOUS_API_KEY"):
        return {
            "api_key": os.environ["NOUS_API_KEY"],
            "base_url": "https://inference-api.nousresearch.com/v1",
        }
    if os.getenv("OPENAI_API_KEY"):
        return {
            "api_key": os.environ["OPENAI_API_KEY"],
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        }
    return {}


def _get_model() -> str:
    """Get the council model from env or use default."""
    return os.getenv("COUNCIL_MODEL", _DEFAULT_MODEL)


async def _llm_call(system_prompt: str, user_message: str, model: str = None) -> str:
    """Make a single LLM call via the OpenAI-compatible API."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return json.dumps({"error": "openai library not installed"})

    config = _get_api_config()
    if not config:
        return "[Error: No API key found. Set OPENROUTER_API_KEY, NOUS_API_KEY, or OPENAI_API_KEY]"

    client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )

    try:
        response = await client.chat.completions.create(
            model=model or _get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"[Error: LLM call failed: {e}]"


# =============================================================================
# Response parsing
# =============================================================================


def _parse_confidence(text: str) -> float:
    """Extract confidence value from persona response text."""
    patterns = [
        r"CONFIDENCE:\s*([\d.]+)",
        r"confidence[:\s]+([\d.]+)",
        r"(\d+)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
    return 0.5  # default


def _parse_dissent(text: str) -> bool:
    """Extract dissent flag from persona response text."""
    match = re.search(r"DISSENT:\s*(true|false)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return False


def _parse_key_points(text: str) -> List[str]:
    """Extract bullet points from persona response text."""
    points = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "* ", "  - ", "  * ")):
            point = line.lstrip("-* ").strip()
            if len(point) > 10:
                points.append(point)
    return points[:10]  # cap at 10


def _extract_sources(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
    return list(set(re.findall(url_pattern, text)))


def _build_persona_response(persona_name: str, raw_text: str) -> PersonaResponse:
    """Parse raw LLM output into a structured PersonaResponse."""
    return PersonaResponse(
        persona_name=persona_name,
        content=raw_text,
        confidence=_parse_confidence(raw_text),
        dissents=_parse_dissent(raw_text),
        key_points=_parse_key_points(raw_text),
        sources=_extract_sources(raw_text),
    )


# =============================================================================
# DPO pair extraction
# =============================================================================


def _extract_dpo_pairs(
    question: str, responses: Dict[str, PersonaResponse]
) -> List[Dict]:
    """Extract DPO preference pairs from council responses.

    The Arbiter's synthesis represents the "chosen" response.
    The lowest-confidence non-Arbiter persona that was overruled
    represents the "rejected" response.
    """
    pairs = []
    non_arbiter = {
        k: v for k, v in responses.items() if k != "arbiter" and v.content
    }
    if not non_arbiter:
        return pairs

    # Sort by confidence (ascending) -- lowest confidence = most likely rejected
    sorted_personas = sorted(non_arbiter.values(), key=lambda r: r.confidence)

    arbiter = responses.get("arbiter")
    if not arbiter or not arbiter.content:
        return pairs

    # Pair 1: Arbiter (chosen) vs lowest-confidence dissenter (rejected)
    dissenters = [r for r in sorted_personas if r.dissents]
    if dissenters:
        pairs.append({
            "question": question,
            "chosen": arbiter.content,
            "rejected": dissenters[0].content,
            "confidence": arbiter.confidence,
            "source": "council_evaluation",
            "chosen_persona": "arbiter",
            "rejected_persona": dissenters[0].persona_name,
        })

    # Pair 2: Highest-confidence aligned vs lowest-confidence persona
    if len(sorted_personas) >= 2:
        aligned = [r for r in sorted_personas if not r.dissents]
        if aligned and sorted_personas[0].confidence < aligned[-1].confidence - 0.2:
            pairs.append({
                "question": question,
                "chosen": aligned[-1].content,
                "rejected": sorted_personas[0].content,
                "confidence": aligned[-1].confidence,
                "source": "council_evaluation",
                "chosen_persona": aligned[-1].persona_name,
                "rejected_persona": sorted_personas[0].persona_name,
            })

    return pairs


# =============================================================================
# Core deliberation logic
# =============================================================================


async def _run_council(
    question: str,
    context: str = "",
    persona_names: List[str] = None,
    evidence_search: bool = True,
    model: str = None,
) -> CouncilVerdict:
    """Run the full council deliberation.

    1. Resolve personas (default 5 or user-specified subset)
    2. Run 4 non-Arbiter personas in parallel
    3. Collect responses, detect conflicts
    4. Run Arbiter with all responses as context
    5. Score confidence, extract DPO pairs
    """
    all_personas = load_custom_personas()
    if persona_names:
        selected = {
            name.lower(): all_personas[name.lower()]
            for name in persona_names
            if name.lower() in all_personas
        }
    else:
        selected = dict(all_personas)

    # Separate Arbiter from deliberators
    arbiter_persona = selected.pop("arbiter", DEFAULT_PERSONAS["arbiter"])
    deliberators = selected

    # Build user message with context
    user_msg = f"Question: {question}"
    if context:
        user_msg += f"\n\nContext:\n{context}"
    if evidence_search:
        user_msg += (
            "\n\nNote: If you have access to web search, use it to find "
            "supporting evidence or counter-evidence for your analysis."
        )

    # Run all deliberators in parallel
    async def _run_one(persona):
        raw = await _llm_call(persona.system_prompt, user_msg, model=model)
        return _build_persona_response(persona.name, raw)

    tasks = [_run_one(p) for p in deliberators.values()]
    deliberator_responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect valid responses
    responses: Dict[str, PersonaResponse] = {}
    for resp in deliberator_responses:
        if isinstance(resp, PersonaResponse):
            responses[resp.persona_name] = resp
        elif isinstance(resp, Exception):
            logger.error("Persona failed: %s", resp)

    # Detect conflicts (confidence spread > 0.3)
    confidences = [r.confidence for r in responses.values()]
    conflict_detected = False
    if len(confidences) >= 2:
        conflict_detected = (max(confidences) - min(confidences)) > 0.3

    # Build Arbiter context from all responses
    arbiter_context = f"Question: {question}\n\n"
    if context:
        arbiter_context += f"Original Context:\n{context}\n\n"
    arbiter_context += "=== COUNCIL DELIBERATION ===\n\n"
    for name, resp in responses.items():
        arbiter_context += f"--- {name.upper()} ({resp.confidence:.0%} confidence) ---\n"
        arbiter_context += f"{resp.content}\n\n"
    if conflict_detected:
        arbiter_context += (
            "\n[NOTE: Significant disagreement detected among council members. "
            "Pay special attention to reconciling conflicting views.]\n"
        )

    # Run Arbiter
    arbiter_raw = await _llm_call(
        arbiter_persona.system_prompt, arbiter_context, model=model
    )
    arbiter_response = _build_persona_response("arbiter", arbiter_raw)
    responses["arbiter"] = arbiter_response

    # Aggregate sources
    all_sources = []
    for resp in responses.values():
        all_sources.extend(resp.sources)
    all_sources = list(set(all_sources))

    # Compute overall confidence (weighted by Arbiter's assessment)
    confidence_score = int(arbiter_response.confidence * 100)

    # Extract DPO pairs
    dpo_pairs = _extract_dpo_pairs(question, responses)

    return CouncilVerdict(
        question=question,
        responses=responses,
        arbiter_synthesis=arbiter_raw,
        confidence_score=confidence_score,
        conflict_detected=conflict_detected,
        dpo_pairs=dpo_pairs,
        sources=all_sources,
    )


# =============================================================================
# Tool 1: council_query
# =============================================================================

COUNCIL_QUERY_SCHEMA = {
    "name": "council_query",
    "description": (
        "Submit a question to the adversarial council for multi-perspective deliberation. "
        "Five personas (Advocate, Skeptic, Oracle, Contrarian, Arbiter) debate the question. "
        "Returns structured verdict with confidence score and evidence links."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to deliberate on",
            },
            "context": {
                "type": "string",
                "description": "Optional context or prior research to inform the debate",
            },
            "personas": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional: specific persona names to include "
                    "(default: all 5 -- advocate, skeptic, oracle, contrarian, arbiter)"
                ),
            },
            "evidence_search": {
                "type": "boolean",
                "description": "If true, personas are encouraged to cite evidence (default: true)",
            },
        },
        "required": ["question"],
    },
}


async def council_query_handler(args: dict, **kwargs) -> str:
    """Handle council_query tool calls."""
    question = args.get("question", "")
    if not question:
        return json.dumps({"error": "question is required"})

    context = args.get("context", "")
    persona_names = args.get("personas")
    evidence_search = args.get("evidence_search", True)

    verdict = await _run_council(
        question=question,
        context=context,
        persona_names=persona_names,
        evidence_search=evidence_search,
    )

    # Serialize to JSON
    result = {
        "success": True,
        "question": verdict.question,
        "confidence_score": verdict.confidence_score,
        "conflict_detected": verdict.conflict_detected,
        "arbiter_synthesis": verdict.arbiter_synthesis,
        "persona_responses": {
            name: {
                "confidence": resp.confidence,
                "dissents": resp.dissents,
                "key_points": resp.key_points,
                "content": resp.content[:2000],  # truncate for token efficiency
                "sources": resp.sources,
            }
            for name, resp in verdict.responses.items()
        },
        "dpo_pairs": verdict.dpo_pairs,
        "sources": verdict.sources,
        "available_personas": list_personas(),
    }
    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# Tool 2: council_evaluate
# =============================================================================

COUNCIL_EVALUATE_SCHEMA = {
    "name": "council_evaluate",
    "description": (
        "Evaluate any content through the adversarial council. "
        "Use this to assess quality of research, analysis, or agent output. "
        "Returns confidence score (0-100) and structured feedback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to evaluate",
            },
            "question": {
                "type": "string",
                "description": "The original question/task the content addresses",
            },
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Evaluation criteria "
                    "(default: accuracy, depth, falsifiability, evidence)"
                ),
            },
        },
        "required": ["content"],
    },
}


async def council_evaluate_handler(args: dict, **kwargs) -> str:
    """Handle council_evaluate tool calls."""
    content = args.get("content", "")
    if not content:
        return json.dumps({"error": "content is required"})

    question = args.get("question", "Evaluate the quality of this content")
    criteria = args.get("criteria", ["accuracy", "depth", "falsifiability", "evidence"])

    eval_question = (
        f"Evaluate the following content against these criteria: {', '.join(criteria)}.\n\n"
        f"Original question/task: {question}\n\n"
        f"Content to evaluate:\n{content[:4000]}"
    )

    verdict = await _run_council(
        question=eval_question,
        context="This is an evaluation task. Each persona should critique the content "
        "from their intellectual tradition and assign a quality assessment.",
        evidence_search=False,
    )

    result = {
        "success": True,
        "confidence_score": verdict.confidence_score,
        "conflict_detected": verdict.conflict_detected,
        "criteria": criteria,
        "arbiter_synthesis": verdict.arbiter_synthesis,
        "persona_feedback": {
            name: {
                "confidence": resp.confidence,
                "dissents": resp.dissents,
                "key_points": resp.key_points,
                "content": resp.content[:1500],
            }
            for name, resp in verdict.responses.items()
        },
        "dpo_pairs": verdict.dpo_pairs,
        "sources": verdict.sources,
    }
    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# Tool 3: council_gate
# =============================================================================

COUNCIL_GATE_SCHEMA = {
    "name": "council_gate",
    "description": (
        "Safety gate: run quick council review before high-stakes actions. "
        "Use before deploying code, sending messages, or making irreversible changes. "
        "Returns allow/deny with reasoning. Uses abbreviated council (Skeptic + Oracle + Arbiter)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Description of the action to review",
            },
            "risk_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Risk level (default: medium)",
            },
            "context": {
                "type": "string",
                "description": "Context about why this action is being taken",
            },
        },
        "required": ["action"],
    },
}


async def council_gate_handler(args: dict, **kwargs) -> str:
    """Handle council_gate tool calls. Abbreviated council for speed."""
    action = args.get("action", "")
    if not action:
        return json.dumps({"error": "action is required"})

    risk_level = args.get("risk_level", "medium")
    context = args.get("context", "")

    gate_question = (
        f"SAFETY REVIEW (Risk level: {risk_level})\n"
        f"Proposed action: {action}\n"
    )
    if context:
        gate_question += f"Context: {context}\n"
    gate_question += (
        "\nShould this action be allowed? Consider risks, reversibility, "
        "and potential negative outcomes. Be concise."
    )

    # Abbreviated council: Skeptic + Oracle + Arbiter only
    verdict = await _run_council(
        question=gate_question,
        persona_names=["skeptic", "oracle", "arbiter"],
        evidence_search=False,
    )

    # Determine allow/deny based on Arbiter confidence
    threshold = {"low": 30, "medium": 50, "high": 70}.get(risk_level, 50)
    allowed = verdict.confidence_score >= threshold

    result = {
        "success": True,
        "allowed": allowed,
        "confidence": verdict.confidence_score,
        "risk_level": risk_level,
        "threshold": threshold,
        "reasoning": verdict.arbiter_synthesis[:1000],
        "skeptic_concerns": (
            verdict.responses.get("skeptic", PersonaResponse(
                persona_name="skeptic", content="", confidence=0, dissents=False
            )).key_points
        ),
    }
    return json.dumps(result, ensure_ascii=False)


# =============================================================================
# Availability check
# =============================================================================


def check_council_requirements() -> bool:
    """Returns True if any LLM API key is available."""
    return bool(
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("NOUS_API_KEY")
    )


# =============================================================================
# Module-level registration
# =============================================================================

registry.register(
    name="council_query",
    toolset="council",
    schema=COUNCIL_QUERY_SCHEMA,
    handler=council_query_handler,
    check_fn=check_council_requirements,
    is_async=True,
)

registry.register(
    name="council_evaluate",
    toolset="council",
    schema=COUNCIL_EVALUATE_SCHEMA,
    handler=council_evaluate_handler,
    check_fn=check_council_requirements,
    is_async=True,
)

registry.register(
    name="council_gate",
    toolset="council",
    schema=COUNCIL_GATE_SCHEMA,
    handler=council_gate_handler,
    check_fn=check_council_requirements,
    is_async=True,
)
