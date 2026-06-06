#!/usr/bin/env python3
"""LLM Council tool.

A deliberative multi-model reasoning tool inspired by multi-agent debate,
Mixture-of-Agents, and judge/evaluator aggregation patterns. Unlike the MoA
synthesizer, this tool asks a small council to produce independent proposals,
critique/vote on the proposals, then asks a judge model to produce a final
answer with consensus and dissent surfaced.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.auxiliary_client import extract_content_or_reasoning
from tools.debug_helpers import DebugSession
from tools.openrouter_client import (
    check_api_key as check_openrouter_api_key,
    get_async_client as _get_openrouter_client,
)
from tools.registry import registry

logger = logging.getLogger(__name__)

# Keep aligned with the current MoA defaults, but use only three council seats
# by default to control cost/latency: 3 member calls + 3 critique calls + judge.
DEFAULT_COUNCIL_MEMBERS: List[Dict[str, str]] = [
    {
        "role": "Architect",
        "model": "anthropic/claude-opus-4.6",
        "persona": "Solve the problem end-to-end. Optimize for correctness, structure, and practical action.",
    },
    {
        "role": "Skeptic",
        "model": "google/gemini-2.5-pro",
        "persona": "Find hidden assumptions, edge cases, missing evidence, and reasons the obvious answer may be wrong.",
    },
    {
        "role": "Implementer",
        "model": "deepseek/deepseek-v3.2",
        "persona": "Focus on concrete execution details, failure modes, and the simplest shippable path.",
    },
]

DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4.6"
PROPOSAL_TEMPERATURE = 0.5
CRITIQUE_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.25
MIN_SUCCESSFUL_MEMBERS = 2

_debug = DebugSession("llm_council", env_var="LLM_COUNCIL_DEBUG")


def _member_name(member: Dict[str, str], index: int) -> str:
    role = (member.get("role") or "Member").strip()
    model = (member.get("model") or "unknown-model").strip()
    return f"{index + 1}. {role} ({model})"


def _normalize_council_members(council_models: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Return council member configs, optionally replacing models only."""
    if not council_models:
        return [dict(member) for member in DEFAULT_COUNCIL_MEMBERS]

    roles = ["Architect", "Skeptic", "Implementer", "Domain Expert", "Risk Officer"]
    personas = [member["persona"] for member in DEFAULT_COUNCIL_MEMBERS] + [
        "Bring specialized domain knowledge and identify non-obvious constraints.",
        "Evaluate safety, operational risk, cost, privacy, and blast radius.",
    ]
    members: List[Dict[str, str]] = []
    for idx, model in enumerate(council_models[:5]):
        model = (model or "").strip()
        if not model:
            continue
        members.append({
            "role": roles[idx] if idx < len(roles) else f"Member {idx + 1}",
            "model": model,
            "persona": personas[idx] if idx < len(personas) else personas[-1],
        })
    return members or [dict(member) for member in DEFAULT_COUNCIL_MEMBERS]


def _render_proposals(proposals: List[Dict[str, Any]]) -> str:
    chunks = []
    for idx, item in enumerate(proposals):
        chunks.append(
            f"### Proposal {idx + 1}: {item['role']} ({item['model']})\n{item['content']}"
        )
    return "\n\n".join(chunks)


def _parse_vote(text: str) -> Dict[str, Any]:
    """Parse a compact vote/confidence block from a critique response.

    The model is instructed to include lines such as:
      Vote: Proposal 2
      Confidence: 0.74
    Parsing is best-effort; raw critique remains available to the judge.
    """
    vote_match = re.search(r"(?im)^\s*vote\s*:\s*(.+?)\s*$", text or "")
    confidence_match = re.search(r"(?im)^\s*confidence\s*:\s*([0-9]*\.?[0-9]+)\s*$", text or "")

    confidence: Optional[float] = None
    if confidence_match:
        try:
            confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
        except ValueError:
            confidence = None

    vote = vote_match.group(1).strip() if vote_match else "unparsed"
    return {"vote": vote, "confidence": confidence}


def _should_omit_temperature(model: str) -> bool:
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare.startswith("gpt-") or bare.startswith("kimi-") or bare == "kimi"


async def _call_model_safe(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int = 8000,
    max_retries: int = 2,
) -> Tuple[str, str, bool]:
    """Call a single OpenRouter model with bounded retries."""
    for attempt in range(max_retries):
        try:
            params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "extra_body": {"reasoning": {"enabled": True, "effort": "high"}},
            }
            if not _should_omit_temperature(model):
                params["temperature"] = temperature

            response = await _get_openrouter_client().chat.completions.create(**params)
            content = extract_content_or_reasoning(response)
            if content:
                return model, content, True
            logger.warning("%s returned empty council content (attempt %s/%s)", model, attempt + 1, max_retries)
        except Exception as exc:
            logger.warning("%s council call failed (attempt %s/%s): %s", model, attempt + 1, max_retries, exc)

        if attempt < max_retries - 1:
            await asyncio.sleep(min(2 ** attempt, 8))

    return model, f"{model} failed after {max_retries} attempts", False


async def _run_member_proposal(member: Dict[str, str], user_prompt: str) -> Dict[str, Any]:
    system = (
        f"You are the {member['role']} in an LLM council. {member['persona']}\n"
        "Give an independent proposal. Do not mention other council members.\n"
        "Format:\nSummary:\nReasoning:\nRisks / assumptions:\nRecommended answer:"
    )
    model, content, success = await _call_model_safe(
        member["model"],
        [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        PROPOSAL_TEMPERATURE,
    )
    return {"role": member["role"], "model": model, "content": content, "success": success}


async def _run_member_critique(member: Dict[str, str], user_prompt: str, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    proposal_text = _render_proposals(proposals)
    system = (
        f"You are the {member['role']} in an LLM council. {member['persona']}\n"
        "Critique the council proposals. Pick the best proposal or combination.\n"
        "Be concise and include exactly one Vote line and one Confidence line.\n"
        "Format:\nVote: Proposal N or Combination\nConfidence: 0.00-1.00\nCritique:\nNeeded fixes:"
    )
    user = f"Original user request:\n{user_prompt}\n\nCouncil proposals:\n{proposal_text}"
    model, content, success = await _call_model_safe(
        member["model"],
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        CRITIQUE_TEMPERATURE,
        max_tokens=5000,
    )
    parsed = _parse_vote(content if success else "")
    return {
        "role": member["role"],
        "model": model,
        "content": content,
        "success": success,
        "vote": parsed["vote"],
        "confidence": parsed["confidence"],
    }


async def _run_judge(
    judge_model: str,
    user_prompt: str,
    proposals: List[Dict[str, Any]],
    critiques: List[Dict[str, Any]],
) -> str:
    votes = [
        {
            "role": c["role"],
            "model": c["model"],
            "vote": c.get("vote"),
            "confidence": c.get("confidence"),
        }
        for c in critiques
        if c.get("success")
    ]
    system = (
        "You are the neutral judge of an LLM council. Produce the final answer.\n"
        "Use the proposals and critiques as evidence, but do not blindly average them.\n"
        "Surface consensus, important dissent, assumptions, and the practical conclusion.\n"
        "If the council is wrong or under-evidenced, correct it."
    )
    user = (
        f"Original user request:\n{user_prompt}\n\n"
        f"Council proposals:\n{_render_proposals(proposals)}\n\n"
        f"Votes:\n{json.dumps(votes, ensure_ascii=False, indent=2)}\n\n"
        "Critiques:\n"
        + "\n\n".join(
            f"### {c['role']} ({c['model']})\n{c['content']}" for c in critiques if c.get("success")
        )
    )
    model, content, success = await _call_model_safe(
        judge_model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        JUDGE_TEMPERATURE,
        max_tokens=10000,
        max_retries=3,
    )
    if not success:
        raise RuntimeError(content)
    return content


async def llm_council_tool(
    user_prompt: str,
    council_models: Optional[List[str]] = None,
    judge_model: Optional[str] = None,
    include_transcript: bool = False,
) -> str:
    """Run a deliberative LLM council and return a JSON result string."""
    start_time = datetime.datetime.now()
    members = _normalize_council_members(council_models)
    judge = (judge_model or DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL

    debug_call_data: Dict[str, Any] = {
        "parameters": {
            "user_prompt": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "council_models": [m["model"] for m in members],
            "judge_model": judge,
            "include_transcript": include_transcript,
        },
        "success": False,
        "error": None,
    }

    try:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt is required")

        logger.info("Starting LLM council: %s members + judge %s", len(members), judge)
        proposals = await asyncio.gather(*[_run_member_proposal(member, user_prompt) for member in members])
        successful_proposals = [p for p in proposals if p.get("success")]
        if len(successful_proposals) < MIN_SUCCESSFUL_MEMBERS:
            raise ValueError(
                f"Insufficient successful council members ({len(successful_proposals)}/{len(members)}). "
                f"Need at least {MIN_SUCCESSFUL_MEMBERS}."
            )

        successful_members = [m for m, p in zip(members, proposals) if p.get("success")]
        critiques = await asyncio.gather(*[
            _run_member_critique(member, user_prompt, successful_proposals)
            for member in successful_members
        ])
        successful_critiques = [c for c in critiques if c.get("success")]
        if not successful_critiques:
            raise ValueError("No successful council critiques/votes")

        final_response = await _run_judge(judge, user_prompt, successful_proposals, successful_critiques)
        elapsed = (datetime.datetime.now() - start_time).total_seconds()

        votes = [
            {
                "role": c["role"],
                "model": c["model"],
                "vote": c.get("vote"),
                "confidence": c.get("confidence"),
            }
            for c in successful_critiques
        ]
        result: Dict[str, Any] = {
            "success": True,
            "response": final_response,
            "council_size": len(successful_proposals),
            "votes": votes,
            "models_used": {
                "council_models": [p["model"] for p in successful_proposals],
                "judge_model": judge,
            },
            "processing_time": elapsed,
        }
        if include_transcript:
            result["transcript"] = {
                "proposals": successful_proposals,
                "critiques": successful_critiques,
                "failed_members": [p["model"] for p in proposals if not p.get("success")],
            }

        debug_call_data.update({"success": True, "processing_time": elapsed, "votes": votes})
        _debug.log_call("llm_council", debug_call_data)
        _debug.save()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        error_msg = f"Error in LLM council processing: {exc}"
        logger.error("%s", error_msg, exc_info=True)
        debug_call_data.update({"error": error_msg, "processing_time": elapsed})
        _debug.log_call("llm_council", debug_call_data)
        _debug.save()
        return json.dumps(
            {
                "success": False,
                "response": "LLM council processing failed. Please try again or use a single model for this query.",
                "models_used": {
                    "council_models": [m["model"] for m in members],
                    "judge_model": judge,
                },
                "error": error_msg,
            },
            ensure_ascii=False,
            indent=2,
        )


def check_llm_council_requirements() -> bool:
    return check_openrouter_api_key()


LLM_COUNCIL_SCHEMA = {
    "name": "llm_council",
    "description": (
        "Run a deliberative council of multiple frontier LLMs: independent proposals, "
        "critique/vote round, then a neutral judge synthesis. Higher cost/latency than "
        "a normal answer; use for high-stakes strategy, architecture, debugging, or reasoning "
        "where dissent and consensus matter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The hard question or decision for the LLM council to deliberate on.",
            },
            "council_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional OpenRouter model slugs for council members. Defaults to 3 diverse frontier models. Max 5.",
            },
            "judge_model": {
                "type": "string",
                "description": "Optional OpenRouter model slug for the final judge/synthesizer.",
            },
            "include_transcript": {
                "type": "boolean",
                "description": "Include successful proposals/critiques in the JSON result. Defaults false to save tokens.",
            },
        },
        "required": ["user_prompt"],
    },
}


registry.register(
    name="llm_council",
    toolset="moa",
    schema=LLM_COUNCIL_SCHEMA,
    handler=lambda args, **kw: llm_council_tool(
        user_prompt=args.get("user_prompt", ""),
        council_models=args.get("council_models"),
        judge_model=args.get("judge_model"),
        include_transcript=bool(args.get("include_transcript", False)),
    ),
    check_fn=check_llm_council_requirements,
    requires_env=["OPENROUTER_API_KEY"],
    is_async=True,
    emoji="⚖️",
)
