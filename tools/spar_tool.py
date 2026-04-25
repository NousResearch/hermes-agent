#!/usr/bin/env python3
"""Spar — bounded builder/reviewer loop for material quality gates."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import (
    async_call_llm,
    extract_content_or_reasoning,
    resolve_provider_client,
)
from agent.failure_registry import record_failure
from hermes_cli.models import detect_provider_for_model
from tools.registry import registry

logger = logging.getLogger(__name__)

NON_MATERIAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnaming\b",
        r"\bwording\b",
        r"\bphrasing\b",
        r"\bformat(?:ting)?\b",
        r"\bstyle\b",
        r"\breadability\b",
        r"\bcosmetic\b",
        r"\bnit(?:pick)?\b",
        r"\bcomment\b",
        r"\brename\b",
    )
]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

SPAR_REVIEW_CONTRACT = [
    "Review the following builder result.",
    "Approve only if it fully completes the user's request.",
    "Reject for material correctness, missing work, regression, safety, or unmet requirements.",
    "Ignore naming, wording, formatting, readability, and other cosmetic-only feedback.",
    "Return JSON only with keys approved, summary, issues, fix.",
]

SPAR_FIX_CONTRACT = [
    "Spar review found material issues. Revise the previous answer once and fix them all.",
    "Do not mention spar mode or the review process unless the user asks.",
]

DEFAULT_BUILDER_ROUTE = {
    "provider": "xiaomi",
    "model": "mimo-v2.5-pro",
    "label": "xiaomi/mimo-v2.5-pro",
}

DEFAULT_REVIEWER_ROUTE = {
    "provider": "openrouter",
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "label": "nvidia/nemotron-3-super-120b-a12b",
}

DEFAULT_JUDGE_ROUTE = {
    "provider": "xiaomi",
    "model": "mimo-v2.5-pro",
    "label": "xiaomi/mimo-v2.5-pro",
}

SPAR_CALL_TIMEOUT_SECONDS = 90.0
SPAR_CALL_RETRIES = 3


@dataclass
class SparReview:
    approved: bool
    summary: str
    issues: List[str] = field(default_factory=list)
    fix: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_route_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._/-]+", "-", str(value or "").strip()) or "route"


def _route_label(route: Dict[str, str]) -> str:
    label = str(route.get("label") or "").strip()
    if label:
        return label
    provider = str(route.get("provider") or "").strip()
    model = str(route.get("model") or "").strip()
    return f"{provider}/{model}" if provider else model


def _normalize_route(spec: Optional[str], *, fallback: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if spec in (None, ""):
        if fallback is None:
            raise ValueError("missing model route")
        return dict(fallback)

    raw = str(spec).strip()
    if "/" in raw:
        provider, model = raw.split("/", 1)
        route = {"provider": provider.strip().lower(), "model": model.strip()}
    else:
        provider, model = detect_provider_for_model(raw, "")
        route = {"provider": provider.strip().lower(), "model": model.strip()}
    route["label"] = _route_label(route)
    return route


def _default_builder_route() -> Dict[str, str]:
    return dict(DEFAULT_BUILDER_ROUTE)


def _route_is_available(route: Dict[str, str]) -> bool:
    try:
        client, _ = resolve_provider_client(
            route.get("provider"),
            route.get("model"),
            explicit_base_url=route.get("base_url"),
            explicit_api_key=route.get("api_key"),
            async_mode=True,
        )
        return client is not None
    except Exception:
        return False


def check_spar_requirements() -> bool:
    return (
        _route_is_available(_default_builder_route())
        and _route_is_available(DEFAULT_REVIEWER_ROUTE)
        and _route_is_available(DEFAULT_JUDGE_ROUTE)
    )


def _is_non_material(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in NON_MATERIAL_PATTERNS)


def filter_spar_review(review: SparReview) -> SparReview:
    if review.approved:
        return review
    issues = [item for item in review.issues if not _is_non_material(item)]
    fix = review.fix if review.fix and not _is_non_material(review.fix) else None
    if issues or fix:
        return SparReview(
            approved=False,
            summary=review.summary,
            issues=issues,
            fix=fix,
        )
    return SparReview(
        approved=True,
        summary="Approved after non-material signoff filter.",
        issues=[],
        fix=None,
    )


def parse_spar_review(text: str) -> SparReview:
    data = _extract_review_object(text)
    if not isinstance(data, dict):
        raise ValueError("review payload must be an object")
    if not isinstance(data.get("approved"), bool):
        raise ValueError("review payload must include boolean approved")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("review payload must include non-empty summary")
    raw_issues = data.get("issues", [])
    if raw_issues is None:
        raw_issues = []
    elif isinstance(raw_issues, str):
        raw_issues = [raw_issues]
    elif isinstance(raw_issues, dict):
        raw_issues = [json.dumps(raw_issues, ensure_ascii=False)]
    elif not isinstance(raw_issues, list):
        raw_issues = [str(raw_issues)]
    issues = [str(item).strip() for item in raw_issues if str(item).strip()]
    fix = data.get("fix")
    if fix is not None:
        fix = str(fix).strip() or None
    return filter_spar_review(
        SparReview(
            approved=data["approved"],
            summary=summary.strip(),
            issues=issues,
            fix=fix,
        )
    )


def _extract_review_object(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    candidates = [match.group(1).strip() for match in _JSON_FENCE_RE.finditer(raw_text)]
    candidates.append(raw_text)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        for start in [idx for idx, char in enumerate(candidate) if char == "{"]:
            try:
                payload, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    raise ValueError("review payload did not contain a valid JSON object")


def format_spar_fix(review: SparReview) -> str:
    lines = ["<system-reminder>", *SPAR_FIX_CONTRACT, f"Summary: {review.summary}"]
    if review.issues:
        lines.append("Issues:")
        lines.extend(f"{idx}. {issue}" for idx, issue in enumerate(review.issues, start=1))
    if review.fix:
        lines.extend(["", f"Fix instruction: {review.fix}"])
    lines.append("</system-reminder>")
    return "\n".join(lines)


async def _call_route(
    route: Dict[str, str],
    messages: List[Dict[str, str]],
    *,
    task: str,
    temperature: float,
    max_tokens: int = 4096,
) -> str:
    route_label = _route_label(route)
    last_error: Exception | None = None
    for attempt in range(SPAR_CALL_RETRIES):
        try:
            response = await async_call_llm(
                task=task,
                provider=route.get("provider"),
                model=route.get("model"),
                base_url=route.get("base_url"),
                api_key=route.get("api_key"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=SPAR_CALL_TIMEOUT_SECONDS,
            )
            content = extract_content_or_reasoning(response)
            if content:
                return content.strip()
            last_error = RuntimeError(f"{route_label} returned empty content")
        except Exception as exc:
            last_error = exc
        logger.warning(
            "%s call failed attempt %s/%s: %s",
            route_label,
            attempt + 1,
            SPAR_CALL_RETRIES,
            last_error,
        )
        if attempt < SPAR_CALL_RETRIES - 1:
            await asyncio.sleep(min(2 ** (attempt + 1), 15))
    raise RuntimeError(f"{route_label} failed after {SPAR_CALL_RETRIES} attempts: {last_error}")


async def _run_build(prompt: str, route: Dict[str, str]) -> str:
    return await _call_route(
        route,
        messages=[{"role": "user", "content": prompt}],
        task="spar",
        temperature=0.3,
    )


async def _run_review(prompt: str, candidate: str, route: Dict[str, str]) -> SparReview:
    review_prompt = "\n".join(SPAR_REVIEW_CONTRACT)
    content = await _call_route(
        route,
        messages=[
            {"role": "system", "content": review_prompt},
            {
                "role": "user",
                "content": (
                    "User request:\n"
                    f"{prompt}\n\n"
                    "Builder result:\n"
                    f"{candidate}"
                ),
            },
        ],
        task="spar",
        temperature=0.0,
        max_tokens=1200,
    )
    try:
        return parse_spar_review(content)
    except Exception as exc:
        preview = content.replace("\n", "\\n")[:240]
        raise ValueError(
            f"{_route_label(route)} returned invalid review JSON: {exc}. Preview: {preview}"
        ) from exc


async def _run_optional_judge(
    prompt: str,
    candidate: str,
    route: Dict[str, str],
) -> Optional[SparReview]:
    try:
        return await asyncio.wait_for(_run_review(prompt, candidate, route), timeout=60.0)
    except Exception as exc:
        logger.warning("Judge review failed for %s: %s", _route_label(route), exc)
        return None


def _trace_entry(
    phase: str,
    route: Dict[str, str],
    started: float,
    *,
    verdict: Optional[str] = None,
    summary: str = "",
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "phase": phase,
        "model": _route_label(route),
        "at": _now_iso(),
        "duration": round(time.monotonic() - started, 3),
    }
    if verdict:
        entry["verdict"] = verdict
    if summary:
        entry["summary"] = summary
    return entry


async def _prepare_candidate(
    user_prompt: str,
    builder_route: Dict[str, str],
    candidate_response: str,
) -> tuple[str, List[Dict[str, Any]]]:
    current_response = str(candidate_response or "").strip()
    if current_response:
        return current_response, [
            {
                "phase": "candidate.input",
                "model": _route_label(builder_route),
                "at": _now_iso(),
                "duration": 0.0,
                "verdict": "completed",
                "summary": "Used provided candidate_response.",
            }
        ]

    started = time.monotonic()
    current_response = await _run_build(user_prompt, builder_route)
    return current_response, [
        _trace_entry("build.1", builder_route, started, verdict="completed")
    ]


async def _run_review_round(
    *,
    user_prompt: str,
    current_response: str,
    review_round: int,
    reviewer_route: Dict[str, str],
    judge_route: Dict[str, str],
    judge_enabled: bool,
) -> tuple[SparReview, Optional[SparReview], bool, List[Dict[str, Any]]]:
    started = time.monotonic()
    if judge_enabled:
        review, judge = await asyncio.gather(
            _run_review(user_prompt, current_response, reviewer_route),
            _run_optional_judge(user_prompt, current_response, judge_route),
        )
    else:
        review = await _run_review(user_prompt, current_response, reviewer_route)
        judge = None

    trace = [
        _trace_entry(
            f"spar.review.{review_round}",
            reviewer_route,
            started,
            verdict="approved" if review.approved else "rejected",
            summary=review.summary,
        )
    ]
    disagreement = False
    if judge is not None:
        trace.append(
            _trace_entry(
                f"judge.review.{review_round}",
                judge_route,
                started,
                verdict="approved" if judge.approved else "rejected",
                summary=judge.summary,
            )
        )
        disagreement = review.approved != judge.approved
    return review, judge, disagreement, trace


def _record_spar_rejection(
    *,
    review: SparReview,
    current_response: str,
    builder_route: Dict[str, str],
    reviewer_route: Dict[str, str],
    judge_route: Dict[str, str],
    judge_enabled: bool,
    disagreement: bool,
    user_prompt: str,
) -> None:
    if review.approved:
        return
    try:
        record_failure(
            trigger="spar_rejection",
            symptom=current_response,
            root_cause=" | ".join(
                item for item in [review.summary, *review.issues] if item
            ),
            fix=review.fix or "",
            prevention=(
                "Tighten the builder contract or acceptance checks before shipping "
                "the response without Spar."
            ),
            related_skills=["spar"],
            metadata={
                "builder_model": _route_label(builder_route),
                "reviewer_model": _route_label(reviewer_route),
                "judge_model": _route_label(judge_route) if judge_enabled else "",
                "disagreement": disagreement,
                "user_prompt": user_prompt,
            },
        )
    except Exception:
        logger.debug("Failed to persist Spar rejection scar", exc_info=True)


async def spar_tool(
    user_prompt: str,
    *,
    candidate_response: str = "",
    builder_model: str = "",
    reviewer_model: str = "",
    judge_model: str = "",
    judge_enabled: bool = True,
    max_fix_rounds: int = 1,
) -> str:
    """Run one build → review → optional single-fix loop."""
    builder_route = _normalize_route(builder_model, fallback=_default_builder_route())
    reviewer_route = _normalize_route(reviewer_model, fallback=DEFAULT_REVIEWER_ROUTE)
    judge_route = _normalize_route(judge_model, fallback=DEFAULT_JUDGE_ROUTE)
    fix_budget = max(0, min(int(max_fix_rounds), 1))

    current_response, trace = await _prepare_candidate(
        user_prompt,
        builder_route,
        candidate_response,
    )

    last_review: Optional[SparReview] = None
    last_judge: Optional[SparReview] = None
    disagreement = False
    for review_round in range(1, 2 + fix_budget):
        last_review, last_judge, round_disagreement, round_trace = await _run_review_round(
            user_prompt=user_prompt,
            current_response=current_response,
            review_round=review_round,
            reviewer_route=reviewer_route,
            judge_route=judge_route,
            judge_enabled=judge_enabled,
        )
        trace.extend(round_trace)
        disagreement = disagreement or round_disagreement
        if last_review.approved:
            break
        if review_round > fix_budget:
            break

        started = time.monotonic()
        current_response = await _call_route(
            builder_route,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": current_response},
                {"role": "user", "content": format_spar_fix(last_review)},
            ],
            task="spar",
            temperature=0.2,
        )
        trace.append(_trace_entry(f"spar.fix.{review_round}", builder_route, started, verdict="completed"))

    if last_review is None:
        raise RuntimeError("spar review did not run")

    result = {
        "approved": last_review.approved,
        "summary": last_review.summary,
        "issues": last_review.issues,
        "fix": last_review.fix,
        "final_response": current_response,
        "trace": trace,
        "builder_model": _route_label(builder_route),
        "reviewer_model": _route_label(reviewer_route),
        "judge_enabled": bool(judge_enabled),
        "judge_model": _route_label(judge_route),
        "judge_verdict": (
            None
            if last_judge is None
            else {
                "approved": last_judge.approved,
                "summary": last_judge.summary,
                "issues": last_judge.issues,
                "fix": last_judge.fix,
            }
        ),
        "disagreement": disagreement,
    }
    _record_spar_rejection(
        review=last_review,
        current_response=current_response,
        builder_route=builder_route,
        reviewer_route=reviewer_route,
        judge_route=judge_route,
        judge_enabled=judge_enabled,
        disagreement=disagreement,
        user_prompt=user_prompt,
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


SPAR_SCHEMA = {
    "name": "spar",
    "description": (
        "Run a bounded builder → reviewer → single-fix quality gate. "
        "Use for ship/no-ship checks where material correctness matters more than diverse brainstorming."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The task or request that should be built and then materially reviewed.",
            },
            "candidate_response": {
                "type": "string",
                "description": "Optional prebuilt answer to review instead of generating a fresh builder response.",
            },
            "builder_model": {
                "type": "string",
                "description": "Optional builder model route like 'xiaomi/mimo-v2.5-pro'. Defaults to the current main model.",
            },
            "reviewer_model": {
                "type": "string",
                "description": "Optional Spar reviewer model route. Defaults to nvidia/nemotron-3-super-120b-a12b.",
            },
            "judge_model": {
                "type": "string",
                "description": "Optional judge model route. Defaults to xiaomi/mimo-v2.5-pro.",
            },
            "judge_enabled": {
                "type": "boolean",
                "description": "Run an independent judge review in parallel. Defaults to true.",
            },
            "max_fix_rounds": {
                "type": "integer",
                "description": "Maximum repair rounds after rejection. Hermes clamps this to 1.",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["user_prompt"],
    },
}


registry.register(
    name="spar",
    toolset="spar",
    schema=SPAR_SCHEMA,
    handler=lambda args, **kw: spar_tool(
        user_prompt=args.get("user_prompt", ""),
        candidate_response=args.get("candidate_response", ""),
        builder_model=args.get("builder_model", ""),
        reviewer_model=args.get("reviewer_model", ""),
        judge_model=args.get("judge_model", ""),
        judge_enabled=args.get("judge_enabled", True),
        max_fix_rounds=args.get("max_fix_rounds", 1),
    ),
    check_fn=check_spar_requirements,
    requires_env=["XIAOMI_API_KEY", "OPENROUTER_API_KEY"],
    is_async=True,
    emoji="🛡️",
)
