#!/usr/bin/env python3
"""
Analyst Council Tool — multi-perspective adversarial research review.

Ports the Swift ``AnalystCouncilTool`` (hermes-companion) to Python. Runs
3–5 specialist reviewer personas in parallel, optionally does anonymized
peer review, then a chairman LLM synthesizes a consensus report.

Based on Karpathy's LLM Council pattern (github.com/karpathy/llm-council):
    poll → (peer review) → synthesize

Three stages:
    Stage 1 (Poll):  Parallel per-persona review calls via asyncio.gather.
    Stage 2 (Peer):  Full depth only. Each reviewer reviews others'
                     anonymized opinions and rates 1–5.
    Stage 3 (Chair): Single chairman LLM merges opinions + peer reviews
                     into a structured consensus report.

Four domain persona sets (ported verbatim from Swift):
    - finance:    Bull / Bear / Technical / Fundamental / Risk
    - medicine:   Clinical / Skeptic / Safety / Patient / Evidence
    - technology: Architect / Security / Devil's Advocate / Domain / Pragmatist
    - general:    Subject / Contrarian / Fact Checker / Risk / Synthesizer

Depth modes:
    - quick: 3 reviewers, no peer review        (4 total LLM calls)
    - full:  5 reviewers + peer review          (11 total LLM calls)

Cost control:
    Reviewer calls default to ``council.reviewer_model`` from config, which
    falls back to ``smart_model_routing.cheap_model``, which falls back to
    ``google/gemini-2.5-flash``. Only the chairman synthesis uses the
    configured ``council.chairman_model`` (defaults to a strong model).

Config (``~/.hermes/cli-config.yaml``)::

    council:
      enabled: true
      default_depth: quick
      reviewer_model: null      # → smart_model_routing.cheap_model
      chairman_model: null      # → FALLBACK_CHAIRMAN_MODEL
      skip_for_trivial: true
      trivial_keywords: [weather, time, calculator, hello, thanks]
"""

import asyncio
import contextvars
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.openrouter_client import (
    get_async_client as _get_openrouter_client,
    check_api_key as check_openrouter_api_key,
)
from agent.auxiliary_client import extract_content_or_reasoning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

REVIEWER_TEMPERATURE = 0.4
PEER_REVIEW_TEMPERATURE = 0.3
CHAIRMAN_TEMPERATURE = 0.3

REVIEWER_MAX_TOKENS = 600
PEER_REVIEW_MAX_TOKENS = 400
CHAIRMAN_MAX_TOKENS = 1500

# Truncation limits for reviewer prompts (mirrors Swift tool)
MAX_DATA_CHARS = 8000
MAX_WEB_CONTEXT_CHARS = 4000

# Retry policy per LLM call
MAX_RETRIES = 4

# Reviewer / chairman model fallbacks (used when config does not specify)
FALLBACK_REVIEWER_MODEL = "google/gemini-2.5-flash"
FALLBACK_CHAIRMAN_MODEL = "anthropic/claude-opus-4.6"

# Recursion guard — prevents council from calling itself inside a council call.
_COUNCIL_IN_FLIGHT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "council_in_flight", default=False
)


# ---------------------------------------------------------------------------
# Reviewer persona schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewerPersona:
    """A single reviewer in the council — name, role, focus, system prompt."""
    name: str
    role: str
    focus: str
    system_prompt: str


# ---------------------------------------------------------------------------
# Persona sets — ported verbatim from
# hermes-companion/Sources/HermesCompanion/Tools/AnalystCouncilTool.swift
# ---------------------------------------------------------------------------

FINANCE_PERSONAS: List[ReviewerPersona] = [
    ReviewerPersona(
        name="Bull Analyst",
        role="Bull Case Analyst",
        focus="Focus on: growth catalysts, competitive moat, upside scenarios, market opportunity.",
        system_prompt=(
            "You are a bull-case equity analyst. Find the strongest investment "
            "thesis. Focus on growth catalysts, competitive advantages, and upside "
            "scenarios. Be specific with numbers."
        ),
    ),
    ReviewerPersona(
        name="Bear Analyst",
        role="Bear Case Analyst",
        focus="Focus on: overvaluation, structural weaknesses, competitive threats, downside scenarios.",
        system_prompt=(
            "You are a bear-case equity analyst. Find every reason NOT to invest. "
            "Focus on risks, overvaluation, structural weaknesses, and downside "
            "scenarios. Be specific with evidence."
        ),
    ),
    ReviewerPersona(
        name="Technical Analyst",
        role="Technical Analyst",
        focus="Focus on: P&F patterns, RSI/MACD, moving averages, volume, support/resistance.",
        system_prompt=(
            "You are a technical analyst. Focus on chart patterns, indicators, and "
            "price action. Read P&F data for structural signals. Identify "
            "support/resistance and trend direction."
        ),
    ),
    ReviewerPersona(
        name="Fundamental Analyst",
        role="Fundamental Analyst",
        focus="Focus on: earnings quality, ROCE trend, cash flow, accounting red flags, peer valuation.",
        system_prompt=(
            "You are a deep-value fundamental analyst. Scrutinize earnings quality, "
            "cash flow sustainability, return on capital trends, and balance sheet "
            "strength."
        ),
    ),
    ReviewerPersona(
        name="Risk Manager",
        role="Risk Manager",
        focus="Focus on: governance, geopolitical exposure, regulatory risk, supply chain, promoter concerns.",
        system_prompt=(
            "You are a risk manager. Look for tail risks: governance red flags, "
            "geopolitical exposure, regulatory changes, concentration risk. Flag "
            "anything that could cause significant loss."
        ),
    ),
]

MEDICINE_PERSONAS: List[ReviewerPersona] = [
    ReviewerPersona(
        name="Clinical Expert",
        role="Clinical Expert",
        focus="Focus on: clinical evidence quality, trial design, effect sizes, patient outcomes.",
        system_prompt=(
            "You are a clinical medicine expert. Evaluate evidence quality: trial "
            "design (RCT vs observational), sample sizes, effect sizes, confidence "
            "intervals, NNT. Flag weak evidence presented as strong."
        ),
    ),
    ReviewerPersona(
        name="Skeptic",
        role="Scientific Skeptic",
        focus="Focus on: methodological flaws, confounders, p-hacking, publication bias, conflicts of interest.",
        system_prompt=(
            "You are a scientific skeptic. Challenge every claim. Look for: "
            "methodological flaws, uncontrolled confounders, cherry-picked data, "
            "publication bias, industry funding conflicts. Apply Cochrane-level "
            "scrutiny."
        ),
    ),
    ReviewerPersona(
        name="Safety Reviewer",
        role="Safety & Regulatory Reviewer",
        focus="Focus on: adverse effects, drug interactions, contraindications, regulatory status, off-label use.",
        system_prompt=(
            "You are a drug safety reviewer. Focus on: adverse effect profiles, "
            "drug interactions, contraindications, black box warnings, FDA/EMA "
            "regulatory status, off-label claims without evidence."
        ),
    ),
    ReviewerPersona(
        name="Patient Advocate",
        role="Patient Outcome Advocate",
        focus="Focus on: real-world effectiveness vs efficacy, quality of life impact, accessibility, cost.",
        system_prompt=(
            "You are a patient advocate. Evaluate from the patient perspective: "
            "does this actually help patients in the real world? Consider "
            "accessibility, cost, quality of life, and whether surrogate endpoints "
            "translate to meaningful outcomes."
        ),
    ),
    ReviewerPersona(
        name="Evidence Synthesizer",
        role="Evidence Synthesis Specialist",
        focus="Focus on: systematic review quality, meta-analysis validity, evidence hierarchy, guideline alignment.",
        system_prompt=(
            "You are an evidence synthesis specialist. Evaluate where claims sit "
            "in the evidence hierarchy (systematic reviews > RCTs > cohort > case "
            "studies > expert opinion). Check if conclusions align with current "
            "clinical guidelines."
        ),
    ),
]

TECHNOLOGY_PERSONAS: List[ReviewerPersona] = [
    ReviewerPersona(
        name="Architect",
        role="Systems Architect",
        focus="Focus on: scalability, reliability, technical debt, architecture patterns, performance.",
        system_prompt=(
            "You are a systems architect. Evaluate technical claims for: "
            "scalability, reliability, maintainability, appropriate technology "
            "choices, hidden complexity, and performance characteristics."
        ),
    ),
    ReviewerPersona(
        name="Security Reviewer",
        role="Security Analyst",
        focus="Focus on: security vulnerabilities, attack surface, data privacy, compliance, threat model.",
        system_prompt=(
            "You are a security analyst. Identify: attack vectors, data exposure "
            "risks, authentication weaknesses, compliance gaps, supply chain "
            "risks, and unaddressed threat scenarios."
        ),
    ),
    ReviewerPersona(
        name="Devil's Advocate",
        role="Devil's Advocate",
        focus="Focus on: why this could fail, hidden assumptions, competitive threats, adoption barriers.",
        system_prompt=(
            "You are a devil's advocate. Actively argue against the presented "
            "conclusions. Find hidden assumptions, optimistic projections, "
            "competitive threats, and reasons for failure."
        ),
    ),
    ReviewerPersona(
        name="Domain Expert",
        role="Domain Expert",
        focus="Focus on: domain accuracy, state-of-the-art comparison, practical feasibility, edge cases.",
        system_prompt=(
            "You are a domain expert. Verify technical accuracy: are claims "
            "consistent with current state-of-the-art? Are benchmarks fair? Are "
            "edge cases addressed? Flag outdated or incorrect technical "
            "assertions."
        ),
    ),
    ReviewerPersona(
        name="Pragmatist",
        role="Pragmatic Evaluator",
        focus="Focus on: real-world applicability, cost-benefit, implementation complexity, alternatives.",
        system_prompt=(
            "You are a pragmatic evaluator. Focus on: does this work in practice? "
            "What's the real cost? Are there simpler alternatives? What's the "
            "implementation risk? Cut through hype to practical reality."
        ),
    ),
]

GENERAL_PERSONAS: List[ReviewerPersona] = [
    ReviewerPersona(
        name="Subject Expert",
        role="Subject Matter Expert",
        focus="Focus on: factual accuracy, domain knowledge, current state of the field, nuance.",
        system_prompt=(
            "You are a subject matter expert. Verify factual claims, check for "
            "outdated information, identify oversimplifications, and add important "
            "nuance that may be missing."
        ),
    ),
    ReviewerPersona(
        name="Contrarian",
        role="Contrarian Reviewer",
        focus="Focus on: opposing viewpoints, counterarguments, blind spots, alternative explanations.",
        system_prompt=(
            "You are a contrarian reviewer. Actively construct the strongest "
            "counterargument to every major claim. Find what the analysis ignores "
            "or underweights. Challenge consensus assumptions."
        ),
    ),
    ReviewerPersona(
        name="Fact Checker",
        role="Fact Checker",
        focus="Focus on: verifiable claims, source quality, logical fallacies, unsupported assertions.",
        system_prompt=(
            "You are a rigorous fact checker. For each factual claim: is it "
            "verifiable? Is the source reliable? Is the logic sound? Flag: "
            "unsupported assertions, logical fallacies, correlation-causation "
            "errors, cherry-picked data."
        ),
    ),
    ReviewerPersona(
        name="Risk Assessor",
        role="Risk Assessor",
        focus="Focus on: what could go wrong, tail risks, unintended consequences, worst-case scenarios.",
        system_prompt=(
            "You are a risk assessor. Identify: what could go wrong? What are the "
            "tail risks? What are the unintended consequences? What's the "
            "worst-case scenario and how likely is it?"
        ),
    ),
    ReviewerPersona(
        name="Synthesizer",
        role="Cross-Disciplinary Synthesizer",
        focus="Focus on: connections across domains, second-order effects, systemic implications.",
        system_prompt=(
            "You are a cross-disciplinary synthesizer. Look for: connections to "
            "other fields, second-order effects, systemic implications, and "
            "insights that specialists might miss by being too narrowly focused."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Config + model selection helpers
# ---------------------------------------------------------------------------

def _load_council_config() -> Dict[str, Any]:
    """Load council section of hermes-agent config, with safe fallbacks.

    Returns a dict with keys: council, smart_model_routing (full config so
    helpers can reach into smart_model_routing for the cheap model fallback).
    """
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception as e:
        logger.debug("Council config load failed, using defaults: %s", e)
        return {}


def _pick_reviewer_model(cfg: Dict[str, Any]) -> str:
    """Select the model used for the 3–5 reviewer calls.

    Precedence:
        1. council.reviewer_model      (explicit per-feature config)
        2. smart_model_routing.cheap_model.model  (shared cheap-tier setting)
        3. FALLBACK_REVIEWER_MODEL     (hardcoded cheap default)
    """
    explicit = (cfg.get("council") or {}).get("reviewer_model")
    if explicit:
        return str(explicit)
    cheap = ((cfg.get("smart_model_routing") or {}).get("cheap_model") or {}).get("model")
    if cheap:
        return str(cheap)
    return FALLBACK_REVIEWER_MODEL


def _pick_chairman_model(cfg: Dict[str, Any]) -> str:
    """Select the model used for the single chairman synthesis call."""
    explicit = (cfg.get("council") or {}).get("chairman_model")
    if explicit:
        return str(explicit)
    return FALLBACK_CHAIRMAN_MODEL


def _select_personas(domain: str, full: bool) -> List[ReviewerPersona]:
    """Pick the persona set for a domain, and trim to 3 if not full depth."""
    key = (domain or "general").lower()
    if key in ("finance", "stock", "market", "equity", "investment"):
        personas = FINANCE_PERSONAS
    elif key in ("medicine", "health", "pharma", "clinical", "medical"):
        personas = MEDICINE_PERSONAS
    elif key in ("technology", "tech", "software", "engineering"):
        personas = TECHNOLOGY_PERSONAS
    else:
        personas = GENERAL_PERSONAS
    return list(personas) if full else list(personas[:3])


def _is_trivial_query(data: str, council_cfg: Dict[str, Any]) -> bool:
    """Bypass council for trivial queries — avoids wasted cost.

    Returns True only for very short inputs that also contain a trivial keyword,
    so "calculator" in a long analytical prompt does NOT bypass the council.
    """
    if not council_cfg.get("skip_for_trivial", True):
        return False
    triggers = council_cfg.get("trivial_keywords") or [
        "weather", "time", "calculator", "hello", "thanks",
    ]
    lowered = (data or "").lower().strip()
    if len(lowered) >= 40:
        return False
    return any(k in lowered for k in triggers)


# ---------------------------------------------------------------------------
# Low-level LLM call helper with retry + backoff
# ---------------------------------------------------------------------------

async def _call_llm(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    label: str,
) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
    """Single LLM call with retry loop. Returns ``(content, usage)`` or ``(None, None)``.

    ``label`` is used in log messages to identify the caller (persona name,
    "chairman", etc.) so the council's parallel calls stay distinguishable.
    """
    for attempt in range(MAX_RETRIES):
        try:
            api_params: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            }
            # GPT models reject custom temperature; match MoA tool's guard
            if not model.lower().startswith("gpt-"):
                api_params["temperature"] = temperature

            response = await _get_openrouter_client().chat.completions.create(**api_params)
            content = extract_content_or_reasoning(response)

            usage: Optional[Dict[str, int]] = None
            raw_usage = getattr(response, "usage", None)
            if raw_usage is not None:
                usage = {
                    "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                }

            if not content:
                logger.warning(
                    "Council[%s] empty content (attempt %s/%s), retrying",
                    label, attempt + 1, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(min(2 ** (attempt + 1), 30))
                    continue
                return None, usage

            logger.info(
                "Council[%s] responded (%s chars, attempt %s)",
                label, len(content), attempt + 1,
            )
            return content, usage

        except Exception as e:  # noqa: BLE001 — openai client raises many types
            err = str(e)
            logger.warning(
                "Council[%s] error (attempt %s/%s): %s",
                label, attempt + 1, MAX_RETRIES, err,
            )
            if attempt < MAX_RETRIES - 1:
                sleep_s = min(2 ** (attempt + 1), 30)
                await asyncio.sleep(sleep_s)
            else:
                logger.error(
                    "Council[%s] failed after %s attempts: %s",
                    label, MAX_RETRIES, err,
                )
                return None, None
    return None, None


# ---------------------------------------------------------------------------
# Stage 1 — independent opinions (parallel)
# ---------------------------------------------------------------------------

async def _collect_opinions(
    *,
    data: str,
    web_context: str,
    personas: List[ReviewerPersona],
    reviewer_model: str,
) -> List[Tuple[str, str, Optional[Dict[str, int]]]]:
    """Run every persona reviewer in parallel. Skips failed reviewers."""

    trimmed_data = data[:MAX_DATA_CHARS]
    trimmed_ctx = web_context[:MAX_WEB_CONTEXT_CHARS] if web_context else ""
    ctx_block = f"\n\nSupplementary Context:\n{trimmed_ctx}" if trimmed_ctx else ""

    async def _review(persona: ReviewerPersona):
        user_prompt = (
            f"You are a {persona.role}. Review this research data and provide "
            f"your assessment.\n\n"
            f"{persona.focus}\n\n"
            f"Research Data:\n{trimmed_data}"
            f"{ctx_block}\n\n"
            f"Provide a focused {persona.role} assessment in 200-400 words. "
            f"Be specific with evidence. Flag errors, gaps, or unsupported claims. "
            f"State your confidence level (Low/Medium/High)."
        )
        content, usage = await _call_llm(
            model=reviewer_model,
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            temperature=REVIEWER_TEMPERATURE,
            max_tokens=REVIEWER_MAX_TOKENS,
            label=persona.name,
        )
        return persona.name, content, usage

    results = await asyncio.gather(
        *[_review(p) for p in personas],
        return_exceptions=True,
    )
    opinions: List[Tuple[str, str, Optional[Dict[str, int]]]] = []
    for item in results:
        if isinstance(item, BaseException):
            logger.error("Council reviewer raised: %s", item)
            continue
        name, content, usage = item
        if content:
            opinions.append((name, content, usage))
    return opinions


# ---------------------------------------------------------------------------
# Stage 2 — peer review (parallel, full depth only)
# ---------------------------------------------------------------------------

async def _collect_peer_reviews(
    *,
    opinions: List[Tuple[str, str, Optional[Dict[str, int]]]],
    reviewer_model: str,
) -> List[Tuple[str, str, Optional[Dict[str, int]]]]:
    """Each reviewer rates the anonymized opinions of the others."""

    async def _peer(idx: int, current_name: str):
        others: List[str] = []
        for j, (_, opinion, _) in enumerate(opinions):
            if j == idx:
                continue
            label = chr(ord("A") + j)  # A, B, C, ...
            others.append(f"Reviewer {label}: {opinion}")
        other_block = "\n\n---\n\n".join(others)

        user_prompt = (
            f"You previously reviewed this research as a {current_name}. "
            f"Now review these anonymized opinions from other reviewers:\n\n"
            f"{other_block}\n\n"
            f"For each opinion:\n"
            f"1. Rate it 1-5 (quality of reasoning and evidence)\n"
            f"2. Where do you agree or disagree?\n"
            f"3. What critical point did they miss?\n"
            f"4. Are there factual errors or unsupported claims?\n\n"
            f"Be concise (150-250 words). Focus on substantive disagreements."
        )
        content, usage = await _call_llm(
            model=reviewer_model,
            system_prompt=(
                "You are a rigorous peer reviewer. Prioritize evidence quality, "
                "logical consistency, and factual accuracy over style."
            ),
            user_prompt=user_prompt,
            temperature=PEER_REVIEW_TEMPERATURE,
            max_tokens=PEER_REVIEW_MAX_TOKENS,
            label=f"peer:{current_name}",
        )
        return current_name, content, usage

    results = await asyncio.gather(
        *[_peer(i, name) for i, (name, _, _) in enumerate(opinions)],
        return_exceptions=True,
    )
    reviews: List[Tuple[str, str, Optional[Dict[str, int]]]] = []
    for item in results:
        if isinstance(item, BaseException):
            logger.error("Council peer reviewer raised: %s", item)
            continue
        name, content, usage = item
        if content:
            reviews.append((name, content, usage))
    return reviews


# ---------------------------------------------------------------------------
# Stage 3 — chairman synthesis (single call)
# ---------------------------------------------------------------------------

async def _run_chairman(
    *,
    domain: str,
    opinions: List[Tuple[str, str, Optional[Dict[str, int]]]],
    reviews: List[Tuple[str, str, Optional[Dict[str, int]]]],
    chairman_model: str,
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Synthesize opinions + peer reviews into a final consensus report."""

    opinions_text = "\n\n".join(
        f"### {name}\n{opinion}" for name, opinion, _ in opinions
    )
    reviews_text = ""
    if reviews:
        reviews_text = "\n\n## Peer Reviews\n\n" + "\n\n".join(
            f"### {name}'s Review\n{review}" for name, review, _ in reviews
        )

    user_prompt = (
        f"You are the Chairman of a review council. Below are independent "
        f"analyses and peer reviews. Synthesize into a final consensus report.\n\n"
        f"## Independent Analyses\n{opinions_text}\n"
        f"{reviews_text}\n\n"
        f"Produce a structured synthesis:\n\n"
        f"## Council Consensus\n\n"
        f"### High-Confidence Findings\n"
        f"[Findings all/most reviewers agree on — these are reliable]\n\n"
        f"### Contested Points\n"
        f"[Where reviewers disagree — present both sides with evidence]\n\n"
        f"### Factual Corrections\n"
        f"[Errors or unsupported claims identified during peer review]\n\n"
        f"### Risk Assessment\n"
        f"[Key risks identified, rated Low/Medium/High/Critical]\n\n"
        f"### Chairman's Verdict\n"
        f"[Your balanced assessment. Weight evidence quality over majority "
        f"opinion. Be direct about what is well-supported vs speculative.]\n\n"
        f"### Disclaimer\n"
        f"This is AI-generated multi-perspective analysis. Verify critical "
        f"claims independently."
    )

    content, usage = await _call_llm(
        model=chairman_model,
        system_prompt=(
            "You are the Chairman of a research review council. Synthesize "
            "multiple independent analyses into a balanced, evidence-weighted "
            "consensus. Do not favor any single reviewer. Resolve conflicts by "
            "evidence quality, not by majority vote. Flag unsupported claims. "
            "Be direct about uncertainty."
        ),
        user_prompt=user_prompt,
        temperature=CHAIRMAN_TEMPERATURE,
        max_tokens=CHAIRMAN_MAX_TOKENS,
        label="chairman",
    )
    if content is None:
        # Degraded fallback: return the raw opinions so the user still gets signal.
        fallback = (
            "Council synthesis failed. Individual opinions follow:\n\n"
            f"{opinions_text}"
        )
        return fallback, usage
    return content, usage


# ---------------------------------------------------------------------------
# Cost/usage aggregation
# ---------------------------------------------------------------------------

def _sum_usage(*usage_lists: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
    total_in = 0
    total_out = 0
    for usage_list in usage_lists:
        for u in usage_list:
            if not u:
                continue
            total_in += int(u.get("input_tokens", 0) or 0)
            total_out += int(u.get("output_tokens", 0) or 0)
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
    }


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------

async def analyst_council_tool(
    data: str,
    domain: str = "general",
    depth: str = "quick",
    web_context: str = "",
) -> str:
    """Multi-perspective adversarial review council — main entry point.

    Args:
        data: The research data or analysis to review (JSON, text, report).
        domain: Review domain: ``finance``, ``medicine``, ``technology``,
                ``policy``, ``science``, or ``general``. Selects the persona set.
        depth: ``quick`` (3 reviewers, no peer review) or ``full`` (5 + peer review).
        web_context: Optional supplementary web research context.

    Returns:
        A JSON string with the synthesized consensus report and metadata.
    """
    if not data:
        return json.dumps({
            "success": False,
            "error": "Missing required parameter: data",
        })

    # Recursion guard — prevent nested council calls from the same async context.
    if _COUNCIL_IN_FLIGHT.get():
        logger.warning("Council recursion detected; skipping nested invocation")
        return json.dumps({
            "success": False,
            "error": "Council recursion detected; nested call skipped",
        })

    cfg = _load_council_config()
    council_cfg = cfg.get("council") or {}

    # Trivial-query bypass (short + keyword match)
    if _is_trivial_query(data, council_cfg):
        logger.info("Council: trivial query detected, bypassing")
        return json.dumps({
            "success": True,
            "skipped": True,
            "reason": "trivial_query",
            "synthesis": "",
        })

    # API key available?
    if not check_openrouter_api_key():
        return json.dumps({
            "success": False,
            "error": "OPENROUTER_API_KEY not set — council tool requires an API key",
        })

    # Depth handling
    depth_norm = (depth or "").lower()
    is_full_depth = depth_norm == "full"
    effective_depth = "full" if is_full_depth else "quick"

    reviewer_model = _pick_reviewer_model(cfg)
    chairman_model = _pick_chairman_model(cfg)
    personas = _select_personas(domain, full=is_full_depth)

    logger.info(
        "Council start: domain=%s depth=%s reviewers=%s reviewer_model=%s chairman_model=%s",
        domain, effective_depth, len(personas), reviewer_model, chairman_model,
    )

    token = _COUNCIL_IN_FLIGHT.set(True)
    try:
        # Stage 1: independent opinions (parallel)
        opinions = await _collect_opinions(
            data=data,
            web_context=web_context or "",
            personas=personas,
            reviewer_model=reviewer_model,
        )
        if not opinions:
            return json.dumps({
                "success": False,
                "error": "Council failed: no reviewer opinions were collected",
            })
        logger.info("Council stage 1: %s/%s opinions collected", len(opinions), len(personas))

        # Stage 2: anonymized peer review (full depth only, need ≥3 reviewers)
        reviews: List[Tuple[str, str, Optional[Dict[str, int]]]] = []
        if is_full_depth and len(opinions) >= 3:
            reviews = await _collect_peer_reviews(
                opinions=opinions,
                reviewer_model=reviewer_model,
            )
            logger.info("Council stage 2: %s peer reviews collected", len(reviews))

        # Stage 3: chairman synthesis (single call)
        synthesis, chairman_usage = await _run_chairman(
            domain=domain,
            opinions=opinions,
            reviews=reviews,
            chairman_model=chairman_model,
        )
    finally:
        _COUNCIL_IN_FLIGHT.reset(token)

    usage_total = _sum_usage(
        [u for _, _, u in opinions],
        [u for _, _, u in reviews],
        [chairman_usage],
    )

    result = {
        "success": True,
        "synthesis": synthesis,
        "domain": domain,
        "depth": effective_depth,
        "reviewers": [name for name, _, _ in opinions],
        "peer_reviews_collected": len(reviews),
        "models": {
            "reviewer_model": reviewer_model,
            "chairman_model": chairman_model,
        },
        "usage": usage_total,
    }
    return json.dumps(result, ensure_ascii=False)


def check_council_requirements() -> bool:
    """Return True if the council tool is ready to run.

    The council needs an OpenRouter-compatible API key to dispatch reviewer
    and chairman calls.
    """
    return check_openrouter_api_key()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402 — register at module import

COUNCIL_SCHEMA: Dict[str, Any] = {
    "name": "analyst_council",
    "description": (
        "Run a multi-perspective adversarial review council on any research "
        "data or analysis. 3-5 specialist reviewers independently evaluate, "
        "optionally peer-review each other, then a chairman synthesizes the "
        "consensus. Use for thorough, fact-checked research on any topic: "
        "stocks, medicine, technology, policy, science, or any domain. "
        "Call this tool AFTER you have completed research work and want an "
        "adversarial quality check before presenting the final answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": (
                    "Research data or analysis to review (JSON, text, or "
                    "report). The council will evaluate this content."
                ),
            },
            "domain": {
                "type": "string",
                "description": (
                    "Research domain: 'finance', 'medicine', 'technology', "
                    "'policy', 'science', or 'general'. Selects the persona "
                    "set used for review."
                ),
            },
            "depth": {
                "type": "string",
                "description": (
                    "'quick' (3 reviewers, no peer review — ~4 LLM calls) or "
                    "'full' (5 reviewers + peer review — ~11 LLM calls). "
                    "Default: quick."
                ),
            },
            "web_context": {
                "type": "string",
                "description": (
                    "Optional supplementary web research context to give the "
                    "reviewers additional grounding."
                ),
            },
        },
        "required": ["data"],
    },
}

registry.register(
    name="analyst_council",
    toolset="council",
    schema=COUNCIL_SCHEMA,
    handler=lambda args, **kw: analyst_council_tool(
        data=args.get("data", ""),
        domain=args.get("domain", "general"),
        depth=args.get("depth", "quick"),
        web_context=args.get("web_context", ""),
    ),
    check_fn=check_council_requirements,
    requires_env=["OPENROUTER_API_KEY"],
    is_async=True,
    description=(
        "Multi-perspective adversarial review council — 3-5 specialist "
        "reviewers + chairman synthesis. Research integrity layer."
    ),
    emoji="⚖️",
    mutates=False,
)
