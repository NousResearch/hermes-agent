#!/usr/bin/env python3
"""
Research Pipeline Runner — legacy single-entry tool for simple web research.

Heavy RESEARCH, DECISION, and RESEARCH_DECISION tasks must use
task_engine_runner instead. This legacy runner fails closed when such a mode is
detected, so the old DeepSeek/Flash path cannot bypass the canonical engine.

Pipeline:
  Stage 1: agy (Gemini 3.1 Pro) → intelligence_report
  Stage 2: ddgs → supplementary_search_results (only after Stage 1)
  Stage 3: synthesis (R1-32B or Gemini, planned for v2)
  Stage 4: structured result returned to DeepSeek

For quick mode, Stage 1 is skipped and only ddgs runs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.task_engine_contracts import detect_task_engine_mode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RESEARCH_PIPELINE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "research_pipeline_runner",
        "description": (
            "Legacy path for simple web research. Do NOT use this for Hermes "
            "RESEARCH, DECISION, or RESEARCH_DECISION tasks; those must use "
            "task_engine_runner. Pipeline: agy (Gemini) intelligence report -> "
            "ddgs supplementary -> returns structured results with artifact paths. "
            "Use mode='quick' only for simple factual lookups that do not need "
            "the canonical multi-model task engine."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or search query. For research mode, make this a comprehensive question covering all aspects."
                },
                "mode": {
                    "type": "string",
                    "enum": ["research", "quick"],
                    "description": "research: full pipeline (agy→ddgs→synthesis). quick: ddgs-only for simple lookups.",
                    "default": "research"
                },
                "subtopics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: 3-5 subtopic queries for supplementary ddgs search. If omitted, the runner generates them from the main query."
                }
            },
            "required": ["query"]
        }
    }
}

# ---------------------------------------------------------------------------
# Artifact storage
# ---------------------------------------------------------------------------

ARTIFACT_DIR = Path(os.path.expanduser("~/.hermes/research_artifacts"))


def _save_artifact(name: str, content: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Stage 1: agy (Gemini) Intelligence Report
# ---------------------------------------------------------------------------

def _run_agy_intelligence_report(query: str, timeout_seconds: int = 180) -> Dict[str, Any]:
    """Run agy with Gemini 3.1 Pro for the intelligence report."""
    agy_path = "/opt/homebrew/bin/agy"

    if not os.path.exists(agy_path):
        return {
            "status": "error",
            "stage": "agy_intelligence_report",
            "error": f"agy not found at {agy_path}",
            "blocked": True
        }

    # Build a research-oriented prompt
    prompt = (
        f"You are an intelligence analyst. Research the following topic thoroughly. "
        f"Include: key facts, latest developments (2024-2026), competing perspectives, "
        f"authoritative sources, and areas of uncertainty. Be comprehensive but concise.\n\n"
        f"TOPIC: {query}\n\n"
        f"Format your response as a structured intelligence report."
    )

    try:
        logger.info("Stage 1: Running agy intelligence report for query: %s", query[:80])
        result = subprocess.run(
            [agy_path, "--model", "Gemini 3.1 Pro (High)", "-p", prompt,
             "--print-timeout", f"{timeout_seconds}s"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 30,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "agy exited with non-zero status"
            return {
                "status": "error",
                "stage": "agy_intelligence_report",
                "error": error_msg,
                "returncode": result.returncode
            }

        output = result.stdout.strip()
        stderr_output = result.stderr.strip()

        # Detect authentication required
        if "Authentication required" in output or "Authentication required" in stderr_output:
            # Extract the auth URL
            combined = output + "\n" + stderr_output
            import re
            auth_url_match = re.search(r'https://accounts\.google\.com/o/oauth2/auth\S+', combined)
            auth_url = auth_url_match.group(0) if auth_url_match else None

            return {
                "status": "auth_required",
                "stage": "agy_intelligence_report",
                "error": "agy requires browser Google authentication",
                "auth_url": auth_url,
                "deepseek_actions": {
                    "primary": {
                        "tool": "browser_navigate",
                        "args": {"url": auth_url} if auth_url else None,
                        "purpose": "Open the Google auth page in browser. User signs in, token auto-exchanges via agy local redirect."
                    },
                    "fallback": {
                        "tool": "clarify",
                        "args": {
                            "question": f"agy需要Google认证。请打开这个链接完成登录，然后把浏览器地址栏里的验证码粘贴到这里：\n{auth_url}" if auth_url else "agy需要Google认证但无法获取URL。请手动运行 agy -p 'test' 完成首次认证。"
                        },
                        "purpose": "If browser_navigate doesn't work or user needs to paste the auth code manually."
                    },
                    "after_auth": "Retry research_pipeline_runner with the same query. agy will use the newly cached token."
                },
                "blocked": False,
                "retry_after_auth": True
            }

        if not output:
            return {
                "status": "error",
                "stage": "agy_intelligence_report",
                "error": "agy returned empty output (likely timed out)",
                "blocked": True
            }

        # Save artifact
        artifact_path = _save_artifact(
            f"intelligence_report_{int(time.time())}.md",
            output
        )

        return {
            "status": "ok",
            "stage": "agy_intelligence_report",
            "content": output[:2000] + ("..." if len(output) > 2000 else ""),
            "artifact_path": str(artifact_path),
            "content_length": len(output),
            "model": "Gemini 3.1 Pro (via agy)"
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "stage": "agy_intelligence_report",
            "error": f"agy timed out after {timeout_seconds}s",
            "blocked": True
        }
    except Exception as e:
        return {
            "status": "error",
            "stage": "agy_intelligence_report",
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# Stage 2: ddgs Supplementary Search
# ---------------------------------------------------------------------------

def _run_ddgs_supplementary(query: str, subtopics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run ddgs as supplementary search (only called after agy completes)."""
    try:
        from ddgs import DDGS

        searches = subtopics or [query]
        if query not in searches:
            searches.insert(0, query)

        all_hits = []
        with DDGS() as ddgs:
            for q in searches[:5]:  # Cap at 5 queries
                try:
                    hits = list(ddgs.text(q, max_results=5))
                    for h in hits:
                        all_hits.append({
                            "query": q[:80],
                            "title": h.get("title", ""),
                            "url": h.get("href", ""),
                            "snippet": h.get("body", "")[:300]
                        })
                    time.sleep(1)
                except Exception as e:
                    logger.warning("ddgs query failed for '%s': %s", q[:60], e)

        content_lines = []
        for i, hit in enumerate(all_hits):
            content_lines.append(f"[{i+1}] {hit['title']}")
            content_lines.append(f"    URL: {hit['url']}")
            content_lines.append(f"    {hit['snippet']}")
            content_lines.append("")

        content = "\n".join(content_lines)
        artifact_path = _save_artifact(
            f"supplementary_search_{int(time.time())}.md",
            content
        )

        return {
            "status": "ok",
            "stage": "ddgs_supplementary",
            "hits_count": len(all_hits),
            "searches_run": len(searches[:5]),
            "content": content[:2000] + ("..." if len(content) > 2000 else ""),
            "artifact_path": str(artifact_path)
        }

    except ImportError:
        return {
            "status": "error",
            "stage": "ddgs_supplementary",
            "error": "ddgs package not installed"
        }
    except Exception as e:
        return {
            "status": "error",
            "stage": "ddgs_supplementary",
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def research_pipeline_runner(
    query: str,
    mode: str = "research",
    subtopics: Optional[List[str]] = None
) -> str:
    """Execute the research pipeline and return structured results.

    Args:
        query: The research question
        mode: "research" (full pipeline) or "quick" (ddgs only)
        subtopics: Optional list of subtopic queries for ddgs

    Returns:
        JSON string with status, stages, and artifact paths
    """
    mode = mode or "research"
    stages = {}
    start_time = time.time()
    heavy_mode = detect_task_engine_mode(query)
    if heavy_mode:
        return json.dumps({
            "pipeline_status": "PIPELINE_BLOCKED",
            "blocked_stage": "legacy_research_pipeline_runner",
            "required_tool": "task_engine_runner",
            "detected_mode": heavy_mode,
            "message": (
                "Legacy research_pipeline_runner is blocked for RESEARCH, "
                "DECISION, and RESEARCH_DECISION tasks. Use task_engine_runner "
                "to generate the canonical 72B-first contract and validate the "
                "full fail-closed pipeline."
            ),
            "duration_seconds": round(time.time() - start_time, 1),
        }, ensure_ascii=False, indent=2)

    # Stage 1: agy (Gemini) intelligence report
    if mode == "research":
        stages["agy_report"] = _run_agy_intelligence_report(query)
        agy_status = stages["agy_report"].get("status")

        if agy_status == "error":
            # If agy fails hard, we return blocked — DeepSeek must report the block
            return json.dumps({
                "pipeline_status": "blocked",
                "blocked_stage": "agy_intelligence_report",
                "error": stages["agy_report"].get("error", "unknown agy error"),
                "message": (
                    "RESEARCH PIPELINE BLOCKED: agy (Gemini) intelligence report failed. "
                    "The pipeline cannot proceed without Stage 1. Report this blocker to the user. "
                    "Do NOT use ddgs/web_search as a replacement."
                ),
                "stages": stages,
                "duration_seconds": round(time.time() - start_time, 1)
            }, ensure_ascii=False, indent=2)

        if agy_status == "auth_required":
            # Auth needed — return immediately, DeepSeek handles auth flow, then retries
            return json.dumps({
                "pipeline_status": "auth_required",
                "blocked_stage": "agy_intelligence_report",
                "stages": stages,
                "deepseek_actions": stages["agy_report"].get("deepseek_actions", {}),
                "message": (
                    "agy needs Google authentication. "
                    "Use browser_navigate to open the auth URL, or clarify() to ask user for the code. "
                    "Then retry research_pipeline_runner."
                ),
                "duration_seconds": round(time.time() - start_time, 1)
            }, ensure_ascii=False, indent=2)

    # Stage 2: ddgs supplementary search
    if mode == "research":
        # Generate subtopics if not provided
        if not subtopics:
            subtopics = [
                f"{query} latest research 2024 2025 2026",
                f"{query} guidelines recommendations",
                f"{query} controversies limitations"
            ]
        stages["ddgs_supplementary"] = _run_ddgs_supplementary(query, subtopics)
    elif mode == "quick":
        stages["ddgs_supplementary"] = _run_ddgs_supplementary(query)

    # Build final result
    duration = round(time.time() - start_time, 1)
    all_ok = all(
        s.get("status") == "ok"
        for s in stages.values()
        if s.get("status") is not None
    )

    result = {
        "pipeline_status": "ok" if all_ok else "partial",
        "mode": mode,
        "query": query,
        "duration_seconds": duration,
        "stages": stages,
        "artifacts": {
            name: s.get("artifact_path")
            for name, s in stages.items()
            if s.get("artifact_path")
        },
        "next_steps": (
            "DeepSeek: synthesize findings from the intelligence report and "
            "supplementary search results above. Structure the output as a "
            "comprehensive report with inline source references. "
            "Then produce ARTIFACT_AUDIT + USER_REPORT."
        ) if mode == "research" else "DeepSeek: use these search results to answer the user's query."
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry


def _research_pipeline_handler(args: Dict[str, Any], **kw) -> str:
    """Tool handler — bridges the JSON Schema args to the implementation."""
    return research_pipeline_runner(
        query=args.get("query", ""),
        mode=args.get("mode", "research"),
        subtopics=args.get("subtopics")
    )


def _check_research_pipeline_requirements() -> bool:
    """Check if the research pipeline can run. No blocking preconditions — always available."""
    return True  # Always available; heavy tasks fail closed at runtime.


registry.register(
    name="research_pipeline_runner",
    toolset="research",
    schema=RESEARCH_PIPELINE_SCHEMA,
    handler=_research_pipeline_handler,
    check_fn=_check_research_pipeline_requirements,
    emoji="🔬",
    description="Single-entry research pipeline: agy(Gemini)→ddgs→synthesis. Only path for web research."
)
