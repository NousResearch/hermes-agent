"""Browser autonomy guardrails.

This module gives browser tools a small, structured read on whether a page is
safe to keep driving autonomously or whether the agent has hit a human
verification / anti-bot wall. It is intentionally detection + handoff only:
Hermes must not solve, bypass, or outsource CAPTCHA-style challenges.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.registry import registry


_MAX_EXCERPT = 160


_HUMAN_VERIFICATION_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("captcha", re.compile(r"\b(?:captcha|hcaptcha|recaptcha)\b", re.I)),
    ("turnstile", re.compile(r"\bturnstile\b|\bcf-chl\b|challenge-platform", re.I)),
    (
        "human_verification",
        re.compile(
            r"(?:verify|confirm|prove)\s+(?:that\s+)?(?:you\s+are|you're|yourself)\s+"
            r"(?:a\s+)?human|i\s+am\s+not\s+a\s+robot|are\s+you\s+(?:a\s+)?robot",
            re.I,
        ),
    ),
    (
        "challenge",
        re.compile(
            r"complete\s+(?:the\s+)?(?:security\s+)?(?:verification|challenge)|"
            r"security\s+check|checking\s+your\s+browser|just\s+a\s+moment",
            re.I,
        ),
    ),
)


_BOT_WALL_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("access_denied", re.compile(r"\baccess\s+denied\b|request\s+blocked", re.I)),
    ("bot_detection", re.compile(r"\bbot\s+detected\b|automated\s+(?:traffic|queries)", re.I)),
    ("rate_or_abuse_wall", re.compile(r"unusual\s+traffic|temporarily\s+blocked|too\s+many\s+requests", re.I)),
    ("cloudflare_wall", re.compile(r"\bcloudflare\b|ddos\s+protection|attention\s+required", re.I)),
)


def _clip(text: str, match: re.Match[str]) -> str:
    start = max(match.start() - 60, 0)
    end = min(match.end() + 60, len(text))
    excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
    if len(excerpt) <= _MAX_EXCERPT:
        return excerpt
    return excerpt[: _MAX_EXCERPT - 1].rstrip() + "..."


def _iter_matches(
    fields: Dict[str, str],
    rules: Iterable[Tuple[str, re.Pattern[str]]],
) -> List[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field, text in fields.items():
        if not text:
            continue
        for rule_name, pattern in rules:
            match = pattern.search(text)
            if match is None:
                continue
            key = (field, rule_name)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "field": field,
                    "rule": rule_name,
                    "excerpt": _clip(text, match),
                }
            )
    return matches


def assess_browser_state(
    *,
    url: str = "",
    title: str = "",
    snapshot: str = "",
    console_text: str = "",
) -> Dict[str, Any]:
    """Classify whether the current browser state can proceed autonomously.

    Returns a stable JSON-serializable shape. The result is deliberately
    conservative: human-verification and CAPTCHA-like pages produce
    ``status='needs_human'`` and instructions to pause, not to bypass.
    """

    fields = {
        "url": url or "",
        "title": title or "",
        "snapshot": snapshot or "",
        "console": console_text or "",
    }

    human_matches = _iter_matches(fields, _HUMAN_VERIFICATION_RULES)
    if human_matches:
        return {
            "status": "needs_human",
            "kind": "human_verification",
            "confidence": "high",
            "matches": human_matches[:6],
            "message": (
                "This page appears to require CAPTCHA or human verification. "
                "Pause autonomous browser actions, ask the user to complete "
                "the check or provide an official API/auth route, then resume "
                "only after a fresh browser_snapshot."
            ),
            "policy": "do_not_solve_or_bypass_captcha",
            "safe_next_steps": [
                "Ask the user to complete the verification in the browser session.",
                "Use an official API, OAuth flow, export, webhook, or integration instead of browser scraping.",
                "After the user confirms completion, call browser_snapshot before continuing.",
            ],
        }

    bot_matches = _iter_matches(fields, _BOT_WALL_RULES)
    if bot_matches:
        return {
            "status": "blocked",
            "kind": "bot_or_access_wall",
            "confidence": "medium",
            "matches": bot_matches[:6],
            "message": (
                "The site appears to have blocked automated access. Do not try "
                "to evade the wall. Prefer an official API/auth path, reduce "
                "request volume, or ask the user how they want to proceed."
            ),
            "policy": "do_not_evade_access_controls",
            "safe_next_steps": [
                "Look for a documented API or account export route.",
                "Ask the user for permission to continue manually if the site allows it.",
                "Stop browser automation for this site when access is denied.",
            ],
        }

    return {
        "status": "ready",
        "kind": "normal",
        "confidence": "low",
        "matches": [],
        "message": "No obvious human-verification or bot-wall blocker detected.",
        "safe_next_steps": [],
    }


def annotate_browser_response(
    response: Dict[str, Any],
    *,
    url: str = "",
    title: str = "",
    snapshot: str = "",
    console_text: str = "",
) -> Dict[str, Any]:
    """Attach an autonomy assessment to a browser tool response."""

    assessment = assess_browser_state(
        url=url,
        title=title,
        snapshot=snapshot,
        console_text=console_text,
    )
    response["autonomy"] = assessment
    if assessment["status"] in {"needs_human", "blocked"}:
        response["human_handoff_required"] = assessment["status"] == "needs_human"
        response["bot_detection_warning"] = assessment["message"]
    return response


BROWSER_AUTONOMY_SCHEMA: Dict[str, Any] = {
    "name": "browser_autonomy_check",
    "description": (
        "Assess the current browser page for autonomy blockers such as CAPTCHA, "
        "human verification, bot walls, and access-denied pages. This tool does "
        "not solve or bypass verification challenges; it returns a safe handoff "
        "plan when human input or an official API route is required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "snapshot": {
                "type": "string",
                "description": "Optional page snapshot/text to classify.",
            },
            "title": {
                "type": "string",
                "description": "Optional current page title.",
            },
            "url": {
                "type": "string",
                "description": "Optional current page URL.",
            },
        },
        "required": [],
    },
}


def browser_autonomy_check(
    snapshot: Optional[str] = None,
    title: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    return json.dumps(
        {
            "success": True,
            "autonomy": assess_browser_state(
                url=url or "",
                title=title or "",
                snapshot=snapshot or "",
            ),
        },
        ensure_ascii=False,
    )


registry.register(
    name="browser_autonomy_check",
    toolset="browser",
    schema=BROWSER_AUTONOMY_SCHEMA,
    handler=lambda args, **_kw: browser_autonomy_check(
        snapshot=args.get("snapshot"),
        title=args.get("title"),
        url=args.get("url"),
    ),
    emoji="🧭",
)
