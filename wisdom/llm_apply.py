"""Safe LLM-assisted Wisdom application proposal generation."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.auxiliary_client import call_llm, extract_content_or_reasoning
from wisdom.apply import deterministic_application_dicts
from wisdom.models import CaptureRecord, VALID_APPLICATION_TYPES
from wisdom.redaction import detect_secret_like_text


_EXPECTED_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "business": ("client_language", "principle", "task_proposal"),
    "investing": ("investment_rule", "checklist", "decision_rule"),
    "health": ("health_experiment", "decision_rule", "principle"),
    "life": ("principle", "writing_idea", "decision_rule"),
    "inbox": ("principle", "writing_idea"),
}


def llm_application_proposals(
    capture: CaptureRecord,
    *,
    timeout: float = 30.0,
) -> list[dict[str, object]] | None:
    """Return validated LLM proposals, or None when fallback is required."""
    try:
        response = call_llm(
            task="wisdom_apply",
            messages=_messages(capture),
            temperature=0.2,
            max_tokens=1200,
            timeout=timeout,
        )
        content = extract_content_or_reasoning(response)
        proposals = _validate_payload(_json_object(content), capture)
        model_used = str(getattr(response, "model", "") or "")
        for proposal in proposals:
            metadata = dict(proposal.get("metadata") or {})
            metadata.update(
                {
                    "generator_version": 3,
                    "generator": "llm",
                    "model_used": model_used,
                }
            )
            proposal["metadata"] = metadata
        return proposals
    except Exception:
        return None


def _messages(capture: CaptureRecord) -> list[dict[str, str]]:
    allowed = _EXPECTED_BY_CATEGORY.get(capture.category, _EXPECTED_BY_CATEGORY["inbox"])
    metadata = {key: value for key, value in capture.metadata.items() if key in {"source", "context", "context_note"}}
    deterministic_examples = deterministic_application_dicts(capture)
    example_body = "\n".join(
        f"- {item['application_type']}: {item['body']}" for item in deterministic_examples
    )
    return [
        {
            "role": "system",
            "content": (
                "Generate safe, domain-specific application proposals for Hermes Wisdom. "
                "Return only JSON. Do not create tasks, reminders, files, messages, SQL, "
                "or external actions. Do not modify or restate the original as if it were "
                "the source of truth; generate proposals only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original capture:\n"
                f"{capture.original_text}\n\n"
                f"Category: {capture.category}\n"
                f"Source type: {capture.source_type}\n"
                f"Metadata: {json.dumps(metadata, sort_keys=True)}\n"
                f"Allowed application types: {', '.join(allowed)}\n\n"
                "Use the same JSON shape exactly:\n"
                '{"applications":[{"application_type":"client_language",'
                '"title":"Client language","body":"..."}]}\n\n'
                "Quality bar:\n"
                "- business/x10x: client language, operating principle, report/process proposal\n"
                "- investing: investment rule, checklist, risk or decision rule\n"
                "- health: health experiment, personal rule, tracking question, decision boundary\n"
                "- life: principle, reflection prompt, writing seed, decision rule\n\n"
                "Deterministic baseline to beat without becoming verbose:\n"
                f"{example_body}"
            ),
        },
    ]


def _json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM Wisdom apply output must be a JSON object")
    return data


def _validate_payload(data: dict[str, Any], capture: CaptureRecord) -> list[dict[str, object]]:
    raw_apps = data.get("applications")
    if not isinstance(raw_apps, list):
        raise ValueError("LLM Wisdom apply output missing applications list")

    expected = _EXPECTED_BY_CATEGORY.get(capture.category, _EXPECTED_BY_CATEGORY["inbox"])
    expected_set = set(expected)
    proposals_by_type: dict[str, dict[str, object]] = {}
    for item in raw_apps[:5]:
        if not isinstance(item, dict):
            continue
        app_type = str(item.get("application_type") or "").strip()
        if app_type not in VALID_APPLICATION_TYPES or app_type not in expected_set:
            continue
        title = _clean_text(item.get("title"), limit=90)
        body = _clean_text(item.get("body"), limit=1400)
        if len(body) < 30 or detect_secret_like_text(body):
            continue
        proposals_by_type[app_type] = {
            "application_type": app_type,
            "title": title or _title_for(app_type),
            "body": body,
            "status": "proposed",
            "metadata": {},
        }

    if set(proposals_by_type) != expected_set:
        raise ValueError("LLM Wisdom apply output did not cover required application types")
    return [proposals_by_type[app_type] for app_type in expected]


def _clean_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit].rstrip()


def _title_for(application_type: str) -> str:
    return application_type.replace("_", " ").title()
