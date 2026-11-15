"""Regression tests for skill_view dedup seeding when tool outputs carry trailing suffixes.

In production, post-tool hooks (such as ``[skill-error-router]...``) and tool-loop
guardrails (``[Tool loop warning: idempotent_no_progress_warning; ...]``) append
non-JSON text after the closing ``}`` of a ``skill_view`` JSON payload. Both
``parse_unpruned_skill_view_payload`` and ``seed_skill_view_dedup_from_messages``
must parse the leading JSON object via ``raw_decode`` and hash the exact JSON
prefix so surviving tail skills remain deduplicated across compaction boundaries.
"""

import json

from tools.skills_tool_dedup import (
    dedup_skill_view,
    parse_unpruned_skill_view_payload,
    reset_skill_view_dedup,
)


def _make_skill_view_json(name: str = "uop-gws-gam-ops", body: str = "# GAM Ops\nInstructions") -> tuple[dict, str]:
    payload = {
        "success": True,
        "name": name,
        "description": "Use when working with U&P Google Workspace via GAM.",
        "category": "",
        "source": "user",
        "skill_dir": f"/Users/emirsaffar/.hermes/skills/{name}",
        "content": body,
        "linked_files": {"references": ["references/gam-recipes.md"]},
        "readiness_status": "ready",
        "setup_needed": False,
    }
    return payload, json.dumps(payload, ensure_ascii=False)


def test_parse_unpruned_skill_view_payload_ignores_trailing_hook_and_guardrail_suffixes():
    _payload, raw_json = _make_skill_view_json("google-oauth-ops")
    suffixed = (
        raw_json
        + "\n\n[skill-error-router] Relevant skill hint — you do NOT need to switch tasks\n"
        + "[[skill-error-router] end]"
        + "\n\n[Tool loop warning: idempotent_no_progress_warning; count=2; "
        + "skill_view returned the same result 2 times with identical arguments.]"
    )
    parsed = parse_unpruned_skill_view_payload(suffixed)
    assert parsed is not None
    parsed_payload, json_prefix = parsed
    assert parsed_payload["name"] == "google-oauth-ops"
    assert json_prefix == raw_json


def test_reset_skill_view_dedup_reseeds_surviving_tail_with_trailing_suffixes():
    task_id = "test_suffix_reseed_task"
    reset_skill_view_dedup(task_id)
    payload, raw_json = _make_skill_view_json("uop-gws-gam-ops")
    suffixed_tool_msg = (
        raw_json
        + "\n\n[skill-error-router] Matched keywords: invalid_grant, RAPT\n"
        + "[[skill-error-router] end]"
    )
    surviving_messages = [
        {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\n..."},
        {"role": "tool", "tool_call_id": "call_gam_1", "content": suffixed_tool_msg},
    ]
    reset_skill_view_dedup(task_id, surviving_messages=surviving_messages)

    # Because the seeded digest hashes the raw_decode JSON prefix (without the
    # post-tool hook suffix), a subsequent redundant skill_view call before any
    # hook runs matches the digest and returns the short unchanged stub.
    dedup_stub = dedup_skill_view(task_id, "uop-gws-gam-ops", None, payload, raw_json)
    assert dedup_stub is not None
    stub_obj = json.loads(dedup_stub)
    assert stub_obj["dedup"] is True
    assert stub_obj["status"] == "unchanged"
    assert stub_obj["name"] == "uop-gws-gam-ops"
    reset_skill_view_dedup(task_id)
