"""Current-session goal control for model tool calls."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from tools.registry import registry


_ACCEPTANCE_EVIDENCE_UNSET = object()
_PORTABLE_ASSERTIONS = {
    "file_exists": "exists",
    "test_result": "passes",
    "command_exit": "exit=0",
    "url_status": "status=2xx",
}
_ARTIFACT_HASH_ASSERTION = re.compile(r"sha256=[0-9a-fA-F]{64}\Z")


def _validate_acceptance_evidence(
    condition: str,
    evidence: Any,
) -> List[Dict[str, str]]:
    if not isinstance(evidence, list):
        raise ValueError("acceptance_evidence must be an array")

    validated: List[Dict[str, str]] = []
    for entry in evidence:
        if not isinstance(entry, dict) or set(entry) != {
            "kind", "locator", "assertion",
        }:
            raise ValueError(
                "each acceptance evidence entry requires only kind, locator, and assertion"
            )
        kind = entry.get("kind")
        locator = entry.get("locator")
        assertion = entry.get("assertion")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (kind, locator, assertion)
        ):
            raise ValueError("acceptance evidence fields must be nonblank strings")
        expected = _PORTABLE_ASSERTIONS.get(kind)
        assertion_valid = (
            bool(_ARTIFACT_HASH_ASSERTION.fullmatch(assertion))
            if kind == "artifact_hash"
            else expected is not None and assertion == expected
        )
        if not assertion_valid:
            raise ValueError("unsupported acceptance evidence kind or assertion")
        if locator not in condition or assertion not in condition:
            raise ValueError(
                "goal condition must contain the evidence locator and assertion"
            )
        validated.append(
            {"kind": kind, "locator": locator, "assertion": assertion}
        )
    return validated


def _error(code: str, message: str, *, session_id: str = "") -> str:
    return json.dumps(
        {
            "success": False,
            "session_id": session_id or None,
            "error": {"code": code, "message": message},
        },
        ensure_ascii=False,
    )


def _state_payload(state: Optional[Any], goal: Dict[str, Any]) -> Dict[str, Any]:
    if state is None:
        return {
            **goal,
            "active": False,
            "paused": False,
            "turns_used": None,
            "max_turns": None,
            "revision": None,
            "stop_reason": None,
            "error_reason": None,
        }

    status = goal["status"]
    has_judge_error = status != "cleared" and bool(
        state.consecutive_parse_failures
        or state.consecutive_transport_failures
        or state.last_verdict == "skipped"
    )
    stop_reason = state.paused_reason
    if stop_reason is None and status in {"done", "cleared"}:
        stop_reason = state.last_reason
    return {
        **goal,
        "active": status == "active",
        "paused": status == "paused",
        "turns_used": state.turns_used,
        "max_turns": state.max_turns,
        "revision": state.revision,
        "stop_reason": stop_reason,
        "error_reason": state.last_reason if has_judge_error else None,
    }


def _configured_max_turns() -> int:
    """Resolve the same goal budget used by the human control surfaces."""
    from hermes_cli.goals import DEFAULT_MAX_TURNS

    try:
        from hermes_cli.config import load_config

        value = int(
            ((load_config() or {}).get("goals") or {}).get(
                "max_turns", DEFAULT_MAX_TURNS
            )
        )
        return value if value > 0 else DEFAULT_MAX_TURNS
    except Exception:
        return DEFAULT_MAX_TURNS


def goal_control_tool(
    *,
    action: str,
    condition: Optional[str] = None,
    max_turns: Optional[int] = None,
    acceptance_evidence: Any = _ACCEPTANCE_EVIDENCE_UNSET,
    session_id: Optional[str] = None,
    requested_session_id: Optional[str] = None,
) -> str:
    """Inspect or mutate only the calling session's canonical goal state."""
    caller_session_id = (session_id or "").strip()
    if not caller_session_id:
        return _error("missing_session_identity", "calling session identity is required")
    if requested_session_id is not None:
        return _error(
            "cross_session_forbidden",
            "goal control is restricted to the calling session",
            session_id=caller_session_id,
        )

    action = (action or "status").strip().lower()
    if action not in {"status", "set", "update", "pause", "resume", "clear"}:
        return _error(
            "invalid_action",
            f"unsupported goal action: {action}",
            session_id=caller_session_id,
        )
    if (
        acceptance_evidence is not _ACCEPTANCE_EVIDENCE_UNSET
        and action not in {"set", "update"}
    ):
        return _error(
            "invalid_acceptance_evidence",
            "acceptance_evidence is supported only for set/update",
            session_id=caller_session_id,
        )

    try:
        from hermes_cli.goals import (
            GoalConflictError,
            GoalManager,
            GoalMutationOutcomeUnknownError,
            GoalPostconditionError,
            goal_state_payload,
            load_goal_snapshot_authoritative,
        )

        # This pre-read distinguishes "no goal" from unavailable/corrupt
        # persistence before any mutation is attempted.
        before, before_raw = load_goal_snapshot_authoritative(caller_session_id)
        configured_max_turns = _configured_max_turns()
        manager = GoalManager.from_authoritative_snapshot(
            session_id=caller_session_id,
            state=before,
            persisted_raw=before_raw,
            default_max_turns=configured_max_turns,
        )

        intended = before
        expected_after_raw = before_raw
        if action in {"set", "update"}:
            text = (condition or "").strip()
            if not text:
                return _error(
                    "invalid_condition",
                    "condition is required for set/update",
                    session_id=caller_session_id,
                )
            if max_turns is not None and int(max_turns) <= 0:
                return _error(
                    "invalid_budget",
                    "max_turns must be a positive integer",
                    session_id=caller_session_id,
                )
            if (
                action == "set"
                and max_turns is not None
                and int(max_turns) > configured_max_turns
            ):
                return _error(
                    "invalid_budget",
                    "max_turns cannot exceed the configured goal budget",
                    session_id=caller_session_id,
                )
            if action == "set" and before is not None:
                return _error(
                    "invalid_transition",
                    "set requires a session with no persisted goal",
                    session_id=caller_session_id,
                )
            if action == "update" and (
                before is None or before.status not in {"active", "paused"}
            ):
                return _error(
                    "invalid_transition",
                    "only an active or paused goal can be updated",
                    session_id=caller_session_id,
                )
            if (
                action == "update"
                and max_turns is not None
                and int(max_turns) > before.max_turns
            ):
                return _error(
                    "invalid_budget",
                    "update cannot increase the current goal budget",
                    session_id=caller_session_id,
                )
            try:
                if acceptance_evidence is _ACCEPTANCE_EVIDENCE_UNSET:
                    validated_evidence = (
                        _validate_acceptance_evidence(
                            text, before.acceptance_evidence
                        )
                        if action == "update"
                        else None
                    )
                else:
                    validated_evidence = _validate_acceptance_evidence(
                        text, acceptance_evidence
                    )
            except ValueError as exc:
                return _error(
                    "invalid_acceptance_evidence",
                    str(exc),
                    session_id=caller_session_id,
                )
            if action == "update":
                intended = manager.update(
                    text,
                    max_turns=max_turns,
                    acceptance_evidence=validated_evidence,
                    expected_raw=before_raw,
                )
            else:
                intended = manager.set(
                    text,
                    max_turns=max_turns,
                    acceptance_evidence=validated_evidence or [],
                    expected_raw=before_raw,
                )
            expected_after_raw = intended.to_json() if intended else None
        elif action == "pause":
            if before is None or before.status != "active":
                return _error(
                    "invalid_transition",
                    "only an active goal can be paused",
                    session_id=caller_session_id,
                )
            intended = manager.pause(
                reason="model-paused", expected_raw=before_raw
            )
            expected_after_raw = intended.to_json() if intended else None
        elif action == "resume":
            if before is None or before.status != "paused":
                return _error(
                    "invalid_transition",
                    "only a paused goal can be resumed",
                    session_id=caller_session_id,
                )
            intended = manager.resume(
                reset_budget=False, expected_raw=before_raw
            )
            expected_after_raw = intended.to_json() if intended else None
        elif action == "clear":
            if before is not None and before.status != "cleared":
                intended = manager.clear(
                    reason="model-cleared", expected_raw=before_raw
                )
                expected_after_raw = intended.to_json() if intended else None

        persisted, persisted_raw = load_goal_snapshot_authoritative(
            caller_session_id
        )
        if action != "status" and persisted_raw != expected_after_raw:
            code = (
                "transition_conflict"
                if action in {"pause", "resume"}
                else "mutation_conflict"
            )
            return _error(
                code,
                "persisted goal state does not match the requested action",
                session_id=caller_session_id,
            )
    except GoalConflictError:
        code = "transition_conflict" if action in {"pause", "resume"} else "mutation_conflict"
        return _error(
            code,
            "persisted goal changed while applying the requested action",
            session_id=caller_session_id,
        )
    except GoalMutationOutcomeUnknownError:
        return _error(
            "mutation_outcome_unknown",
            (
                "goal mutation may have committed, but persisted state could not be "
                "verified; do not retry blindly without first reading authoritative status"
            ),
            session_id=caller_session_id,
        )
    except GoalPostconditionError:
        code = "transition_conflict" if action in {"pause", "resume"} else "mutation_conflict"
        return _error(
            code,
            "persisted goal state could not be verified after mutation",
            session_id=caller_session_id,
        )
    except (TypeError, ValueError) as exc:
        return _error("invalid_request", str(exc), session_id=caller_session_id)
    except Exception:
        return _error(
            "persistence_unavailable",
            "goal state could not be read back from canonical storage",
            session_id=caller_session_id,
        )

    goal_payload = goal_state_payload(persisted)
    state_payload = _state_payload(persisted, goal_payload)
    revision = state_payload["revision"]
    return json.dumps(
        {
            "success": True,
            "session_id": caller_session_id,
            "action": action,
            "state": state_payload,
            "goal_readback": {
                "kind": "goal-status-readback",
                "receipt_id": (
                    f"goal_control:{caller_session_id}:revision:{revision}"
                    if revision is not None
                    else f"goal_control:{caller_session_id}:absent"
                ),
                "session_id": caller_session_id,
                "revision": revision,
                "active": state_payload["active"],
                "condition": state_payload["condition"],
                "goal": goal_payload,
                "observed_via": "goal_control",
                "acceptance_evidence": (
                    [dict(item) for item in persisted.acceptance_evidence]
                    if persisted is not None
                    else []
                ),
            },
        },
        ensure_ascii=False,
    )


GOAL_CONTROL_SCHEMA = {
    "name": "goal_control",
    "description": (
        "Inspect or manage the persisted goal for the current session. "
        "Every successful call returns authoritative persisted state. "
        "Set is allowed only when no goal row exists; update preserves active "
        "or paused lifecycle progress and cannot increase its budget; model "
        "resume preserves consumed turns. The calling session is the only "
        "permitted scope."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "set", "update", "pause", "resume", "clear"],
            },
            "condition": {
                "type": "string",
                "description": "Exact goal condition; required for set/update.",
            },
            "max_turns": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional continuation-turn budget. Set may not exceed the "
                    "configured goal budget; update may only decrease it."
                ),
            },
            "acceptance_evidence": {
                "type": "array",
                "description": (
                    "Portable acceptance checks named verbatim in the goal condition. "
                    "On update, omit to preserve, pass [] to clear, or pass a list to replace."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "file_exists",
                                "test_result",
                                "command_exit",
                                "url_status",
                                "artifact_hash",
                            ],
                        },
                        "locator": {"type": "string", "minLength": 1},
                        "assertion": {"type": "string", "minLength": 1},
                    },
                    "required": ["kind", "locator", "assertion"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _handle_goal_control(args: Dict[str, Any], **kwargs: Any) -> str:
    return goal_control_tool(
        action=args.get("action", "status"),
        condition=args.get("condition"),
        max_turns=args.get("max_turns"),
        acceptance_evidence=(
            args["acceptance_evidence"]
            if "acceptance_evidence" in args
            else _ACCEPTANCE_EVIDENCE_UNSET
        ),
        session_id=kwargs.get("session_id"),
        # The schema intentionally omits session_id. Preserve this explicit
        # guard because not every provider enforces additionalProperties.
        requested_session_id=args.get("session_id") if "session_id" in args else None,
    )


registry.register(
    name="goal_control",
    toolset="goal",
    schema=GOAL_CONTROL_SCHEMA,
    handler=_handle_goal_control,
    check_fn=lambda: True,
)
