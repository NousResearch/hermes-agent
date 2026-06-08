"""Plan 058-F — Hermes-side ``plan_lifecycle`` tool.

Turns a Slack goal into a fully-decomposed plan by running an adaptive
5-dimension coverage loop in-thread, then POSTing the completed spec to
the agentic-hub orchestrator sensor's ``POST /plan/decompose`` endpoint
(in-VPC) with an HMAC signature.

Design notes
------------
- **Vendored coverage helpers** — this fork cannot import agentic-hub so
  the 5-dimension coverage check is self-contained (same pattern as
  ``cron/run_record.py`` 056-D vendor). The canonical dimension list mirrors
  ``orchestrator/registry/spec_intake.py``.

- **Adaptive questioning loop** — the tool returns a question string when
  coverage is incomplete so the *agent* asks it in-thread. The conversation
  IS the loop; there is no internal recursion.

- **Budget gate** — the tool refuses to POST if ``budget_usd`` is absent
  or None, mirroring the orchestrator's own budget warning. This prevents
  a dispatch that would immediately stall the drain.

- **HTTP client** — synchronous ``httpx.post`` (same library as
  ``tools/skills_hub.py``). The tool handler is registered as synchronous
  (no ``is_async=True``) because the POST is a single short-lived network
  call and blocking the thread is fine for this low-frequency tool.

- **HMAC signing** — ``Linear-Signature: <hex>`` exactly as verified by
  ``orchestrator/linear/sensors.py::_verify_signature``:
      mac = hmac.new(secret.encode("utf-8"), raw_body, sha256).hexdigest()
  Header name: ``Linear-Signature``.

- **Registration** — ``registry.register(...)`` at module level, discovered
  automatically by ``tools/registry.py::discover_builtin_tools()`` (AST scan
  for top-level ``registry.register()`` calls). No existing file touched.

Environment variables
---------------------
``LINEAR_WEBHOOK_SECRET``   (required at POST time) — HMAC signing key.
                             Populated from Secrets Manager in prod. Never
                             hardcoded.
``ORCHESTRATOR_SENSOR_URL`` (optional) — base URL of the sensor service.
                             Default: ``http://orchestrator-sensor.agentic-stack.internal:8000``
                             (Cloud Map DNS name + port Blake configures in
                             the ECS task definition / terraform variable).
                             Override in dev/test via env.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from typing import Any, Optional

import httpx

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

SENSOR_BASE_URL: str = os.environ.get(
    "ORCHESTRATOR_SENSOR_URL",
    "http://orchestrator-sensor.agentic-stack.internal:8000",
)
"""Base URL of the orchestrator sensor service (Cloud Map).

Override via the ``ORCHESTRATOR_SENSOR_URL`` env var. The default resolves
inside the VPC via Cloud Map DNS; it will not work from a developer laptop
without a tunnel or local override.
"""

# ---------------------------------------------------------------------------
# 5-dimension coverage helpers (vendored — no agentic-hub import allowed)
#
# Mirrors orchestrator/registry/spec_intake.py dimension definitions.
# A dimension is "present" when:
#   operator         — non-blank string
#   budget_usd       — numeric value >= 0  (None / missing = absent)
#   repos            — non-empty list      (empty list = absent)
#   success_criteria — non-empty list      (empty list = absent; ["none"] ok)
#   constraints      — non-empty list      (empty list = absent; ["none"] ok)
# ---------------------------------------------------------------------------

_DIMENSIONS = ("operator", "success_criteria", "repos", "budget_usd", "constraints")

# Priority order for the questioning loop (first absent → first asked).
# operator → success_criteria → repos → budget_usd → constraints
_QUESTION_ORDER = ("operator", "success_criteria", "repos", "budget_usd", "constraints")

_NEXT_QUESTION_TEXT: dict[str, str] = {
    "operator": (
        "Who is the operator (person or team) responsible for this plan? "
        "Please provide a name or handle (e.g. 'blake', 'platform-team')."
    ),
    "success_criteria": (
        "What are the success criteria for this plan? "
        "Please list at least one concrete, measurable outcome "
        "(e.g. ['CI is green', 'feature is behind a flag']). "
        "Reply with ['none'] if there are no explicit criteria."
    ),
    "repos": (
        "Which GitHub repository (or repositories) does this plan target? "
        "Please provide the full slug(s), e.g. ['blakeaber/agentic-hub']. "
        "At least one repo is required."
    ),
    "budget_usd": (
        "What is the approved spend ceiling in USD for this plan? "
        "Please reply with a number (e.g. 10 for $10.00). "
        "This gates the autonomous drain — no work will begin without it."
    ),
    "constraints": (
        "Are there any constraints the plan must respect "
        "(e.g. ['no new AWS resources', 'must not modify prod DB schema'])? "
        "Reply with ['none'] if there are no constraints."
    ),
}


def _spec_coverage(spec: dict[str, Any]) -> dict[str, bool]:
    """Return {dimension: is_present} for each of the 5 required dimensions.

    Pure function — no I/O, no side effects. Testable in isolation.

    Dimension presence rules (mirrors spec_intake.py):
      operator         str, non-blank
      budget_usd       numeric (int/float) >= 0
      repos            non-empty list
      success_criteria non-empty list
      constraints      non-empty list

    >>> _spec_coverage({})
    {'operator': False, 'budget_usd': False, 'repos': False, 'success_criteria': False, 'constraints': False}
    >>> _spec_coverage({'operator': 'blake', 'budget_usd': 10, 'repos': ['r/r'], 'success_criteria': ['x'], 'constraints': ['none']})
    {'operator': True, 'budget_usd': True, 'repos': True, 'success_criteria': True, 'constraints': True}
    """
    operator = spec.get("operator")
    budget_usd = spec.get("budget_usd")
    repos = spec.get("repos")
    success_criteria = spec.get("success_criteria")
    constraints = spec.get("constraints")

    return {
        "operator": bool(operator and str(operator).strip()),
        "budget_usd": (
            isinstance(budget_usd, (int, float))
            and not isinstance(budget_usd, bool)
            and budget_usd >= 0
        ),
        "repos": bool(repos and isinstance(repos, list) and len(repos) > 0),
        "success_criteria": bool(
            success_criteria
            and isinstance(success_criteria, list)
            and len(success_criteria) > 0
        ),
        "constraints": bool(
            constraints
            and isinstance(constraints, list)
            and len(constraints) > 0
        ),
    }


def _next_question(coverage: dict[str, bool]) -> Optional[str]:
    """Return the question text for the highest-priority absent dimension.

    Priority order: operator → success_criteria → repos → budget_usd → constraints
    Returns None when all dimensions are covered (spec is complete).

    >>> _next_question({'operator': False, 'budget_usd': True, 'repos': True, 'success_criteria': True, 'constraints': True})
    'Who is the operator...'  # (abbreviated)
    >>> _next_question({'operator': True, 'budget_usd': True, 'repos': True, 'success_criteria': True, 'constraints': True}) is None
    True
    """
    for dim in _QUESTION_ORDER:
        if not coverage.get(dim, False):
            return _NEXT_QUESTION_TEXT[dim]
    return None


# ---------------------------------------------------------------------------
# HMAC signing helper
# ---------------------------------------------------------------------------


def _sign_body(secret: str, raw_body: bytes) -> str:
    """Compute the Linear-Signature header value for *raw_body*.

    Exact algorithm verified by orchestrator/linear/sensors.py::_verify_signature:
        hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    """
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Sensor POST helper
# ---------------------------------------------------------------------------


def _post_to_sensor(
    *,
    sensor_base_url: str,
    secret: str,
    payload: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST *payload* to ``/plan/decompose`` and return the parsed response body.

    Signs with ``Linear-Signature: <hex>`` (HMAC-SHA256 of the raw JSON body).
    Raises ``RuntimeError`` on network failure, HTTP error, or JSON decode error
    so the caller can surface a clean error message to the agent.

    Returns the parsed response dict on 200.
    """
    url = f"{sensor_base_url.rstrip('/')}/plan/decompose"
    raw_body: bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = _sign_body(secret, raw_body)

    try:
        resp = httpx.post(
            url,
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "Linear-Signature": signature,
            },
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Network error contacting orchestrator sensor: {exc}") from exc

    if resp.status_code == 401:
        raise RuntimeError(
            "Orchestrator sensor rejected the request: 401 Unauthorized. "
            "Check that LINEAR_WEBHOOK_SECRET matches the sensor's configured secret."
        )
    if resp.status_code == 422:
        detail = resp.text[:400]
        raise RuntimeError(f"Orchestrator sensor rejected the payload: 422 Unprocessable Entity — {detail}")
    if resp.status_code >= 500:
        raise RuntimeError(
            f"Orchestrator sensor returned {resp.status_code}. "
            "The sensor may be unhealthy; check its logs."
        )
    if not resp.is_success:
        raise RuntimeError(
            f"Orchestrator sensor returned unexpected status {resp.status_code}: {resp.text[:400]}"
        )

    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Orchestrator sensor returned non-JSON body: {exc}") from exc


# ---------------------------------------------------------------------------
# Tool entrypoint
# ---------------------------------------------------------------------------


def plan_lifecycle(
    goal: str,
    operator: Optional[str] = None,
    budget_usd: Optional[float] = None,
    repos: Optional[list[str]] = None,
    success_criteria: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    plan_id: Optional[str] = None,
    repo: Optional[str] = None,
    ref: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Turn a Slack goal into a dispatched plan via the orchestrator sensor.

    Call this iteratively from the agent conversation. On each call:

    1. If the 5-dimension spec coverage is incomplete → return the next
       question as a string (the agent asks it in-thread).
    2. If coverage is complete but ``budget_usd`` is not set → return a
       message prompting for the budget (never POST without it).
    3. If coverage is complete AND ``budget_usd`` is set → sign and POST
       the spec to ``POST /plan/decompose`` on the orchestrator sensor;
       return a Slack-friendly summary of the created issues or a clear
       error message.

    Parameters
    ----------
    goal:             Natural-language goal / spec string (required).
    operator:         Responsible person or team (e.g. "blake").
    budget_usd:       Approved spend ceiling in USD (e.g. 10.0).
    repos:            Target repositories, e.g. ["blakeaber/agentic-hub"].
    success_criteria: Measurable outcomes, e.g. ["CI is green"]. Use ["none"]
                      if there are no explicit criteria.
    constraints:      Constraints to respect, e.g. ["no new AWS resources"].
                      Use ["none"] if there are none.
    plan_id:          Optional plan identifier (e.g. "hub-058"). Auto-generated
                      from a UUID prefix if omitted.
    repo:             Primary repo URL for the orchestrator (defaults to the
                      first entry in ``repos``). The orchestrator's required
                      ``repo`` field is the full git URL or slug.
    ref:              Git ref (defaults to "main").
    task_id:          Injected by the tool dispatcher; unused but accepted.
    """
    del task_id  # unused; kept for handler signature compatibility

    # ------------------------------------------------------------------
    # Build the accumulated spec dict from the conversation fields
    # ------------------------------------------------------------------
    spec: dict[str, Any] = {
        "goal": (goal or "").strip(),
        "operator": operator,
        "budget_usd": budget_usd,
        "repos": repos or [],
        "success_criteria": success_criteria or [],
        "constraints": constraints or [],
    }

    if not spec["goal"]:
        return tool_error("goal is required to start a plan")

    # ------------------------------------------------------------------
    # Coverage check: ask for the next missing dimension
    # ------------------------------------------------------------------
    coverage = _spec_coverage(spec)
    question = _next_question(coverage)
    if question:
        # Return the question as plain text so the agent relays it in-thread.
        return json.dumps({"question": question, "coverage": coverage})

    # ------------------------------------------------------------------
    # Budget gate: never POST without a budget
    # ------------------------------------------------------------------
    if budget_usd is None or not coverage["budget_usd"]:
        return json.dumps({
            "question": _NEXT_QUESTION_TEXT["budget_usd"],
            "coverage": coverage,
            "warning": (
                "A budget_usd is required before dispatching — "
                "the autonomous drain will stall otherwise."
            ),
        })

    # ------------------------------------------------------------------
    # Derive POST payload fields
    # ------------------------------------------------------------------
    effective_plan_id: str = (
        plan_id.strip()
        if plan_id and plan_id.strip()
        else f"plan-{uuid.uuid4().hex[:8]}"
    )

    # ``repo`` in the sensor body is the primary repo URL/slug; default to
    # the first entry in ``repos`` (which is a non-empty list at this point).
    effective_repo: str = (
        repo.strip()
        if repo and repo.strip()
        else (spec["repos"][0] if spec["repos"] else "")
    )
    effective_ref: str = (ref or "main").strip() or "main"

    payload: dict[str, Any] = {
        "goal": spec["goal"],
        "plan_id": effective_plan_id,
        "repo": effective_repo,
        "ref": effective_ref,
        "budget_usd": budget_usd,
        "repos": spec["repos"],
        "success_criteria": spec["success_criteria"],
        "constraints": spec["constraints"],
        "operator": spec["operator"],
    }

    # ------------------------------------------------------------------
    # HMAC sign + POST
    # ------------------------------------------------------------------
    secret = os.environ.get("LINEAR_WEBHOOK_SECRET", "")
    if not secret:
        return tool_error(
            "LINEAR_WEBHOOK_SECRET is not set. "
            "The plan cannot be dispatched without the signing secret. "
            "Please ensure the secret is populated from Secrets Manager."
        )

    sensor_url = os.environ.get("ORCHESTRATOR_SENSOR_URL", SENSOR_BASE_URL)

    try:
        response_body = _post_to_sensor(
            sensor_base_url=sensor_url,
            secret=secret,
            payload=payload,
        )
    except RuntimeError as exc:
        return tool_error(str(exc))

    # ------------------------------------------------------------------
    # Format a Slack-friendly success summary
    # ------------------------------------------------------------------
    created_issue_ids: list[str] = response_body.get("created_issue_ids", [])
    phase_ids: list[str] = response_body.get("phase_ids", [])
    signaled: bool = response_body.get("signaled", False)

    issues_summary = (
        ", ".join(created_issue_ids) if created_issue_ids else "(none returned)"
    )
    phases_summary = (
        ", ".join(str(p) for p in phase_ids) if phase_ids else "(none)"
    )

    return tool_result(
        success=True,
        plan_id=effective_plan_id,
        created_issue_ids=created_issue_ids,
        phase_ids=phase_ids,
        signaled=signaled,
        message=(
            f"Plan *{effective_plan_id}* dispatched to the orchestrator.\n"
            f"• Issues created: {issues_summary}\n"
            f"• Phases: {phases_summary}\n"
            f"• Drain signaled: {signaled}"
        ),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

PLAN_LIFECYCLE_SCHEMA: dict[str, Any] = {
    "name": "plan_lifecycle",
    "description": (
        "Turn a Slack goal into a decomposed, dispatched plan by running an "
        "adaptive 5-dimension coverage loop in-thread, then POSTing the "
        "completed spec to the agentic-hub orchestrator sensor.\n\n"
        "Call this tool iteratively from the conversation:\n"
        "  • If the spec is incomplete the tool returns the next question — "
        "relay it to the user and call again with the answer filled in.\n"
        "  • If the spec is complete but ``budget_usd`` is missing the tool "
        "prompts for the budget (no dispatch without it).\n"
        "  • Once all 5 dimensions are present AND a budget is set the tool "
        "signs and POSTs the spec, returning a Slack-friendly summary of the "
        "created Linear issues.\n\n"
        "The 5 required dimensions are: operator, budget_usd, repos, "
        "success_criteria, constraints.  Each must be non-empty (use "
        "['none'] for criteria/constraints when genuinely absent).\n\n"
        "Scope: team  Role: orchestration"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "Natural-language goal or spec string. Required. "
                    "This becomes the primary task description dispatched to the orchestrator."
                ),
            },
            "operator": {
                "type": "string",
                "description": (
                    "Person or team responsible for this plan (e.g. 'blake', 'platform-team'). "
                    "Required before dispatch."
                ),
            },
            "budget_usd": {
                "type": "number",
                "description": (
                    "Approved spend ceiling in USD (e.g. 10 for $10.00). "
                    "Required — the autonomous drain will not start without it."
                ),
            },
            "repos": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Target repositories as slugs or URLs, "
                    "e.g. ['blakeaber/agentic-hub']. "
                    "At least one is required before dispatch."
                ),
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Measurable success outcomes, e.g. ['CI is green', 'feature behind flag']. "
                    "Use ['none'] if there are no explicit criteria. "
                    "Required before dispatch."
                ),
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Constraints the plan must respect, e.g. ['no new AWS resources']. "
                    "Use ['none'] if there are none. Required before dispatch."
                ),
            },
            "plan_id": {
                "type": "string",
                "description": (
                    "Optional explicit plan identifier (e.g. 'hub-058'). "
                    "Auto-generated (plan-<uuid8>) when omitted."
                ),
            },
            "repo": {
                "type": "string",
                "description": (
                    "Primary repository URL/slug for the orchestrator's 'repo' field. "
                    "Defaults to the first entry in ``repos`` when omitted."
                ),
            },
            "ref": {
                "type": "string",
                "description": "Git ref to target (defaults to 'main').",
            },
        },
        "required": ["goal"],
    },
}


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _check_plan_lifecycle_requirements() -> bool:
    """Available in interactive (CLI) and gateway (Slack) sessions.

    Uses the same env-var check pattern as ``check_cronjob_requirements``
    (``hermes_constants`` env_var_enabled helper).
    """
    from utils import env_var_enabled

    return (
        env_var_enabled("HERMES_INTERACTIVE")
        or env_var_enabled("HERMES_GATEWAY_SESSION")
        or env_var_enabled("HERMES_EXEC_ASK")
    )


# ---------------------------------------------------------------------------
# Registration — auto-discovered by tools/registry.py::discover_builtin_tools
# via AST scan for top-level registry.register() calls. No existing file
# needs to be modified.
# ---------------------------------------------------------------------------

registry.register(
    name="plan_lifecycle",
    toolset="orchestration",
    schema=PLAN_LIFECYCLE_SCHEMA,
    handler=lambda args, **kw: plan_lifecycle(
        goal=args.get("goal", ""),
        operator=args.get("operator"),
        budget_usd=args.get("budget_usd"),
        repos=args.get("repos"),
        success_criteria=args.get("success_criteria"),
        constraints=args.get("constraints"),
        plan_id=args.get("plan_id"),
        repo=args.get("repo"),
        ref=args.get("ref"),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_plan_lifecycle_requirements,
    requires_env=["LINEAR_WEBHOOK_SECRET", "ORCHESTRATOR_SENSOR_URL"],
    description="Turn a Slack goal into a dispatched plan via the orchestrator sensor.",
    emoji="📋",
)
