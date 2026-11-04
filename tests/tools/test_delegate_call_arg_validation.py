"""Regression tests: delegate_task must refuse model call arguments it can never forward.

Both model-facing dispatch sites pick a fixed field list out of the LLM's argument dict, so any
other top-level key silently vanishes before routing. The live failure (model_map backend):
``option`` sent beside a single ``goal`` instead of inside a ``tasks`` entry dropped the route
choice and spawned the child on the backend's default route while the call reported success.
Sibling of #118825 (the same swallow one level down on per-task fields). The ACP transport pins
stay a deliberate silent drop — the trust-boundary contract pinned by
test_delegate.py::test_model_acp_args_not_forwarded.

Every test drives one of the two MODEL-FACING surfaces (the live intercept, or the registry
entry's handler) so the same file is RED on the unfixed base and GREEN on the fix.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import tools.delegate_tool  # registers the delegate_task tool entry that _registry() resolves
from tools.registry import registry

GOOD_GOAL = "Refactor the login handler to use the new session helper"


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.session_id = "root"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _live(args, parent=None):
    """Drive the model-facing live path (the INLINE_TOOL_EXECUTORS entry for delegate_task)."""
    import run_agent

    return json.loads(run_agent.AIAgent._dispatch_delegate_task(parent or _make_mock_parent(), args))


def _registry(args, parent=None):
    """Drive the registry handler (the fallback path when the live intercept is bypassed)."""
    entry = registry.get_entry("delegate_task")
    assert entry is not None
    return json.loads(entry.handler(args, parent_agent=parent or _make_mock_parent()))


class TestStrayTopLevelOptionRefused:
    """The live incident: a routing field at top level beside `goal` was swallowed and the
    child silently ran the backend's default route."""

    def test_live_dispatch_refuses_option_beside_goal(self):
        with patch("tools.delegate_tool.delegate_task") as spy:
            out = _live({"goal": GOOD_GOAL, "option": "Self"})
        spy.assert_not_called()
        assert "option" in out["error"]
        assert "No subagent was started" in out["error"]
        # actionable: names the per-task position the routing field actually belongs to
        assert "tasks=" in out["error"]

    def test_registry_handler_refuses_option_beside_goal(self):
        with patch("tools.delegate_tool.delegate_task") as spy:
            out = _registry({"goal": GOOD_GOAL, "option": "Self"})
        spy.assert_not_called()
        assert "option" in out["error"] and "tasks=" in out["error"]

    def test_any_undeclared_key_refused(self):
        with patch("tools.delegate_tool.delegate_task") as spy:
            out = _registry({"goal": GOOD_GOAL, "temperature": 0.2})
        spy.assert_not_called()
        assert "temperature" in out["error"] and "No subagent was started" in out["error"]
        # a non-routing key gets no per-task hint (would be misleading)
        assert "tasks=" not in out["error"]

    def test_raw_route_field_gets_no_per_task_hint(self):
        # model/provider/base_url are RESERVED_TASK_PARAMETERS — never model-facing at any level,
        # so the refusal must not steer the caller to move them into a task entry.
        with patch("tools.delegate_tool.delegate_task") as spy:
            out = _registry({"goal": GOOD_GOAL, "model": "gpt-whatever"})
        spy.assert_not_called()
        assert "model" in out["error"]
        assert "tasks=" not in out["error"]
        # mixed strays (valid routing field + reserved field) keep the hint off too: moving the
        # whole set per-task would earn a second refusal from _route_one
        with patch("tools.delegate_tool.delegate_task") as spy:
            out = _registry({"goal": GOOD_GOAL, "option": "Self", "model": "gpt-whatever"})
        spy.assert_not_called()
        assert "option" in out["error"] and "model" in out["error"] and "tasks=" not in out["error"]

    def test_internal_caller_params_never_model_sendable(self):
        # credentials_cfg/parent_agent are excluded from the accepted set on purpose: a model
        # naming an internal-caller parameter is refused, not silently dropped or honoured.
        for internal in ("credentials_cfg", "parent_agent"):
            with patch("tools.delegate_tool.delegate_task") as spy:
                out = _registry({"goal": GOOD_GOAL, internal: {"model": "x"}})
            spy.assert_not_called()
            assert internal in out["error"], f"{internal} not refused: {out}"

    def test_legacy_toolsets_window_pinned(self):
        # `toolsets` was model-facing until ba0bc01d1f (#56386); a session whose tool schema was
        # built before that update can still emit it per its cached schema. The deliberate call:
        # refuse loudly (self-correcting) rather than re-open a silent drop — the window closes
        # when the session rebuilds. Pinned here so the behavior change is a decision, not drift.
        with patch("tools.delegate_tool.delegate_task") as spy:
            out = _live({"goal": GOOD_GOAL, "toolsets": ["web"]})
        spy.assert_not_called()
        assert "toolsets" in out["error"]


class TestValidShapesUnaffected:
    """The gate must not over-reject the shapes that legitimately arrive."""

    def test_plain_single_goal_reaches_the_callee(self):
        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = "{}"
            out = _live({"goal": GOOD_GOAL})
        assert out == {}
        spy.assert_called_once()
        assert spy.call_args.kwargs.get("goal") == GOOD_GOAL

    def test_option_inside_task_entry_is_forwarded_verbatim(self):
        """The per-task form is the documented shape (it works on base too — this pins that the
        gate never mangles it): the call layer must hand the task entry to the callee untouched,
        whatever the route layer then decides."""
        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = "{}"
            out = _live({"tasks": [{"goal": GOOD_GOAL, "option": "Self"}]})
        assert out == {}
        sent = spy.call_args.kwargs["tasks"]
        assert sent[0].get("option") == "Self"

    def test_control_actions_pass_through(self):
        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = '{"status": "ok"}'
            out = _live({"action": "list"})
        assert out == {"status": "ok"}
        assert spy.call_args.kwargs.get("action") == "list"

    def test_live_site_forwards_top_level_output_schema(self):
        """The legacy single-goal shape accepts top-level output_schema (see the handler doc);
        the live site used to drop it (the child spawned schema-less, silently) while the
        registry mirror forwarded it."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = "{}"
            _live({"goal": GOOD_GOAL, "output_schema": schema})
        assert spy.call_args.kwargs.get("output_schema") == schema

    def test_accepted_set_pinned_to_schema_plus_legacy(self):
        """MODEL_TOP_LEVEL_KEYS must equal the advertised top-level schema keys plus the
        deliberate unadvertised-legacy set. Two drift directions fail loudly: a new schema
        field (must reach the signature — else it can never be forwarded) and a new signature
        param (must be declared _INTERNAL_PARAMS or listed here — else a model could silently
        name an internal-caller knob)."""
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA, MODEL_TOP_LEVEL_KEYS

        advertised = set(DELEGATE_TASK_SCHEMA["parameters"]["properties"])
        legacy = {"goal", "context", "output_schema", "images", "max_iterations", "role",
                  "background"}
        assert MODEL_TOP_LEVEL_KEYS == advertised | legacy


class TestDispatchSitesInSync:
    """The gate accepts exactly what both sites forward: an accepted key NO site forwards is the
    swallow this PR exists to kill, creeping back at one site (found in review: output_schema
    was accepted + forwarded by the registry handler but dropped by the live site). Behavioural,
    not source-shape: send every accepted key, require every one to reach the callee."""

    def test_live_site_forwards_every_accepted_key(self):
        from tools.delegate_tool import MODEL_TOP_LEVEL_KEYS

        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = "{}"
            _live({"goal": GOOD_GOAL, "context": "ctx", "tasks": [{"goal": GOOD_GOAL}],
                   "max_iterations": 9, "role": "leaf", "output_schema": {"type": "object"},
                   "background": False, "images": [], "action": "spawn",
                   "subagent_id": "sa-x", "message": "m"})
        kwargs = spy.call_args.kwargs
        missing = {k for k in MODEL_TOP_LEVEL_KEYS if k not in kwargs}
        assert not missing, f"live dispatch accepts but never forwards: {sorted(missing)}"

    def test_registry_handler_forwards_every_accepted_key(self):
        from tools.delegate_tool import MODEL_TOP_LEVEL_KEYS

        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = "{}"
            _registry({"goal": GOOD_GOAL, "context": "ctx", "tasks": [{"goal": GOOD_GOAL}],
                       "max_iterations": 9, "role": "leaf", "output_schema": {"type": "object"},
                       "background": False, "images": [], "action": "spawn",
                       "subagent_id": "sa-x", "message": "m"})
        kwargs = spy.call_args.kwargs
        missing = {k for k in MODEL_TOP_LEVEL_KEYS if k not in kwargs}
        assert not missing, f"registry handler accepts but never forwards: {sorted(missing)}"


class TestAcpSilentDropPreserved:
    """acp_command/acp_args are stripped, never refused: a refusal would advertise the trusted
    transport fields to the model. Parity with test_delegate.py::test_model_acp_args_not_forwarded."""

    def test_top_level_acp_fields_stripped_not_refused(self):
        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
            out = _live({"goal": "test", "acp_command": "claude", "acp_args": ["--acp", "--stdio"]})
        assert out == {}
        assert captured.get("goal") == "test"
        assert "acp_command" not in captured and "acp_args" not in captured

    def test_registry_path_acp_silent_drop_unchanged(self):
        with patch("tools.delegate_tool.delegate_task") as spy:
            spy.return_value = "{}"
            out = _registry({"goal": "test", "acp_command": "claude"})
        assert out == {}
        spy.assert_called_once()
        assert "acp_command" not in spy.call_args.kwargs
