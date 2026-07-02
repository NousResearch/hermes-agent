from types import SimpleNamespace
from unittest.mock import MagicMock

from tools import delegate_tool


def _parent(**overrides):
    parent = MagicMock()
    parent.base_url = "https://provider.example.invalid/v1"
    parent.api_key = "test-key"
    parent.provider = "custom:test-provider"
    parent.api_mode = "codex_responses"
    parent.model = "gpt-5.5"
    parent.enabled_toolsets = ["terminal"]
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._delegate_spinner = None
    parent.tool_progress_callback = None
    parent._active_children = []
    parent._active_children_lock = None
    parent._print_fn = None
    parent.session_id = "parent-session"
    parent.request_overrides = {}
    for key, value in overrides.items():
        setattr(parent, key, value)
    return parent


def _capture_child_kwargs(monkeypatch):
    captured = {}

    class FakeAgent(SimpleNamespace):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(session_id="child-session", **kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    return captured


def test_subagent_inherits_parent_request_overrides(monkeypatch):
    captured = _capture_child_kwargs(monkeypatch)
    parent_overrides = {
        "extra_headers": {"User-Agent": "codex_cli_rs/0.138.0"},
        "extra_body": {"trace": "parent"},
    }

    delegate_tool._build_child_agent(
        task_index=0,
        goal="audit",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=3,
        task_count=1,
        parent_agent=_parent(request_overrides=parent_overrides),
    )

    assert captured["request_overrides"] == parent_overrides
    assert captured["request_overrides"] is not parent_overrides


def test_subagent_does_not_leak_parent_overrides_to_explicit_provider(monkeypatch):
    captured = _capture_child_kwargs(monkeypatch)

    delegate_tool._build_child_agent(
        task_index=0,
        goal="audit",
        context=None,
        toolsets=None,
        model="mini",
        max_iterations=3,
        task_count=1,
        parent_agent=_parent(
            request_overrides={"extra_headers": {"User-Agent": "toolcode-only"}},
        ),
        override_provider="custom:other-provider",
        override_base_url="https://other-provider.example.invalid/anthropic",
        override_api_key="child-key",
        override_api_mode="anthropic_messages",
    )

    assert captured["request_overrides"] == {}


def test_subagent_uses_explicit_delegation_request_overrides(monkeypatch):
    captured = _capture_child_kwargs(monkeypatch)
    child_overrides = {"extra_headers": {"User-Agent": "child-route"}}

    delegate_tool._build_child_agent(
        task_index=0,
        goal="audit",
        context=None,
        toolsets=None,
        model="gpt-5.5",
        max_iterations=3,
        task_count=1,
        parent_agent=_parent(
            request_overrides={"extra_headers": {"User-Agent": "parent-route"}},
        ),
        override_provider="custom:test-provider",
        override_base_url="https://provider.example.invalid/v1",
        override_api_key="child-key",
        override_api_mode="codex_responses",
        override_request_overrides=child_overrides,
    )

    assert captured["request_overrides"] == child_overrides
    assert captured["request_overrides"] is not child_overrides
