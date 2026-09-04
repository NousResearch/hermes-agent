"""delegate_task wiring for delegation.profiles (T3, Phase 1: config-driven).

Per-task profile routing inside the spawn loop — a batch can mix profiles, default_profile
applies when a task is silent, profile fallback lists replace the parent's chain,
supports_tools=False profiles are rejected before AIAgent construction, and legacy behavior
is byte-identical when no profiles are configured.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


# ── helpers (mirrors tests/tools/test_delegate.py fixtures) ──────────────────

def _make_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
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
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-fallback"}]
    return parent


def _make_child():
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "done", "completed": True, "api_calls": 1, "messages": [],
    }
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = "test"
    return child


PROFILES = {
    "small": {"provider": "openrouter", "model": "prof/small-model", "max_iterations": 7, "fallback": []},
    "big": {
        "provider": "openrouter", "model": "prof/big-model",
        "fallback": [{"provider": "nous", "model": "prof/backup"}],
    },
    "tiny": {"provider": "openrouter", "model": "prof/tiny-model", "reasoning_effort": "low"},
}


def _fake_runtime(requested=None, target_model=None, **_kw):
    return {
        "provider": requested or "openrouter", "base_url": "https://rt.example/v1",
        "api_key": "rt-key", "api_mode": "chat_completions",
        "request_overrides": {}, "max_output_tokens": None,
    }


class _Harness:
    """Patches the delegation config + provider resolution + AIAgent construction."""

    def __init__(self, cfg, capabilities=None):
        self.built = []
        self.cfg = cfg
        self._patches = [
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_fake_runtime),
            patch("agent.models_dev.get_model_capabilities", return_value=capabilities),
            patch("run_agent.AIAgent", side_effect=self._factory),
        ]

    def _factory(self, *args, **kwargs):
        self.built.append(kwargs)
        return _make_child()

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


# ── T3: per-task profile routing (config-driven) ─────────────────────────────

def test_batch_mixes_profiles_per_task():
    """Two tasks with two different profiles resolve two different models in ONE batch."""
    cfg = {"profiles": PROFILES}
    with _Harness(cfg) as h:
        delegate_task(
            tasks=[
                {"goal": "Summarize the release notes for QA", "model_profile": "small"},
                {"goal": "Design the storage migration plan", "model_profile": "big"},
            ],
            parent_agent=_make_parent(),
        )
    assert [b["model"] for b in h.built] == ["prof/small-model", "prof/big-model"]
    assert all(b["api_key"] == "rt-key" for b in h.built)
    assert all(b["provider"] == "openrouter" for b in h.built)


def test_default_profile_applies_when_task_silent():
    cfg = {"profiles": PROFILES, "default_profile": "small"}
    with _Harness(cfg) as h:
        delegate_task(goal="Do the maintenance work carefully", parent_agent=_make_parent())
    assert h.built[0]["model"] == "prof/small-model"


def test_profile_max_iterations_clamps_child_budget():
    cfg = {"profiles": PROFILES, "default_profile": "small"}
    with _Harness(cfg) as h:
        delegate_task(goal="Do the maintenance work carefully", parent_agent=_make_parent())
    assert h.built[0]["max_iterations"] == 7


def test_profile_reasoning_effort_overrides_parent():
    from hermes_constants import parse_reasoning_effort
    cfg = {"profiles": PROFILES, "default_profile": "tiny"}
    with _Harness(cfg) as h:
        delegate_task(goal="Do the maintenance work carefully", parent_agent=_make_parent())
    assert h.built[0]["reasoning_config"] == parse_reasoning_effort("low")


def test_profile_fallback_list_replaces_parent_chain():
    cfg = {"profiles": PROFILES}
    with _Harness(cfg) as h:
        delegate_task(
            tasks=[{"goal": "Design the storage migration plan", "model_profile": "big"}],
            parent_agent=_make_parent(),
        )
    assert h.built[0]["fallback_model"] == [{"provider": "nous", "model": "prof/backup"}]


def test_profile_empty_fallback_isolates_from_parent_chain():
    """fallback: [] = no model promotion — the parent's chain must NOT leak in."""
    cfg = {"profiles": PROFILES}
    with _Harness(cfg) as h:
        delegate_task(
            tasks=[{"goal": "Summarize the release notes for QA", "model_profile": "small"}],
            parent_agent=_make_parent(),
        )
    assert not h.built[0]["fallback_model"]


def test_no_profiles_legacy_byte_identical():
    """No profiles configured → parent inherit exactly as before (model + fallback chain)."""
    parent = _make_parent()
    with _Harness({}) as h:
        delegate_task(goal="Do the maintenance work carefully", parent_agent=parent)
    assert h.built[0]["model"] == parent.model
    assert h.built[0]["fallback_model"] == parent._fallback_chain


def test_unknown_profile_fails_before_any_spawn():
    cfg = {"profiles": PROFILES}
    with _Harness(cfg) as h:
        result = delegate_task(
            tasks=[
                {"goal": "Summarize the release notes for QA", "model_profile": "small"},
                {"goal": "Design the storage migration plan", "model_profile": "nope"},
            ],
            parent_agent=_make_parent(),
        )
    assert "nope" in result
    assert "small" in result  # actionable: configured names listed
    assert h.built == []  # no AIAgent construction at all


def test_supports_tools_false_profile_rejected_with_toolsets():
    cfg = {"profiles": PROFILES}
    caps = SimpleNamespace(supports_tools=False)
    with _Harness(cfg, capabilities=caps) as h:
        parent = _make_parent()
        parent.enabled_toolsets = ["terminal", "file"]
        result = delegate_task(
            tasks=[{"goal": "Summarize the release notes for QA", "model_profile": "small"}],
            parent_agent=parent,
        )
    assert "small" in result and "tool" in result.lower()
    assert h.built == []
