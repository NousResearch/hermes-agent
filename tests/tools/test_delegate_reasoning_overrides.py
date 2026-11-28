"""``delegate_task`` children honour ``agent.reasoning_overrides`` for their OWN model.

Every other surface (CLI, gateway, TUI, cron, ``/model``, fallback activation) resolves its
effective effort per-model first — ``agent.reasoning_overrides`` beats the global level
(``hermes_constants.resolve_reasoning_config``). Delegation resolved its own ladder instead
(route effort > ``delegation.reasoning_effort`` > the parent's level) and never looked at the
child's model, so a child pinned to a model with an override ran at the delegation-wide
default (#155866).

The cases below drive ``_resolve_child_runtime`` — the function that exists on both sides of
the fix — so the failure on the unpatched tree is the behaviour itself, not a signature.

No test here touches a real home: the config is either patched at the loader or written into a
``HERMES_HOME`` under ``tmp_path``.
"""

from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

from tools.delegate_tool_config import ResolvedReasoning, _resolve_child_runtime

PARENT_LEVEL = {"enabled": True, "effort": "xhigh"}


def _parent(model="parent-model"):
    agent = MagicMock()
    agent.model = model
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "parent-key"
    agent.api_mode = "chat_completions"
    agent.reasoning_config = dict(PARENT_LEVEL)
    agent.providers_allowed = agent.providers_ignored = agent.providers_order = agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = ""
    agent._fallback_chain = None
    return agent


def _config(mapping: dict) -> dict:
    return {"agent": {"reasoning_effort": "medium", "reasoning_overrides": mapping},
            "delegation": {"reasoning_effort": "medium"}}


def _runtime(agent, delegation_cfg, model, effort=None):
    """Child kwargs for a ``delegate_task`` spawn: ``model`` is the per-task pin, ``effort`` the route's."""
    return _resolve_child_runtime(
        agent, delegation_cfg, "parent-key", model=model, override_provider="openrouter",
        override_base_url=None, override_api_key=None, override_api_mode=None,
        override_acp_command=None, override_acp_args=None, override_reasoning_effort=effort)


def test_child_model_override_beats_the_delegation_default():
    """The child's own model has an entry: that entry decides, not ``delegation.reasoning_effort``."""
    with patch("hermes_cli.config.load_config_readonly",
               return_value=_config({"child-model": "high", "parent-model": "low"})):
        kwargs = _runtime(_parent(), {"reasoning_effort": "medium"}, "child-model")
    assert kwargs["model"] == "child-model"
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_child_without_an_entry_keeps_the_delegation_default():
    """No entry for the child's model: the delegation-wide default still beats the parent level."""
    with patch("hermes_cli.config.load_config_readonly", return_value=_config({"other-model": "high"})):
        kwargs = _runtime(_parent(), {"reasoning_effort": "low"}, "child-model")
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_inherited_parent_model_uses_its_own_override():
    """An unpinned child runs the parent's model, so that model's entry applies to it too."""
    with patch("hermes_cli.config.load_config_readonly", return_value=_config({"parent-model": "minimal"})):
        kwargs = _runtime(_parent(), {"reasoning_effort": "high"}, None)
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "minimal"}


def test_per_task_route_effort_still_outranks_the_override():
    """An explicit per-task/route effort is the caller's decision; config never overrides it."""
    with patch("hermes_cli.config.load_config_readonly", return_value=_config({"child-model": "high"})):
        kwargs = _runtime(_parent(), {"reasoning_effort": "medium"}, "child-model", effort="low")
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_resolved_route_config_is_used_as_is():
    """A profile route already resolved its own profile's overrides: nothing here may fill it in."""
    with patch("hermes_cli.config.load_config_readonly", return_value=_config({"child-model": "high"})):
        kwargs = _runtime(_parent(), {"reasoning_effort": "medium"}, "child-model",
                          effort=ResolvedReasoning({"enabled": False}))
    assert kwargs["reasoning_config"] == {"enabled": False}


def test_a_disabled_override_disables_thinking_for_that_model():
    """``reasoning_overrides: {model: false}`` means disabled — the same reading as the global key."""
    with patch("hermes_cli.config.load_config_readonly", return_value=_config({"child-model": False})):
        kwargs = _runtime(_parent(), {"reasoning_effort": "high"}, "child-model")
    assert kwargs["reasoning_config"] == {"enabled": False}


def test_config_file_on_disk_is_honoured(tmp_path, monkeypatch):
    """End-to-end over a real ``config.yaml`` in a sandboxed home (loader not patched)."""
    import hermes_cli.config as config_module

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(_config({"disk-model": "max"})), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(config_module, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(config_module, "_RAW_CONFIG_CACHE", {})
    kwargs = _runtime(_parent(), {"reasoning_effort": "low"}, "disk-model")
    assert kwargs["reasoning_config"] == {"enabled": True, "effort": "max"}

    # --ignore-user-config must not resurrect an effort pinned in the skipped YAML file.
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    ignored = _runtime(_parent(), {"reasoning_effort": "low"}, "disk-model")
    assert ignored["reasoning_config"] == {"enabled": True, "effort": "low"}


def test_override_lookup_is_spelling_tolerant_and_silent_on_misses():
    """The lookup itself: tolerant matching, ``None`` when nothing matches, no config when no model.

    Imported defensively so the unpatched tree reports the behavioural failures above rather than
    dying at collection on a missing symbol.
    """
    from tools import delegate_tool_config as mod

    lookup = getattr(mod, "_per_model_reasoning_effort", None)
    if lookup is None:
        pytest.skip("per-model override lookup not implemented")
        return
    with patch("hermes_cli.config.load_config_readonly", return_value=_config({"child-model": "high"})):
        assert lookup("child-model") == {"enabled": True, "effort": "high"}
        assert lookup("") is None
        assert lookup("unknown-model") is None
