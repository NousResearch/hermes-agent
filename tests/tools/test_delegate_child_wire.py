"""A subagent's ``delegation.model`` gets its own wire on a multi-wire provider.

OpenCode Zen/Go, and any provider whose profile declares ``api_mode_for_model``, serve several wire
formats behind one provider and pick one per model. With ``delegation.model`` set and
``delegation.provider`` left empty, the child kept the parent's ``api_mode`` (and, on OpenCode, the
base_url shaped for it), so a child of another model family posted to the wrong endpoint and every
delegated task failed.
"""

from unittest.mock import MagicMock, patch

import pytest

from providers.base import ProviderProfile
from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent

_GO = "https://opencode.ai/zen/go"
_ACME = "https://api.acme-wire.test"


class _PerModelWire(ProviderProfile):
    def api_mode_for_model(self, model, *, api_key=None):
        return "codex_responses" if model.startswith("gpt-5") else "chat_completions"


@pytest.fixture(autouse=True)
def _acme_wire_profile(monkeypatch):
    acme = _PerModelWire(name="acme-wire", base_url=_ACME)
    monkeypatch.setattr("providers.get_provider_profile", lambda name: acme if name == "acme-wire" else None)


def _child_kwargs(provider, parent_model, parent_mode, base_url, child_model):
    parent = _make_mock_parent(depth=0)
    parent.provider, parent.model, parent.api_mode, parent.base_url = provider, parent_model, parent_mode, base_url
    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(task_index=0, goal="do it", context=None, toolsets=None, model=child_model,
                           max_iterations=10, parent_agent=parent, task_count=1)
        return MockAgent.call_args.kwargs


@pytest.mark.parametrize(
    ("provider", "parent_model", "parent_mode", "base_url", "child_model", "mode", "child_base_url"),
    [
        ("acme-wire", "gpt-5.4", "codex_responses", _ACME, "claude-sonnet-4.5", "chat_completions", _ACME),
        ("acme-wire", "claude-sonnet-4.5", "chat_completions", _ACME, "gpt-5.4", "codex_responses", _ACME),
        ("opencode-go", "claude-sonnet-4.5", "anthropic_messages", _GO, "deepseek-v4-flash",
         "chat_completions", f"{_GO}/v1"),
        ("opencode-go", "deepseek-v4-flash", "chat_completions", f"{_GO}/v1", "claude-sonnet-4.5",
         "anthropic_messages", _GO),
    ],
)
def test_child_model_of_another_family_gets_its_own_wire(
        provider, parent_model, parent_mode, base_url, child_model, mode, child_base_url):
    kwargs = _child_kwargs(provider, parent_model, parent_mode, base_url, child_model)

    assert (kwargs["provider"], kwargs["model"]) == (provider, child_model)
    assert kwargs["api_mode"] == mode
    assert kwargs["base_url"] == child_base_url


def test_child_on_the_parents_model_keeps_the_parents_wire():
    # No delegation.model: the parent's (possibly configured) api_mode is kept as before.
    kwargs = _child_kwargs("acme-wire", "gpt-5.4", "chat_completions", _ACME, None)

    assert kwargs["api_mode"] == "chat_completions"


def test_single_wire_provider_keeps_the_parents_wire():
    kwargs = _child_kwargs("openrouter", "anthropic/claude-sonnet-4", "chat_completions",
                           "https://openrouter.ai/api/v1", "openai/gpt-5.4")

    assert kwargs["api_mode"] == "chat_completions"
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
