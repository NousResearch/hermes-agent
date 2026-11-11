"""A pinned command keeps the provider whose runtime resolved it through child command resolution."""
import os
import sys
from types import SimpleNamespace

import pytest

from providers.base import ProviderProfile
from tools.delegate_tool_config import _resolve_child_runtime


@pytest.fixture
def process_profiles(monkeypatch):
    """Swap the provider registry the resolver reads for a fixed set of profiles."""
    import providers

    def install(*profiles):
        by_name = {p.name: p for p in profiles}
        monkeypatch.setattr(providers, "get_provider_profile", by_name.get)
        monkeypatch.setattr(providers, "list_providers", lambda: list(profiles))
    return install


@pytest.mark.parametrize("external", [True, False])
def test_pinned_command_retains_the_selected_provider(external, process_profiles):
    profile = ProviderProfile(name="test-process-provider", display_name="Test",
                              auth_type="external_process" if external else "api_key")
    # Another installed external-process profile launching the very same binary must not capture the child.
    same_binary = ProviderProfile(name="harness-acp", auth_type="external_process",
                                  process_command=os.path.basename(sys.executable), api_mode="anthropic_messages")
    process_profiles(profile, same_binary)
    parent = SimpleNamespace(model="test-model", provider=profile.name, base_url="process://test",
                             api_mode="chat_completions", acp_args=[])
    result = _resolve_child_runtime(
        parent, {}, "external-process", model=None, override_provider=profile.name,
        override_base_url=None, override_api_key=None, override_api_mode="chat_completions",
        override_acp_command=sys.executable, override_acp_args=[],
    )
    assert (result["provider"], result["api_mode"]) == (profile.name, "chat_completions")
    assert result["requested_provider"] == result["provider"]
    assert result["acp_command"] == sys.executable
