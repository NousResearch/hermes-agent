"""A selected external-process provider must survive child command resolution."""
import sys
from types import SimpleNamespace

import pytest

from providers import register_provider
from providers.base import ProviderProfile
from tools.delegate_tool_config import _resolve_child_runtime


@pytest.mark.parametrize("external", [True, False])
def test_pinned_command_retains_selected_external_provider(external):
    profile = ProviderProfile(name="test-process-provider", display_name="Test", auth_type="external_process" if external else "api_key")
    register_provider(profile)
    parent = SimpleNamespace(model="test-model", provider=profile.name, base_url="process://test", api_mode="chat_completions", acp_args=[])
    result = _resolve_child_runtime(
        parent, {}, "external-process", model=None, override_provider=profile.name,
        override_base_url=None, override_api_key=None, override_api_mode=None,
        override_acp_command=sys.executable, override_acp_args=[],
    )
    assert result["provider"] == (profile.name if external else "copilot-acp")
    assert result["acp_command"] == sys.executable
