"""Tests for the model_switch agent-callable tool.

TDD GREEN phase — tests for the model_switch tool that allows an agent
to change its own model mid-session.

Issue: #16525
"""

import pytest
import tools.model_switch_tool as mod
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(model="gpt-4o", provider="openai", api_key="sk-test"):
    """Create a mock agent with the minimal surface area the tool needs."""
    agent = MagicMock()
    agent.model = model
    agent.provider = provider
    agent.api_key = api_key
    agent.base_url = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._primary_runtime = MagicMock()
    return agent


def _mock_pipeline_result(**overrides):
    """Create a mock ModelSwitchResult."""
    defaults = dict(
        success=True,
        new_model="claude-sonnet-4",
        target_provider="anthropic",
        provider_label="Anthropic",
        api_key="sk-ant-test",
        base_url="",
        api_mode="anthropic_messages",
        error_message="",
        warning_message="",
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

class TestSchema:
    """The tool schema must declare the right name, params, and types."""

    def test_schema_exists(self):
        assert hasattr(mod, "MODEL_SWITCH_SCHEMA")
        schema = mod.MODEL_SWITCH_SCHEMA
        assert schema["name"] == "model_switch"

    def test_schema_has_model_param(self):
        props = mod.MODEL_SWITCH_SCHEMA["parameters"]["properties"]
        assert "model" in props
        assert props["model"]["type"] == "string"

    def test_schema_has_provider_param(self):
        props = mod.MODEL_SWITCH_SCHEMA["parameters"]["properties"]
        assert "provider" in props
        assert props["provider"]["type"] == "string"

    def test_model_is_required(self):
        required = mod.MODEL_SWITCH_SCHEMA["parameters"].get("required", [])
        assert "model" in required

    def test_provider_is_optional(self):
        required = mod.MODEL_SWITCH_SCHEMA["parameters"].get("required", [])
        assert "provider" not in required


# ---------------------------------------------------------------------------
# 2. Successful switch
# ---------------------------------------------------------------------------

class TestSuccessfulSwitch:
    """When the switch succeeds, the tool returns a confirmation."""

    @patch.object(mod, "_run_runtime_switch")
    @patch.object(mod, "_run_pipeline")
    @patch.object(mod, "_get_custom_providers", return_value=None)
    @patch.object(mod, "_get_user_providers", return_value=None)
    def test_returns_success_on_valid_switch(self, mock_up, mock_cp, mock_pipeline, mock_runtime):
        mock_pipeline.return_value = _mock_pipeline_result()
        mock_runtime.return_value = None

        agent = _make_agent()
        result = mod.model_switch(model="claude-sonnet-4", provider="anthropic", parent_agent=agent)

        assert result["success"] is True
        assert "claude-sonnet-4" in result["message"]

    @patch.object(mod, "_run_runtime_switch")
    @patch.object(mod, "_run_pipeline")
    @patch.object(mod, "_get_custom_providers", return_value=None)
    @patch.object(mod, "_get_user_providers", return_value=None)
    def test_calls_runtime_switch_with_credentials(self, mock_up, mock_cp, mock_pipeline, mock_runtime):
        mock_pipeline.return_value = _mock_pipeline_result(
            new_model="gemini-2.5-flash",
            target_provider="google",
            provider_label="Google",
            api_key="AIza-test",
            api_mode="openai",
        )

        agent = _make_agent()
        mod.model_switch(model="gemini-2.5-flash", provider="google", parent_agent=agent)

        mock_runtime.assert_called_once_with(
            agent,
            new_model="gemini-2.5-flash",
            new_provider="google",
            api_key="AIza-test",
            base_url="",
            api_mode="openai",
        )

    @patch.object(mod, "_run_pipeline")
    @patch.object(mod, "_get_custom_providers", return_value=None)
    @patch.object(mod, "_get_user_providers", return_value=None)
    def test_returns_error_when_pipeline_fails(self, mock_up, mock_cp, mock_pipeline):
        mock_pipeline.return_value = _mock_pipeline_result(
            success=False,
            new_model="",
            target_provider="",
            error_message="Unknown provider 'foobar'",
        )

        agent = _make_agent()
        result = mod.model_switch(model="test", provider="foobar", parent_agent=agent)

        assert result["success"] is False
        assert "Unknown provider" in result["message"]


# ---------------------------------------------------------------------------
# 3. Provider auto-detection
# ---------------------------------------------------------------------------

class TestProviderAutoDetection:
    """When provider is omitted, the pipeline should auto-detect."""

    @patch.object(mod, "_run_runtime_switch")
    @patch.object(mod, "_run_pipeline")
    @patch.object(mod, "_get_custom_providers", return_value=None)
    @patch.object(mod, "_get_user_providers", return_value=None)
    def test_auto_detect_provider_when_omitted(self, mock_up, mock_cp, mock_pipeline, mock_runtime):
        mock_pipeline.return_value = _mock_pipeline_result()
        mock_runtime.return_value = None

        agent = _make_agent()
        result = mod.model_switch(model="claude-sonnet-4", parent_agent=agent)

        # Pipeline should be called with empty explicit_provider
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs.get("explicit_provider") == ""
        assert result["success"] is True


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Edge cases and error paths."""

    def test_returns_error_when_no_parent_agent(self):
        result = mod.model_switch(model="test", parent_agent=None)
        assert result["success"] is False
        assert "no agent" in result["message"].lower() or "parent_agent" in result["message"].lower()

    @patch.object(mod, "_run_pipeline")
    @patch.object(mod, "_get_custom_providers", return_value=None)
    @patch.object(mod, "_get_user_providers", return_value=None)
    def test_returns_success_on_same_model(self, mock_up, mock_cp, mock_pipeline):
        """Switching to the same model+provider is a no-op with a message."""
        agent = _make_agent(model="gpt-4o", provider="openai")
        result = mod.model_switch(model="gpt-4o", provider="openai", parent_agent=agent)

        mock_pipeline.assert_not_called()
        assert result["success"] is True
        assert "already" in result["message"].lower() or "same" in result["message"].lower()

    @patch.object(mod, "_run_runtime_switch")
    @patch.object(mod, "_run_pipeline")
    @patch.object(mod, "_get_custom_providers", return_value=None)
    @patch.object(mod, "_get_user_providers", return_value=None)
    def test_returns_warning_when_pipeline_warns(self, mock_up, mock_cp, mock_pipeline, mock_runtime):
        mock_pipeline.return_value = _mock_pipeline_result(
            new_model="NousResearch/Hermes-3-Llama-3.1-70B",
            target_provider="openrouter",
            provider_label="OpenRouter",
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            api_mode="openai",
            warning_message="Hermes models are NOT agentic",
        )

        agent = _make_agent()
        result = mod.model_switch(model="NousResearch/Hermes-3-Llama-3.1-70B", parent_agent=agent)

        assert result["success"] is True
        assert result.get("warning") is not None
        assert "agentic" in result["warning"].lower() or "Hermes" in result["warning"]


# ---------------------------------------------------------------------------
# 5. Registry integration
# ---------------------------------------------------------------------------

class TestRegistry:
    """The tool must self-register in the tool registry."""

    def test_registered_in_registry(self):
        from tools.registry import registry
        assert "model_switch" in registry._tools

    def test_registered_in_delegation_toolset(self):
        from tools.registry import registry
        entry = registry._tools.get("model_switch")
        assert entry is not None
        assert entry.toolset == "delegation"
