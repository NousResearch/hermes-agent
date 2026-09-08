"""Regression coverage for provider-aware context sizing in the tool-search gate.

``model_tools._resolve_active_context_length()`` feeds ``should_activate``'s
window-fraction check. Providers like Codex OAuth enforce a lower context
window than the direct API for the same slug (e.g. gpt-5.5 is 1.05M on the
API but 272K on the Codex route), and ``get_model_context_length()`` only
applies those provider-aware resolutions when it receives the provider,
base_url, and credential. Before this coverage existed the gate called the
resolver with the model id alone, so Codex sessions sized activation against
generic direct-API metadata.
"""

from unittest.mock import patch


def _model_cfg(**overrides):
    cfg = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "",
    }
    cfg.update(overrides)
    return {"model": cfg}


class TestResolveActiveContextLengthProviderAware:
    def test_passes_provider_base_url_and_key_from_runtime(self):
        """Resolved runtime credentials must reach get_model_context_length."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(
                model=model_id, base_url=base_url, api_key=api_key,
                config_ctx=config_context_length, provider=provider,
            )
            return 272_000

        with patch("hermes_cli.config.load_config", return_value=_model_cfg()), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"base_url": "https://chatgpt.com/backend-api/codex",
                                 "api_key": "tok-live"}) as mock_rt, \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 272_000
        assert captured["provider"] == "openai-codex"
        assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert captured["api_key"] == "tok-live"
        mock_rt.assert_called_once_with(
            requested="openai-codex", target_model="gpt-5.6-sol"
        )

    def test_offline_credential_failure_degrades_to_config_values(self):
        """Runtime resolution raising must not zero the gate — the resolver is
        still called with the configured provider/base_url and an empty key so
        static provider-aware fallbacks apply."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(base_url=base_url, api_key=api_key, provider=provider)
            return 272_000

        with patch("hermes_cli.config.load_config",
                   return_value=_model_cfg(base_url="https://chatgpt.com/backend-api/codex")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   side_effect=RuntimeError("no credentials")), \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 272_000
        assert captured["provider"] == "openai-codex"
        assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert captured["api_key"] == ""

    def test_no_provider_configured_skips_runtime_resolution(self):
        """Without a provider in config, behavior matches the legacy path: no
        runtime resolution attempt, resolver called with empty routing."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured.update(base_url=base_url, provider=provider)
            return 200_000

        with patch("hermes_cli.config.load_config",
                   return_value={"model": {"model": "some-model"}}), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_rt, \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 200_000
        assert captured["provider"] == ""
        mock_rt.assert_not_called()

    def test_config_context_length_still_short_circuits(self):
        """Explicit model.context_length must keep winning (issue #46620)."""
        import model_tools

        captured = {}

        def fake_get_ctx(model_id, base_url="", api_key="", config_context_length=None, provider=""):
            captured["config_ctx"] = config_context_length
            return config_context_length or 0

        with patch("hermes_cli.config.load_config",
                   return_value=_model_cfg(context_length=150_000)), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"base_url": "https://chatgpt.com/backend-api/codex",
                                 "api_key": "tok"}), \
             patch("agent.model_metadata.get_model_context_length", side_effect=fake_get_ctx):
            ctx = model_tools._resolve_active_context_length()

        assert ctx == 150_000
        assert captured["config_ctx"] == 150_000
