"""Regression test for #11884: _make_agent must resolve runtime provider.

Without resolve_runtime_provider(), bare-slug models in config
(e.g. ``claude-opus-4-6`` with ``model.provider: anthropic``) leave
provider/base_url/api_key empty in AIAgent, causing HTTP 404.
"""

import os
from unittest.mock import patch




def test_probe_config_health_flags_null_sections():
    """Bare YAML keys (`agent:` with no value) parse as None and silently
    drop nested settings; probe must surface them so users can fix."""
    from tui_gateway.server import _probe_config_health

    assert _probe_config_health({"agent": {"x": 1}}) == ""
    assert _probe_config_health({}) == ""

    msg = _probe_config_health({"agent": None, "display": None, "model": {}})
    assert "agent" in msg and "display" in msg
    assert "model" not in msg


def test_apply_model_switch_does_not_leak_process_env():
    """Core fix for cross-session contamination: an in-session /model switch
    must mutate only the target session (record a per-session override + switch
    that session's agent in place) and must NOT write process-global env vars,
    which the single-process desktop backend shares across every live session.
    """
    from tui_gateway import server

    class _FakeResult:
        success = True
        error_message = ""
        warning_message = ""
        new_model = "zai/glm-5.1"
        target_provider = "zai"
        base_url = "https://api.z.ai/v1"
        api_key = "sk-glm"
        api_mode = "chat_completions"

    class _FakeAgent:
        def __init__(self):
            self.model = "minimax/m3"
            self.provider = "minimax"
            self.base_url = ""
            self.api_key = ""

        def switch_model(self, **kw):
            self.model = kw["new_model"]
            self.provider = kw["new_provider"]

    env_keys = (
        "HERMES_MODEL",
        "HERMES_INFERENCE_MODEL",
        "HERMES_TUI_PROVIDER",
        "HERMES_INFERENCE_PROVIDER",
    )

    sess_b = {
        "agent": _FakeAgent(), "session_key": "k-B", "model_override": None,
        "follow_profile_config": True,
    }
    sess_a = {"agent": _FakeAgent(), "session_key": "k-A", "model_override": None}

    persisted_composer_profiles = []
    with (
        patch("hermes_cli.model_switch.parse_model_flags",
              return_value=("glm-5.1", None, False, False, True)),
        patch("hermes_cli.model_switch.resolve_persist_behavior",
              return_value=False),
        patch("hermes_cli.model_switch.switch_model", return_value=_FakeResult()),
        patch("tui_gateway.server._emit"),
        patch("tui_gateway.server._restart_slash_worker"),
        patch("tui_gateway.server._session_info", return_value={}),
        patch("hermes_cli.model_switch.persist_model_selection") as mock_persist,
        patch(
            "tui_gateway.server._persist_live_session_runtime",
            side_effect=lambda session: persisted_composer_profiles.append(
                session.get("composer_override_profile")),
        ) as persist_runtime,
        patch("tui_gateway.server._config_model_target", return_value=("minimax/m3", "minimax")),
    ):
        before = {k: os.environ.get(k) for k in env_keys}
        result = server._apply_model_switch("sidB", sess_b, "glm-5.1")
        after = {k: os.environ.get(k) for k in env_keys}

    assert result["value"] == "zai/glm-5.1"
    # No process-global env mutation (the contamination vector).
    assert before == after
    # persist_global was False → config untouched.
    mock_persist.assert_not_called()
    # Target session recorded a per-session override.
    assert sess_b["model_override"]["model"] == "zai/glm-5.1"
    assert sess_b["model_override"]["provider"] == "zai"
    assert sess_b["composer_override_profile"] == {"model": "minimax/m3", "provider": "minimax"}
    # _commit_agent_switch owns the runtime transaction; provenance must be present
    # on its first (and only) DB write rather than relying on a second best-effort write.
    persist_runtime.assert_called_once_with(sess_b)
    assert persisted_composer_profiles == [{"model": "minimax/m3", "provider": "minimax"}]
    # The switched agent mutated in place.
    assert sess_b["agent"].model == "zai/glm-5.1"
    # Sibling session is completely untouched.
    assert sess_a["model_override"] is None
    assert sess_a["agent"].model == "minimax/m3"


def test_resumed_row_cannot_pin_stale_wire_onto_per_model_provider():
    """#96066: a persisted opencode-go row written while the session ran an anthropic_messages model must not
    route deepseek-v4-flash-vision-exp through the Anthropic wire or the other family's relay URL on resume;
    the route is re-derived from the target model. Fixed-wire providers keep honoring their row."""
    from tui_gateway import server

    def fake_resolve(**kwargs):
        provider = kwargs["requested"]
        fresh = {"opencode-go": ("chat_completions", "https://opencode.ai/zen/go/v1"),
                 "anthropic": ("anthropic_messages", "https://api.anthropic.com")}[provider]
        return {"provider": provider, "requested_provider": provider, "api_mode": fresh[0], "base_url": fresh[1],
                "api_key": "k", "source": "config"}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve):
        for stale_url in ("https://opencode.ai/zen/go", "https://opencode.ai/zen/v1"):
            _, runtime = server._resolve_agent_model_runtime(
                {"model": "deepseek-v4-flash-vision-exp", "provider": "opencode-go",
                 "base_url": stale_url, "api_mode": "anthropic_messages"}, None)
            assert (runtime["api_mode"], runtime["base_url"]) == ("chat_completions", "https://opencode.ai/zen/go/v1")
        _, runtime = server._resolve_agent_model_runtime(
            {"model": "claude-opus-4-6", "provider": "anthropic",
             "base_url": "https://my-proxy.example", "api_mode": "anthropic_messages"}, None)
        assert (runtime["api_mode"], runtime["base_url"]) == ("anthropic_messages", "https://my-proxy.example")
