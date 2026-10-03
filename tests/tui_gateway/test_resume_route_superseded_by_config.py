"""A resume must not resurrect a session route that only mirrored a since-changed config default.

The state.db row's ``model_config`` now carries a ``config_default`` provenance stamp (written by
``_runtime_model_config``): the config.yaml (model.default, model.provider) the row's route was
captured against. Resume drops the route override when the row merely mirrored that default and
config has since explicitly changed — an explicit per-chat divergence survives (#122016).
"""

import json

import pytest

from tui_gateway import server


def _stamp(cfg_model: str, cfg_provider: str) -> dict:
    return {"config_default": {"model": cfg_model, "provider": cfg_provider}}


def _row(model: str, provider: str, extra_config: dict | None = None) -> dict:
    return {"model": model, "model_config": json.dumps(
        {"model": model, "provider": provider, **(extra_config or {})})}


@pytest.fixture
def config_pins(monkeypatch):
    def _pin(model: str, provider: str):
        monkeypatch.setattr(server, "_load_cfg", lambda: {
            "model": {"default": model, "provider": provider}})
    return _pin


def test_resume_drops_route_that_mirrored_a_changed_config_default(config_pins):
    config_pins("gpt-5.6-luna", "openai-codex")
    # Row captured while config pinned the deepseek default; the row still mirrors that default.
    row = _row("deepseek-flash", "deepseek", _stamp("deepseek-flash", "deepseek"))

    overrides = server._stored_session_runtime_overrides(row)

    assert "model_override" not in overrides
    assert "provider_override" not in overrides


def test_resume_keeps_explicit_pick_across_config_change(config_pins):
    config_pins("gpt-5.6-luna", "openai-codex")
    # The chat explicitly picked deepseek while config pinned the luna default: deliberate
    # divergence, not an inherited default — it must survive the config edit.
    row = _row("deepseek-flash", "deepseek", _stamp("gpt-5.6-luna", "openai-codex"))

    overrides = server._stored_session_runtime_overrides(row)

    assert overrides["model_override"]["model"] == "deepseek-flash"
    assert overrides["model_override"]["provider"] == "deepseek"


def test_resume_keeps_route_when_config_unchanged(config_pins):
    config_pins("gpt-5.6-luna", "openai-codex")
    row = _row("gpt-5.6-luna", "openai-codex", _stamp("gpt-5.6-luna", "openai-codex"))

    overrides = server._stored_session_runtime_overrides(row)

    assert overrides["model_override"]["model"] == "gpt-5.6-luna"


def test_resume_legacy_row_without_stamp_keeps_old_restore(config_pins):
    config_pins("gpt-5.6-luna", "openai-codex")
    # Rows persisted before the stamp exist forever; no stamp, old behavior.
    row = _row("deepseek-flash", "deepseek")

    overrides = server._stored_session_runtime_overrides(row)

    assert overrides["model_override"]["model"] == "deepseek-flash"


def test_persist_stamp_names_the_config_target(config_pins):
    """_runtime_model_config stamps the config target it captured the runtime against."""
    from types import SimpleNamespace

    config_pins("gpt-5.6-luna", "openai-codex")
    agent = SimpleNamespace(model="gpt-5.6-luna", provider="openai-codex", base_url="",
                            api_mode="", reasoning_config=None, service_tier=None)
    persisted = server._runtime_model_config(agent)
    assert persisted["config_default"] == {"model": "gpt-5.6-luna", "provider": "openai-codex"}
