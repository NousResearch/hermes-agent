"""image.generate binds the routed profile's secret scope on a multiplexed backend.

The Desktop avatar picker probes and calls ``image.generate`` directly. The handler was
registered unscoped, so once ``hermes serve`` hosted more than one profile the first scoped
credential read inside the configured image backend (``HERMES_CODEX_BASE_URL`` for
``openai-codex``) raised ``UnscopedSecretError``; the handler's availability guard swallowed
it and the picker reported "No image model available" while chat-side ``image_generate``
kept working. Same shape as #112061 (readiness) and #117544 (/review).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_yaml as yaml
from tui_gateway import server


class _ScopedReadProvider:
    """Stands in for a plugin backend: availability performs the scoped read the openai-codex
    plugin hits through a pooled credential (``PooledCredential.runtime_base_url`` →
    ``get_secret_str("HERMES_CODEX_BASE_URL")``), so an unbound scope raises."""

    name = "openai-codex"

    def is_available(self) -> bool:
        from agent.secret_scope import get_secret_str

        get_secret_str("HERMES_CODEX_BASE_URL", "")
        return True


def _write_cfg(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"image_gen": {"provider": "openai-codex"}}), encoding="utf-8"
    )


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    """Launch home + named profile ``bot``, both on openai-codex image gen, with multi-profile
    hosting active the way ``hermes serve`` activates it at boot."""
    import tools.image_generation_tool as image_tool
    import tui_gateway.launch_profile_policy as policy
    from agent.secret_scope import set_multiplex_active

    launch = tmp_path / ".hermes"
    bot = launch / "profiles" / "bot"
    _write_cfg(launch)
    _write_cfg(bot)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_profile_home", lambda name: bot if (name or "").strip() == "bot" else None)
    monkeypatch.setattr(image_tool, "check_fal_api_key", lambda: False)
    monkeypatch.setattr(image_tool, "_get_plugin_provider", lambda name, force=False: _ScopedReadProvider())
    monkeypatch.setattr(policy, "_snapshot", None)
    policy.activate_multi_profile_hosting()
    try:
        yield launch, bot
    finally:
        set_multiplex_active(False)


def _probe(params: dict) -> dict:
    resp = server._methods["image.generate"]("rid-probe", {"probe": True, **params})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.mark.parametrize("params", [{}, {"profile": "bot"}], ids=["launch-profile", "secondary-profile"])
def test_image_generate_probe_is_available_under_multiplex(hosted, params):
    assert _probe(params) == {"available": True}


def test_unscoped_backend_read_still_fails_closed(hosted):
    """Control: the same read outside any scope must keep raising (isolation is not weakened)."""
    from agent.secret_scope import UnscopedSecretError

    with pytest.raises(UnscopedSecretError):
        _ScopedReadProvider().is_available()
