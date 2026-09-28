"""Profile-scope regression for user TTS command-provider lifecycle hooks."""

import threading

import tools.tts_tool_lifecycle as lifecycle
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def test_command_provider_hook_keeps_routed_profile_context(tmp_path, monkeypatch):
    """The async warm/release hook must act for the profile that requested it."""
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    observed = {}
    done = threading.Event()

    monkeypatch.setattr(
        lifecycle, "_get_named_provider_config",
        lambda _tts_config, _name: {"warm_command": "echo warm"},
    )
    monkeypatch.setattr(lifecycle, "_is_command_provider_config", lambda _cfg: True)
    monkeypatch.setattr(
        lifecycle, "_render_command_tts_template", lambda template, _values: template,
    )
    monkeypatch.setattr(lifecycle, "_get_command_tts_timeout", lambda _cfg: 1)
    monkeypatch.setattr(lifecycle, "_command_provider_env_passthrough", lambda _cfg: ())

    def fake_run_command_provider(*_args, **_kwargs):
        observed["home"] = str(get_hermes_home())
        done.set()

    monkeypatch.setattr(
        lifecycle.tts_command_provider, "run_command_provider", fake_run_command_provider,
    )

    token = set_hermes_home_override(served_home)
    try:
        assert lifecycle._signal_user_tts_provider("custom", {}, "warm") == "warm"
        assert done.wait(2), "TTS command hook worker did not run"
    finally:
        reset_hermes_home_override(token)

    assert observed["home"] == str(served_home)
