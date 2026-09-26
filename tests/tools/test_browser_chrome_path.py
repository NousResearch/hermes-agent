from pathlib import Path

from hermes_cli.config import DEFAULT_CONFIG
from tools import browser_tool


def _stub_scrubbed_env(monkeypatch):
    monkeypatch.setattr(
        "tools.environments.local.hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )


def test_configured_chrome_path_follows_active_profile_scope(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    executable_a = tmp_path / "Chrome for Testing A"
    executable_b = tmp_path / "Chrome for Testing B"
    for executable in (executable_a, executable_b):
        executable.write_text("binary")
        executable.chmod(0o755)
    (home_a / "config.yaml").write_text(
        f"browser:\n  chrome_path: {executable_a}\n"
    )
    (home_b / "config.yaml").write_text(
        f"browser:\n  chrome_path: {executable_b}\n"
    )
    _stub_scrubbed_env(monkeypatch)

    resolved = []
    for home in (home_a, home_b, home_a):
        token = set_hermes_home_override(str(home))
        try:
            resolved.append(
                browser_tool._build_browser_env()["AGENT_BROWSER_EXECUTABLE_PATH"]
            )
        finally:
            reset_hermes_home_override(token)

    assert DEFAULT_CONFIG["browser"]["chrome_path"] == ""
    assert resolved == [str(executable_a), str(executable_b), str(executable_a)]


def test_unresolvable_configured_chrome_path_is_ignored(monkeypatch):
    _stub_scrubbed_env(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"chrome_path": "~unresolvable/Chrome"}},
    )

    def raise_unresolvable(_path):
        raise RuntimeError("home directory cannot be resolved")

    monkeypatch.setattr(Path, "expanduser", raise_unresolvable)

    env = browser_tool._build_browser_env()

    assert "AGENT_BROWSER_EXECUTABLE_PATH" not in env
