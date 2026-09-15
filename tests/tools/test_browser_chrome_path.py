from pathlib import Path

from hermes_cli.config import DEFAULT_CONFIG
from tools import browser_tool


def _stub_scrubbed_env(monkeypatch):
    monkeypatch.setattr(
        "tools.environments.local.hermes_subprocess_env",
        lambda **_kwargs: {"PATH": "/usr/bin"},
    )


def test_configured_chrome_path_is_recognized_expanded_and_exported(
    monkeypatch, tmp_path
):
    executable = tmp_path / "Chrome for Testing"
    executable.write_text("binary")
    executable.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    _stub_scrubbed_env(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"chrome_path": "~/Chrome for Testing"}},
    )

    env = browser_tool._build_browser_env()

    assert DEFAULT_CONFIG["browser"]["chrome_path"] == ""
    assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == str(executable)


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
