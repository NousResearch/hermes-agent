"""Unit tests for the windows-sandbox backend wrapper integration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.environments.windows_sandbox as windows_sandbox


pytestmark = pytest.mark.skipif(
    not windows_sandbox._IS_WINDOWS,
    reason="windows-sandbox backend only runs on Windows hosts",
)


def _make_wrapper(tmp_path: Path) -> Path:
    wrapper = tmp_path / "hermes-windows-sandbox-wrapper.exe"
    wrapper.write_text("stub", encoding="utf-8")
    return wrapper


def _make_setup_helper(tmp_path: Path) -> Path:
    helper = tmp_path / "codex-windows-sandbox-setup.exe"
    helper.write_text("stub", encoding="utf-8")
    return helper


def test_find_wrapper_executable_prefers_configured_bin_dir(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    found = windows_sandbox.find_wrapper_executable(str(tmp_path))
    assert found == wrapper


def test_find_wrapper_executable_prefers_hermes_bin_dir(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    hermes_bin = hermes_home / "bin"
    hermes_bin.mkdir(parents=True)
    wrapper = _make_wrapper(hermes_bin)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    found = windows_sandbox.find_wrapper_executable()
    assert found == wrapper

def test_find_wrapper_executable_ignores_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(windows_sandbox.shutil, "which", lambda _name: str(tmp_path / "path-wrapper.exe"))

    found = windows_sandbox.find_wrapper_executable()
    assert found is None


def test_find_setup_helper_executable_prefers_wrapper_sibling(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)

    found = windows_sandbox.find_setup_helper_executable(
        wrapper_path=wrapper,
    )
    assert found == helper

def test_find_setup_helper_executable_ignores_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(windows_sandbox.shutil, "which", lambda _name: str(tmp_path / "path-helper.exe"))

    found = windows_sandbox.find_setup_helper_executable()
    assert found is None


def test_provision_windows_sandbox_binaries_copies_wrapper_and_helper(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    target_home = tmp_path / "hermes-home"
    source_dir.mkdir()
    wrapper = _make_wrapper(source_dir)
    helper = _make_setup_helper(source_dir)
    monkeypatch.setenv("HERMES_HOME", str(target_home))

    provisioned = windows_sandbox.provision_windows_sandbox_binaries(str(source_dir))

    assert provisioned["wrapper"] == target_home / "bin" / wrapper.name
    assert provisioned["setup_helper"] == target_home / "bin" / helper.name
    assert provisioned["wrapper"].read_text(encoding="utf-8") == "stub"
    assert provisioned["setup_helper"].read_text(encoding="utf-8") == "stub"


def test_default_codex_home_uses_windows_sandbox_state_root(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=60,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )

    assert env.codex_home == str(
        Path(tmp_path / "hermes-home" / "sandboxes" / "windows-sandbox" / "codex-home")
    )


def test_execute_invokes_wrapper_and_flattens_stderr(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "stdout": "hello from sandbox\r\n",
                "stderr": "warning text",
                "exit_code": 7,
                "timed_out": False,
                "error": None,
                "error_type": None,
                "diagnostics": {"shell": "powershell"},
            }),
            stderr="",
        )

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=60,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )
    result = env.execute("Get-ChildItem", cwd=str(tmp_path / "repo"), timeout=12)

    request = json.loads(captured["input"])
    assert captured["cmd"] == [str(wrapper), "exec"]
    assert request["command"] == "Get-ChildItem"
    assert request["cwd"] == str(tmp_path / "repo")
    assert request["timeout_secs"] == 12
    assert request["mode"] == "workspace-write"
    assert result["returncode"] == 7
    assert "hello from sandbox" in result["output"]
    assert "[stderr]" in result["output"]
    assert "warning text" in result["output"]


def test_execute_returns_timeout_result_when_wrapper_times_out(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)

    def fake_run(*_args, **_kwargs):
        raise windows_sandbox.subprocess.TimeoutExpired("wrapper", timeout=5)

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=9,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )
    result = env.execute("Get-ChildItem")
    assert result["returncode"] == 124
    assert "timed out after 9s" in result["output"]


def test_execute_maps_wrapper_timeout_error_type(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "stdout": "",
                "stderr": "",
                "exit_code": 124,
                "timed_out": False,
                "error": "Command timed out after 30 seconds",
                "error_type": "timeout",
                "diagnostics": {},
            }),
            stderr="",
        )

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=30,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )
    result = env.execute("Get-ChildItem")
    assert result["returncode"] == 124
    assert "timed out after 30s" in result["output"]


def test_execute_rejects_stdin_piping(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=30,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )
    result = env.execute("Get-ChildItem", stdin_data="hello")
    assert result["returncode"] == 1
    assert "does not support stdin piping" in result["output"]


def test_execute_reports_invalid_wrapper_json(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="not-json", stderr="broken wrapper")

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=30,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )
    result = env.execute("Get-ChildItem")
    assert result["returncode"] == -1
    assert "invalid JSON" in result["output"]
    assert "broken wrapper" in result["output"]


def test_execute_surfaces_setup_required_error(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "stdout": "",
                "stderr": "Windows sandbox setup is required before execution.",
                "exit_code": -1,
                "timed_out": False,
                "error": "Windows sandbox setup is required before execution.",
                "error_type": "setup_required",
                "diagnostics": {"setup_complete": False, "setup_code": "helper_log_failed"},
            }),
            stderr="",
        )

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=30,
        mode="workspace-write",
        bin_dir=str(tmp_path),
    )
    result = env.execute("Get-ChildItem")
    assert result["returncode"] == -1
    assert "setup is required" in result["output"]


def test_missing_wrapper_error_mentions_hermes_bin_dir(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: None)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: None,
    )

    with pytest.raises(RuntimeError) as excinfo:
        windows_sandbox.WindowsSandboxEnvironment(
            cwd=str(tmp_path),
            timeout=30,
            mode="workspace-write",
        )

    assert str(hermes_home / "bin") in str(excinfo.value)


def test_missing_setup_helper_error_mentions_hermes_bin_dir(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    wrapper = _make_wrapper(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: None,
    )

    with pytest.raises(RuntimeError) as excinfo:
        windows_sandbox.WindowsSandboxEnvironment(
            cwd=str(tmp_path),
            timeout=30,
            mode="workspace-write",
        )

    assert str(hermes_home / "bin") in str(excinfo.value)


def test_get_windows_sandbox_status_invokes_status_subcommand(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "ok": True,
                "error": None,
                "error_type": None,
                "diagnostics": {"setup_complete": True},
            }),
            stderr="",
        )

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    result = windows_sandbox.get_windows_sandbox_status(
        bin_dir=str(tmp_path),
        cwd=str(tmp_path / "repo"),
        mode="workspace-write",
        network_enabled=False,
        writable_roots=[str(tmp_path / "repo")],
        codex_home=str(tmp_path / "codex-home"),
    )

    request = json.loads(captured["input"])
    assert captured["cmd"] == [str(wrapper), "status"]
    assert request["cwd"] == str(tmp_path / "repo")
    assert request["mode"] == "workspace-write"
    assert request["writable_roots"] == [str(tmp_path / "repo")]
    assert result["diagnostics"]["setup_complete"] is True


def test_run_windows_sandbox_setup_invokes_setup_subcommand(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "ok": True,
                "error": None,
                "error_type": None,
                "diagnostics": {"setup_complete": True},
            }),
            stderr="",
        )

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    result = windows_sandbox.run_windows_sandbox_setup(
        bin_dir=str(tmp_path),
        cwd=str(tmp_path / "repo"),
        codex_home=str(tmp_path / "codex-home"),
    )

    request = json.loads(captured["input"])
    assert captured["cmd"] == [str(wrapper), "setup"]
    assert request["cwd"] == str(tmp_path / "repo")
    assert request["codex_home"] == str(tmp_path / "codex-home")
    assert result["diagnostics"]["setup_complete"] is True

def test_execute_invokes_wrapper_with_policy_fields(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    captured: dict = {}
    extra_root = tmp_path / "extra-root"
    codex_home = tmp_path / "codex-home"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "stdout": "policy ok",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "error": None,
                "error_type": None,
                "diagnostics": {},
            }),
            stderr="",
        )

    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )
    monkeypatch.setattr(windows_sandbox.subprocess, "run", fake_run)

    env = windows_sandbox.WindowsSandboxEnvironment(
        cwd=str(tmp_path),
        timeout=60,
        mode="workspace-write",
        network_enabled=True,
        writable_roots=[str(extra_root)],
        bin_dir=str(tmp_path),
        codex_home=str(codex_home),
    )
    result = env.execute("Get-ChildItem", cwd=str(tmp_path / "repo"), timeout=12)

    request = json.loads(captured["input"])
    assert captured["cmd"] == [str(wrapper), "exec"]
    assert request["mode"] == "workspace-write"
    assert request["network_enabled"] is True
    assert request["writable_roots"] == [str(extra_root)]
    assert request["codex_home"] == str(codex_home)
    assert result["returncode"] == 0
    assert "policy ok" in result["output"]






def test_windows_sandbox_environment_rejects_unsupported_architecture(monkeypatch, tmp_path):
    wrapper = _make_wrapper(tmp_path)
    helper = _make_setup_helper(tmp_path)
    monkeypatch.setattr(windows_sandbox.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_sandbox.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(windows_sandbox, "find_wrapper_executable", lambda _bin_dir=None: wrapper)
    monkeypatch.setattr(
        windows_sandbox,
        "find_setup_helper_executable",
        lambda _bin_dir=None, wrapper_path=None: helper,
    )

    with pytest.raises(RuntimeError, match="x64 Windows hosts"):
        windows_sandbox.WindowsSandboxEnvironment(
            cwd=str(tmp_path),
            timeout=60,
            mode="workspace-write",
            bin_dir=str(tmp_path),
        )
