"""File tools on a container (or SSH) backend driven from a native Windows host.

The backend's paths are POSIX; the host's are ``C:\\...``. The resolver used to read
a drive path as relative and anchor it with POSIX rules: an absolute host path under
the env's mount (``host_cwd`` -> ``host_cwd_mount``, as Docker binds a Windows
workspace) landed at ``<container cwd>/C:\\...``, a session's first relative write
anchored to the host ``TERMINAL_CWD`` the same way, and a correct relative write
after a terminal command came back with a false "OUTSIDE the active workspace"
warning because the check compared POSIX paths as Windows ones.

The backend here runs commands in a real shell (Git Bash on this host), which sees
the host workspace at its POSIX path the way WSL sees ``C:\\`` at ``/mnt/c``. Writes
land on disk, so a misresolved path shows up as a stray directory tree.
"""

import json
from pathlib import Path

import pytest

import model_tools
import tools.file_tools_paths as paths
from agent import terminal_env_registry
from agent.terminal_env_provider import TerminalEnvironmentProvider
from tools import file_tools, terminal_tool
from tools.environments.local import LocalEnvironment, _bash_safe_path

TASK = "sess-host-paths"
PLUGIN = "host_mount_sandbox"


class _ShellBackend:
    """A non-local backend whose commands run in a real shell on this host."""

    def __init__(self, cwd: str, **attrs):
        self._shell = LocalEnvironment(cwd=str(Path.cwd()), timeout=60)
        self.cwd = cwd
        self.__dict__.update(attrs)

    def execute(self, command, cwd="", **kwargs):
        return self._shell.execute(command, cwd=cwd or self.cwd, **kwargs)

    def cleanup(self):
        self._shell.cleanup()


class _MountedSandboxProvider(TerminalEnvironmentProvider):
    name = PLUGIN
    display_name = "Host-mount sandbox"

    def __init__(self, host_dir: Path):
        self.host_dir = host_dir

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, task_id="default", **kwargs):
        mount = _bash_safe_path(str(self.host_dir))
        return _ShellBackend(mount, env_type=PLUGIN, host_cwd=str(self.host_dir), host_cwd_mount=mount)


@pytest.fixture
def isolated_envs(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(paths, "_env_bringup_failed_at", {})
    yield
    for env in list(terminal_tool._active_environments.values()):
        env.cleanup()


@pytest.fixture
def sandbox(tmp_path, monkeypatch, isolated_envs):
    """Plugin container backend with the host workspace bound at its POSIX path;
    ``TERMINAL_CWD`` names the host workspace, as a Windows config does."""
    proj = tmp_path / "proj"
    proj.mkdir()
    provider = _MountedSandboxProvider(proj)
    previous = terminal_env_registry.get_provider(PLUGIN)
    terminal_env_registry.register_provider(provider)
    monkeypatch.setenv("TERMINAL_ENV", PLUGIN)
    monkeypatch.setenv("TERMINAL_CWD", str(proj))
    yield proj, _bash_safe_path(str(proj))
    terminal_env_registry.restore_registration(PLUGIN, provider, previous)


def _call(name: str, args: dict) -> dict:
    return json.loads(model_tools.handle_function_call(name, args, task_id=TASK))


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


@pytest.mark.platforms("windows")
def test_host_paths_resolve_through_the_container_mount(sandbox):
    proj, mount = sandbox

    first = _call("write_file", {"path": "notes/plan.md", "content": "first\n"})
    absolute = _call("write_file", {"path": str(proj / "abs.md"), "content": "abs\n"})

    assert first["resolved_path"] == f"{mount}/notes/plan.md", first
    assert absolute["resolved_path"] == f"{mount}/abs.md", absolute
    assert "_warning" not in first and "_warning" not in absolute
    assert (proj / "notes" / "plan.md").read_text(encoding="utf-8-sig") == "first\n"
    assert (proj / "abs.md").read_text(encoding="utf-8-sig") == "abs\n"
    assert _tree(proj) == ["abs.md", "notes", "notes/plan.md"]
    assert "abs" in _call("read_file", {"path": str(proj / "abs.md")})["content"]


@pytest.mark.platforms("windows")
def test_relative_writes_compare_against_the_container_cwd(sandbox):
    proj, mount = sandbox
    _call("write_file", {"path": "seed.md", "content": "seed\n"})  # brings the backend up
    terminal_tool.record_session_cwd(TASK, mount)  # what a completed terminal command leaves

    inside = _call("write_file", {"path": "notes/plan2.md", "content": "second\n"})
    outside = _call("write_file", {"path": "../outside.md", "content": "out\n"})

    assert inside["resolved_path"] == f"{mount}/notes/plan2.md"
    assert "_warning" not in inside, inside["_warning"]
    assert (proj / "notes" / "plan2.md").read_text(encoding="utf-8-sig") == "second\n"
    assert outside["resolved_path"] == f"{mount.rsplit('/', 1)[0]}/outside.md"
    assert f"'{mount}'" in outside["_warning"]


@pytest.mark.platforms("windows")
def test_host_path_outside_the_mount_is_refused(sandbox, tmp_path):
    """No container path names it: writing it anyway lands in the sandbox's own
    filesystem (or the workspace) while the result reports the host path."""
    proj, mount = sandbox
    outside = tmp_path / "elsewhere" / "x.md"

    written = _call("write_file", {"path": str(outside), "content": "x\n"})
    patched = _call("patch", {"mode": "patch",
                              "patch": f"*** Begin Patch\n*** Add File: {outside}\n+x\n*** End Patch\n"})

    for out in (written, patched):
        assert mount in out.get("error", ""), out
    assert not outside.exists()
    assert _tree(proj) == []


@pytest.mark.platforms("windows")
def test_ssh_relative_write_under_the_remote_cwd_has_no_warning(tmp_path, isolated_envs):
    remote = tmp_path / "remote"
    (remote / "proj").mkdir(parents=True)
    home = _bash_safe_path(str(remote))
    key = terminal_tool._resolve_container_task_id(TASK)
    terminal_tool._active_environments[key] = _ShellBackend(
        f"{home}/proj", env_type="ssh", _hermes_backend_name="ssh",
        _remote_home=home, _remote_home_detected=True)
    terminal_tool.record_session_cwd(TASK, "~/proj")

    out = _call("write_file", {"path": "x.txt", "content": "x\n"})

    assert out["resolved_path"] == f"{home}/proj/x.txt", out
    assert "_warning" not in out, out["_warning"]
    assert (remote / "proj" / "x.txt").read_text(encoding="utf-8-sig") == "x\n"
