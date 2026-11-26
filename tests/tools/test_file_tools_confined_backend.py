"""A backend that confines execution keeps the file tools inside it.

The file tools answer reads, writes, patches and searches on the host when the terminal backend runs
on the host. A backend that subclasses ``LocalEnvironment`` but confines its commands (a bubblewrap
sandbox: bind mounts, read-only paths, its own ``$TMPDIR``) sets ``uses_host_filesystem = False``, and
then every file tool goes through ``env.execute`` and sees what the sandbox sees.

The test backend exposes a real directory under a second path, the way a bind mount does: the view
path exists only for commands it runs, so a tool result that names it came through ``execute``.
"""

import json

import pytest

from agent.runtime_scope import bind_profile_scope
from agent.terminal_env_provider import TerminalEnvironmentProvider
from agent.terminal_env_registry import _registry as terminal_env_registry
from agent.verification_evidence import record_verify_run, verification_status
from tools import file_state, terminal_tool
from tools.environments.local import LocalEnvironment
from tools.file_tools import _get_file_ops, clear_file_ops_cache
from tools.file_tools_read_tracking import reset_file_dedup
from tools.registry import registry
from tools.terminal_tool_lifecycle import cleanup_vm

# The host fast path is POSIX-only (``_native_read_enabled``); Windows always takes the shell path.
pytestmark = pytest.mark.platforms("posix")


class _MountedViewEnv(LocalEnvironment):
    """Commands see ``real`` at ``view``; the host sees only ``real``."""

    def __init__(self, view: str, real: str):
        super().__init__(cwd=real, timeout=30, persistent=False)
        self._view, self._real = view, real

    def execute(self, command, cwd="", **kwargs):
        result = super().execute(command.replace(self._view, self._real),
                                 cwd=cwd.replace(self._view, self._real), **kwargs)
        result["output"] = result.get("output", "").replace(self._real, self._view)
        return result


class _ConfinedEnv(_MountedViewEnv):
    uses_host_filesystem = False


class _CardProvider(TerminalEnvironmentProvider):
    name = "test_card"
    is_remote = False
    is_container = False

    def __init__(self, workspace):
        self.default_workspace = str(workspace)

    @property
    def default_cwd(self):
        return self.default_workspace

    def is_available(self):
        return True

    def create_environment(self, *, cwd, timeout, **kwargs):
        return _ConfinedEnv(self.default_workspace, self.default_workspace)


def _dispatch(name, task_id, *, session_id=None, **arguments):
    result = registry.dispatch(name, arguments, task_id=task_id, session_id=session_id)
    assert isinstance(result, str)
    return json.loads(result)


@pytest.fixture
def mounted(tmp_path, monkeypatch, request):
    """``(env, view, real, task_id)`` with the env active for the task; ``a.txt`` holds two lines."""
    monkeypatch.delenv("HERMES_NATIVE_FILE_READ", raising=False)  # the fast path at its default: on
    view, real = tmp_path / "view", tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    env = request.param(str(view), str(real))
    task_id = f"confined-{request.node.name}"
    monkeypatch.setitem(terminal_tool._active_environments, terminal_tool._resolve_container_task_id(task_id), env)
    yield env, view, real, task_id
    clear_file_ops_cache()
    env.cleanup()


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
def test_confined_backend_runs_every_file_tool_through_execute(mounted):
    _env, view, real, task = mounted

    read = _dispatch("read_file", task, path=str(view / "a.txt"))
    assert "alpha" in read.get("content", ""), read

    written = _dispatch("write_file", task, path=str(view / "b.txt"), content="gamma\n")
    assert "error" not in written, written
    assert (real / "b.txt").read_text(encoding="utf-8") == "gamma\n"

    patched = _dispatch("patch", task, path=str(view / "a.txt"), old_string="beta", new_string="delta")
    assert "error" not in patched, patched
    assert (real / "a.txt").read_text(encoding="utf-8") == "alpha\ndelta\n"

    found = _dispatch("search_files", task, pattern="delta", path=str(view))
    assert [m["path"] for m in found.get("matches", [])] == [str(view / "a.txt")], found

    assert not view.exists(), "a file tool touched the host path instead of the sandbox"


@pytest.mark.parametrize("mounted", [_MountedViewEnv], indirect=True)
def test_host_backend_keeps_the_host_fast_path(mounted):
    _env, view, real, task = mounted

    read = _dispatch("read_file", task, path=str(view / "a.txt"))
    assert "alpha" not in json.dumps(read), read

    written = _dispatch("write_file", task, path=str(view / "b.txt"), content="gamma\n")
    assert "error" not in written, written
    assert (view / "b.txt").read_text(encoding="utf-8") == "gamma\n"
    assert not (real / "b.txt").exists()


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
def test_confined_overwrite_checks_backend_file_instead_of_host_decoy(mounted, tmp_path):
    _env, view, real, task = mounted
    view.mkdir()
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("host copy\n", encoding="utf-8")
    path = view / "a.txt"
    path.symlink_to(decoy)

    refused = _dispatch("write_file", task, path=str(path), content="unread\n")
    assert refused.get("stale_write_blocked"), refused
    read = _dispatch("read_file", task, path=str(path))
    assert read.get("content") == "1|alpha\n2|beta", read
    written = _dispatch("write_file", task, path=str(path), content="backend copy\n")
    assert "error" not in written, written
    assert (real / "a.txt").read_text(encoding="utf-8") == "backend copy\n"

    (real / "a.txt").write_text("external edit\n", encoding="utf-8")
    refused = _dispatch("write_file", task, path=str(path), content="stale\n")
    assert refused.get("stale_write_blocked"), refused
    assert (real / "a.txt").read_text(encoding="utf-8") == "external edit\n"
    assert decoy.read_text(encoding="utf-8") == "host copy\n"


@pytest.fixture
def card_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text("terminal:\n  backend: test_card\n", encoding="utf-8")
    path = workspace / "draft.json"
    path.write_text('{\n  "status": "draft"\n}\n', encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_SINGLE_QUERY", "1")
    task = "card-workspace"
    with bind_profile_scope(profile, hydrate_secrets=False):
        provider = _CardProvider(workspace)
        previous = terminal_env_registry.snapshot_registration(provider.name)
        terminal_env_registry.register(provider)
        terminal_tool.record_session_cwd(task, str(workspace))
        try:
            yield path, task
        finally:
            cleanup_vm(terminal_tool._resolve_container_task_id(task))
            terminal_tool.clear_session_cwd(task)
            clear_file_ops_cache(task)
            terminal_env_registry.restore_registration(provider.name, provider, previous)


@pytest.mark.parametrize("patch_first", [False, True], ids=["full-read", "full-reread-after-patch"])
def test_card_workspace_full_read_allows_overwrite(card_workspace, patch_first):
    path, task = card_workspace
    read = _dispatch("read_file", task, path=str(path))
    assert read.get("truncated") is False and read.get("total_lines") == 3, read
    if patch_first:
        patched = _dispatch("patch", task, path=str(path), old_string='"draft"', new_string='"reviewed"')
        assert patched["success"], patched
        read = _dispatch("read_file", task, path=str(path))
        assert read.get("truncated") is False and '"reviewed"' in read["content"], read

    content = '{"status": "final"}\n'
    written = _dispatch("write_file", task, path=str(path), content=content)
    assert "error" not in written, written.get("error")
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("prior_read", ["none", "partial", "stale"])
def test_card_workspace_overwrite_requires_current_full_read(card_workspace, prior_read):
    path, task = card_workspace
    if prior_read == "partial":
        read = _dispatch("read_file", task, path=str(path), limit=1)
        assert read.get("truncated") is True, read
    if prior_read == "stale":
        read = _dispatch("read_file", task, path=str(path))
        assert read.get("truncated") is False, read
        path.write_text('{"status": "changed externally"}\n', encoding="utf-8")
    saved = path.read_text(encoding="utf-8")

    refused = _dispatch("write_file", task, path=str(path), content='{"status": "final"}\n')
    assert refused.get("stale_write_blocked"), refused
    assert path.read_text(encoding="utf-8") == saved


def test_card_workspace_paged_read_uses_task_cwd_and_survives_compaction(card_workspace, tmp_path):
    path, task = card_workspace
    other = tmp_path / "other"
    other.mkdir()
    (other / path.name).write_text('{"status": "other"}\n', encoding="utf-8")
    _get_file_ops(task).env.cwd = str(other)

    first = _dispatch("read_file", task, path=path.name, limit=1)
    assert first.get("truncated") is True, first
    rest = _dispatch("read_file", task, path=path.name, offset=2)
    assert '"draft"' in rest.get("content", ""), rest
    reset_file_dedup(task)

    content = '{"status": "final"}\n'
    written = _dispatch("write_file", task, path=path.name, content=content)
    assert "error" not in written, written
    assert path.read_text(encoding="utf-8") == content
    assert (other / path.name).read_text(encoding="utf-8") == '{"status": "other"}\n'


@pytest.mark.parametrize("tool", ["write_file", "patch"])
def test_confined_edit_stales_verification(card_workspace, monkeypatch, tool):
    path, task = card_workspace
    workspace = path.parent
    (workspace / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    session = "card-verification"
    record_verify_run(root=workspace, session_id=session, ok=True)
    assert verification_status(session_id=session, cwd=workspace)["status"] == "passed"

    if tool == "write_file":
        edited = workspace / "new.txt"
        result = _dispatch(tool, task, session_id=session, path=str(edited), content="new\n")
    else:
        edited = path
        result = _dispatch(tool, task, session_id=session, path=str(edited), old_string="draft", new_string="final")
    assert "error" not in result, result
    status = verification_status(session_id=session, cwd=workspace)
    assert status["status"] == "stale", status
    assert status["changed_paths"] == [str(edited)]


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
@pytest.mark.parametrize("filename", ["a.txt", "a.ipynb"])
def test_confined_read_is_visible_to_delegation(mounted, filename):
    _env, view, real, task = mounted
    if filename.endswith(".ipynb"):
        (real / filename).write_text(
            json.dumps({"cells": [{"cell_type": "markdown", "source": ["alpha"]}]}),
            encoding="utf-8",
        )
    path = str(view / filename)
    result = _dispatch("read_file", task, path=path)
    assert "alpha" in result["content"]
    assert file_state.known_reads(task) == [path]


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
@pytest.mark.parametrize("tool", ["write_file", "patch"])
def test_confined_edit_is_visible_to_delegation(mounted, tool):
    _env, view, _real, task = mounted
    if tool == "write_file":
        path = str(view / "new.txt")
        result = _dispatch(tool, task, path=path, content="new\n")
    else:
        path = str(view / "a.txt")
        result = _dispatch(tool, task, path=path, old_string="beta", new_string="gamma")
    assert "error" not in result, result
    assert file_state.writes_since("parent", 0, [path]) == {task: [path]}
    assert file_state.writes_since("parent", 0)[task] == [path]


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
def test_confined_patch_warns_about_sibling_write(mounted, monkeypatch):
    env, view, _real, task = mounted
    path = str(view / "a.txt")
    sibling = f"{task}-sibling"
    monkeypatch.setitem(terminal_tool._active_environments, terminal_tool._resolve_container_task_id(sibling), env)
    assert "alpha" in _dispatch("read_file", task, path=path)["content"]
    changed = _dispatch("patch", sibling, path=path, old_string="beta", new_string="gamma")
    assert changed["success"], changed
    result = _dispatch("patch", task, path=path, old_string="alpha", new_string="delta")
    assert result["success"], result
    assert f"modified by sibling subagent {sibling!r}" in result.get("_warning", ""), result


def test_confined_relative_edit_uses_symlinked_workspace(card_workspace):
    path, task = card_workspace
    alias = path.parent.with_name("workspace-link")
    alias.symlink_to(path.parent, target_is_directory=True)
    terminal_tool.record_session_cwd(task, str(alias))

    result = _dispatch("patch", task, path=path.name, old_string="draft", new_string="final")
    assert result["success"], result
    assert result["resolved_path"] == str(alias / path.name)
    assert "_warning" not in result, result

    outside = _dispatch("write_file", task, path="../outside.txt", content="outside\n")
    assert "error" not in outside, outside
    assert "OUTSIDE the active workspace" in outside.get("_warning", ""), outside


@pytest.fixture
def unrelated_terminal_provider(tmp_path):
    class UnrelatedProvider(_CardProvider):
        name = "test_unrelated_card"

    provider = UnrelatedProvider(tmp_path)
    previous = terminal_env_registry.snapshot_registration(provider.name)
    terminal_env_registry.register(provider)
    yield provider
    registered = terminal_env_registry.get_provider(provider.name)
    terminal_env_registry.restore_registration(provider.name, provider, previous)
    assert registered is provider


def test_card_workspace_preserves_other_terminal_providers(unrelated_terminal_provider, request):
    request.getfixturevalue("card_workspace")
    assert terminal_env_registry.get_provider(unrelated_terminal_provider.name) is unrelated_terminal_provider


@pytest.mark.parametrize("mounted", [_ConfinedEnv], indirect=True)
def test_confined_overwrite_explains_backend_without_hash_tools(mounted, monkeypatch):
    env, view, real, task = mounted
    execute = env.execute

    def execute_without_hash_tools(command, **kwargs):
        if "sha256sum " in command:
            command = "export PATH=\n" + command
        return execute(command, **kwargs)

    monkeypatch.setattr(env, "execute", execute_without_hash_tools)
    path = str(view / "a.txt")
    assert "alpha" in _dispatch("read_file", task, path=path)["content"]

    result = _dispatch("write_file", task, path=path, content="replacement\n")
    assert result.get("stale_write_blocked"), result
    assert "backend cannot hash" in result["error"], result
    assert (real / "a.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"
