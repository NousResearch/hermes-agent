"""Binary-document overwrite guard must decide WHERE THE WRITE WILL EXECUTE.

A text write can never produce a valid SQLite/PDF payload, so write_file/patch
refuse to overwrite an EXISTING binary document. The existence check used to
stat the CONTROLLER's disk only, so a target that existed only in the task's
execution target (Docker/SSH/... filesystem namespace) was treated as new and
destroyed (#122662). These contracts drive real registry dispatch and real
shell/file I/O: the namespace boundary is emulated by a transport stub around
``LocalEnvironment.execute`` that rewrites the host-visible view path to a
target directory — only PATHS are substituted, never probe results.
"""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.file_tools as file_tools_mod  # noqa: F401 — registers the file tools
import tools.terminal_tool as terminal_tool
from tools.environments.local import LocalEnvironment, _bash_safe_path
from tools.registry import registry

_KEEP = "KEEPME line"
_BROKEN = "CHANGED line"


def _binary_payload(ext: str) -> bytes:
    """Real-format bytes for the target file: a genuine SQLite / PDF header plus
    one matchable text line (the V4A/replace edit anchor)."""
    if ext == ".pdf":
        # Pure-text PDF body: raw PDF syntax is text-authorable, and the
        # text-vs-binary read sniff must NOT be what saves the file — only the
        # write guard does.
        return (b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
                + _KEEP.encode() + b"\nend\n%%EOF\n")
    return b"SQLite format 3\x00\x01\x02\x03\xff\xfe\n" + _KEEP.encode() + b"\nend\n\x00\x07tail\n"


class VercelSandboxEnvironment:
    """Non-local backend double whose CLASS NAME no name-hint table classifies
    (the real ``VercelSandboxEnvironment`` is missing from
    ``_ENV_CLASS_NAME_HINTS``), so any locality decision taken from env_type
    tags, scoped config or class-name hints misclassifies it as local.

    Its transport is a real ``LocalEnvironment`` bash whose commands have every
    form of the host-visible view dir rewritten to the execution-target dir —
    a filesystem namespace the controller cannot stat. Paths only; the shell
    answers every probe for real. ``echo $HOME`` is answered with the target
    dir: the execution target's home is its own namespace's, never the host's.
    """

    env_type = "vercel_sandbox"

    def __init__(self, view_dir: Path, target_dir: Path, inner: LocalEnvironment):
        self.cwd = str(view_dir)
        self._inner = inner
        self.failure_mode = None  # None | "raise" | "error"
        self._home_answer = str(target_dir)
        src, dst = str(view_dir), str(target_dir)
        self._sub_pairs = [
            (src, dst),
            (src.replace("\\", "/"), dst.replace("\\", "/")),
            (_bash_safe_path(src), _bash_safe_path(dst)),
        ]

    def _sub(self, text: str) -> str:
        for src, dst in self._sub_pairs:
            text = text.replace(src, dst)
        return text

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        if self.failure_mode == "raise":
            raise RuntimeError("transport down")
        if self.failure_mode == "error":
            return {"output": "bash: transport unreachable", "returncode": 1}
        if command.strip() == "echo $HOME":
            return {"output": self._home_answer + "\n", "returncode": 0}
        return self._inner.execute(self._sub(command), cwd=self._sub(cwd), **kwargs)


@pytest.fixture()
def remote_target(tmp_path: Path, monkeypatch):
    """A task whose writes EXECUTE against ``target/`` while the controller-side
    ``view/`` namespace is empty: real namespace boundary, real shell I/O."""
    view = tmp_path / "view"
    target = tmp_path / "target"
    view.mkdir()
    target.mkdir()
    # Scoped config claims 'local' — it must not decide locality (T2c pin).
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda *a, **k: {"env_type": "local", "cwd": None})
    inner = LocalEnvironment(cwd=str(target))
    env = VercelSandboxEnvironment(view, target, inner)
    task_id = f"guard-{uuid.uuid4().hex}"
    # Own env key (RL/benchmark isolation override), never the shared "default".
    terminal_tool.register_task_env_overrides(task_id, {"env_type": "vercel_sandbox"})
    with terminal_tool._env_lock:
        terminal_tool._active_environments[task_id] = env
    try:
        yield SimpleNamespace(task_id=task_id, view=view, target=target, env=env)
    finally:
        file_tools_mod.clear_file_ops_cache(task_id)
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(task_id, None)
        terminal_tool._task_env_overrides.pop(task_id, None)
        terminal_tool._creation_locks.pop(task_id, None)
        terminal_tool._last_activity.pop(task_id, None)


def _dispatch(name: str, args: dict, task_id: str) -> dict:
    result = registry.dispatch(name, args, task_id=task_id)
    return json.loads(result) if isinstance(result, str) else result


def _write_file(view_path: Path, task_id: str,
                content: str = "plain text replacement") -> dict:
    return _dispatch("write_file", {"path": str(view_path), "content": content},
                     task_id)


def _patch_replace(view_path: Path, task_id: str) -> dict:
    return _dispatch("patch",
                     {"mode": "replace", "path": str(view_path),
                      "old_string": _KEEP, "new_string": _BROKEN},
                     task_id)


def _patch_v4a_update(view_path: Path, task_id: str) -> dict:
    patch = ("*** Begin Patch\n"
             f"*** Update File: {view_path}\n"
             "@@\n"
             f"-{_KEEP}\n"
             f"+{_BROKEN}\n"
             "*** End Patch")
    return _dispatch("patch", {"mode": "patch", "patch": patch}, task_id)


_OPERATIONS = {
    "write_file": _write_file,
    "patch_replace": _patch_replace,
    "patch_v4a_update": _patch_v4a_update,
}


class TestRemoteExistingBinaryRefused:
    """T1: an existing binary in the EXECUTION target is protected even though
    the controller's own filesystem says the path is free; a NEW one is not."""

    @pytest.mark.parametrize("ext", [".sqlite", ".pdf"])
    @pytest.mark.parametrize("op", sorted(_OPERATIONS))
    def test_existing_target_binary_refused_and_untouched(self, remote_target, op, ext):
        view_path = remote_target.view / f"data{ext}"
        target_path = remote_target.target / f"data{ext}"
        target_path.write_bytes(_binary_payload(ext))
        original = target_path.read_bytes()
        assert not view_path.exists(), "target must exist ONLY in the execution target"

        result = _OPERATIONS[op](view_path, remote_target.task_id)

        error = result.get("error") or ""
        assert "Refusing" in error, f"{op} must refuse a remote-only binary overwrite: {result}"
        if ext == ".pdf":
            assert "Refusing to overwrite existing PDF" in error, error
        else:
            assert "Refusing to overwrite existing binary file" in error, error
        assert target_path.read_bytes() == original, "target bytes must be untouched"
        assert not view_path.exists(), "the controller namespace must stay clean"

        # A proven-absent sibling is NOT refused: creating a NEW file with a
        # binary extension on the remote backend stays allowed.
        created = _write_file(remote_target.view / f"new{ext}", remote_target.task_id)
        assert not created.get("error"), f"creating a NEW remote {ext} must be allowed: {created}"
        assert (remote_target.target / f"new{ext}").read_text() == "plain text replacement"
        assert not (remote_target.view / f"new{ext}").exists()


def test_remote_probe_failure_fails_closed(remote_target):
    """An unreachable/erroring backend cannot prove absence: refuse, bytes intact."""
    target_path = remote_target.target / "data.sqlite"
    target_path.write_bytes(_binary_payload(".sqlite"))
    original = target_path.read_bytes()
    for failure_mode in ("raise", "error"):
        remote_target.env.failure_mode = failure_mode

        result = _write_file(remote_target.view / "data.sqlite", remote_target.task_id)

        error = result.get("error") or ""
        assert "Refusing" in error and "establish" in error, (failure_mode, result)
        assert target_path.read_bytes() == original
