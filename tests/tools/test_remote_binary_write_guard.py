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
from tools.environments.local import LocalEnvironment
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
        self._sub_pairs = []
        for src, dst in ((str(view_dir), str(target_dir)),):
            self._sub_pairs.append((src, dst))
            self._sub_pairs.append((src.replace("\\", "/"), dst.replace("\\", "/")))
            self._sub_pairs.append((_bash_safe(src), _bash_safe(dst)))

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


def _bash_safe(path: str) -> str:
    from tools.environments.local import _bash_safe_path
    return _bash_safe_path(path)


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
    the controller's own filesystem says the path is free."""

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


class TestTriStateContract:
    """T2: absent -> creation allowed; probe failure -> fail closed; locality
    comes from the LIVE file-ops environment, never name-hint tables."""

    @pytest.mark.parametrize("ext", [".sqlite", ".pdf"])
    def test_absent_on_target_creation_allowed(self, remote_target, ext):
        view_path = remote_target.view / f"new{ext}"
        content = "%PDF-1.4\n%%EOF\n" if ext == ".pdf" else "new db text fixture\n"

        result = _write_file(view_path, remote_target.task_id, content)

        assert not result.get("error"), f"creation must stay allowed: {result}"
        written = (remote_target.target / f"new{ext}").read_bytes()
        assert written == content.encode(), "the real write must land in the execution target"
        assert not view_path.exists()

    def test_absent_on_target_v4a_add_allowed(self, remote_target):
        view_path = remote_target.view / "fresh.pdf"
        patch = ("*** Begin Patch\n"
                 f"*** Add File: {view_path}\n"
                 "+%PDF-1.4\n"
                 "+%%EOF\n"
                 "*** End Patch")

        result = _dispatch("patch", {"mode": "patch", "patch": patch}, remote_target.task_id)

        assert not result.get("error"), f"V4A Add of a new .pdf must stay allowed: {result}"
        # V4A Add joins the '+' lines with '\n' (no trailing newline).
        assert (remote_target.target / "fresh.pdf").read_bytes() == b"%PDF-1.4\n%%EOF"

    @pytest.mark.parametrize("failure_mode", ["raise", "error"])
    def test_probe_transport_failure_fails_closed(self, remote_target, failure_mode):
        view_path = remote_target.view / "data.sqlite"
        target_path = remote_target.target / "data.sqlite"
        target_path.write_bytes(_binary_payload(".sqlite"))
        original = target_path.read_bytes()
        remote_target.env.failure_mode = failure_mode

        result = _write_file(view_path, remote_target.task_id)

        error = result.get("error") or ""
        assert "Refusing" in error, f"an unstat-able target must fail closed: {result}"
        assert "establish" in error and "retry" in error.lower(), error
        assert target_path.read_bytes() == original, "target bytes must be untouched"

    def test_locality_authority_is_the_live_file_ops_env(self, remote_target):
        """LOCALITY AUTHORITY pin: the backend is non-local and its env_type tag
        ('vercel_sandbox') is unclassified by every name-hint table while scoped
        config reports 'local'. A guard that classifies locality from env_type
        strings, config or class-name hints probes the WRONG filesystem and
        destroys the remote-only binary — only the live file-ops environment
        object may decide."""
        view_path = remote_target.view / "data.sqlite"
        target_path = remote_target.target / "data.sqlite"
        target_path.write_bytes(_binary_payload(".sqlite"))
        original = target_path.read_bytes()
        assert remote_target.env.env_type == "vercel_sandbox"
        assert not view_path.exists()

        result = _write_file(view_path, remote_target.task_id)

        error = result.get("error") or ""
        assert "Refusing" in error, f"guard must probe the execution target: {result}"
        assert target_path.read_bytes() == original, "target bytes must be untouched"


class TestResolutionFailureFallbackParity:
    """T3: when task resolution fails, the guard must probe the EXACT string the
    write would land on — ``write_file(_resolved or path)`` plus the file-ops
    layer's own ``_expand_path``, i.e. the BACKEND's ``$HOME`` for a tilde path,
    never the host's (the same wrong-filesystem bug class as #122662)."""

    def test_fallback_probe_expands_tilde_on_the_backend(self, remote_target, monkeypatch):
        import tools.file_tools_write_guards as write_guards
        from tools.file_tools_paths import _expand_tilde

        name = f"fallback-{uuid.uuid4().hex}.sqlite"
        raw_path = f"~/{name}"
        target_path = remote_target.target / name
        target_path.write_bytes(_binary_payload(".sqlite"))
        original = target_path.read_bytes()
        # The divergence under test: the HOST tilde expansion is free while the
        # write's fallback lands in the BACKEND home, where the binary lives.
        assert not Path(_expand_tilde(raw_path)).exists()
        assert str(remote_target.target) not in _expand_tilde(raw_path)

        def _resolver_down(*args, **kwargs):
            raise OSError("forced resolution failure")

        # Both import bindings of the SAME resolver: the guard helper's and
        # write_file's ``_resolve_or_none`` — in production they fail together,
        # so both paths take their documented raw-string fallback.
        monkeypatch.setattr(write_guards, "_resolve_path_for_task", _resolver_down)
        monkeypatch.setattr(file_tools_mod, "_resolve_path_for_task", _resolver_down)

        result = _dispatch("write_file", {"path": raw_path, "content": "plain text replacement"},
                           remote_target.task_id)

        error = result.get("error") or ""
        assert "Refusing to overwrite existing binary file" in error, (
            f"fallback probe must hit the backend home and find the binary: {result}")
        assert target_path.read_bytes() == original, "target bytes must be untouched"
        assert not Path(_expand_tilde(raw_path)).exists(), "the host home must stay clean"
