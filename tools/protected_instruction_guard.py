"""Protected agent-instruction files: the write guard for execute_code's child process.

``AGENTS.md`` / ``CLAUDE.md`` / ``SOUL.md`` / ``.cursorrules`` and project-local ``.hermes``
config steer future agent behavior, so ``write_file`` / ``patch`` route them through an
always-ask, no-yolo approval gate (``tools/file_tools_write_guards.py``). That gate lives on
the file tools only: ``execute_code`` runs arbitrary Python in a child process, where a plain
``open(path, "w")`` rewrites the same files with no prompt at all (#143266).

This module closes that route at the syscall level rather than by guessing at source text.
The child installs a ``sys.addaudithook`` guard, so ``open``, ``os.open``, ``os.replace``,
``pathlib.Path.write_text``, ``shutil.copyfile`` and every other in-process spelling reduce to
the single audit event the hook sees. Shell escapes (``subprocess`` / ``os.system``) are
matched lexically, because a spawned process's writes never cross this process's hooks.

Contract:

* ``guard_spec`` resolves config + the ``~/.hermes`` exemptions on the host and hands them to
  the child as JSON in ``SPEC_ENV``. The child never imports Hermes, so the guard works under
  a project venv interpreter.
* ``CHILD_GUARD_SOURCE`` ships as ``CHILD_GUARD_FILENAME`` next to the generated tool stubs
  and is imported before the first cell runs.
* Fail closed both ways: ``SPEC_ENV`` is ALWAYS set by the host (to ``"off"`` when the gate is
  disabled), so a child that finds it missing was not spawned by a guarded host and refuses to
  run; a spec that is present but unreadable also aborts the child. An unguarded
  execute_code is the hole itself.
* Deny, don't prompt. execute_code has no per-operation approval channel at syscall time, so
  the refusal names the sanctioned route — the ``write_file`` / ``patch`` tools, which prompt.
* Cost: the per-open check is a basename compare (microseconds); ``realpath`` runs only on a
  name hit. Ordinary file I/O in a cell pays a few percent, a protected write pays one
  resolution and is refused.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re

# The child-side module, shipped verbatim into every sandbox staging dir.
CHILD_GUARD_FILENAME = "hermes_protected_instruction_guard.py"
SPEC_ENV = "HERMES_PROTECTED_INSTRUCTION_SPEC"

# Kept in sync with tools/file_tools_write_guards.py by
# tests/tools/test_protected_instruction_execute_code.py.
BASENAMES = ("agents.md", "claude.md", "soul.md", ".cursorrules")

# Shell writes into a protected basename. Two-part pre-filter, then the token scan does the
# real work: (1) the command contains a write SHAPE at all, (2) some token names a protected
# file. Keeping the shape match loose (any redirect/`sed -i`/`tee`/`cp`/`mv`/`install`) is
# deliberate — quoting, `$VAR`, Windows backslashes and drive letters make a tight
# path-glued regex miss real writes, and the basename check is what decides.
SHELL_WRITE_SHAPE = (
    r"(?:>>?|\bsed\b[^\n]*\s(?:-[A-Za-z]*i[A-Za-z]*|--in-place)\b|\btee\b"
    r"|\b(?:cp|mv|install)\b|\btruncate\b|\bdd\b[^\n]*\bof=|\bpython[0-9.]*\b[^\n]*-c\b)"
)


def _alternation(names) -> str:
    """Regex alternation over protected names, longest first so ``.cursorrules`` wins ties."""
    escaped = [re.escape(str(n)) for n in names if n]
    return "(?:" + "|".join(sorted(dict.fromkeys(escaped), key=len, reverse=True)) + ")"


def guard_spec(task_id: str = "default") -> dict:
    """The JSON-able guard spec for one sandbox child.

    ``enabled`` follows ``security.protected_instruction_files``; a config read failure keeps
    the gate ON, matching the file-tool gate. ``exempt`` is the Hermes home tree, where these
    names are the agent's own store governed by their own guards.
    """
    from tools.file_tools_write_guards import (
        _hermes_exempt_homes,
        _protected_instruction_config,
    )

    enabled, extra = _protected_instruction_config()
    try:
        exempt = [h for h in _hermes_exempt_homes() if h]
    except Exception:
        # An unresolvable home must not widen the gate over the agent's own store; an empty
        # exemption list only makes the guard stricter.
        exempt = []
    return {
        "enabled": bool(enabled),
        "basenames": list(BASENAMES),
        "extra": [str(p) for p in extra if p],
        "exempt": exempt,
        "task_id": task_id or "default",
    }


def guard_spec_json(task_id: str = "default") -> str:
    """``guard_spec`` serialized for ``SPEC_ENV``.

    Always non-empty: ``"off"`` is the explicit gate-disabled marker, so the child can tell
    "the operator turned the gate off" from "no guarded host spawned me" and fail closed on the
    latter.
    """
    spec = guard_spec(task_id)
    if not spec["enabled"]:
        return "off"
    return json.dumps(spec, ensure_ascii=False)


def child_guard_source(spec: dict | None = None) -> str:
    """Source of the child-side guard module (``CHILD_GUARD_FILENAME``).

    The basenames and extra patterns ride in ``SPEC_ENV`` at run time, so one generated file
    serves every sandbox; *spec* is accepted for symmetry with ``guard_spec_json`` and future
    name-baking.
    """
    return (
        CHILD_GUARD_SOURCE.replace("__SPEC_ENV__", repr(SPEC_ENV))
        .replace("__BASENAMES__", repr(tuple(BASENAMES)))
        .replace("__SHELL_WRITE_RE__", repr(SHELL_WRITE_SHAPE))
    )


# --- terminal side -------------------------------------------------------------------------
# `terminal` writes the same files through the shell, where no audit hook of ours is running.
# The check is lexical (shell is Turing-complete) but it covers the mechanical spellings, and
# it routes through the SAME one-operation approval gate the file tools use, so an approved
# shell write looks and behaves exactly like an approved `write_file`.

_TERMINAL_GATED_ENV_TYPES = frozenset({"local", "docker", "podman"})


def terminal_write_targets(
    command: str,
    cwd: str | None = None,
    *,
    env_type: str = "local",
    workdir: str | None = None,
) -> list[str]:
    """Protected instruction paths a terminal *command* appears to write.

    Empty for a remote backend (ssh/modal/...): the path it writes lives on that host's
    filesystem, not this one, so resolving it here would compare against the wrong tree.
    """
    if env_type not in _TERMINAL_GATED_ENV_TYPES:
        return []
    spec = guard_spec()
    if not spec.get("enabled"):
        return []
    if not re.search(SHELL_WRITE_SHAPE, command or "", re.IGNORECASE):
        return []
    base_dir = workdir or cwd or os.getcwd()
    targets: list[str] = []
    for token in _command_path_tokens(command):
        base = os.path.basename(token.rstrip("/\\")).lower()
        if not _basename_matches(base, spec):
            continue
        path = os.path.normpath(
            os.path.expanduser(
                token if os.path.isabs(token) else os.path.join(base_dir, token)
            )
        )
        if _is_exempt(path, spec) or path in targets:
            continue
        targets.append(path)
    return targets


def terminal_write_block(
    command: str,
    cwd: str | None = None,
    *,
    env_type: str = "local",
    workdir: str | None = None,
    task_id: str = "default",
) -> str | None:
    """Approval-gate a terminal command that writes a protected instruction file.

    ``None`` means "proceed" (no protected target, or the human approved). A returned string is
    the tool's blocked-result body. Deliberately independent of ``--yolo`` / ``force``: the
    file-tool gate is, and a shell command that skips it would be the same hole one level up.
    """
    targets = terminal_write_targets(command, cwd, env_type=env_type, workdir=workdir)
    if not targets:
        return None
    from tools.file_tools_write_guards import _request_protected_instruction_approval

    return _request_protected_instruction_approval(targets, task_id)


def _command_path_tokens(command: str) -> list[str]:
    """Word-split *command* into candidate paths (shell-aware, falling back to whitespace)."""
    import shlex

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = (command or "").split()
    out = []
    for token in tokens:
        # `>file`, `>>file` and `2>file` keep the redirect glued to the path; strip it so the
        # basename compare sees the path itself.
        token = token.lstrip("<>|&")
        if token.startswith("2>"):
            token = token[2:].lstrip("<>|&")
        if token and not token.startswith("-"):
            out.append(token)
    return out


def _basename_matches(base: str, spec: dict) -> bool:
    if not base:
        return False
    if base in (spec.get("basenames") or BASENAMES):
        return True
    return any(fnmatch.fnmatch(base, str(p).lower()) for p in (spec.get("extra") or []))


def _is_exempt(path: str, spec: dict) -> bool:
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        real = path
    for home in spec.get("exempt") or ():
        if real == home or real.startswith(home + os.sep):
            return True
    return False


CHILD_GUARD_SOURCE = '''\
"""Protected agent-instruction write guard for a Hermes execute_code child.

Installed before the first cell runs. Writes to AGENTS.md / CLAUDE.md / SOUL.md /
.cursorrules (and project-local .hermes config) are refused here: this process has no
per-operation approval channel, and the sanctioned route is the write_file / patch tool,
which prompts the human. Reads are untouched.

Generated by Hermes; overwritten on every sandbox start. Do not edit in place.
"""
import fnmatch
import json
import os
import re
import sys
import threading

_SPEC_ENV = __SPEC_ENV__
_BASENAMES = __BASENAMES__
_SHELL_WRITE = re.compile(__SHELL_WRITE_RE__, re.IGNORECASE)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_STATE = threading.local()


def _load_spec():
    """Parse the host-provided spec. Fails CLOSED: a missing or unusable spec raises."""
    raw = os.environ.get(_SPEC_ENV)
    if raw is None or raw == "":
        raise RuntimeError(
            "Hermes protected-instruction guard: no guard spec in the environment, so this "
            "execute_code child was not spawned by a guarded host; refusing to run. Start a "
            "new session, or report this as a Hermes bug.")
    if raw == "off":
        return None
    try:
        spec = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(
            "Hermes protected-instruction guard: unreadable guard spec; refusing to run "
            "unguarded execute_code code") from exc
    if not isinstance(spec, dict):
        raise RuntimeError(
            "Hermes protected-instruction guard: malformed guard spec; refusing to run "
            "unguarded execute_code code")
    if not spec.get("enabled"):
        return None
    spec.setdefault("basenames", list(_BASENAMES))
    spec.setdefault("extra", [])
    spec.setdefault("exempt", [])
    return spec


def _matches(path, spec):
    """The cheap test: does this path name a protected instruction file? Returns a label."""
    try:
        text = os.fspath(path)
    except TypeError:
        return ""
    if isinstance(text, bytes):
        try:
            text = os.fsdecode(text)
        except (UnicodeDecodeError, ValueError):
            return ""
    if not isinstance(text, str) or not text:
        return ""
    base = os.path.basename(text.rstrip("/\\\\")).lower()
    if not base:
        return ""
    if base in spec["basenames"]:
        return base
    for pattern in spec["extra"]:
        if fnmatch.fnmatch(base, str(pattern).lower()):
            return base
    # Project-local .hermes config dir: only the IMMEDIATE parent counts (same rule as the
    # file-tool gate), so a checkout living under ~/.hermes is not swept in wholesale.
    parts = text.replace("\\\\", "/").rstrip("/").split("/")
    if len(parts) >= 2 and parts[-2] == ".hermes":
        return text
    return ""


def _exempt(path, spec):
    """True when *path* lives in the Hermes home tree, which has its own guards."""
    if not spec["exempt"]:
        return False
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for home in spec["exempt"]:
        if real == home or real.startswith(home + os.sep):
            return True
    return False


def _refuse(label, path, what):
    raise PermissionError(
        "BLOCKED: %s to protected agent-instruction file %r (%s) from execute_code. "
        "These files steer future agent behavior, so every write needs one-operation human "
        "approval, and execute_code has no approval channel. Call hermes_tools.write_file or "
        "hermes_tools.patch instead: those route through the approval prompt. Do not retry "
        "this write through another spelling, a subprocess, or a script."
        % (what, str(path), label))


def _guard(path, spec, what):
    """Refuse when *path* is a protected instruction file; re-entrancy safe."""
    if getattr(_STATE, "busy", False):
        return
    label = _matches(path, spec)
    if not label:
        return
    _STATE.busy = True
    try:
        if _exempt(path, spec):
            return
        _refuse(label, path, what)
    finally:
        _STATE.busy = False


def _is_write_open(mode, flags):
    if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


def _guard_command(command, spec):
    """A spawned process's writes never reach this process's hook, so the shell text is
    matched lexically. Best-effort by construction: it stops the mechanical spellings."""
    if isinstance(command, (list, tuple)):
        command = " ".join(str(part) for part in command)
    if not isinstance(command, str) or not command:
        return
    if not _SHELL_WRITE.search(command):
        return
    cwd = os.getcwd()
    for token in _shell_tokens(command):
        label = _matches(token, spec)
        if not label:
            continue
        path = token if os.path.isabs(token) else os.path.join(cwd, token)
        if not _exempt(path, spec):
            _refuse(label, path, "spawned shell write")


def _shell_tokens(command):
    """Split a shell command into candidate path words.

    Two passes: ``shlex`` unwraps quoting, then every part is split on whitespace again. The
    second pass matters because a wrapper spelling puts the real command inside quotes —
    ``cmd.exe /c "echo x > AGENTS.md"`` and ``sh -c '... > AGENTS.md'`` are one shlex token
    otherwise, and the protected name hides inside it. Redirect operators are stripped off the
    glued word (``>AGENTS.md``, ``2>AGENTS.md``); ``$VAR``/``%VAR%`` spellings are skipped
    rather than guessed at.
    """
    import shlex
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        parts = command.split()
    out = []
    for part in parts:
        for word in part.split():
            word = word.lstrip("<>|&")
            if word.startswith("2>"):
                word = word[2:].lstrip("<>|&")
            word = word.strip("\\\"'")
            if not word or word.startswith("-") or "$" in word or "%" in word:
                continue
            out.append(word)
    return out


def _hook(event, args, _spec):
    if event == "open":
        if len(args) >= 3 and _is_write_open(args[1], args[2]):
            _guard(args[0], _spec, "write")
        return
    if event in ("os.rename", "os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.chmod",
                 "os.chown", "os.utime", "os.truncate", "os.link", "os.symlink",
                 "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.move",
                 "shutil.chown", "shutil.rmtree"):
        for path in args[:2]:
            if isinstance(path, (str, bytes, os.PathLike)):
                _guard(path, _spec, event)
        return
    if event == "subprocess.Popen":
        _guard_command(args[1], _spec)
        return
    if event in ("os.system", "os.exec", "os.posix_spawn", "os.spawn"):
        _guard_command(args[0], _spec)


def install():
    """Install the audit hook. Returns False when the gate is disabled."""
    spec = _load_spec()
    if spec is None:
        return False
    # The spec is bound as a default argument so cell code that rebinds this module's globals
    # cannot retune the gate. (An adversary already inside execute_code is out of scope here —
    # see SECURITY.md §2.2: the OS, not an in-process heuristic, is the boundary.)
    sys.addaudithook(lambda event, args: _hook(event, args, spec))
    return True


install()
'''
