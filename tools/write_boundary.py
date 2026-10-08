"""Profile-scoped write boundary for designated authority files.

Plugins register basenames and the exact writer for one Hermes home. Other
homes are unaffected. The command scanner is only a fast refusal. A name built
at runtime is stopped by ``guard_command``, which wraps the process that opens
the final path. A command is never trusted because its text mentions a writer.
"""

from __future__ import annotations

import shlex
from pathlib import Path


_BY_HOME: dict[str, set[str]] = {}
_OFFICIAL_PATHS: set[str] = set()
_OFFICIAL_FLAGS = ("--kind", "--run-dir", "--relative", "--input")
_SHELL_META = set("\n;&|<>`$()\\")


def register_protected_basenames(home: str, names) -> None:
    bucket = _BY_HOME.setdefault(str(Path(home).expanduser()), set())
    bucket.update(str(name) for name in names if name)


def protected_basenames(home: str | None = None) -> frozenset[str]:
    if home is None:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
    return frozenset(_BY_HOME.get(str(Path(home).expanduser()), ()))


def clear_protected_basenames(home: str | None = None) -> None:
    if home is None:
        _BY_HOME.clear()
        return
    _BY_HOME.pop(str(Path(home).expanduser()), None)


def register_official_writer(path: str) -> None:
    text = str(path or "")
    if text.startswith("/") and ".." not in Path(text).parts:
        _OFFICIAL_PATHS.add(text)


def clear_official_writers() -> None:
    _OFFICIAL_PATHS.clear()


def official_exec_argv(command: str) -> list[str] | None:
    """Exact interpreter argv of a registered writer. Extra shell text is not official.

    The four flags may appear in any order, each once, with a value that does
    not start a new flag. A relative script path, a copy of the writer, or a
    command joined with another statement returns None.
    """
    text = command or ""
    if not text or any(char in text for char in _SHELL_META):
        return None
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    if len(tokens) != 2 + len(_OFFICIAL_FLAGS) * 2:
        return None
    if Path(tokens[0]).name not in {"python", "python3"}:
        return None
    script = tokens[1]
    if script not in _OFFICIAL_PATHS:
        return None
    seen: dict[str, str] = {}
    pairs = tokens[2:]
    for index in range(0, len(pairs), 2):
        flag, value = pairs[index], pairs[index + 1]
        if flag not in _OFFICIAL_FLAGS or flag in seen or not value or value.startswith("-"):
            return None
        seen[flag] = value
    if set(seen) != set(_OFFICIAL_FLAGS):
        return None
    return tokens


def refuse_paths(paths, home: str | None = None) -> str | None:
    names = protected_basenames(home)
    for path in paths or []:
        base = Path(str(path or "")).name
        if base in names:
            return (
                f"Refusing to create, truncate, or patch {base}. "
                "Authority files are written only by authority_write.py."
            )
    return None


def official_writer_command(command: str) -> bool:
    """True only for an exact registered writer argv, never a surrounding command."""
    return official_exec_argv(command) is not None


def authority_write_block(tool_name: str, paths, home: str | None = None) -> str | None:
    """Block a file tool when the required plugin is down or the path is protected.

    Used by the file tools themselves so a skipped pre-tool hook cannot create
    an authority file. Profiles that do not declare ``plugins.required`` and
    have not registered basenames are unchanged.
    """
    try:
        from tools.required_plugin_gate import required_tool_block
        blocked = required_tool_block(tool_name)
    except Exception:
        blocked = f"Required plugin check failed; refusing {tool_name}."
    if blocked:
        return blocked
    return refuse_paths(paths, home)


def command_write_targets(command: str) -> list[str]:
    """Literal shell write destinations; runtime writes remain OS-guarded."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    targets, segment = [], []
    def finish(parts):
        if not parts:
            return
        cmd = Path(parts[0]).name
        args = parts[1:]
        operands, stop_options = [], False
        target_dir = None
        for i, arg in enumerate(args):
            if i and args[i - 1] in ("-t", "--target-directory"):
                target_dir = arg
                continue
            if arg.startswith("--target-directory="):
                target_dir = arg.split("=", 1)[1]
                continue
            if arg == "--":
                stop_options = True
            elif stop_options or not arg.startswith("-"):
                operands.append(arg)
        if cmd in {"cp", "mv", "install", "ln"} and len(operands) >= 2:
            if target_dir:
                targets.extend(str(Path(target_dir) / Path(src).name) for src in operands)
            else:
                targets.append(operands[-1])
        elif cmd in {"tee", "truncate", "rm", "unlink", "touch"}:
            targets.extend(operands)
        elif cmd == "sed" and any(arg == "--in-place" or arg.startswith("-i") or arg.startswith("--in-place=") for arg in args):
            targets.extend(operands[1:])
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {">", ">>", ">|", "&>", "&>>"} and i + 1 < len(tokens):
            targets.append(tokens[i + 1])
            i += 2
            continue
        if token in {";", "&&", "||", "|", "&"}:
            finish(segment)
            segment = []
        else:
            segment.append(token)
        i += 1
    finish(segment)
    return targets


def refuse_command(command: str, home: str | None = None) -> str | None:
    """Reject literal protected write destinations, allowing ordinary reads."""
    if not command or official_exec_argv(command) is not None:
        return None
    names = protected_basenames(home)
    for target in command_write_targets(command):
        base = Path(target).name
        if base in names:
            return (f"Refusing to create or truncate {base}. "
                    "Authority files are written only by authority_write.py.")
    return None


def guard_command(command: str, *, env_type: str, home: str | None = None) -> str | None:
    """Return the command to execute.

    Unchanged when this home has no protected names. An exact writer argv is
    replaced by a check that execs it only when that script refuses a write
    open. Every other command is wrapped. None means the command must not run.
    """
    names = protected_basenames(home)
    if not names:
        return command
    argv = official_exec_argv(command)
    if argv is not None:
        from tools.authority_os_guard import immutable_exec_command
        return immutable_exec_command(argv)
    from tools.authority_os_guard import wrap_shell
    return wrap_shell(command, names, env_type=env_type)


def prelude(names: frozenset[str]) -> str:
    literal = ", ".join(repr(name) for name in sorted(names))
    return (
        "import os as _hermes_os, sys as _hermes_sys\n"
        f"_HERMES_PROTECTED = frozenset({{{literal}}})\n"
        "def _hermes_writing(mode):\n"
        "    if isinstance(mode, int):\n"
        "        flags = _hermes_os.O_WRONLY | _hermes_os.O_RDWR | _hermes_os.O_APPEND | _hermes_os.O_TRUNC\n"
        "        return bool(mode & flags)\n"
        "    text = str(mode or '')\n"
        "    return any(flag in text for flag in ('w', 'a', 'x', '+'))\n"
        "def _hermes_blob(value):\n"
        "    if isinstance(value, (list, tuple)):\n"
        "        return '\\n'.join(_hermes_blob(item) for item in value)\n"
        "    if isinstance(value, bytes):\n"
        "        return value.decode('utf-8', 'replace')\n"
        "    if value is None:\n"
        "        return ''\n"
        "    return str(value)\n"
        "def _hermes_audit(event, args):\n"
        "    if event in ('open', 'os.open'):\n"
        "        path = args[0] if args else ''\n"
        "        mode = args[1] if len(args) > 1 else 'r'\n"
        "        name = _hermes_os.path.basename(str(path))\n"
        "        if name in _HERMES_PROTECTED and _hermes_writing(mode):\n"
        "            raise PermissionError('protected authority file: ' + name)\n"
        "        return\n"
        "    if event not in ('os.system', 'os.exec', 'os.posix_spawn', 'os.spawn', 'subprocess.Popen'):\n"
        "        return\n"
        "    blob = _hermes_blob(args)\n"
        "    for name in _HERMES_PROTECTED:\n"
        "        if name in blob:\n"
        "            raise PermissionError('protected authority file: ' + name)\n"
        "_hermes_sys.addaudithook(_hermes_audit)\n"
    )


def wrap_code(code: str, home: str | None = None) -> str:
    names = protected_basenames(home)
    if not names:
        return code
    return prelude(names) + "\n" + (code or "")
