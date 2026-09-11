"""Effect-aware self-repo mutation analysis for interpreter-backed terminal calls.

The existing Windows guard correctly blocks a literal ``git reset``/``checkout``
against the checkout backing the running Hermes process.  This module closes the
next execution shape without widening the terminal backend itself: inspect local
interpreter inputs before the canonical terminal handler runs, recover statically
visible child-process commands, and feed those effects through the existing
self-repo guard.

This remains a bounded pre-execution heuristic, not a sandbox.  It intentionally
does not execute user code or guess dynamic values.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tools.approval import (
    _bash_exec_payload,
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_starts,
    _read_shell_word,
)
from tools.self_repo_guard import detect_self_repo_git_mutation


_MAX_SCRIPT_BYTES = 512 * 1024
_MAX_EFFECT_DEPTH = 4
_MAX_SHELL_WORDS = 128
_PYTHON_EXE_RE = re.compile(
    r"^(?:pythonw?|pypy)(?:\d+(?:\.\d+)*)?$",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=(.*)", re.DOTALL)

_SUDO_OPTIONS_WITH_ARG = frozenset({
    "-C", "--chdir", "-c", "--close-from", "-g", "--group", "-h", "--host",
    "-p", "--prompt", "-R", "--chroot", "-T", "--command-timeout", "-u", "--user",
})
_ENV_OPTIONS_WITH_ARG = frozenset({
    "-a", "--argv0", "-C", "--chdir", "-S", "--split-string", "-u", "--unset",
})
_WRAPPER_OPTIONS_WITH_ARG = {
    "exec": frozenset({"-a"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}
_SIMPLE_WRAPPERS = frozenset({"builtin", "exec", "nohup", "setsid", "time"})
_SHELL_EXECUTABLES = frozenset({"bash", "dash", "ksh", "sh", "zsh"})


@dataclass(frozen=True)
class MutationEffect:
    """One statically recovered operation that would rewrite the live checkout."""

    operation: str
    message: str
    origin: str
    script_path: Path | None = None


def _resolve_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(os.path.expanduser(os.fspath(value)))
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return path


def _executable_name(value: str) -> str:
    return Path(value.replace("\\", "/")).name.removesuffix(".exe").lower()


def _shell_word_value(raw_word: str) -> str:
    """Decode one shell word without corrupting quoted interpreter payload data."""

    try:
        values = shlex.split(raw_word, posix=True)
    except ValueError:
        values = []
    if len(values) == 1:
        return values[0]
    # The existing detector deliberately recognizes a narrow class of
    # command-word obfuscations (for example literal command substitutions).
    # Keep that behavior only when ordinary shell decoding cannot produce one
    # semantic word; using it unconditionally destroys quotes *inside* data
    # arguments such as Python ``-c`` source.
    return _deobfuscate_shell_word_for_detection(raw_word)


def _shell_words_at(command: str, start: int) -> list[str]:
    words: list[str] = []
    cursor = start
    for _ in range(_MAX_SHELL_WORDS):
        word_start, word_end, raw_word = _read_shell_word(command, cursor)
        if word_start == word_end:
            break
        if words and "\n" in command[cursor:word_start]:
            break
        words.append(_shell_word_value(raw_word))
        cursor = word_end
    return words


def _consume_options(
    words: list[str],
    start: int,
    options_with_arg: frozenset[str],
) -> int:
    index = start
    while index < len(words):
        option = words[index]
        if option == "--":
            return index + 1
        if not option.startswith("-") or option == "-":
            break
        option_name = option.split("=", 1)[0]
        if "=" not in option and option_name in options_with_arg:
            index += 2
        else:
            index += 1
    return index


def _command_parts(words: list[str]) -> tuple[str | None, list[str]]:
    index = 0
    while index < len(words):
        if _ASSIGNMENT_RE.fullmatch(words[index]):
            index += 1
            continue
        executable = _executable_name(words[index])
        if executable == "sudo":
            index = _consume_options(words, index + 1, _SUDO_OPTIONS_WITH_ARG)
            continue
        if executable == "env":
            index = _consume_options(words, index + 1, _ENV_OPTIONS_WITH_ARG)
            continue
        if executable == "command":
            if index + 1 < len(words) and words[index + 1] in {"-v", "-V"}:
                return None, []
            index = _consume_options(words, index + 1, frozenset())
            continue
        if executable in _SIMPLE_WRAPPERS:
            index = _consume_options(
                words,
                index + 1,
                _WRAPPER_OPTIONS_WITH_ARG.get(executable, frozenset()),
            )
            continue
        return words[index], words[index + 1 :]
    return None, []


def _shell_script_arg(args: list[str]) -> str | None:
    has_c, payload = _bash_exec_payload(args)
    if has_c:
        return payload
    for index, arg in enumerate(args):
        if arg == "--":
            break
        if arg.startswith("-") and "c" in arg[1:]:
            return args[index + 1] if index + 1 < len(args) else None
        if not arg.startswith("-"):
            break
    return None


def _python_source_arg(args: list[str]) -> tuple[str, str | None]:
    """Return ``(\"code\"|\"path\"|\"none\", value)`` for Python/py launcher argv."""

    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return (
                ("path", args[index + 1])
                if index + 1 < len(args)
                else ("none", None)
            )
        if arg == "-":
            return "none", None
        if arg == "-c":
            return (
                ("code", args[index + 1])
                if index + 1 < len(args)
                else ("none", None)
            )
        if arg == "-m":
            return "none", None
        if arg in {"-W", "-X"}:
            index += 2
            continue
        if arg.startswith(("-W", "-X")) and len(arg) > 2:
            index += 1
            continue
        # Windows py launcher selectors: -3, -3.12, -V:Company/Tag.
        if re.fullmatch(r"-\d+(?:\.\d+)?(?:-\d+)?", arg) or arg.startswith("-V:"):
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return "path", arg
    return "none", None


def _unwrap_runner(executable: str, args: list[str]) -> tuple[str, list[str]]:
    """Unwrap common environment runners that preserve a nested command argv."""

    name = _executable_name(executable)
    if name == "uv" and args[:1] == ["run"]:
        index = 1
        options_with_arg = {
            "--directory",
            "--project",
            "--python",
            "--with",
            "--with-editable",
        }
        while index < len(args):
            arg = args[index]
            if arg == "--":
                index += 1
                break
            option = arg.split("=", 1)[0]
            if not arg.startswith("-"):
                break
            index += 2 if "=" not in arg and option in options_with_arg else 1
        if index < len(args):
            return args[index], args[index + 1 :]
    if name in {"poetry", "pipenv"} and args[:1] == ["run"] and len(args) > 1:
        return args[1], args[2:]
    return executable, args


class MutationEffectGuard:
    """Recover terminal-visible interpreter effects and reuse the live-repo guard."""

    def __init__(
        self,
        source_root: Path,
        *,
        max_script_bytes: int = _MAX_SCRIPT_BYTES,
        max_depth: int = _MAX_EFFECT_DEPTH,
        command_detector: Callable[
            [str, str | None, Path | None],
            tuple[bool, str | None],
        ] = detect_self_repo_git_mutation,
    ) -> None:
        self.source_root = _resolve_path(source_root, Path("/"))
        self.max_script_bytes = max(1, int(max_script_bytes))
        self.max_depth = max(1, int(max_depth))
        self._command_detector = command_detector

    def detect(
        self,
        command: str,
        cwd: str | os.PathLike[str] | None,
    ) -> MutationEffect | None:
        """Return the first live-checkout mutation effect, otherwise ``None``."""

        if not isinstance(command, str) or not command.strip():
            return None
        base = _resolve_path(cwd or os.getcwd(), Path("/"))
        return self._detect_command(command, base, depth=0, origin="terminal command")

    def _detect_command(
        self,
        command: str,
        cwd: Path,
        *,
        depth: int,
        origin: str,
    ) -> MutationEffect | None:
        if depth > self.max_depth:
            return MutationEffect(
                operation=f"interpreter effect depth >{self.max_depth}",
                message=(
                    "Blocked: interpreter indirection exceeded the self-repo mutation "
                    "analysis depth while Hermes is running. Run the operation from a "
                    "separate checkout, or stop Hermes and execute it externally."
                ),
                origin=origin,
            )

        hit, message = self._command_detector(command, str(cwd), self.source_root)
        if hit:
            operation = self._operation_from_message(message)
            return MutationEffect(
                operation=operation,
                message=message or self._fallback_message(operation),
                origin=origin,
            )

        current_cwd = cwd
        for start in sorted(set(_iter_shell_command_starts(command))):
            words = _shell_words_at(command, start)
            executable, args = _command_parts(words)
            if executable is None:
                continue

            name = _executable_name(executable)
            if name in {"cd", "pushd"}:
                target = next((arg for arg in args if not arg.startswith("-")), None)
                if target:
                    candidate = _resolve_path(target, current_cwd)
                    if candidate.is_dir():
                        current_cwd = candidate
                continue

            executable, args = _unwrap_runner(executable, args)
            name = _executable_name(executable)

            if _PYTHON_EXE_RE.fullmatch(name) or name == "py":
                effect = self._inspect_python_invocation(
                    args,
                    current_cwd,
                    depth=depth + 1,
                    origin=origin,
                )
                if effect:
                    return effect
                continue

            if name in _SHELL_EXECUTABLES:
                payload = _shell_script_arg(args)
                if payload:
                    effect = self._detect_command(
                        payload,
                        current_cwd,
                        depth=depth + 1,
                        origin=f"{origin} via {name} -c",
                    )
                    if effect:
                        return effect
                continue

            if name in {"cmd", "cmd.exe"}:
                for flag in ("/c", "/k"):
                    if flag in [arg.lower() for arg in args]:
                        index = [arg.lower() for arg in args].index(flag)
                        payload = " ".join(args[index + 1 :])
                        if payload:
                            effect = self._detect_command(
                                payload,
                                current_cwd,
                                depth=depth + 1,
                                origin=f"{origin} via cmd {flag}",
                            )
                            if effect:
                                return effect
                        break
                continue

            if name in {"powershell", "pwsh"}:
                lowered = [arg.lower() for arg in args]
                for flag in ("-command", "-c"):
                    if flag in lowered:
                        index = lowered.index(flag)
                        payload = " ".join(args[index + 1 :])
                        if payload:
                            effect = self._detect_command(
                                payload,
                                current_cwd,
                                depth=depth + 1,
                                origin=f"{origin} via {name} {flag}",
                            )
                            if effect:
                                return effect
                        break

        return None

    def _inspect_python_invocation(
        self,
        args: list[str],
        cwd: Path,
        *,
        depth: int,
        origin: str,
    ) -> MutationEffect | None:
        source_kind, value = _python_source_arg(args)
        if source_kind == "none" or value is None:
            return None
        if source_kind == "code":
            return self._inspect_python_source(
                value,
                cwd,
                script_path=None,
                depth=depth,
                origin=f"{origin} via Python -c",
            )

        script_path = _resolve_path(value, cwd)
        try:
            stat_result = script_path.stat()
        except OSError:
            # A missing/unreadable script cannot be executed successfully. Let
            # the terminal owner report its ordinary execution error.
            return None
        if not script_path.is_file():
            return MutationEffect(
                operation="unscannable interpreter input",
                message=self._unscannable_message(script_path, "is not a regular file"),
                origin=origin,
                script_path=script_path,
            )
        if stat_result.st_size > self.max_script_bytes:
            return MutationEffect(
                operation="oversized interpreter input",
                message=self._unscannable_message(
                    script_path,
                    f"exceeds the {self.max_script_bytes}-byte analysis limit",
                ),
                origin=origin,
                script_path=script_path,
            )
        try:
            source = script_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return MutationEffect(
                operation="unscannable interpreter input",
                message=self._unscannable_message(
                    script_path,
                    "could not be read as UTF-8",
                ),
                origin=origin,
                script_path=script_path,
            )
        return self._inspect_python_source(
            source,
            cwd,
            script_path=script_path,
            depth=depth,
            origin=f"{origin} via {script_path}",
        )

    def _inspect_python_source(
        self,
        source: str,
        cwd: Path,
        *,
        script_path: Path | None,
        depth: int,
        origin: str,
    ) -> MutationEffect | None:
        from tools.python_mutation_effects import scan_python_effect

        return scan_python_effect(
            source,
            cwd=cwd,
            script_path=script_path,
            depth=depth,
            origin=origin,
            detect_command=self._detect_command,
        )

    @staticmethod
    def _operation_from_message(message: str | None) -> str:
        if not message:
            return "self-repo mutation"
        match = re.search(r"Blocked: `([^`]+)`", message)
        return match.group(1) if match else "self-repo mutation"

    def _fallback_message(self, operation: str) -> str:
        return (
            f"Blocked: `{operation}` would rewrite Hermes's live source checkout "
            f"({self.source_root}). Interpreter or script indirection does not change "
            "that boundary. Use a separate checkout, or stop Hermes and run the "
            "operation externally."
        )

    def _unscannable_message(self, path: Path, detail: str) -> str:
        return (
            f"Blocked: interpreter input {path} {detail}, so Hermes cannot prove it "
            f"will leave the live source checkout ({self.source_root}) unchanged. "
            "Run it from a separate checkout, or stop Hermes and execute it externally."
        )
