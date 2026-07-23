"""Detect Git operations that can rewrite the checkout backing this process."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tools.approval import (
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_starts,
    _read_shell_word,
)


_WORKTREE_MUTATIONS = frozenset({
    "checkout",
    "switch",
    "rebase",
    "merge",
    "pull",
    "restore",
    "clean",
    "cherry-pick",
    "revert",
})
_STASH_SAFE_ACTIONS = frozenset({"list", "show", "create", "store", "drop", "clear"})
_RESET_WORKTREE_MODES = frozenset({"--hard", "--merge", "--keep"})
_KNOWN_GIT_BUILTINS = frozenset({
    "add",
    "am",
    "apply",
    "bisect",
    "blame",
    "branch",
    "bundle",
    "clone",
    "commit",
    "config",
    "describe",
    "diff",
    "fetch",
    "format-patch",
    "grep",
    "help",
    "init",
    "log",
    "maintenance",
    "mv",
    "notes",
    "push",
    "range-diff",
    "remote",
    "repack",
    "replace",
    "rev-list",
    "rev-parse",
    "rm",
    "show",
    "status",
    "submodule",
    "tag",
    "worktree",
})
_SHELL_EXECUTABLES = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=(.*)", re.DOTALL)
_SUDO_OPTIONS_WITH_ARG = frozenset({
    "-C",
    "--chdir",
    "-c",
    "--close-from",
    "-g",
    "--group",
    "-h",
    "--host",
    "-p",
    "--prompt",
    "-R",
    "--chroot",
    "-T",
    "--command-timeout",
    "-u",
    "--user",
})
_ENV_OPTIONS_WITH_ARG = frozenset({
    "-a",
    "--argv0",
    "-C",
    "--chdir",
    "-S",
    "--split-string",
    "-u",
    "--unset",
})
_WRAPPER_OPTIONS_WITH_ARG = {
    "exec": frozenset({"-a"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
}
_SIMPLE_WRAPPERS = frozenset({"builtin", "exec", "nohup", "setsid", "time"})
_MAX_RECURSION = 4


@dataclass
class _Heredoc:
    delimiter: str
    strip_tabs: bool
    execute_as_shell: bool
    body: list[str] = field(default_factory=list)


@dataclass
class _ShellContext:
    kind: str
    opener: int
    quote: str | None = None


def get_running_source_root() -> Path | None:
    """Return the source checkout backing this process, if there is one."""
    try:
        root = Path(__file__).resolve().parent.parent
    except (OSError, RuntimeError):
        return None
    return root if (root / ".git").exists() else None


def _resolve(path_str: str, base: Path) -> Path:
    path = Path(os.path.expanduser(path_str))
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path == root or path.is_relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False


def _executable_name(value: str) -> str:
    return Path(value.replace("\\", "/")).name.removesuffix(".exe").lower()


def _shell_words_at(command: str, start: int) -> list[str]:
    words: list[str] = []
    cursor = start
    for _ in range(64):
        word_start, word_end, raw_word = _read_shell_word(command, cursor)
        if word_start == word_end:
            break
        if words and "\n" in command[cursor:word_start]:
            break
        words.append(_deobfuscate_shell_word_for_detection(raw_word))
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


def _command_parts(words: list[str]) -> tuple[dict[str, str], str | None, list[str]]:
    env: dict[str, str] = {}
    index = 0

    while index < len(words):
        if _ASSIGNMENT_RE.fullmatch(words[index]):
            name, value = words[index].split("=", 1)
            env[name] = value
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
                return env, None, []
            index = _consume_options(words, index + 1, frozenset())
            continue
        if executable in _SIMPLE_WRAPPERS:
            index = _consume_options(
                words,
                index + 1,
                _WRAPPER_OPTIONS_WITH_ARG.get(executable, frozenset()),
            )
            continue
        return env, words[index], words[index + 1 :]

    return env, None, []


def _scope_keys(command: str, starts: list[int]) -> dict[int, tuple[int, ...]]:
    contexts = [_ShellContext("root", -1)]
    scopes: dict[int, tuple[int, ...]] = {}
    cursor = 0

    for start in sorted(set(starts)):
        while cursor < start:
            context = contexts[-1]
            quote = context.quote
            char = command[cursor]

            if quote == "'":
                if char == "'":
                    context.quote = None
                cursor += 1
                continue
            if quote == '"':
                if char == "\\" and cursor + 1 < start:
                    cursor += 2
                    continue
                if char == '"':
                    context.quote = None
                    cursor += 1
                    continue
                if command.startswith("$(", cursor):
                    contexts.append(_ShellContext("$(", cursor))
                    cursor += 2
                    continue
                if char == "`":
                    contexts.append(_ShellContext("`", cursor))
                cursor += 1
                continue

            if char in {"'", '"'}:
                context.quote = char
                cursor += 1
                continue
            if char == "\\" and cursor + 1 < start:
                cursor += 2
                continue
            if command.startswith("$(", cursor):
                contexts.append(_ShellContext("$(", cursor))
                cursor += 2
                continue
            if char == "(":
                contexts.append(_ShellContext("(", cursor))
                cursor += 1
                continue
            if char == ")" and len(contexts) > 1 and contexts[-1].kind in {"(", "$("}:
                contexts.pop()
                cursor += 1
                continue
            if char == "`":
                if len(contexts) > 1 and contexts[-1].kind == "`":
                    contexts.pop()
                else:
                    contexts.append(_ShellContext("`", cursor))
            cursor += 1

        scopes[start] = tuple(item.opener for item in contexts[1:])

    return scopes


def _operator_before(command: str, start: int) -> str | None:
    index = start - 1
    saw_newline = False
    while index >= 0 and command[index].isspace():
        saw_newline = saw_newline or command[index] == "\n"
        index -= 1
    if index < 0:
        return "\n" if saw_newline else None
    if index > 0 and command[index - 1 : index + 1] in {"&&", "||"}:
        return command[index - 1 : index + 1]
    if command[index] in {";", "|", "&", "(", "{"}:
        return command[index]
    return "\n" if saw_newline else None


def _cd_target(executable: str, args: list[str], cwd: Path) -> Path | None:
    if _executable_name(executable) not in {"cd", "pushd"}:
        return None
    index = _consume_options(args, 0, frozenset())
    if index >= len(args) or args[index] == "-":
        return None
    target = _resolve(args[index], cwd)
    return target if target.is_dir() else None


def _shell_script_arg(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "--":
            break
        if arg.startswith("-") and "c" in arg[1:]:
            return args[index + 1] if index + 1 < len(args) else None
        if not arg.startswith("-"):
            break
    return None


def _heredoc_specs(line: str) -> list[_Heredoc]:
    specs: list[_Heredoc] = []
    quote: str | None = None
    index = 0

    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and quote == '"' and index + 1 < len(line):
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if not line.startswith("<<", index) or line.startswith("<<<", index):
            index += 1
            continue

        operator_at = index
        index += 2
        strip_tabs = index < len(line) and line[index] == "-"
        if strip_tabs:
            index += 1
        while index < len(line) and line[index] in {" ", "\t"}:
            index += 1
        if index >= len(line):
            break

        delimiter_quote = line[index] if line[index] in {"'", '"'} else None
        if delimiter_quote:
            index += 1
            end = line.find(delimiter_quote, index)
            if end == -1:
                break
            delimiter = line[index:end]
            index = end + 1
        else:
            end = index
            while (
                end < len(line) and not line[end].isspace() and line[end] not in ";|&<>"
            ):
                end += 1
            delimiter = line[index:end]
            index = end
        if not delimiter:
            continue

        header = line[:operator_at]
        starts = list(_iter_shell_command_starts(header))
        words = _shell_words_at(header, starts[-1]) if starts else []
        _, executable, args = _command_parts(words)
        execute_as_shell = bool(
            executable
            and _executable_name(executable) in _SHELL_EXECUTABLES
            and _shell_script_arg(args) is None
            and not any(arg and not arg.startswith("-") for arg in args)
        )
        specs.append(_Heredoc(delimiter, strip_tabs, execute_as_shell))

    return specs


def _masked_line(line: str) -> str:
    return "".join(char if char in {"\r", "\n"} else " " for char in line)


def _mask_heredocs(command: str) -> tuple[str, list[str]]:
    output: list[str] = []
    pending: list[_Heredoc] = []
    shell_scripts: list[str] = []

    for line in command.splitlines(keepends=True):
        if pending:
            current = pending[0]
            candidate = line.rstrip("\r\n")
            if current.strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == current.delimiter:
                if current.execute_as_shell:
                    shell_scripts.append("".join(current.body))
                pending.pop(0)
            else:
                current.body.append(line)
            output.append(_masked_line(line))
            continue

        output.append(line)
        pending.extend(_heredoc_specs(line))

    for current in pending:
        if current.execute_as_shell:
            shell_scripts.append("".join(current.body))
    return "".join(output), shell_scripts


def _git_target_and_subcommand(
    args: list[str],
    current_dir: Path,
    env: dict[str, str],
) -> tuple[Path, str | None, list[str], dict[str, str]]:
    target = current_dir
    work_tree: str | None = None
    aliases: dict[str, str] = {}
    index = 0

    while index < len(args):
        arg = args[index]
        if arg == "--":
            index += 1
            break
        if arg == "-C" and index + 1 < len(args):
            target = _resolve(args[index + 1], target)
            index += 2
            continue
        if arg.startswith("-C") and len(arg) > 2:
            target = _resolve(arg[2:], target)
            index += 1
            continue
        if arg in {"--work-tree", "--git-dir", "--namespace", "--exec-path"}:
            if arg == "--work-tree" and index + 1 < len(args):
                work_tree = args[index + 1]
            index += 2
            continue
        if arg.startswith("--work-tree="):
            work_tree = arg.split("=", 1)[1]
            index += 1
            continue
        if arg == "-c" and index + 1 < len(args):
            config = args[index + 1]
            if config.lower().startswith("alias.") and "=" in config:
                key, value = config.split("=", 1)
                aliases[key[6:].lower()] = value
            index += 2
            continue
        if arg.startswith("-calias.") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            aliases[key[6:].lower()] = value
            index += 1
            continue
        if arg.startswith("-"):
            index += 1
            continue
        break

    explicit_work_tree = work_tree or env.get("GIT_WORK_TREE")
    if explicit_work_tree:
        target = _resolve(explicit_work_tree, target)
    subcommand = args[index].lower() if index < len(args) else None
    return target, subcommand, args[index + 1 :], aliases


def _mutates_worktree(subcommand: str, args: list[str]) -> bool:
    if subcommand == "reset":
        hard = re.compile(r"--h(?:a(?:r(?:d)?)?)?\Z")
        return any(arg in _RESET_WORKTREE_MODES or hard.fullmatch(arg) for arg in args)
    if subcommand == "stash":
        action = next((arg for arg in args if not arg.startswith("-")), "push")
        return action not in _STASH_SAFE_ACTIONS
    if subcommand == "clean":
        dry_run = any(
            arg == "--dry-run"
            or (arg.startswith("-") and not arg.startswith("--") and "n" in arg[1:])
            for arg in args
        )
        return not dry_run
    if subcommand == "restore":
        staged = any(
            arg == "--staged" or (arg.startswith("-") and "S" in arg[1:])
            for arg in args
        )
        worktree = any(
            arg == "--worktree" or (arg.startswith("-") and "W" in arg[1:])
            for arg in args
        )
        return worktree or not staged
    return subcommand in _WORKTREE_MUTATIONS


def _read_git_alias(executable: str, target: Path, alias: str) -> str | None:
    try:
        result = subprocess.run(
            [executable, "-C", str(target), "config", "--get", f"alias.{alias}"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _inspect_git(
    executable: str,
    args: list[str],
    current_dir: Path,
    env: dict[str, str],
    root: Path,
    depth: int,
) -> str | None:
    target, subcommand, sub_args, inline_aliases = _git_target_and_subcommand(
        args, current_dir, env
    )
    if subcommand is None or not _is_within(target, root):
        return None
    if _mutates_worktree(subcommand, sub_args):
        return f"git {subcommand}"
    if subcommand in _KNOWN_GIT_BUILTINS:
        return None
    if depth >= _MAX_RECURSION:
        return None

    alias = inline_aliases.get(subcommand)
    if alias is None:
        alias = _read_git_alias(executable, target, subcommand)
    if not alias:
        return None
    if alias.startswith("!"):
        return _find_mutation(alias[1:], target, root, depth + 1)
    try:
        alias_args = shlex.split(alias, posix=True)
    except ValueError:
        return None
    return _inspect_git(
        executable,
        [*alias_args, *sub_args],
        target,
        {},
        root,
        depth + 1,
    )


def _inspect_github_cli(
    executable: str,
    args: list[str],
    current_dir: Path,
    root: Path,
) -> str | None:
    if not _is_within(current_dir, root):
        return None
    name = _executable_name(executable)
    index = _consume_options(args, 0, frozenset({"-R", "--repo", "--hostname"}))
    if args[index : index + 2] == ["pr", "checkout"]:
        return f"{name} pr checkout"
    return None


def _find_mutation(command: str, cwd: Path, root: Path, depth: int = 0) -> str | None:
    if depth > _MAX_RECURSION:
        return None

    masked_command, heredoc_scripts = _mask_heredocs(command)
    for script in heredoc_scripts:
        operation = _find_mutation(script, cwd, root, depth + 1)
        if operation:
            return operation

    starts = sorted(set(_iter_shell_command_starts(masked_command)))
    scopes = _scope_keys(masked_command, starts)
    cwd_by_scope: dict[tuple[int, ...], Path] = {(): cwd}
    pending_cd: dict[tuple[int, ...], Path] = {}

    for start in starts:
        scope = scopes[start]
        if scope not in cwd_by_scope:
            cwd_by_scope[scope] = cwd_by_scope.get(scope[:-1], cwd)

        operator = _operator_before(masked_command, start)
        pending = pending_cd.pop(scope, None)
        if pending is not None and operator in {"&&", ";", "\n"}:
            cwd_by_scope[scope] = pending

        words = _shell_words_at(masked_command, start)
        env, executable, args = _command_parts(words)
        if executable is None:
            continue

        current_dir = cwd_by_scope[scope]
        cd_target = _cd_target(executable, args, current_dir)
        if cd_target is not None:
            pending_cd[scope] = cd_target
            continue

        executable_name = _executable_name(executable)
        if executable_name == "git":
            operation = _inspect_git(executable, args, current_dir, env, root, depth)
            if operation:
                return operation
        elif executable_name in {"gh", "hub"}:
            operation = _inspect_github_cli(executable, args, current_dir, root)
            if operation:
                return operation
        elif executable_name in _SHELL_EXECUTABLES:
            script = _shell_script_arg(args)
            if script:
                operation = _find_mutation(script, current_dir, root, depth + 1)
                if operation:
                    return operation

    return None


def detect_self_repo_git_mutation(
    command: str,
    cwd: str | None,
    source_root: Path | None = None,
) -> tuple[bool, str | None]:
    """Return whether a command would rewrite the live source checkout."""
    root = source_root if source_root is not None else get_running_source_root()
    if root is None or not command:
        return False, None

    root = _resolve(str(root), Path("/"))
    base = _resolve(cwd, Path("/")) if cwd else Path("/")
    operation = _find_mutation(command, base, root)
    if operation is None:
        return False, None
    return True, _block_message(operation, root)


def _block_message(operation: str, root: Path) -> str:
    return (
        f"Blocked: `{operation}` would rewrite Hermes's live source checkout "
        f"({root}) and can mix module versions in this running process. "
        "Use a separate worktree or temporary clone. To change this checkout, "
        "stop Hermes, run the command externally, then restart Hermes."
    )
