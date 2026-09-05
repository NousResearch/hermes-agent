"""Detect Git operations that can rewrite the checkout backing this process."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tools.approval_detection import (
    _bash_exec_payload, _deobfuscate_shell_word_for_detection, _iter_shell_command_starts,
    _read_shell_word)

# bisect drives repeated checkouts of the running root — the exact skew hazard guarded here.
_WORKTREE_MUTATIONS = frozenset({
    "checkout", "switch", "rebase", "merge", "pull", "restore", "clean", "cherry-pick", "revert",
    "bisect"})
_WORKTREE_TARGET_ACTIONS = frozenset({"move", "remove"})
_STASH_SAFE_ACTIONS = frozenset({"list", "show", "create", "store", "drop", "clear"})
_RESET_WORKTREE_MODES = frozenset({"--hard", "--merge", "--keep"})
# `reset`/`stash`/`clean`/`restore` reach this set only in their SAFE forms (_mutates_worktree
# runs first); listing them skips a pointless `git config --get alias.<sub>` subprocess.
_KNOWN_GIT_BUILTINS = frozenset({
    "add", "am", "apply", "blame", "branch", "bundle", "cat-file", "clean", "clone", "commit",
    "config", "describe", "diff", "fetch", "format-patch", "grep", "help", "init", "log",
    "ls-files", "ls-remote", "ls-tree", "maintenance", "merge-base", "mv", "notes", "push",
    "range-diff", "reflog", "remote", "repack", "replace", "reset", "restore", "rev-list",
    "rev-parse", "rm", "shortlog", "show", "show-ref", "stash", "status", "submodule", "tag",
    "worktree"})
_SHELL_EXECUTABLES = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=(.*)", re.DOTALL)
_RESET_HARD_RE = re.compile(r"--h(?:a(?:r(?:d)?)?)?\Z")
# ``<<-`` opener + optional blanks + a quoted delimiter (closing quote required) or a bare word.
_HEREDOC_OPENER_RE = re.compile(
    r"<<(?P<dash>-?)[ \t]*(?:(?P<q>['\"])(?P<quoted>.*?)(?P=q)|(?!['\"])(?P<bare>[^\s;|&<>]*))")
_NO_OPTIONS: frozenset[str] = frozenset()
# Wrapper executables skipped to reach the real command -> options that consume an argument.
_WRAPPER_OPTIONS_WITH_ARG: dict[str, frozenset[str]] = {
    "sudo": frozenset({
        "-C", "--chdir", "-c", "--close-from", "-g", "--group", "-h", "--host",
        "-p", "--prompt", "-R", "--chroot", "-T", "--command-timeout", "-u", "--user"}),
    "env": frozenset({"-a", "--argv0", "-C", "--chdir", "-S", "--split-string", "-u", "--unset"}),
    "command": _NO_OPTIONS, "builtin": _NO_OPTIONS, "nohup": _NO_OPTIONS, "setsid": _NO_OPTIONS,
    "exec": frozenset({"-a"}),
    "time": frozenset({"-f", "--format", "-o", "--output"})}
_MAX_RECURSION = 4
# git global options that consume the next argument (-C/--work-tree/-c are acted on).
_GIT_GLOBAL_OPTIONS_WITH_ARG = frozenset(
    {"-C", "-c", "--work-tree", "--git-dir", "--namespace", "--exec-path"})


@dataclass
class _Heredoc:
    delimiter: str
    strip_tabs: bool
    execute_as_shell: bool
    body: list[str] = field(default_factory=list)


@dataclass
class _ShellContext:  # one `(` / `$(` / backtick nesting level and its live quote state
    kind: str
    opener: int
    quote: str | None = None


def get_running_source_root() -> Path | None:
    """The source checkout backing this process, if there is one."""
    try:
        root = Path(__file__).resolve().parent.parent
    except (OSError, RuntimeError):
        return None
    return root if (root / ".git").exists() else None


def _resolve(path_str: str, base: Path) -> Path:
    path = base / Path(os.path.expanduser(path_str))  # ``/`` keeps an absolute right operand
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        return path.resolve()
    return path


def _is_within(path: Path, root: Path) -> bool:
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        return path == root or path.is_relative_to(root)
    return False


def _executable_name(value: str) -> str:
    return Path(value.replace("\\", "/")).name.removesuffix(".exe").lower()


def _shell_words_at(command: str, start: int) -> list[str]:
    """Deobfuscated words of the simple command at ``start`` (stops at a newline; max 64)."""
    words: list[str] = []
    cursor = start
    for _ in range(64):
        word_start, word_end, raw_word = _read_shell_word(command, cursor)
        if word_start == word_end or (words and "\n" in command[cursor:word_start]):
            break
        words.append(_deobfuscate_shell_word_for_detection(raw_word))
        cursor = word_end
    return words


def _consume_options(
    words: list[str], start: int, options_with_arg: frozenset[str] = _NO_OPTIONS) -> int:
    """Index of the first positional at/after ``start`` (``--`` ends options)."""
    index = start
    while index < len(words) and words[index].startswith("-") and words[index] != "-":
        if words[index] == "--":
            return index + 1
        index += 2 if "=" not in words[index] and words[index] in options_with_arg else 1
    return index


def _command_parts(words: list[str]) -> tuple[dict[str, str], str | None, list[str]]:
    """Split leading VAR=value assignments and wrappers off -> (env, executable, args)."""
    env: dict[str, str] = {}
    index = 0
    while index < len(words):
        if _ASSIGNMENT_RE.fullmatch(words[index]):
            name, value = words[index].split("=", 1)
            env[name] = value
            index += 1
            continue
        executable = _executable_name(words[index])
        wrapper_options = _WRAPPER_OPTIONS_WITH_ARG.get(executable)
        if wrapper_options is None:
            return env, words[index], words[index + 1 :]
        if executable == "command" and words[index + 1 : index + 2] in (["-v"], ["-V"]):
            break  # `command -v/-V` only reports; nothing runs
        index = _consume_options(words, index + 1, wrapper_options)
    return env, None, []


def _scope_keys(command: str, starts: list[int]) -> dict[int, tuple[int, ...]]:
    """Map each command start to the tuple of enclosing ``(``/``$(``/backtick openers."""
    contexts = [_ShellContext("root", -1)]
    scopes: dict[int, tuple[int, ...]] = {}
    cursor = 0
    for start in sorted(set(starts)):
        while cursor < start:
            context = contexts[-1]
            quote = context.quote
            char = command[cursor]
            closes = quote is None and len(contexts) > 1  # an unquoted closer may pop a scope
            if quote is not None and char == quote:
                context.quote = None
            elif quote == "'":
                pass  # single quotes: no escapes, no substitutions
            elif char == "\\" and cursor + 1 < start:
                cursor += 1
            elif quote is None and char in "'\"":
                context.quote = char
            # Unquoted or inside double quotes: substitutions still open scopes.
            elif command.startswith("$(", cursor):
                contexts.append(_ShellContext("$(", cursor))
                cursor += 1
            elif quote is None and char == "(":
                contexts.append(_ShellContext("(", cursor))
            elif (char == ")" and closes and context.kind in {"(", "$("}) or (
                    char == "`" and closes and context.kind == "`"):
                contexts.pop()
            elif char == "`":
                contexts.append(_ShellContext("`", cursor))
            cursor += 1
        scopes[start] = tuple(item.opener for item in contexts[1:])
    return scopes


def _operator_before(command: str, start: int) -> str | None:
    """The list/grouping operator (or newline) immediately preceding a command start."""
    head = command[:start].rstrip()
    for tail in (head[-2:], head[-1:]):
        if tail in {"&&", "||", ";", "|", "&", "(", "{"}:
            return tail
    return "\n" if "\n" in command[len(head):start] else None


def _cd_target(executable: str, args: list[str], cwd: Path) -> Path | None:
    """Directory a ``cd``/``pushd`` would land in (existing dirs only), else None."""
    if _executable_name(executable) not in {"cd", "pushd"}:
        return None
    index = _consume_options(args, 0)
    if index >= len(args) or args[index] == "-":
        return None
    target = _resolve(args[index], cwd)
    return target if target.is_dir() else None


def _shell_script_arg(args: list[str]) -> str | None:
    """Script string owned by a shell's ``-c``, if present. ``_bash_exec_payload`` parses bash's
    real option grammar (``-o pipefail -c '<script>'``); when it finds no ``-c``, fall back to a
    permissive scan: zsh/dash/ksh letters (``zsh -yc``) would otherwise fail this guard open."""
    has_c, payload = _bash_exec_payload(args)
    if has_c:
        return payload
    for index, arg in enumerate(args):
        if arg == "--" or not arg.startswith("-"):
            return None
        if "c" in arg[1:]:
            return args[index + 1] if index + 1 < len(args) else None
    return None


def _heredoc_specs(line: str) -> list[_Heredoc]:
    """Heredoc openers on one line; ``execute_as_shell`` when a bare shell consumes the body."""
    specs: list[_Heredoc] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and quote == '"' and index + 1 < len(line):
                index += 1  # skip the escaped character too
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        if quote or not line.startswith("<<", index) or line.startswith("<<<", index):
            index += 1
            continue
        opener = _HEREDOC_OPENER_RE.match(line, index)
        if opener is None:  # unterminated quoted delimiter: give up on this line
            break
        header, index = line[:index], opener.end()
        delimiter = opener.group("quoted") if opener.group("q") else opener.group("bare")
        if not delimiter:
            continue
        starts = list(_iter_shell_command_starts(header))
        _, executable, args = _command_parts(_shell_words_at(header, starts[-1]) if starts else [])
        # A bare shell (no -c script, no script operand) executes the body itself.
        execute_as_shell = bool(
            executable and _executable_name(executable) in _SHELL_EXECUTABLES
            and _shell_script_arg(args) is None
            and not any(arg and not arg.startswith("-") for arg in args))
        specs.append(_Heredoc(delimiter, bool(opener.group("dash")), execute_as_shell))
    return specs


def _mask_heredocs(command: str) -> tuple[str, list[str]]:
    """Blank heredoc bodies -> (masked command, bodies a bare shell would execute). Unterminated
    heredocs run to end of input and are still reported."""
    output: list[str] = []
    pending: list[_Heredoc] = []
    finished: list[_Heredoc] = []
    for line in command.splitlines(keepends=True):
        if not pending:
            output.append(line)
            pending.extend(_heredoc_specs(line))
            continue
        current = pending[0]
        candidate = line.rstrip("\r\n")
        if (candidate.lstrip("\t") if current.strip_tabs else candidate) == current.delimiter:
            finished.append(pending.pop(0))
        else:
            current.body.append(line)
        output.append(re.sub(r"[^\r\n]", " ", line))
    shell_scripts = ["".join(spec.body) for spec in finished + pending if spec.execute_as_shell]
    return "".join(output), shell_scripts


def _record_alias(config: str, aliases: dict[str, str]) -> None:
    """Record an inline ``-c alias.<name>=<value>`` git config override."""
    key, sep, value = config.partition("=")
    if sep and key.lower().startswith("alias."):
        aliases[key[6:].lower()] = value


def _git_target_and_subcommand(
    args: list[str], current_dir: Path, env: dict[str, str],
) -> tuple[Path, str | None, list[str], dict[str, str]]:
    """Parse git's global options -> (target dir, subcommand, sub args, inline aliases)."""
    target = current_dir
    work_tree: str | None = None
    aliases: dict[str, str] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            index += 1
            break
        if not arg.startswith("-"):
            break
        # Separate-argument form (`-C dir`) vs attached form (`-Cdir`, `--work-tree=dir`, `-cK=V`).
        if arg in _GIT_GLOBAL_OPTIONS_WITH_ARG:
            option, value = arg, args[index + 1] if index + 1 < len(args) else None
            index += 2
        else:
            option, value = next(((o, arg[len(o):]) for o in ("-C", "--work-tree=", "-c")
                                  if arg.startswith(o) and len(arg) > len(o)), (None, None))
            index += 1
        if option == "-C" and value is not None:
            target = _resolve(value, target)
        elif option and option.startswith("--work-tree") and value is not None:
            work_tree = value
        elif option == "-c" and value is not None:
            _record_alias(value, aliases)  # only alias.* overrides matter; others are ignored
    explicit_work_tree = work_tree or env.get("GIT_WORK_TREE")
    if explicit_work_tree:
        target = _resolve(explicit_work_tree, target)
    return target, args[index].lower() if index < len(args) else None, args[index + 1 :], aliases


def _has_flag(args: list[str], long: str, letter: str, short_only: bool = False) -> bool:
    """``--long`` present, or ``letter`` inside a dash-prefixed arg (``-fdn``; ``short_only``
    additionally excludes ``--`` args from the letter scan)."""
    return any(
        arg == long or (arg.startswith("-") and letter in arg[1:]
                        and not (short_only and arg.startswith("--")))
        for arg in args)


# Subcommands whose worktree impact depends on their arguments -> predicate(args).
_CONDITIONAL_MUTATIONS: dict[str, Callable[[list[str]], bool]] = {
    "reset": lambda args: any(
        arg in _RESET_WORKTREE_MODES or _RESET_HARD_RE.fullmatch(arg) for arg in args),
    "stash": lambda args: next(
        (arg for arg in args if not arg.startswith("-")), "push") not in _STASH_SAFE_ACTIONS,
    "clean": lambda args: not _has_flag(args, "--dry-run", "n", short_only=True),
    # `restore` touches the worktree unless ONLY --staged was requested.
    "restore": lambda args: (_has_flag(args, "--worktree", "W")
                             or not _has_flag(args, "--staged", "S"))}


def _mutates_worktree(subcommand: str, args: list[str]) -> bool:
    check = _CONDITIONAL_MUTATIONS.get(subcommand, lambda _args: subcommand in _WORKTREE_MUTATIONS)
    return check(args)


def _inspect_git_worktree(args: list[str], cwd: Path, root: Path) -> str | None:
    """Block `worktree remove|move` aimed at the running root, from any directory."""
    action_index = _consume_options(args, 0)
    action = args[action_index].lower() if action_index < len(args) else None
    target_index = _consume_options(args, action_index + 1)
    if (action in _WORKTREE_TARGET_ACTIONS and target_index < len(args)
            and _resolve(args[target_index], cwd) == root):
        return f"git worktree {action}"
    return None


def _read_git_alias(executable: str, target: Path, alias: str) -> str | None:
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        result = subprocess.run(
            [executable, "-C", str(target), "config", "--get", f"alias.{alias}"],
            capture_output=True, text=True, timeout=1, check=False)
        return (result.stdout.strip() or None) if result.returncode == 0 else None
    return None


def _inspect_git(
    executable: str, args: list[str], current_dir: Path, env: dict[str, str], root: Path, depth: int
) -> str | None:
    target, subcommand, sub_args, inline_aliases = _git_target_and_subcommand(
        args, current_dir, env)
    if subcommand is None:
        return None
    if subcommand == "worktree":  # names its victim as an argument: the cwd check does not apply
        return _inspect_git_worktree(sub_args, target, root)
    if not _is_within(target, root):
        return None
    if _mutates_worktree(subcommand, sub_args):
        return f"git {subcommand}"
    if subcommand in _KNOWN_GIT_BUILTINS or depth >= _MAX_RECURSION:
        return None
    alias = (inline_aliases[subcommand] if subcommand in inline_aliases
             else _read_git_alias(executable, target, subcommand))
    if not alias:
        return None
    if alias.startswith("!"):  # shell alias: scan it as a command
        return _find_mutation(alias[1:], target, root, depth + 1)
    try:
        alias_args = shlex.split(alias, posix=True)
    except ValueError:
        return None
    return _inspect_git(executable, [*alias_args, *sub_args], target, {}, root, depth + 1)


def _inspect_github_cli(
    executable: str, args: list[str], current_dir: Path, env: dict[str, str], root: Path, depth: int
) -> str | None:
    if not _is_within(current_dir, root):
        return None
    index = _consume_options(args, 0, frozenset({"-R", "--repo", "--hostname"}))
    is_checkout = args[index : index + 2] == ["pr", "checkout"]
    return f"{_executable_name(executable)} pr checkout" if is_checkout else None


def _inspect_shell(
    executable: str, args: list[str], current_dir: Path, env: dict[str, str], root: Path, depth: int
) -> str | None:
    script = _shell_script_arg(args)
    return _find_mutation(script, current_dir, root, depth + 1) if script else None


# executable name -> inspector(executable, args, current_dir, env, root, depth)
_INSPECTORS: dict[str, Callable[..., str | None]] = {
    "git": _inspect_git, "gh": _inspect_github_cli, "hub": _inspect_github_cli,
    **{shell: _inspect_shell for shell in _SHELL_EXECUTABLES}}


def _find_mutation(command: str, cwd: Path, root: Path, depth: int = 0) -> str | None:
    """Name of the first command in ``command`` that would rewrite ``root``, else None."""
    if depth > _MAX_RECURSION:
        return None
    masked_command, heredoc_scripts = _mask_heredocs(command)
    for script in heredoc_scripts:
        if operation := _find_mutation(script, cwd, root, depth + 1):
            return operation
    starts = sorted(set(_iter_shell_command_starts(masked_command)))
    scopes = _scope_keys(masked_command, starts)
    # cwd per subshell scope; `cd` applies to the NEXT command only via `&&`, `;`, newline.
    cwd_by_scope: dict[tuple[int, ...], Path] = {(): cwd}
    pending_cd: dict[tuple[int, ...], Path] = {}
    for start in starts:
        scope = scopes[start]
        cwd_by_scope.setdefault(scope, cwd_by_scope.get(scope[:-1], cwd))
        pending = pending_cd.pop(scope, None)
        if pending is not None and _operator_before(masked_command, start) in {"&&", ";", "\n"}:
            cwd_by_scope[scope] = pending
        env, executable, args = _command_parts(_shell_words_at(masked_command, start))
        if executable is None:
            continue
        current_dir = cwd_by_scope[scope]
        if (cd_target := _cd_target(executable, args, current_dir)) is not None:
            pending_cd[scope] = cd_target
        elif (inspect := _INSPECTORS.get(_executable_name(executable))) and (
                operation := inspect(executable, args, current_dir, env, root, depth)):
            return operation
    return None


def guard_active() -> bool:
    """Windows-only: NTFS locks loaded .py/.pyd files, so overwriting the live checkout can
    corrupt the running process. On POSIX open handles keep the old inode alive; the mixed-module
    hazard is limited to later lazy imports — not worth blocking every git workflow for."""
    return os.name == "nt"


def detect_self_repo_git_mutation(
    command: str, cwd: str | None, source_root: Path | None = None) -> tuple[bool, str | None]:
    """-> (blocked, message): whether a command would rewrite the live source checkout."""
    root = source_root if source_root is not None else get_running_source_root()
    if root is None or not command:
        return False, None
    root = _resolve(str(root), Path("/"))
    operation = _find_mutation(command, _resolve(cwd or "/", Path("/")), root)
    return (True, _block_message(operation, root)) if operation is not None else (False, None)


def _block_message(operation: str, root: Path) -> str:
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    scratch = (Path(hermes_home).expanduser() if hermes_home else Path.home() / ".hermes") / "scratch"
    return (
        f"Blocked: `{operation}` would rewrite Hermes's live source checkout "
        f"({root}) and can mix module versions in this running process. "
        f"Use a separate worktree or a shared clone on real disk, e.g. "
        f"`git clone --shared {root} {scratch}/<task>` — avoid /tmp for "
        "clones that install node/python deps: /tmp is usually RAM-backed tmpfs and a few "
        "dependency installs can fill it and ENOSPC other work. Delete the clone when the branch "
        "is pushed. To change this checkout, stop Hermes, run the command externally, then restart "
        "Hermes.")
