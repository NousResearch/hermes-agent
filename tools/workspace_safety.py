from __future__ import annotations

import re
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

"""Workspace-binding guards for side-effecting project tools.

The gateway can bind a chat/channel to an authoritative workspace repo.  Tool
side effects should not silently land in a different checkout just because the
model inferred project context from room names or stale memory.
"""




_MUTATING_GIT_SUBCOMMANDS = frozenset({
    "add",
    "am",
    "apply",
    "bisect",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "fetch",
    "gc",
    "merge",
    "mv",
    "pull",
    "push",
    "rebase",
    "reset",
    "restore",
    "revert",
    "rm",
    "stash",
    "submodule",
    "switch",
    "tag",
    "worktree",
})

_READ_ONLY_GIT_SUBCOMMANDS = frozenset({
    "blame",
    "branch",  # treated as mutating when args imply config/ref changes
    "diff",
    "grep",
    "log",
    "ls-files",
    "remote",  # treated as mutating when args imply config/ref changes
    "rev-parse",
    "show",
    "status",
})

_MUTATING_REMOTE_SUBCOMMANDS = frozenset({
    "add",
    "remove",
    "rm",
    "rename",
    "set-branches",
    "set-head",
    "set-url",
    "prune",
    "update",
})

_MUTATING_BRANCH_FLAGS = frozenset({
    "-d",
    "-D",
    "-m",
    "-M",
    "-c",
    "-C",
    "-u",
    "--delete",
    "--move",
    "--copy",
    "--set-upstream-to",
    "--unset-upstream",
    "--edit-description",
})

_UNSAFE_GIT_CWD_SUBCOMMAND = "__unsafe_explicit_git_dir__"

_UNSAFE_GIT_ENV_VARS = frozenset({
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
})

_UNSAFE_GIT_CONFIG_KEYS = frozenset({
    "core.worktree",
    "include.path",
})

_UNSAFE_GIT_CONFIG_KEY_PREFIXES = (
    "includeif.",
    "alias.",
)

_UNSAFE_GIT_CONFIG_ENV_PREFIXES = (
    "GIT_CONFIG_",
)


@dataclass(frozen=True)
class GitInvocation:
    subcommand: str
    args: list[str]
    cwd: Path


def check_path_side_effect_allowed(
    path: str | Path,
    backend: Optional[str] = None,
) -> Optional[str]:
    """Return an error if a file write targets a repo outside the binding.

    Non-gateway callers and non-repo paths are allowed.  Gateway project writes
    are blocked when they target a git checkout that is not the bound repo, or
    when the session has no workspace binding at all.
    """

    if not _in_gateway_session():
        return None

    backend = (backend or "local").strip().lower() or "local"
    if backend != "local":
        return (
            f"Blocked repo side effect: workspace guard cannot verify file write "
            f"on nonlocal `{backend}` backend paths."
        )

    resolved = _safe_resolve(path)
    repo_root = _find_git_root(resolved if resolved.is_dir() else resolved.parent)
    if repo_root is None:
        return None

    bound_repo = _bound_repo_path()
    if bound_repo is None:
        return _blocked_message(repo_root, None)
    if not _same_path(repo_root, bound_repo):
        return _blocked_message(repo_root, bound_repo)
    return None


def check_terminal_side_effect_allowed(
    command: str,
    cwd: str | Path,
    backend: Optional[str] = None,
) -> Optional[str]:
    """Return an error if a mutating git command is outside the binding."""

    if not _in_gateway_session():
        return None

    cwd_path = _safe_resolve(cwd)
    backend = (backend or "local").strip().lower() or "local"
    unsafe_ambient = any(
        value and _env_name_is_unsafe_git(name)
        for name, value in __import__("os").environ.items()
    )
    for invocation in _iter_git_invocations(command, cwd_path):
        if not _git_invocation_is_mutating(invocation):
            continue
        if backend != "local":
            return (
                f"Blocked repo side effect: workspace guard cannot verify git command "
                f"on nonlocal `{backend}` backend."
            )
        if unsafe_ambient:
            return _unverified_git_message()
        if invocation.subcommand == _UNSAFE_GIT_CWD_SUBCOMMAND:
            return _unverified_git_message()
        if _git_subcommand_target_is_unverifiable(invocation.subcommand):
            return _unverified_git_message()
        repo_root = _find_git_root(invocation.cwd)
        if repo_root is None:
            # Preserve current behavior for non-repo scratch contexts while
            # still checking later git invocations in the same shell command.
            continue

        bound_repo = _bound_repo_path()
        if bound_repo is None:
            return _blocked_message(repo_root, None)
        if not _same_path(repo_root, bound_repo):
            return _blocked_message(repo_root, bound_repo)
    return None


def _unverified_git_message() -> str:
    return (
        "Blocked repo side effect: workspace guard cannot verify the target "
        "repository for this git command. Avoid complex cd/global git-dir/work-tree "
        "forms or run the command from the bound repository."
    )


def _blocked_message(actual_repo: Path, bound_repo: Optional[Path]) -> str:
    if bound_repo is None:
        return (
            "Blocked repo side effect: this gateway session has no authoritative "
            f"workspace binding, but the target is inside git repo `{actual_repo}`. "
            "Add a channel binding to workspaces.yaml or use read-only tools."
        )
    return (
        "Blocked repo side effect outside authoritative workspace binding: "
        f"target repo `{actual_repo}` does not match bound repo `{bound_repo}`."
    )


def _in_gateway_session() -> bool:
    return bool(_session_env("HERMES_SESSION_PLATFORM", "") and _session_env("HERMES_SESSION_CHAT_ID", ""))


def _bound_repo_path() -> Optional[Path]:
    repo_path = _session_env("HERMES_WORKSPACE_REPO_PATH", "")
    if not repo_path:
        return None
    return _safe_resolve(repo_path)


def _session_env(name: str, default: str = "") -> str:
    try:
        from gateway.session_context import get_session_env

        return get_session_env(name, default)
    except Exception:
        return os.getenv(name, default)


def _find_git_root(start: Path) -> Optional[Path]:
    current = _safe_resolve(start)
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)


def _safe_resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _iter_git_invocations(command: str, cwd: Path) -> Iterable[GitInvocation]:
    current_cwd = cwd
    unsafe_export = False
    if _has_unverifiable_shell_git(command):
        yield GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, [], current_cwd)
        return
    for segment in _split_shell_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            if "git" in segment:
                yield GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, [], current_cwd)
            continue
        if not tokens:
            continue
        if tokens[0] == "export":
            for token in tokens[1:]:
                name = _env_assignment_name(token) or token
                if _env_name_is_unsafe_git(name):
                    unsafe_export = True
            continue
        unsafe_env, tokens = _strip_env_prefix(tokens)
        if (unsafe_env or unsafe_export) and any(_is_git_executable_token(token) for token in tokens):
            yield GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, [], current_cwd)
            continue
        if not tokens:
            continue
        if tokens[0] == "cd":
            if len(tokens) != 2 or _path_has_shell_expansion(tokens[1]):
                yield GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, [], current_cwd)
                continue
            current_cwd = _safe_resolve(current_cwd / tokens[1])
            continue
        for index, token in enumerate(tokens):
            if not _is_git_executable_token(token):
                continue
            if unsafe_export:
                yield GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, [], current_cwd)
                continue
            invocation = _parse_git_invocation(tokens[index:], current_cwd)
            if invocation:
                yield invocation


def _is_git_executable_token(token: str) -> bool:
    return Path(token).name == "git"


def _path_has_shell_expansion(value: str) -> bool:
    return "$" in value or "`" in value or "$(" in value or value.startswith("~")


def _has_unquoted_shell_paren(value: str) -> bool:
    in_single = False
    in_double = False
    escaped = False

    for ch in value:
        if escaped:
            escaped = False
            continue

        if ch == "\\" and not in_single:
            escaped = True
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            continue

        if ch in "()" and not in_single and not in_double:
            return True

    return False


def _has_unverifiable_shell_git(command: str) -> bool:
    # Command substitution can redirect the git target at runtime and is
    # intentionally fail-closed rather than partially shell-parsed.
    if "$(" in command and "git" in command:
        return True

    if _shell_command_executes_git(command):
        return True

    # Only unquoted parentheses imply shell grouping/subshell syntax. Quoted
    # parentheses are common in conventional commit scopes and format strings.
    return _has_unquoted_shell_paren(command) and "git" in command


_SHELL_NAMES = frozenset({"sh", "bash", "zsh", "dash", "ash", "ksh"})


def _shell_command_executes_git(command: str) -> bool:
    """True when a shell -c/-lc payload has git in command position (not echo text)."""
    import shlex
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "git" in command and (" -c " in f" {command} " or " -lc " in f" {command} ")

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        name = Path(tok).name
        if name in _SHELL_NAMES and i + 1 < len(tokens):
            j = i + 1
            # skip shell flags like -e -u -o pipefail until -c/-lc
            while j < len(tokens) and tokens[j].startswith("-") and tokens[j] not in {"-c", "-lc"}:
                # flags that take args
                if tokens[j] in {"-o", "-O"} and j + 1 < len(tokens):
                    j += 2
                    continue
                j += 1
            if j < len(tokens) and tokens[j] in {"-c", "-lc"} and j + 1 < len(tokens):
                payload = tokens[j + 1]
                if _payload_has_git_command(payload):
                    return True
                i = j + 2
                continue
        i += 1
    return False


def _payload_has_git_command(payload: str) -> bool:
    """Return True if git appears as an executable token in a shell payload."""
    import shlex
    if _shell_command_executes_git(payload):
        return True
    try:
        parts = shlex.split(payload)
    except ValueError:
        return bool(re.search(r"(?:^|[;|&\n\t ])git(?:[;|&\n\t ]|$)", payload))

    prev_connector = True  # start of payload is a command position
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        if part in {";", "|", "||", "&&", "&"}:
            prev_connector = True
            idx += 1
            continue
        # skip env assignments before a command
        if "=" in part and not part.startswith("-") and Path(part).name != "git":
            idx += 1
            continue
        if prev_connector and Path(part).name == "git":
            return True
        prev_connector = False
        idx += 1
    return False


def _env_assignment_name(token: str) -> Optional[str]:
    if "=" not in token or token.startswith("="):
        return None
    name, _, _ = token.partition("=")
    if not name or any(ch in name for ch in "-./"):
        return None
    return name


def _env_name_is_unsafe_git(name: str) -> bool:
    return name in _UNSAFE_GIT_ENV_VARS or any(
        name.startswith(prefix) for prefix in _UNSAFE_GIT_CONFIG_ENV_PREFIXES
    )


def _git_config_key(token: str) -> str:
    key, _, _value = token.partition("=")
    return key.strip().lower()


def _is_unsafe_git_config_assignment(token: str) -> bool:
    key = _git_config_key(token)
    return key in _UNSAFE_GIT_CONFIG_KEYS or any(
        key.startswith(prefix) for prefix in _UNSAFE_GIT_CONFIG_KEY_PREFIXES
    )


def _strip_env_prefix(tokens: list[str]) -> tuple[bool, list[str]]:
    unsafe_env = False
    index = 0
    if tokens and tokens[0] == "env":
        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
    while index < len(tokens):
        name = _env_assignment_name(tokens[index])
        if name is None:
            break
        if _env_name_is_unsafe_git(name):
            unsafe_env = True
        index += 1
    return unsafe_env, tokens[index:]


def _parse_git_invocation(tokens: list[str], base_cwd: Path) -> Optional[GitInvocation]:
    if not tokens or not _is_git_executable_token(tokens[0]):
        return None

    git_cwd = base_cwd
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return None
        if token == "-C":
            if index + 1 >= len(tokens) or _path_has_shell_expansion(tokens[index + 1]):
                return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
            git_cwd = _safe_resolve(git_cwd / tokens[index + 1])
            index += 2
            continue
        if token.startswith("-C") and token != "-C":
            if _path_has_shell_expansion(token[2:]):
                return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
            git_cwd = _safe_resolve(git_cwd / token[2:])
            index += 1
            continue
        if token in {"-c", "--config-env"}:
            if index + 1 >= len(tokens):
                return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
            if _is_unsafe_git_config_assignment(tokens[index + 1]):
                return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
            index += 2
            continue
        if token.startswith("-c") and token != "-c":
            inline_config = token[2:]
            if _is_unsafe_git_config_assignment(inline_config):
                return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
            index += 1
            continue
        if token.startswith("--config-env="):
            if _is_unsafe_git_config_assignment(token.removeprefix("--config-env=")):
                return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
            index += 1
            continue
        if token.startswith("--git-dir") or token.startswith("--work-tree"):
            return GitInvocation(_UNSAFE_GIT_CWD_SUBCOMMAND, tokens[index + 1 :], git_cwd)
        if token.startswith("-"):
            index += 1
            continue
        return GitInvocation(token, tokens[index + 1 :], git_cwd)
    return None


def _git_invocation_is_mutating(invocation: GitInvocation) -> bool:
    subcommand = invocation.subcommand
    if subcommand == _UNSAFE_GIT_CWD_SUBCOMMAND:
        return True
    if subcommand == "branch":
        return _branch_is_mutating(invocation.args)
    if subcommand == "remote":
        return _remote_is_mutating(invocation.args)
    if subcommand in _READ_ONLY_GIT_SUBCOMMANDS:
        return False
    if subcommand in _MUTATING_GIT_SUBCOMMANDS:
        return True
    # Unknown git subcommands are potentially side-effecting; guard them.
    return True


def _git_subcommand_target_is_unverifiable(subcommand: str) -> bool:
    """Return true when a git subcommand might be an alias or extension.

    Unknown git subcommands are treated as mutating above. They can also be
    repo-local or global aliases that shell out to a different repository, so
    the workspace guard cannot verify their target from the current cwd alone.
    """

    return (
        subcommand not in _READ_ONLY_GIT_SUBCOMMANDS
        and subcommand not in _MUTATING_GIT_SUBCOMMANDS
        and subcommand not in {"branch", "remote"}
    )


def _branch_is_mutating(args: list[str]) -> bool:
    if any(arg in _MUTATING_BRANCH_FLAGS for arg in args):
        return True
    # `git branch name` creates a branch; `git branch` and flags like `-vv` list.
    return bool(args and not any(arg.startswith("-") for arg in args))


def _remote_is_mutating(args: list[str]) -> bool:
    non_flags = [arg for arg in args if not arg.startswith("-")]
    return bool(non_flags and non_flags[0] in _MUTATING_REMOTE_SUBCOMMANDS)


def _split_shell_segments(command: str) -> Iterable[str]:
    for segment in command.replace("&&", ";").replace("||", ";").split(";"):
        segment = segment.strip()
        if segment:
            yield segment
