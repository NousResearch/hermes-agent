"""Guard against git commands that rewrite the running hermes source checkout.

When hermes runs from a source/editable install (``pip install -e`` on a git
checkout), the interpreter keeps resolving lazy imports from that directory
for the lifetime of the process. A ``git checkout``/``reset``/``pull`` in
that repo swaps the code on disk under the live process: modules imported
before the switch stay at the old version while anything imported later
loads the new one. The resulting version skew surfaces as delayed,
nonsensical failures — signature TypeErrors between a caller and callee of
the same seam, tracebacks whose lines don't match the file on disk — long
after the command that caused them, and typically eats the in-flight turn.

Packaged installs are immune (site-packages has no ``.git``) so the guard
resolves to inert there. Read-only git commands (``status``/``log``/
``diff``/``branch``), commits, and content edits to individual files stay
allowed — editing hermes source *files* is the normal dev loop; the hazard
is wholesale working-tree/ref switches. ``git worktree add`` also stays
allowed: it is the recommended alternative and the block message points to
it.
"""

import os
import re
import shlex
from pathlib import Path
from typing import List, Optional, Tuple

# Working-tree/ref mutations. Not listed (deliberately allowed): commit,
# branch, tag, fetch, worktree, apply/am (file-level edits are the normal
# hermes-on-hermes dev loop), and all read-only subcommands.
_MUTATING_SUBCOMMANDS = frozenset({
    "checkout", "switch", "reset", "rebase", "merge", "pull",
    "restore", "stash", "clean", "cherry-pick", "revert",
})

# stash invocations that only read state.
_STASH_READONLY = frozenset({"list", "show"})

_WRAPPER_COMMANDS = frozenset({"sudo", "env", "exec", "nohup", "setsid", "time", "command"})

# Git global flags that consume the NEXT token as their argument.
_GIT_FLAGS_WITH_ARG = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"})

# Command separators plus subshell boundaries, so `$(git ...)` and
# `(cd repo && git ...)` are scanned as their own segments.
_SEGMENT_SPLIT = re.compile(r"(?:&&|\|\||;|\||\n|\$\(|\(|\)|`)")


def get_running_source_root() -> Optional[Path]:
    """Repo root of the source checkout this process runs from, or None.

    None means a packaged install (no ``.git`` beside the code) and disables
    the guard. ``.git`` may be a directory (normal clone) or a file (linked
    worktree); both count.
    """
    try:
        root = Path(__file__).resolve().parent.parent
    except OSError:
        return None
    return root if (root / ".git").exists() else None


def _tokenize(segment: str) -> List[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(os.path.expanduser(path_str))
    if not p.is_absolute():
        p = base / p
    try:
        return p.resolve()
    except OSError:
        return p


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path == root or path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def _strip_wrappers(tokens: List[str]) -> List[str]:
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _WRAPPER_COMMANDS:
            i += 1
            # env/sudo may carry VAR=VAL assignments and flags before the
            # real command.
            while i < len(tokens) and ("=" in tokens[i] or tokens[i].startswith("-")):
                i += 1
            continue
        if "=" in tok and not tok.startswith("-"):
            i += 1
            continue
        break
    return tokens[i:]


def _git_target_and_subcommand(
    tokens: List[str], current_dir: Path
) -> Tuple[Optional[Path], Optional[str], List[str]]:
    """Parse one git invocation: (target dir, subcommand, subcommand args).

    ``-C`` entries are applied cumulatively against *current_dir*, matching
    git's own sequential ``-C`` semantics.
    """
    target = current_dir
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-C" and i + 1 < len(tokens):
            target = _resolve(tokens[i + 1], target)
            i += 2
        elif tok in _GIT_FLAGS_WITH_ARG and i + 1 < len(tokens):
            i += 2
        elif tok.startswith("--") and "=" in tok:
            i += 1
        elif tok.startswith("-"):
            i += 1
        else:
            return target, tok, tokens[i + 1:]
    return target, None, []


def detect_self_repo_git_mutation(
    command: str,
    cwd: Optional[str],
    source_root: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """Return (True, block message) if *command* would rewrite the source repo.

    *cwd* is the directory the command will run in; ``cd`` segments inside
    the command are tracked so ``cd <repo> && git checkout x`` is caught.
    """
    root = source_root if source_root is not None else get_running_source_root()
    if root is None or not command:
        return False, None

    base = _resolve(cwd, Path("/")) if cwd else Path("/")
    current_dir = base

    for segment in _SEGMENT_SPLIT.split(command):
        tokens = _strip_wrappers(_tokenize(segment))
        if not tokens:
            continue
        if tokens[0] == "cd":
            current_dir = _resolve(tokens[1], current_dir) if len(tokens) > 1 else current_dir
            continue
        if tokens[0] != "git":
            continue
        target, sub, sub_args = _git_target_and_subcommand(tokens, current_dir)
        if sub not in _MUTATING_SUBCOMMANDS:
            continue
        if sub == "stash" and sub_args and sub_args[0] in _STASH_READONLY:
            continue
        if target is not None and _is_within(target, root):
            return True, _block_message(sub, root)

    return False, None


def _block_message(subcommand: str, root: Path) -> str:
    return (
        f"Blocked: `git {subcommand}` would rewrite the working tree of the "
        f"hermes source checkout this process is running from ({root}). "
        "Changing the code on disk under the live interpreter causes version "
        "skew: modules already imported keep the old version while later lazy "
        "imports load the new one, producing delayed crashes (mismatched "
        "signatures, tracebacks that don't match the source) that lose the "
        "in-flight turn. Work in a separate checkout instead, e.g. "
        f"`git -C {root} worktree add <tmpdir> <branch>` or a temp clone. "
        "If this checkout itself must change, ask the user to run the "
        "command outside hermes and restart hermes afterwards."
    )
