"""User skill locks, enforced on every agent write path.

``hermes curator lock <skill>`` (or the Desktop lock switch) records ``locked: true`` beside
``pinned`` in the profile's ``skills/.usage.json``. While a skill is locked the agent cannot change
it: ``skill_manage`` refuses every mutating action, ``write_file``/``patch`` refuse any path inside
the skill's directory, and ``terminal`` refuses commands that write there. The lock record itself
is protected the same way, and the agent cannot run ``hermes curator unlock``. The user's own edits
(Desktop editor, their own editor, the CLI) are not gated.

The terminal check reads the command's write targets (redirections, tee, cp/mv/rm/sed -i and
friends, git in the skill's tree, ``cd`` into it, inline interpreter code naming it). It is a
policy check, not a sandbox: a script that writes there through its own logic is not seen.
"""

from __future__ import annotations

import glob
import itertools
import os
import re
import shlex
from typing import Iterable

# Verbs whose every path operand is written (or removed/renamed).
_WRITES_EVERY_OPERAND = frozenset({
    "rm", "rmdir", "unlink", "truncate", "shred", "touch", "chmod", "chown", "chgrp", "chattr",
    "setfacl", "mkdir", "mv", "patch", "dos2unix", "unix2dos", "xattr", "ed", "ex", "vi", "vim",
    "nvim", "nano", "emacs"})
# Verbs that write only to their LAST operand (the destination).
_WRITES_LAST_OPERAND = frozenset({"cp", "install", "ln", "rsync", "scp"})
# Verbs that reach INTO a directory operand recursively, so an ancestor of a locked skill counts.
_RECURSIVE = frozenset({"rm", "mv", "chmod", "chown", "chgrp", "chattr", "setfacl", "rsync", "find"})
_INPLACE_EDITORS = frozenset({"sed", "perl", "ruby"})
_GIT_MUTATIONS = frozenset({
    "add", "am", "apply", "checkout", "clean", "commit", "merge", "mv", "pull", "rebase", "reset",
    "restore", "revert", "rm", "stash", "switch", "cherry-pick"})
_INTERPRETERS = re.compile(
    r"^(?:python[0-9.]*|node|nodejs|deno|bun|ruby|perl|php|bash|sh|zsh|dash|ksh|pwsh|powershell)$")
_WRAPPERS = frozenset({"sudo", "env", "command", "builtin", "exec", "nohup", "setsid", "time", "nice", "xargs"})
_WRITEISH_CODE = re.compile(
    r"open\s*\(|write|unlink|remove|rmtree|rename|replace|truncate|copy|move|chmod|mkdir|"
    r"writeFile|appendFile|rmSync|>|sed\s+-i|\bmv\b|\brm\b|\bcp\b", re.IGNORECASE)
_SEPARATORS = frozenset({";", "&", "&&", "|", "||", "(", ")", "|&", ";;"})


def _skill_usage():
    from tools import skill_usage
    return skill_usage


def locked_skill_dirs() -> list[tuple[str, str]]:
    """``(name, realpath)`` of every locked skill that resolves in the active profile."""
    names = _skill_usage().locked_skill_names()
    if not names:
        return []
    from tools.skill_manager_tool import _find_skill
    found = []
    for name in names:
        if hit := _find_skill(name):
            found.append((name, os.path.realpath(hit["path"])))
    return found


def is_locked_skill(name: str, locked_names: Iterable[str] | None = None) -> bool:
    """Whether *name* (directory, ``cat/name`` or frontmatter name) is locked. Locks are keyed by
    directory name, while curator rows and some callers carry the frontmatter name."""
    names = set(_skill_usage().locked_skill_names() if locked_names is None else locked_names)
    if not names:
        return False
    if name in names or os.path.basename(name) in names:
        return True
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(name)
    return bool(found) and found["path"].name in names


def _record_path() -> str:
    return os.path.realpath(_skill_usage()._usage_file())


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def skill_refusal(name: str) -> str:
    return (f"Skill '{name}' is locked by the user, so the agent cannot change it: skill_manage, "
            f"write_file, patch and terminal all refuse. Don't retry or work around it (no copying it "
            f"under another name); tell the user what you would change. They can unlock it with "
            f"`hermes curator unlock {name}` or the lock switch in Desktop.")


RECORD_REFUSAL = ("skills/.usage.json holds the user's skill locks, so the agent cannot change it "
                  "while any skill is locked.")
UNLOCK_REFUSAL = ("Only the user can unlock a skill. Ask them to run `hermes curator unlock <skill>` "
                  "themselves or use the lock switch in Desktop.")


def _skills_roots() -> list[str]:
    from agent.skill_utils import get_all_skills_dirs
    return [os.path.realpath(d) for d in get_all_skills_dirs()]


def _hit(path: str, locked: list[tuple[str, str]], record: str, *, ancestor: str = "") -> str | None:
    """Refusal text when *path* is the lock record or inside a locked skill. ``ancestor="any"`` also
    counts a directory ABOVE a locked skill (recursive deletes/moves); ``ancestor="skills"`` counts it
    only inside a skills root (a recursive copy or extraction there can merge into the skill)."""
    real = os.path.realpath(path)
    if real == record:
        return RECORD_REFUSAL
    for name, root in locked:
        if _inside(real, root):
            return skill_refusal(name)
        if ancestor and real != os.sep and _inside(root, real) and (
                ancestor == "any" or any(_inside(real, r) for r in _skills_roots())):
            return skill_refusal(name)
    return None


def path_refusal(paths: Iterable[str]) -> str | None:
    """write_file/patch guard: refusal text when any path lands in a locked skill or the lock record."""
    locked = locked_skill_dirs()
    if not locked:
        return None
    record = _record_path()
    for path in paths:
        if path and (reason := _hit(path, locked, record)):
            return reason
    return None


# ── terminal ──────────────────────────────────────────────────────────────────────────────────

def _expand(word: str, cwd: str) -> list[str]:
    """Shell word -> absolute path(s): ~, $HOME, $HERMES_HOME, relative to *cwd*; a glob expands
    against the filesystem the way the shell would (unmatched: its literal directory prefix)."""
    from hermes_constants import get_hermes_home
    home = os.path.expanduser("~")
    for var, value in (("HERMES_HOME", str(get_hermes_home())), ("HOME", home)):
        word = word.replace("${" + var + "}", value).replace("$" + var, value)
    if word == "~" or word.startswith("~/"):
        word = home + word[1:]
    path = os.path.normpath(os.path.join(cwd, word))
    if not any(ch in word for ch in "*?["):
        return [path]
    if matches := glob.glob(path):
        return matches
    return [path[: min(path.index(ch) for ch in "*?[" if ch in path)].rsplit(os.sep, 1)[0] or os.sep]


def _unquoted_newlines_to_semicolons(command: str) -> str:
    """Newlines separate commands like ``;`` does; shlex would read them as plain whitespace."""
    out, quote = [], None
    for ch in command:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "\n":
            ch = ";"
        out.append(ch)
    return "".join(out)


def _tokens(command: str) -> list[str]:
    lexer = shlex.shlex(_unquoted_newlines_to_semicolons(command), posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes: fall back to whitespace words
        return command.split()


def _segments(tokens: list[str]) -> Iterable[tuple[list[str], list[str], str]]:
    """Yield ``(argv, redirect_targets, separator_that_ended_it)`` per simple command."""
    argv: list[str] = []
    targets: list[str] = []
    it = iter(tokens)
    for tok in it:
        if ">" in tok and set(tok) <= set("&>|0123456789"):
            if not tok.endswith("&"):
                target = next(it, "")
                if target and not target.startswith("&") and target != "/dev/null":
                    targets.append(target)
            continue
        if tok in ("<", "<<", "<<<", "<<-"):
            next(it, "")  # input file / heredoc delimiter / here-string: read, never written
            continue
        if tok and set(tok) <= set(";&|()"):
            yield argv, targets, tok
            argv, targets = [], []
            continue
        argv.append(tok)
    yield argv, targets, ""


_XARGS_OPTIONS_WITH_ARG = frozenset({"-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a"})


def _strip_wrappers(argv: list[str]) -> list[str]:
    from tools.self_repo_guard import _WRAPPER_OPTIONS_WITH_ARG
    while argv:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            argv = argv[1:]
            continue
        head = os.path.basename(argv[0])
        if head not in _WRAPPERS:
            break
        takes_value = _XARGS_OPTIONS_WITH_ARG if head == "xargs" else _WRAPPER_OPTIONS_WITH_ARG.get(head, frozenset())
        argv = argv[1:]
        while argv and argv[0].startswith("-"):
            argv = argv[2:] if argv[0] in takes_value else argv[1:]
    return argv


def _operands(args: list[str]) -> list[str]:
    return [a for a in args if a and not a.startswith("-")]


def _target_directory(args: list[str]) -> tuple[str | None, list[str]]:
    """``cp/mv/install/ln -t DIR`` / ``--target-directory[=]DIR``: (DIR, remaining args)."""
    rest = list(args)
    for i, arg in enumerate(rest):
        if arg.startswith("--target-directory="):
            return arg.split("=", 1)[1], rest[:i] + rest[i + 1:]
        if arg in ("-t", "--target-directory") and i + 1 < len(rest):
            return rest[i + 1], rest[:i] + rest[i + 2:]
        if arg.startswith("-t") and len(arg) > 2 and not arg.startswith("--"):
            return arg[2:], rest[:i] + rest[i + 1:]
    return None, rest


class _Scope:
    """What a command is checked against: its working dir, the locked skills and the lock record."""

    def __init__(self, cwd: str, locked: list[tuple[str, str]], record: str, depth: int) -> None:
        self.cwd, self.locked, self.record, self.depth = cwd, locked, record, depth

    def hit(self, word: str, ancestor: str = "") -> str | None:
        return next((r for path in _expand(word, self.cwd)
                     if (r := _hit(path, self.locked, self.record, ancestor=ancestor))), None)

    def first_hit(self, words: Iterable[str], ancestor: str = "") -> str | None:
        return next((r for w in words if (r := self.hit(w, ancestor))), None)


def _check_unlock(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb == "hermes" or (verb.startswith("python") and "hermes_cli.main" in args):
        if "curator" in args and "unlock" in args[args.index("curator"):]:
            return UNLOCK_REFUSAL
    return None


_RECURSIVE_COPY_FLAGS = re.compile(r"^(?:-[A-Za-z]*[rRa][A-Za-z]*|--recursive|--archive)$")


def _check_file_verbs(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb in _WRITES_EVERY_OPERAND or verb == "tee":
        return scope.first_hit(_operands(args), ancestor="any" if verb in _RECURSIVE else "")
    if verb not in _WRITES_LAST_OPERAND:
        return None
    target, rest = _target_directory(args)
    ops = _operands(rest)
    dest = target if target is not None else (ops[-1] if ops else None)
    if dest is None:
        return None
    # A recursive copy into a directory above the skill merges into it; rsync is recursive with -a/-r.
    merges = verb == "rsync" or any(_RECURSIVE_COPY_FLAGS.match(a) for a in rest)
    return scope.hit(dest, ancestor="skills" if merges else "")


def _check_inplace(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb in _INPLACE_EDITORS and any(
            a.startswith("--in-place") or (a.startswith("-") and not a.startswith("--") and "i" in a) for a in args):
        return scope.first_hit(_operands(args))
    return None


def _check_find(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb != "find" or not any(a in ("-delete", "-exec", "-execdir", "-ok") for a in args):
        return None
    starts = itertools.takewhile(lambda a: not a.startswith("-"), args)
    return scope.first_hit(starts, ancestor="any")


def _check_git(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb != "git":
        return None
    where, rest = scope.cwd, list(args)
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag == "-C" and rest:
            where = _expand(rest.pop(0), scope.cwd)[0]
        elif flag in ("-c", "--git-dir", "--work-tree") and rest:
            rest.pop(0)
    if rest and rest[0] in _GIT_MUTATIONS:
        return _hit(where, scope.locked, scope.record, ancestor="any")
    return None


def _check_extract(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb not in ("tar", "unzip", "bsdtar"):
        return None
    dest = scope.cwd
    for flag, value in itertools.pairwise(args):
        if flag in ("-C", "--directory", "-d"):
            dest = _expand(value, scope.cwd)[0]
    bundled = bool(args) and not args[0].startswith("-") and "x" in args[0]  # `tar xzf a.tgz`
    extracting = verb == "unzip" or bundled or any(
        a == "--extract" or (a.startswith("-") and not a.startswith("--") and "x" in a) for a in args)
    # The archive's paths are unknown: extracting into a directory above the skill can overwrite it.
    return _hit(dest, scope.locked, scope.record, ancestor="skills") if extracting else None


def _check_interpreter(verb: str, args: list[str], scope: _Scope) -> str | None:
    if not _INTERPRETERS.match(verb):
        return None
    for flag, code in itertools.pairwise(args):
        if flag not in ("-c", "-e", "-E", "--eval", "-Command", "-r"):
            continue
        if verb in ("bash", "sh", "zsh", "dash", "ksh") and scope.depth < 3:
            reason = _command_refusal(code, scope.cwd, scope.locked, scope.record, scope.depth + 1)
        else:
            reason = _code_refusal(code, scope.locked, scope.record)
        if reason:
            return reason
    return None


_ARGV_CHECKS = (_check_unlock, _check_file_verbs, _check_inplace, _check_find, _check_git,
                _check_extract, _check_interpreter)


def _argv_refusal(argv: list[str], cwd: str, locked, record, depth: int) -> str | None:
    scope = _Scope(cwd, locked, record, depth)
    verb = os.path.basename(argv[0])
    return next((r for check in _ARGV_CHECKS if (r := check(verb, argv[1:], scope))), None)


def _code_refusal(code: str, locked, record) -> str | None:
    """Inline interpreter code that names a locked skill (or the record) AND looks like it writes."""
    from hermes_constants import get_hermes_home
    if not _WRITEISH_CODE.search(code):
        return None
    home, hermes_home = os.path.expanduser("~"), str(get_hermes_home())
    text = code.replace("$HERMES_HOME", hermes_home).replace("~", home)
    skills_root = os.path.realpath(_skill_usage()._skills_dir())
    if ".usage.json" in text:
        return RECORD_REFUSAL
    for name, root in locked:
        rel = os.path.relpath(root, skills_root) if skills_root and _inside(root, skills_root) else ""
        if root in text or (rel and f"skills/{rel}" in text) or f"skills/{os.path.basename(root)}" in text:
            return skill_refusal(name)
    return None


def _command_refusal(command: str, cwd: str, locked, record, depth: int = 0) -> str | None:
    if "<<" in command and (reason := _code_refusal(command, locked, record)):
        return reason  # heredoc-fed interpreter body
    subshells: list[str] = []  # cwd to restore when each ( ... ) closes
    for argv, targets, separator in _segments(_tokens(command)):
        for target in targets:
            if reason := next((r for path in _expand(target, cwd)
                               if (r := _hit(path, locked, record))), None):
                return reason
        argv = _strip_wrappers(argv)
        if argv and argv[0] in ("cd", "pushd") and len(argv) > 1:
            cwd = _expand(argv[1], cwd)[0]
        elif argv and (reason := _argv_refusal(argv, cwd, locked, record, depth)):
            return reason
        for ch in separator:
            if ch == "(":
                subshells.append(cwd)
            elif ch == ")" and subshells:
                cwd = subshells.pop()
    return None


def command_refusal(command: str, cwd: str) -> str | None:
    """terminal guard: refusal text when *command* writes into a locked skill, the lock record, or
    runs ``hermes curator unlock``. None when nothing is locked."""
    locked = locked_skill_dirs()
    if not locked:
        return None
    return _command_refusal(command, cwd or os.getcwd(), locked, _record_path())
