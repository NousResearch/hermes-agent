"""Fold the resolved user home and HERMES_HOME in a command to ``~/`` / ``~/.hermes/``.

Detection-time normalization for :mod:`tools.approval_detection`: the dangerous-command patterns
name Hermes's own files through ``~/.hermes`` / ``$HERMES_HOME`` spellings, so every spelling a shell
uses for the resolved homes (native, Git Bash ``/c/…``, ``~``, ``$HOME``, ``$LOCALAPPDATA``, …) is
rewritten into those forms first.
"""

import functools
import os
import posixpath
import re
import shlex

from hermes_platform.host import facts


# Shell metacharacters, quotes, and whitespace that terminate a path token.
_PATH_TOKEN_STOP = r"""\s'"`;|&<>()"""
_PATH_TAIL = r"(?P<tail>(?:[/\\][^/\\" + _PATH_TOKEN_STOP + r"]*)+)"


@functools.lru_cache(maxsize=64)
def _home_prefix_fold_regex(path: str):
    """Compile a regex matching *path* as an absolute directory prefix.
    Components match with either separator so native Windows, forward-slash, and mixed forms all
    fold; the caller normalizes the tail's backslashes to ``/``. A non-empty tail is required, so a
    bare home is never folded. Returns ``None`` for an unset/degenerate path (no components, or a
    bare root/drive: ``/``, ``C:``, ``C:\\``, ``""``) so a stray HOME cannot rewrite unrelated
    prefixes.

    A depth-1 home is NOT degenerate: ``/root`` is root's own default HOME, and skipping it left
    ``/root/.ssh/authorized_keys`` unfolded — hence "safe" — while the identical command spelled
    under ``/home/alice`` folded to ``~/.ssh/authorized_keys`` and was denied. A single-component
    home must therefore match a mandatory leading separator, so a relative same-named path
    (``root/file``) is still left alone."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if (not components or (len(components) == 1 and re.fullmatch(r"[A-Za-z]:", components[0]))
            or (path.startswith(("//", "\\\\")) and len(components) < 2)):
        return None
    # Optional leading root separator; a Windows drive letter is a component. Mandatory (and anchored
    # at a path start: ``/data/root/x`` is not ``/data~/x``) for a single-component home, else ``root/file`` folds.
    # A POSIX single-component home splits only on ``/`` or ``\/``: ``/root\*`` is the literal name ``/root*``.
    sep = r"(?:\\?/)" if len(components) == 1 and facts.os_family() != "win32" and path.startswith("/") else r"[/\\]"
    lead = r"(?<![\w.~$}/\\-])" + sep + "+" if len(components) == 1 else sep + "*"
    tail = _PATH_TAIL if sep == r"[/\\]" else r"(?P<tail>(?:\\?/[^/\\" + _PATH_TOKEN_STOP + r"]*)+)"
    return re.compile(lead + (sep + "+").join(re.escape(c) for c in components) + tail)


def _home_path_word_projection(word: str) -> str:
    """Use one separator/escape spelling for both decoding and provenance checks.

    Native-rooted drive/UNC words retain native separators. Forward-drive words
    also accept historical mixed directory separators, but backslashes escaping
    shell syntax remain escapes; a drive prefix alone does not make them native.
    POSIX words keep the shared shell scanner's escape semantics unchanged.
    """
    from tools.approval_detection import _scan_shell, _splice
    unquoted = word.strip("'\"")
    if (re.match(r"^[A-Za-z]:\\", unquoted) or unquoted.startswith("\\\\")
            or re.match(r"^%[A-Za-z_]\w*%\\", unquoted)):
        return word.replace("\\", "/")
    if not re.match(r"^[A-Za-z]:/", unquoted):
        return word
    edits = []
    for kind, i, j, _ in _scan_shell(word):
        if kind == "esc" and word[i + 1] not in "$`*?[]{}'\";&|<>()" and not word[i + 1].isspace():
            edits.append((i, j, "/" + word[i + 1:j].replace("\\", "/")))
        elif kind == "char" and word[i] == "\\":
            edits.append((i, j, "/"))
    return _splice(word, edits)


def _fold_home_prefixes(command: str, paths, replacement: str) -> str:
    """Project whole shell path words, preserving literal and expansion provenance."""
    if not any(marker in command for marker in ("/", "\\", "~", "$", "%")):
        return command
    from tools.approval_detection import _scan_shell, _splice, _strip_shell_word_syntax

    roots = list(dict.fromkeys(
        posixpath.normpath(path.replace("\\", "/"))
        for path in sorted((path for path in paths if path), key=len, reverse=True)
        if _home_prefix_fold_regex(path) is not None
    ))
    edits, start = [], None

    def fold(begin: int, end: int) -> None:
        word = command[begin:end]
        if not any(marker in word for marker in ("/", "\\", "~", "$", "%")):
            return
        provenance = _home_path_word_projection(word)
        value = _strip_shell_word_syntax(provenance)
        assignment = re.match(r"^[A-Za-z_]\w*=", value)
        prefix = assignment.group() if assignment else ""
        value = value[len(prefix):]
        symbolic = re.match(r"^(?:~|\$HOME|\$\{HOME\})(?=/)", value)
        home_syntax = None
        if replacement == "~" and symbolic and paths:
            value = paths[0].replace("\\", "/") + value[symbolic.end():]
            home_syntax = re.match(r"^[\"']?(?:~|\$HOME|\$\{HOME\})(?=/)", word)
        dynamic = any(
            kind == "char" and (home_syntax is None or index >= home_syntax.end())
            and ((quote != "'" and provenance[index] in "$`")
                 or (quote is None and provenance[index] in "*?[]{}"))
            for kind, index, _, quote in _scan_shell(provenance)
        )
        # Keep raw quote boundaries around globs/expansions: the hardline floor needs them.
        if dynamic:
            rendered = word
            for path in sorted((path for path in paths if path), key=len, reverse=True):
                pattern = _home_prefix_fold_regex(path)
                if pattern is not None:
                    rendered = pattern.sub(lambda match: replacement + match.group("tail").replace("\\/", "/").replace("\\", "/"), rendered)
            if rendered != word:
                edits.append((begin, end, rendered))
            return
        value = posixpath.normpath(value)
        for root in roots:
            if value.startswith(root.rstrip("/") + "/"):
                folded = prefix + replacement + value[len(root.rstrip("/")):]
                rendered = (shlex.quote(folded) if re.search(r"[\s'\";&|<>()`\\]", folded)
                            or "'" in word or '"' in word else folded)
                edits.append((begin, end, rendered))
                return

    for kind, index, stop, quote in _scan_shell(command, subst="uq", comments=True):
        if kind == "subst" and stop is not None:
            opener = 2 if command.startswith("$(", index) else 1
            body = command[index + opener:stop - 1]
            projected = _fold_home_prefixes(body, paths, replacement)
            if projected != body:
                edits.append((index, stop, command[index:index + opener] + projected + command[stop - 1:stop]))
            if start is None:
                start = index
            continue
        boundary = kind == "comment" or (kind == "char" and quote is None
                                         and (command[index].isspace() or command[index] in ";&|<>()"))
        if boundary:
            if start is not None:
                fold(start, index)
                start = None
        elif start is None:
            start = index
    if start is not None:
        fold(start, len(command))
    return _splice(command, edits)


_MSYS_DRIVE_PATH_RE = re.compile(r"^([A-Za-z]):[/\\](.+)$")


def _msys_drive_spellings(path: str) -> list[str]:
    """``/c/Users/me`` (and the capital-drive form) for ``C:\\Users\\me``; empty without a drive."""
    match = _MSYS_DRIVE_PATH_RE.match(path) if path else None
    if match is None:
        return []
    letter, rest = match.group(1), match.group(2)
    return [f"/{letter.lower()}/{rest}", f"/{letter.upper()}/{rest}"]


def _path_below(root: str, path: str) -> str | None:
    """*path* relative to *root* when it sits underneath (either separator; Windows folds case),
    else ``None``. Only the comparison folds, so the caller gets the tail as it was written."""
    root_parts = [p for p in re.split(r"[/\\]+", root) if p]
    path_parts = [p for p in re.split(r"[/\\]+", path) if p]
    if not root_parts or len(path_parts) <= len(root_parts):
        return None
    head, wanted = path_parts[:len(root_parts)], root_parts
    if facts.os_family() == "win32":
        head, wanted = [p.lower() for p in head], [p.lower() for p in wanted]
    if head != wanted:
        return None
    return "/".join(path_parts[len(root_parts):])


def _home_spellings(paths, command: str) -> list[str]:
    """Every spelling a shell uses for one of *paths*: as written (native and Git Bash drive form)
    and the same location written through a home variable — ``~``, ``$HOME``, ``$LOCALAPPDATA`` /
    ``%LOCALAPPDATA%``, ``$USERPROFILE``. Windows's default HERMES_HOME nests under the profile,
    which Git Bash reaches as ``/c/Users/<me>/AppData/Local/hermes``; folding only the absolute
    native form left that spelling — and its ``~/…`` and ``$LOCALAPPDATA/…`` siblings — unmatched.
    The variable spellings are only derived when *command* writes one of ``~``, ``$``, ``%``, so a
    command with no variable syntax costs nothing beyond its own absolute prefixes."""
    absolute = list(dict.fromkeys(s for path in paths for s in (path, *_msys_drive_spellings(path))))
    if not any(marker in command for marker in ("~", "$", "%")):
        return absolute  # no variable syntax, so no variable spelling can match either
    # `~` is $HOME in Git Bash, so the variable wins over the native reading of it.
    home = os.environ.get("HOME") or os.path.expanduser("~")
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    roaming = os.environ.get("APPDATA", "")
    profile = os.environ.get("USERPROFILE", "")
    named = (("HOME", home), ("LOCALAPPDATA", local_appdata), ("APPDATA", roaming), ("USERPROFILE", profile))
    roots = [("~", home)] + [(spelled, root) for name, root in named
                             for spelled in (f"${name}", f"${{{name}}}", f"%{name}%")]
    relative = [f"{prefix}/{tail}"
                for prefix, root in roots
                for spelled_root in dict.fromkeys([root, *_msys_drive_spellings(root)])
                for candidate in absolute if (tail := _path_below(spelled_root, candidate))]
    return absolute + list(dict.fromkeys(relative))


def _rewrite_resolved_user_home(command: str) -> str:
    """User home (expanduser / realpath / $HOME) -> ``~/``; no-op when unset, degenerate, or unresolvable."""
    try:
        # expanduser, realpath, and an explicit HOME — Windows expanduser uses USERPROFILE, not HOME.
        home = os.path.expanduser("~")
        paths = [home, os.path.realpath(home), os.environ.get("HOME", "")]
        paths += [spelling for path in list(paths) if path for spelling in _msys_drive_spellings(path)]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~")


def _rewrite_resolved_hermes_home(command: str) -> str:
    """Resolved HERMES_HOME -> ``~/.hermes/`` so the _HERMES_CONFIG_PATH / _HERMES_ENV_PATH rules
    match Docker/gateway deployments and the platform default in every spelling a shell uses for it
    (native, Git Bash ``/c/…``, ``~/AppData/Local/hermes``, ``$LOCALAPPDATA/hermes``)."""
    try:
        from hermes_constants import get_default_hermes_root, get_hermes_home
        home = get_hermes_home().expanduser()
        root = get_default_hermes_root()
        paths = _home_spellings([str(home), str(home.resolve(strict=False)), str(root), str(root.resolve(strict=False))], command)
    except Exception:
        return command
    command = _fold_home_prefixes(command, paths, "~/.hermes")
    # The prefix fold needs a child tail; fold the bare home word too (`mv /opt/data /tmp/x`) so the
    # Hermes-home rules see it.
    for path in dict.fromkeys(sorted((p for p in paths if p), key=len, reverse=True)):
        pattern = _bare_home_fold_regex(path)
        if pattern is not None:
            command = pattern.sub("~/.hermes", command)
    return command


@functools.lru_cache(maxsize=64)
def _bare_home_fold_regex(path: str):
    """*path* as a whole shell word (optional trailing separator); ``None`` when degenerate."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if len(components) < 2:
        return None
    return re.compile(
        r"(?<![^\s'\"`=;|&<>(])[/\\]*" + r"[/\\]+".join(re.escape(c) for c in components)
        + r"[/\\]?(?=[" + _PATH_TOKEN_STOP + r"*]|$)")
