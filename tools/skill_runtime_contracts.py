"""Runtime dependency checks for skill deletion.

Curator can consolidate or prune skills automatically, but deleting a skill is
only safe after runtime entrypoints that invoke Hermes by skill name have been
migrated. This module keeps that check small and deterministic so it can run
before the irreversible filesystem delete.
"""

from __future__ import annotations

import ast
import os
import shlex
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from hermes_constants import get_hermes_home

_MAX_SCAN_BYTES = 1_000_000
_RUNTIME_DIR_NAMES = ("bin", "scripts")
_SHELL_SUFFIXES = {
    ".bash",
    ".fish",
    ".sh",
    ".zsh",
}
_SECRET_NAMES = {".env", "auth.json", "secrets.json"}
_SHELL_SEPARATORS = {";", "&", "&&", "|", "||", "{", "}"}
_SHELL_CONTROL_PREFIXES = {"!", "if", "then", "elif", "else", "while", "until", "do"}
_ENV_OPTIONS_WITH_VALUES = {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}
_EXEC_OPTIONS_WITH_VALUES = {"-a"}
_SUDO_OPTIONS_WITH_VALUES = {
    "-C",
    "--close-from",
    "-D",
    "--chdir",
    "-g",
    "--group",
    "-h",
    "--host",
    "-p",
    "--prompt",
    "-r",
    "--role",
    "-T",
    "--command-timeout",
    "-t",
    "--type",
    "-u",
    "--user",
}
_TIME_OPTIONS_WITH_VALUES = {"-f", "--format", "-o", "--output"}


@dataclass(frozen=True)
class SkillRuntimeReference:
    """A runtime surface that still refers to a skill by name."""

    surface: str
    path: str
    line: Optional[int]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillRuntimeScanError(RuntimeError):
    """A runtime surface could not be inspected before skill deletion."""

    def __init__(self, surface: str, path: str, reason: str) -> None:
        self.surface = surface
        self.path = path
        self.reason = reason
        super().__init__(f"{surface} at {path}: {reason}")

    def to_dict(self) -> dict[str, str]:
        return {
            "surface": self.surface,
            "path": self.path,
            "error": self.reason,
        }


@dataclass(frozen=True)
class _SkillRuntimeTarget:
    """The exact installed skill whose runtime references are being scanned."""

    name: str
    hermes_home: Path
    skill_path: Optional[Path] = None
    skills_root: Optional[Path] = None


def _paths_resolve_equal(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except (OSError, RuntimeError):
        return False


def _skill_identifier_matches(identifier: str, target: _SkillRuntimeTarget) -> bool:
    """Match a bare name or a path that resolves to the exact target skill."""

    raw = str(identifier or "").strip()
    if not raw:
        return False
    if raw == target.name:
        return True
    if target.skill_path is None:
        return False

    # Hermes accepts category-relative and trusted absolute skill paths. Treat
    # backslashes as separators too so the same comparison works on Windows.
    normalized = raw.replace("\\", "/")
    if "/" not in normalized or ".." in Path(normalized).parts:
        return False

    candidate = Path(normalized).expanduser()
    if candidate.is_absolute():
        return _paths_resolve_equal(candidate, target.skill_path)

    roots = [target.skills_root, target.hermes_home / "skills"]
    return any(
        root is not None and _paths_resolve_equal(root / candidate, target.skill_path)
        for root in roots
    )


def _is_runtime_candidate(path: Path, runtime_dir: str) -> bool:
    if path.name in _SECRET_NAMES:
        return False
    # Cron executes every referenced file under scripts/: .sh/.bash via Bash
    # and every other suffix via Python.  Parse all of them according to that
    # same interpreter contract rather than guessing from filename alone.
    if runtime_dir == "scripts":
        return True
    if path.suffix.lower() in _SHELL_SUFFIXES | {".py"}:
        return True
    try:
        return os.access(path, os.X_OK)
    except OSError:
        return False


def _iter_runtime_files(hermes_home: Path) -> Iterable[tuple[str, Path]]:
    for dirname in _RUNTIME_DIR_NAMES:
        root = hermes_home / dirname
        surface = f"hermes.{dirname}"
        try:
            root_stat = root.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SkillRuntimeScanError(
                surface,
                str(root),
                f"could not inspect runtime directory: {exc}",
            ) from exc
        if not stat.S_ISDIR(root_stat.st_mode):
            continue

        def _walk_error(exc: OSError) -> None:
            raise SkillRuntimeScanError(
                surface,
                str(exc.filename or root),
                f"could not inspect runtime directory: {exc}",
            ) from exc

        for dirpath, _dirnames, filenames in os.walk(root, onerror=_walk_error):
            for filename in filenames:
                path = Path(dirpath) / filename
                if not _is_runtime_candidate(path, dirname):
                    continue
                try:
                    path_stat = path.stat()
                except FileNotFoundError:
                    # A concurrently removed file cannot remain a dependency.
                    continue
                except OSError as exc:
                    raise SkillRuntimeScanError(
                        surface,
                        str(path),
                        f"could not inspect runtime file: {exc}",
                    ) from exc
                if not stat.S_ISREG(path_stat.st_mode):
                    continue
                size = path_stat.st_size
                if size > _MAX_SCAN_BYTES:
                    raise SkillRuntimeScanError(
                        surface,
                        str(path),
                        f"runtime file exceeds the {_MAX_SCAN_BYTES}-byte scan limit",
                    )
                yield dirname, path


def _is_assignment(token: str) -> bool:
    """Return whether a shell token is an environment assignment."""

    if "=" not in token:
        return False
    name, _value = token.split("=", 1)
    return (
        bool(name)
        and (name[0].isalpha() or name[0] == "_")
        and all(char.isalnum() or char == "_" for char in name[1:])
    )


def _command_name(token: str) -> str:
    """Return a command basename for POSIX or Windows-style paths."""

    return token.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _skip_prefix_options(
    tokens: Sequence[str],
    index: int,
    options_with_values: set[str],
) -> int:
    """Skip wrapper options, including known options with separate values."""

    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return index + 1
        if token == "-" or not token.startswith("-"):
            return index

        option = token.split("=", 1)[0]
        index += 1
        if option in options_with_values and "=" not in token and index < len(tokens):
            index += 1
    return index


def _command_start(tokens: Sequence[str]) -> Optional[int]:
    """Return the executable index after supported shell launch prefixes."""

    index = 0
    while index < len(tokens):
        while index < len(tokens) and (
            tokens[index] in _SHELL_CONTROL_PREFIXES or _is_assignment(tokens[index])
        ):
            index += 1
        if index >= len(tokens):
            return None

        command = _command_name(tokens[index])
        if command == "env":
            index = _skip_prefix_options(tokens, index + 1, _ENV_OPTIONS_WITH_VALUES)
            continue
        if command == "command":
            index += 1
            while index < len(tokens) and tokens[index].startswith("-"):
                option = tokens[index]
                if option in {"-v", "-V"}:
                    return None
                index += 1
                if option == "--":
                    break
            continue
        if command == "exec":
            index = _skip_prefix_options(tokens, index + 1, _EXEC_OPTIONS_WITH_VALUES)
            continue
        if command == "sudo":
            index = _skip_prefix_options(tokens, index + 1, _SUDO_OPTIONS_WITH_VALUES)
            continue
        if command == "nohup":
            index = _skip_prefix_options(tokens, index + 1, set())
            continue
        if command == "time":
            index = _skip_prefix_options(tokens, index + 1, _TIME_OPTIONS_WITH_VALUES)
            continue
        return index
    return None


def _is_hermes_executable(token: str) -> bool:
    return _command_name(token) in {"hermes", "hermes.exe"}


def _hermes_argv(tokens: Sequence[str]) -> Optional[Sequence[str]]:
    """Return Hermes arguments for a supported process invocation.

    This deliberately recognizes only direct Hermes executable calls and the
    standard ``python -m hermes_cli.main`` equivalent.  A textual mention of a
    skill is not a runtime dependency.
    """

    index = _command_start(tokens)
    if index is None:
        return None
    if _is_hermes_executable(tokens[index]):
        return tokens[index + 1 :]

    executable = _command_name(tokens[index])
    if not executable.startswith("python"):
        return None
    index += 1
    while index < len(tokens) and tokens[index] in {"-u", "-B", "-E", "-s"}:
        index += 1
    if (
        index + 1 < len(tokens)
        and tokens[index] == "-m"
        and tokens[index + 1] == "hermes_cli.main"
    ):
        return tokens[index + 2 :]
    return None


def _loads_skill(tokens: Sequence[str], target: _SkillRuntimeTarget) -> bool:
    """Return whether Hermes argv explicitly preloads the target skill."""

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return False
        value: Optional[str] = None
        if token in {"-s", "--skills"} and index + 1 < len(tokens):
            value = tokens[index + 1]
            index += 1
        elif token.startswith("--skills="):
            value = token.split("=", 1)[1]
        elif token.startswith("-s") and token != "-s":
            value = token[2:]

        if value is not None:
            if any(
                _skill_identifier_matches(part, target)
                for part in value.split(",")
                if part.strip()
            ):
                return True
        index += 1
    return False


def _safe_reference_text(skill_name: str) -> str:
    """Describe a match without returning source text that may hold secrets."""

    return f"Hermes invocation preloads skill {skill_name!r}"


def _shell_segments(command: str) -> Optional[List[List[str]]]:
    """Split a shell command into simple command argv segments."""

    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|;&<>{}")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        # An open quote can legitimately continue on a later physical line.
        return None

    segments: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _heredoc_delimiters(tokens: Sequence[str]) -> List[tuple[str, bool]]:
    """Return literal here-document delimiters and whether tabs are stripped."""

    delimiters: List[tuple[str, bool]] = []
    index = 0
    while index + 1 < len(tokens):
        if tokens[index] != "<<":
            index += 1
            continue

        index += 1
        delimiter = tokens[index]
        strip_tabs = False
        if delimiter == "-":
            strip_tabs = True
            index += 1
            if index >= len(tokens):
                break
            delimiter = tokens[index]
        elif delimiter.startswith("-"):
            strip_tabs = True
            delimiter = delimiter[1:]

        if delimiter:
            delimiters.append((delimiter, strip_tabs))
        index += 1
    return delimiters


def _shell_references(text: str, target: _SkillRuntimeTarget) -> List[tuple[int, str]]:
    """Find explicit Hermes skill-loading commands in shell-like source."""

    refs: List[tuple[int, str]] = []
    pending_heredocs: List[tuple[str, bool]] = []
    logical_line = ""
    start_line = 1
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pending_heredocs:
            delimiter, strip_tabs = pending_heredocs[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                pending_heredocs.pop(0)
            continue

        if not logical_line:
            start_line = line_no
        logical_line += line.rstrip("\\\n")
        if line.rstrip().endswith("\\"):
            logical_line += " "
            continue

        segments = _shell_segments(logical_line)
        if segments is None:
            logical_line += "\n"
            continue
        for tokens in segments:
            pending_heredocs.extend(_heredoc_delimiters(tokens))
            argv = _hermes_argv(tokens)
            if argv is not None and _loads_skill(argv, target):
                refs.append((start_line, _safe_reference_text(target.name)))
        logical_line = ""

    if logical_line:
        segments = _shell_segments(logical_line)
        if segments is not None:
            for tokens in segments:
                argv = _hermes_argv(tokens)
                if argv is not None and _loads_skill(argv, target):
                    refs.append((start_line, _safe_reference_text(target.name)))
    return refs


def _literal_string(node: ast.AST) -> Optional[str]:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _literal_argv(node: ast.AST) -> Optional[List[str]]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: List[str] = []
    for item in node.elts:
        value = _literal_string(item)
        if value is None:
            return None
        values.append(value)
    return values


def _python_references(text: str, target: _SkillRuntimeTarget) -> List[tuple[int, str]]:
    """Find explicit subprocess or os.system Hermes invocations in Python."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    refs: List[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        dotted_name = ""
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            dotted_name = f"{node.func.value.id}.{node.func.attr}"
        if dotted_name in {
            "subprocess.run",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
        }:
            argv = _literal_argv(node.args[0])
            if argv is not None:
                hermes_args = _hermes_argv(argv)
                if hermes_args is not None and _loads_skill(hermes_args, target):
                    refs.append((node.lineno, _safe_reference_text(target.name)))
                    continue
            command = _literal_string(node.args[0])
            shell_true = any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            if command is not None and shell_true:
                refs.extend(
                    (node.lineno, snippet)
                    for _line, snippet in _shell_references(command, target)
                )
        elif dotted_name == "os.system":
            command = _literal_string(node.args[0])
            if command is not None:
                refs.extend(
                    (node.lineno, snippet)
                    for _line, snippet in _shell_references(command, target)
                )
    return refs


def _is_python_source(path: Path, text: str) -> bool:
    if path.suffix.lower() == ".py":
        return True
    first_line = text.splitlines()[0] if text else ""
    return first_line.startswith("#!") and "python" in first_line.lower()


def _scan_runtime_file(
    path: Path,
    target: _SkillRuntimeTarget,
    surface: str,
    runtime_dir: str,
) -> List[SkillRuntimeReference]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SkillRuntimeScanError(
            surface,
            str(path),
            f"could not read runtime file: {exc}",
        ) from exc

    if runtime_dir == "scripts":
        # Keep this aligned with cron.scheduler._run_job_script().
        is_python = path.suffix.lower() not in {".sh", ".bash"}
    else:
        is_python = _is_python_source(path, text)

    if is_python:
        matches = _python_references(text, target)
    else:
        matches = _shell_references(text, target)
    return [
        SkillRuntimeReference(
            surface=surface,
            path=str(path),
            line=line_no,
            text=snippet,
        )
        for line_no, snippet in matches
    ]


def _scan_cron_jobs(target: _SkillRuntimeTarget) -> List[SkillRuntimeReference]:
    jobs_file = target.hermes_home / "cron" / "jobs.json"
    try:
        from cron.jobs import _normalize_skill_list, load_jobs, use_cron_store

        with use_cron_store(target.hermes_home):
            jobs = load_jobs()
        refs: List[SkillRuntimeReference] = []
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue
            skills = _normalize_skill_list(job.get("skill"), job.get("skills"))
            if not any(_skill_identifier_matches(item, target) for item in skills):
                continue
            job_id = job.get("id") or f"index:{index}"
            refs.append(
                SkillRuntimeReference(
                    surface="cron.jobs",
                    path=f"{jobs_file}#{job_id}",
                    line=None,
                    text=f"Cron job references skill {target.name!r}",
                )
            )
        return refs
    except Exception as exc:
        raise SkillRuntimeScanError(
            "cron.jobs",
            str(jobs_file),
            f"could not load Cron jobs: {exc}",
        ) from exc


def find_skill_runtime_references(
    skill_name: str,
    hermes_home: Optional[Path] = None,
    skill_path: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> List[SkillRuntimeReference]:
    """Find runtime entrypoints that refer to ``skill_name``.

    Cron and script references are blocking until their consumers have been
    updated.  Cron rewrites occur after a curator run and cannot make a direct
    deletion safe.  Script references are recognized only when they are actual
    Hermes process invocations, not ordinary text that mentions a skill name.

    Raises ``SkillRuntimeScanError`` when a runtime surface cannot be inspected;
    callers must not interpret an incomplete scan as safe deletion.
    """

    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    target = _SkillRuntimeTarget(
        name=skill_name,
        hermes_home=home,
        skill_path=Path(skill_path) if skill_path is not None else None,
        skills_root=Path(skills_root) if skills_root is not None else None,
    )
    refs: List[SkillRuntimeReference] = []

    refs.extend(_scan_cron_jobs(target))
    for runtime_dir, file_path in _iter_runtime_files(home):
        surface = f"hermes.{runtime_dir}"
        refs.extend(_scan_runtime_file(file_path, target, surface, runtime_dir))

    return refs


def blocking_skill_runtime_references(
    skill_name: str,
    hermes_home: Optional[Path] = None,
    skill_path: Optional[Path] = None,
    skills_root: Optional[Path] = None,
) -> List[SkillRuntimeReference]:
    """Return references that must be cleared before deletion."""

    return find_skill_runtime_references(
        skill_name,
        hermes_home=hermes_home,
        skill_path=skill_path,
        skills_root=skills_root,
    )
