"""Case-sensitivity probes and case-only write-target detection."""

from __future__ import annotations

import ast
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Optional


def filesystem_is_case_sensitive(directory: Path) -> bool:
    """Return the actual case behavior of the filesystem containing *directory*."""
    directory = Path(directory)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    if not directory.is_dir():
        return True
    fd, first_name = tempfile.mkstemp(prefix=".hermes-case-probe-", dir=directory)
    os.close(fd)
    first = Path(first_name)
    second = first.with_name(first.name.swapcase())
    try:
        return not second.exists()
    finally:
        if second.exists() and second != first:
            second.unlink()
        first.unlink(missing_ok=True)


def _case_variant(path: Path) -> Optional[Path]:
    """Return the existing spelling of a missing path component, if any."""
    if path.exists():
        return None
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, component in enumerate(parts):
        if not current.is_dir():
            return None
        exact = current / component
        if exact.exists():
            current = exact
            continue
        matches = [entry for entry in current.iterdir()
                   if entry.name.casefold() == component.casefold()]
        if len(matches) != 1 or matches[0].name == component:
            return None
        return matches[0].joinpath(*parts[index + 1:])
    return None


def case_conflict_for_path(path: Path) -> Optional[Path]:
    """Return a case-only existing path when a write target is ambiguous."""
    path = Path(path)
    if path.exists() or not filesystem_is_case_sensitive(path.parent):
        return None
    return _case_variant(path)


def _path_arg(value: str, cwd: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cwd) / path


def extract_write_targets(command: str, cwd: str) -> list[Path]:
    """Extract destinations for simple commands with unambiguous targets."""
    # Do not tokenize arbitrary commands. Besides avoiding unnecessary work,
    # this preserves the terminal security parser's fail-closed handling for
    # oversized or malformed command strings.
    if not re.match(
        r"^\s*(?:/[^\s;|]+/)?(?:mkdir|touch|install|mv|cp)(?:\s|$)",
        command,
    ):
        return []
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return []
    if not tokens:
        return []
    command_name = Path(tokens[0]).name
    candidates = [token for token in tokens[1:] if not token.startswith("-")]
    if command_name == "mkdir":
        return [_path_arg(token, cwd) for token in candidates]
    if command_name in {"touch", "install"}:
        return [_path_arg(candidates[-1], cwd)] if candidates else []
    if command_name in {"mv", "cp"}:
        return [_path_arg(candidates[-1], cwd)] if len(candidates) >= 2 else []
    return []


def _case_conflict_script(targets: list[Path]) -> str:
    """Build a self-contained probe for execution inside the target backend."""
    payload = repr([str(path) for path in targets])
    return f"""
import os, pathlib, tempfile
items = {payload}
out = []
for raw in items:
    p = pathlib.Path(raw)
    if p.exists():
        continue
    parent = p.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        continue
    try:
        fd, probe_name = tempfile.mkstemp(prefix='.hermes-case-probe-', dir=str(parent))
        os.close(fd)
        probe = pathlib.Path(probe_name)
        variant = probe.with_name(probe.name.swapcase())
        insensitive = variant.exists()
        if variant.exists() and variant != probe:
            variant.unlink()
        probe.unlink(missing_ok=True)
    except OSError:
        continue
    if insensitive:
        continue
    current = pathlib.Path(p.anchor)
    parts = p.parts[1:]
    for index, component in enumerate(parts):
        if not current.is_dir():
            break
        exact = current / component
        if exact.exists():
            current = exact
            continue
        matches = [entry for entry in current.iterdir()
                   if entry.name.casefold() == component.casefold()]
        if len(matches) == 1 and matches[0].name != component:
            existing = matches[0].joinpath(*parts[index + 1:])
            out.append({{'requested_path': str(p), 'existing_path': str(existing)}})
        break
print(repr(out))
"""


def check_case_conflicts(env: Any, command: str, cwd: str) -> list[dict[str, str]]:
    """Check simple write targets inside the target terminal environment.

    The probe runs inside *env*, so remote and container filesystems are tested
    where the command will execute. Unknown shell syntax is left unchanged.
    """
    targets = extract_write_targets(command, cwd)
    if not targets:
        return []
    result = env.execute(
        f"python3 -c {shlex.quote(_case_conflict_script(targets))}",
        cwd=cwd,
    )
    if result.get("returncode", 1) != 0:
        return []
    try:
        findings = ast.literal_eval(result.get("output", "").strip())
    except (SyntaxError, TypeError, ValueError):
        return []
    return findings if isinstance(findings, list) else []


def rewrite_target_path(
    command: str, requested: Path, existing: Path, cwd: Optional[str] = None
) -> str:
    """Replace one destination spelling with the existing spelling."""
    candidates = [os.fspath(requested)]
    if cwd:
        try:
            relative = requested.relative_to(Path(cwd))
            candidates.extend((os.fspath(relative), f"./{relative}"))
        except ValueError:
            pass
    for candidate in candidates:
        for spelling in (candidate, candidate + os.sep):
            if spelling not in command:
                continue
            replacement = os.fspath(existing)
            if not spelling.startswith(("/", "\\")):
                try:
                    replacement = os.fspath(existing.relative_to(Path(cwd)))
                except (ValueError, TypeError):
                    pass
            if spelling.endswith(os.sep):
                replacement += os.sep
            return command.replace(spelling, replacement, 1)
    return command
