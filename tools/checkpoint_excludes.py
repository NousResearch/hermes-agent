"""The user's own git ignore rules, applied to checkpoint staging.

The shadow store runs git with ``GIT_DIR=<store>`` and the user's global
config hidden, so ``git add -A`` sees only the working tree's ``.gitignore``
files. Rules the user keeps in ``.git/info/exclude`` or a configured
``core.excludesFile`` were invisible: those files were snapshotted into the
store and a rollback overwrote them with old content.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from hermes_cli._subprocess_compat import selected_git_env, windows_hide_flags

_TIMEOUT = 10


def _user_git(args: list[str], working_dir: str) -> str:
    """Run a read-only git query in the user's repo with the user's config."""
    env = {k: v for k, v in selected_git_env().items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    git = shutil.which("git", path=env.get("PATH", ""))
    if git is None:
        return ""
    try:
        result = subprocess.run(
            [git, "-C", working_dir, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=_TIMEOUT,
            stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(), check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    out = result.stdout.strip() if result.returncode == 0 else ""
    return "" if "\n" in out else out


def _user_exclude_sources(working_dir: str) -> list[Path]:
    """``core.excludesFile`` (or git's XDG default) plus the repo's ``info/exclude``."""
    sources: list[Path] = []
    configured = _user_git(["config", "--path", "--get", "core.excludesFile"], working_dir)
    if configured:
        sources.append(Path(configured).expanduser())
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        sources.append(Path(xdg) / "git" / "ignore")
    info = _user_git(["rev-parse", "--path-format=absolute", "--git-path", "info/exclude"], working_dir)
    if info:
        sources.append(Path(info))
    return sources


def staging_config_args(store: Path, dir_hash: str, working_dir: str) -> list[str]:
    """``-c core.excludesFile=…`` for a shadow-store ``git add``, or ``[]``.

    Overriding ``core.excludesFile`` replaces git's XDG default, so the merged
    file carries that default too. The store's own ``info/exclude`` (Hermes'
    DEFAULT_EXCLUDES) still applies on top.
    """
    chunks: list[str] = []
    for src in _user_exclude_sources(working_dir):
        try:
            chunks.append(src.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError:
            continue
    merged_text = "\n".join(chunks).strip()
    if not merged_text:
        return []
    merged = store / "excludes" / dir_hash
    try:
        merged.parent.mkdir(parents=True, exist_ok=True)
        existing: Optional[str] = merged.read_text(encoding="utf-8-sig") if merged.exists() else None
        if existing != merged_text + "\n":
            merged.write_text(merged_text + "\n", encoding="utf-8")
    except OSError:
        return []
    return ["-c", f"core.excludesFile={merged}"]
