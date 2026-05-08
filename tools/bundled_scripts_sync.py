"""Sync bundled helper scripts into the user's command path.

Hermes ships a few operational helper scripts under ``scripts/``.  Users often
run them from cron, so updates should refresh the command path symlink instead
of leaving stale copies behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from hermes_cli.config import get_hermes_home

BUNDLED_SCRIPTS: list[tuple[str, str]] = [
    ("scripts/telegram-healthcheck-stateful.sh", "telegram-healthcheck-stateful"),
]
SCRIPTS_MANIFEST = get_hermes_home() / ".bundled_scripts_manifest"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _command_link_dir() -> Path:
    prefix = os.environ.get("PREFIX", "").strip()
    if prefix:
        return Path(prefix) / "bin"
    return Path.home() / ".local" / "bin"


def _read_manifest() -> dict[str, str]:
    if not SCRIPTS_MANIFEST.exists():
        return {}
    result: dict[str, str] = {}
    try:
        for line in SCRIPTS_MANIFEST.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            name, _, target = line.partition(":")
            if name and target:
                result[name] = target
    except OSError:
        return {}
    return result


def _write_manifest(entries: dict[str, str]) -> None:
    SCRIPTS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(f"{name}:{target}" for name, target in sorted(entries.items())) + "\n"
    tmp = SCRIPTS_MANIFEST.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(SCRIPTS_MANIFEST)


def sync_bundled_scripts(quiet: bool = False) -> dict:
    root = _project_root()
    dest_dir = _command_link_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()

    linked: List[str] = []
    updated: List[str] = []
    missing: List[str] = []

    for rel_src, dest_name in BUNDLED_SCRIPTS:
        src = root / rel_src
        dest = dest_dir / dest_name
        desired_target = str(src)

        if not src.exists():
            missing.append(dest_name)
            continue

        src.chmod(src.stat().st_mode | 0o111)

        changed = True
        if dest.is_symlink():
            try:
                changed = dest.resolve() != src.resolve()
            except OSError:
                changed = True
        elif dest.exists():
            changed = True

        if dest.exists() or dest.is_symlink():
            if changed:
                dest.unlink()
                dest.symlink_to(src)
                updated.append(dest_name)
        else:
            dest.symlink_to(src)
            linked.append(dest_name)

        manifest[dest_name] = desired_target

    _write_manifest(manifest)
    return {
        "linked": linked,
        "updated": updated,
        "missing": missing,
        "destination_dir": str(dest_dir),
    }


if __name__ == "__main__":
    result = sync_bundled_scripts(quiet=False)
    for name in result["linked"]:
        print(f"+ linked {name}")
    for name in result["updated"]:
        print(f"↻ updated {name}")
    for name in result["missing"]:
        print(f"! missing source for {name}")
