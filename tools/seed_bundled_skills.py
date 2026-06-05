#!/usr/bin/env python3
"""Seed ~/.hermes/skills/.bundled_manifest at install time.

This script is invoked once by the Hermes installer (scripts/install.sh,
scripts/install.ps1, scripts/install.cmd) immediately after the
``~/.hermes/`` directory tree is created.  It enumerates every SKILL.md
shipped inside the repo (``skills/`` and ``optional-skills/``) and writes
``~/.hermes/skills/.bundled_manifest.json``: a rich JSON sidecar
enumerating ``source`` per entry (``workspace-pack`` | ``hermes-skill`` |
``upstream-hermes-skill``).

The legacy text manifest ``~/.hermes/skills/.bundled_manifest``
(``name:hash`` per line, consumed by ``tools/skill_usage.py``) is owned
by ``tools/skills_sync.py`` and is left untouched here.  This seeder
adds the *source-classified* sidecar that companion catalogs
(intake ``hermes-202606-006``) needs for read-only bundled-vs-hub-vs-user
grouping.  Desktop never writes ``~/.hermes/skills/*`` directly
(desktop boundary policy 2026-05-23); that is why this seeder runs from
the Hermes sidecar/install side, not from the desktop runtime.

Idempotent: safe to re-run.  Atomic writes via tempfile + os.replace so
a partial write never leaves a half-formed manifest behind.

Usage::

    python3 tools/seed_bundled_skills.py [--hermes-home PATH] [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger("seed_bundled_skills")

# Source classification: which subtree did the SKILL.md come from?
SOURCE_WORKSPACE_PACK = "workspace-pack"            # shipped inside the workspace (default skills/ tree)
SOURCE_HERMES_SKILL = "hermes-skill"                 # default Hermes skill (skills/ subtree)
SOURCE_UPSTREAM_HERMES_SKILL = "upstream-hermes-skill"  # optional, opt-in (optional-skills/)

DEFAULT_SUBTREES = (
    ("skills", SOURCE_HERMES_SKILL),
    ("optional-skills", SOURCE_UPSTREAM_HERMES_SKILL),
)

# When ~/.hermes/skill-bundles/<slug>.yaml is present, those entries belong
# to "workspace-pack".  We do not seed them here (the bundle resolver does);
# the constant is exposed so downstream tooling has a canonical name.
SOURCE_WORKSPACE_PACK_DIRNAME = "skill-bundles"


@dataclass(frozen=True)
class BundledSkill:
    name: str
    sha256: str
    source: str
    rel_path: str  # repo-relative path to SKILL.md, e.g. "skills/email/gmail/SKILL.md"


def _resolve_hermes_home(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".hermes"


def _resolve_repo_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    # Default: parent of this file's directory (tools/ -> repo root).
    return Path(__file__).resolve().parent.parent


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Read the ``name:`` field from the YAML front-matter, fallback to dir name."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    # Cheap YAML front-matter parse (no PyYAML dep at install time).
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return fallback
    for line in lines[1:]:
        s = line.strip()
        if s == "---":
            break
        if s.startswith("name:"):
            value = s.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return fallback


def _hash_skill_dir(skill_md: Path) -> str:
    """Stable SHA-256 over SKILL.md content (one hash per skill).

    We hash only SKILL.md (not the full directory) to keep this fast and
    deterministic across platforms.  Re-hashing the entire skill tree is the
    Skills Hub's job, not the installer's.
    """
    h = hashlib.sha256()
    try:
        with skill_md.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError as e:
        logger.warning("could not hash %s: %s", skill_md, e)
        return ""
    return h.hexdigest()


def discover_bundled_skills(repo_root: Path) -> List[BundledSkill]:
    found: List[BundledSkill] = []
    seen_names: set[str] = set()
    for subdir, source in DEFAULT_SUBTREES:
        base = repo_root / subdir
        if not base.is_dir():
            continue
        for skill_md in sorted(base.rglob("SKILL.md")):
            try:
                rel = skill_md.relative_to(repo_root)
            except ValueError:
                continue
            # Skip anything under a dot-dir (e.g. .archive, .hub, .git).
            if any(part.startswith(".") for part in rel.parts):
                continue
            fallback = skill_md.parent.name
            name = _read_skill_name(skill_md, fallback=fallback)
            # Collisions: first-wins (skills/ before optional-skills/ per DEFAULT_SUBTREES order).
            if name in seen_names:
                logger.debug("skipping duplicate skill name %s at %s", name, rel)
                continue
            seen_names.add(name)
            digest = _hash_skill_dir(skill_md)
            if not digest:
                continue
            found.append(
                BundledSkill(
                    name=name,
                    sha256=digest,
                    source=source,
                    rel_path=str(rel).replace(os.sep, "/"),
                )
            )
    return found


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".bundled_manifest.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def render_legacy_text(skills: Iterable[BundledSkill]) -> str:
    lines = [f"{s.name}:{s.sha256}" for s in skills]
    return "\n".join(lines) + ("\n" if lines else "")


def render_json(skills: List[BundledSkill]) -> str:
    payload = {
        "format_version": 1,
        "sources": [SOURCE_WORKSPACE_PACK, SOURCE_HERMES_SKILL, SOURCE_UPSTREAM_HERMES_SKILL],
        "skills": [asdict(s) for s in skills],
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def seed(hermes_home: Path, repo_root: Path) -> List[BundledSkill]:
    skills = discover_bundled_skills(repo_root)
    skills_dir = hermes_home / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(skills_dir / ".bundled_manifest.json", render_json(skills))
    return skills


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--hermes-home", help="Override ~/.hermes (default: $HERMES_HOME or ~/.hermes)")
    p.add_argument("--repo-root", help="Override repo root (default: parent of tools/)")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress info logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    hermes_home = _resolve_hermes_home(args.hermes_home)
    repo_root = _resolve_repo_root(args.repo_root)
    logger.info("seeding bundled manifest under %s from %s", hermes_home, repo_root)
    skills = seed(hermes_home, repo_root)
    logger.info("wrote %d entries to %s/skills/.bundled_manifest{,.json}", len(skills), hermes_home)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
