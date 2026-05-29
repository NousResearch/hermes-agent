from __future__ import annotations

import os
from pathlib import Path

from hermes_constants import get_hermes_home


def is_curator_sandbox_mode() -> bool:
    return str(os.getenv("HERMES_CURATOR_SANDBOX_MODE", "")).strip() == "1"


def get_sandbox_root() -> Path | None:
    skills = os.getenv("HERMES_SKILLS_ROOT", "").strip()
    if not skills:
        return None
    p = Path(skills).expanduser().resolve()
    # expected .../.hermes_bg_curator_lab/sandbox/skills
    return p.parent if p.name == "skills" else p


def get_skills_root() -> Path:
    override = os.getenv("HERMES_SKILLS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (get_hermes_home() / "skills").resolve()


def get_usage_path() -> Path:
    override = os.getenv("HERMES_USAGE_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return get_skills_root() / ".usage.json"


def get_cron_root() -> Path:
    override = os.getenv("HERMES_CRON_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (get_hermes_home() / "cron").resolve()


def resolve_under(root: Path, candidate: Path) -> Path:
    return (root / candidate).resolve()


def assert_under_root(candidate: Path, allowed_root: Path, operation_label: str) -> Path:
    c = candidate.resolve()
    r = allowed_root.resolve()
    try:
        c.relative_to(r)
    except Exception as exc:
        raise ValueError(
            f"{operation_label}: path escapes allowed root; candidate={c} allowed_root={r} err={exc}"
        )
    return c


def assert_sandbox_write_path(candidate: Path, category: str) -> Path:
    if not is_curator_sandbox_mode():
        return candidate.resolve()

    if category == "skills":
        root = get_skills_root()
    elif category == "usage":
        root = get_usage_path().parent
    elif category == "cron":
        root = get_cron_root()
    else:
        root = get_sandbox_root() or candidate.parent

    return assert_under_root(candidate, root, f"sandbox-write[{category}]")
