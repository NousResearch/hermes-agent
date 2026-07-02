#!/usr/bin/env python3
"""Compile wiki `## Agent quick context` sections into git-tracked Hermes skills.

The wiki remains the source of truth. Generated skills are fast runtime caches that
Hermes can route to from the initial prompt. By default this writes skills under
`skills/<category>/<name>/SKILL.md`; use `--profile-symlink-root` to symlink those
repo skills into an active Hermes profile.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIKI_ROOT = REPO_ROOT / "yallaplay-wiki"
DEFAULT_SKILLS_ROOT = REPO_ROOT / "skills"
DEFAULT_SECTION = "Agent quick context"
GENERATED_START = "<!-- BEGIN GENERATED FROM WIKI AGENT QUICK CONTEXT -->"
GENERATED_END = "<!-- END GENERATED FROM WIKI AGENT QUICK CONTEXT -->"


@dataclass(frozen=True)
class WikiSkillSource:
    wiki_path: Path
    wiki_rel: str
    frontmatter: dict[str, Any]
    section_title: str
    section_body: str

    @property
    def skill_config(self) -> dict[str, Any]:
        config = self.frontmatter.get("agent_skill")
        if not isinstance(config, dict):
            raise ValueError(f"{self.wiki_rel}: agent_skill frontmatter must be a mapping")
        return config

    @property
    def skill_name(self) -> str:
        name = self.skill_config.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.wiki_rel}: agent_skill.name is required")
        return name.strip()

    @property
    def category(self) -> str:
        category = self.skill_config.get("category", "yallaplay")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{self.wiki_rel}: agent_skill.category must be a string")
        return category.strip().strip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--skills-root", type=Path, default=DEFAULT_SKILLS_ROOT)
    parser.add_argument("--section", default=DEFAULT_SECTION)
    parser.add_argument("--name", help="compile only one agent_skill.name")
    parser.add_argument("--check", action="store_true", help="fail if generated files or symlinks are stale; do not write")
    parser.add_argument("--list", action="store_true", help="list compilable wiki skill sources")
    parser.add_argument(
        "--profile-symlink-root",
        type=Path,
        help="optional Hermes profile skill category directory, e.g. ~/.hermes/profiles/claudio-lab/skills/claudio-authored",
    )
    parser.add_argument("--no-symlink", action="store_true", help="skip symlink creation even if --profile-symlink-root is set")
    return parser.parse_args()


def split_frontmatter(text: str, rel: str) -> tuple[dict[str, Any], str, str]:
    if not text.startswith("---\n"):
        return {}, "", text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{rel}: frontmatter is not closed")
    raw = text[4:end]
    body = text[end + len("\n---\n") :]
    parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{rel}: frontmatter must parse to a mapping")
    return parsed, raw, body


def heading_level(line: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
    return len(match.group(1)) if match else None


def extract_section(markdown_body: str, section_title: str) -> str | None:
    wanted = section_title.strip().lower()
    lines = markdown_body.splitlines()
    start: int | None = None
    start_level: int | None = None
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        title = match.group(2).strip().rstrip("#").strip().lower()
        if title == wanted:
            start = index + 1
            start_level = len(match.group(1))
            break
    if start is None or start_level is None:
        return None
    end = len(lines)
    for index in range(start, len(lines)):
        level = heading_level(lines[index])
        if level is not None and level <= start_level:
            end = index
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def iter_sources(wiki_root: Path, section: str) -> list[WikiSkillSource]:
    sources: list[WikiSkillSource] = []
    for path in sorted(wiki_root.rglob("*.md")):
        if any(part == ".git" for part in path.parts):
            continue
        rel = path.relative_to(wiki_root).as_posix()
        text = path.read_text(encoding="utf-8")
        if "agent_skill:" not in text:
            continue
        frontmatter, _raw_frontmatter, body = split_frontmatter(text, rel)
        if "agent_skill" not in frontmatter:
            continue
        section_body = extract_section(body, section)
        if section_body is None:
            raise ValueError(f"{rel}: has agent_skill frontmatter but no ## {section} section")
        sources.append(WikiSkillSource(path, rel, frontmatter, section, section_body))
    return sources


def as_string_list(value: Any, *, field: str, rel: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{rel}: {field} must be a list of strings")
    return value


def render_skill(source: WikiSkillSource) -> str:
    config = source.skill_config
    name = source.skill_name
    description = config.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{source.wiki_rel}: agent_skill.description is required")
    if len(description) > 1024:
        raise ValueError(f"{source.wiki_rel}: agent_skill.description exceeds 1024 chars")
    tags = as_string_list(config.get("tags"), field="agent_skill.tags", rel=source.wiki_rel)
    related = as_string_list(config.get("related_skills"), field="agent_skill.related_skills", rel=source.wiki_rel)
    checksum = hashlib.sha256(source.section_body.encode("utf-8")).hexdigest()
    built = date.today().isoformat()
    title = source.frontmatter.get("title") or name.replace("-", " ").title()

    frontmatter = {
        "name": name,
        "description": description.strip(),
        "version": "1.0.0",
        "author": "YallaPlay",
        "license": "Proprietary",
        "metadata": {
            "hermes": {
                "tags": tags,
                "related_skills": related,
            },
            "generated_from": {
                "wiki_path": f"yallaplay-wiki/{source.wiki_rel}",
                "section": source.section_title,
                "source_sha256": checksum,
                "built": built,
            },
        },
    }
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"""---
{yaml_text}
---

# {title} — Compiled Agent Context

{GENERATED_START}

> Generated from `yallaplay-wiki/{source.wiki_rel}` → `## {source.section_title}`.
> Do not hand-edit this compiled body; update the wiki section and rerun `python3 tools/compile_wiki_skills.py`.

{source.section_body.rstrip()}

{GENERATED_END}

## Maintenance

- Canonical source: `yallaplay-wiki/{source.wiki_rel}`.
- If this skill and the wiki disagree, the wiki wins; patch the wiki, then regenerate this skill.
- Keep this skill symlinked into the active Hermes profile so `/skills` and prompt routing see the git-tracked copy.
"""


def skill_path(skills_root: Path, source: WikiSkillSource) -> Path:
    return skills_root / source.category / source.skill_name / "SKILL.md"


def ensure_symlink(profile_root: Path, source: WikiSkillSource, target_dir: Path, *, check: bool) -> bool:
    link = profile_root / source.skill_name
    desired = target_dir.resolve()
    if link.is_symlink() and link.resolve() == desired:
        return False
    if check:
        print(f"stale/missing symlink: {link} -> {desired}", file=sys.stderr)
        return True
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace non-matching profile skill path: {link}")
    profile_root.mkdir(parents=True, exist_ok=True)
    link.symlink_to(desired, target_is_directory=True)
    print(f"symlinked {link} -> {desired}")
    return True


def main() -> int:
    args = parse_args()
    wiki_root = args.wiki_root.resolve()
    skills_root = args.skills_root.resolve()
    sources = iter_sources(wiki_root, args.section)
    if args.name:
        sources = [source for source in sources if source.skill_name == args.name]
    if args.list:
        for source in sources:
            print(f"{source.skill_name}\t{source.category}\t{source.wiki_rel}")
        return 0
    if not sources:
        print("no wiki agent_skill sources found", file=sys.stderr)
        return 1

    changed = False
    for source in sources:
        rendered = render_skill(source)
        path = skill_path(skills_root, source)
        existing = path.read_text(encoding="utf-8") if path.exists() else None
        if existing != rendered:
            changed = True
            if args.check:
                print(f"stale generated skill: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rendered, encoding="utf-8")
                print(f"wrote {path.relative_to(REPO_ROOT)}")
        elif not args.check:
            print(f"up to date {path.relative_to(REPO_ROOT)}")

        if args.profile_symlink_root and not args.no_symlink:
            changed = ensure_symlink(
                args.profile_symlink_root.expanduser().resolve(),
                source,
                path.parent,
                check=args.check,
            ) or changed

    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
