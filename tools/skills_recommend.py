#!/usr/bin/env python3
"""Project stack detection and skill recommendation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.skill_utils import find_git_root, get_project_local_skills_dirs, parse_frontmatter
from tools.skills_hub import create_source_router


RECOMMENDATION_MAP: dict[str, dict[str, list[str]]] = {
    "react": {
        "hub": ["vercel-labs/agent-skills/vercel-react-best-practices"],
    },
    "nextjs": {
        "hub": ["vercel-labs/next-skills/next-best-practices"],
    },
    "playwright": {
        "hub": ["currents-dev/playwright-best-practices-skill/playwright-best-practices"],
    },
    "docker": {
        "official": ["official/devops/docker-management"],
    },
    "fastapi": {
        "hub": ["jezweb/claude-skills/fastapi"],
    },
}

COMBO_RECOMMENDATION_MAP: dict[tuple[str, ...], dict[str, list[str]]] = {
    ("nextjs", "react"): {
        "hub": ["vercel-labs/next-skills/next-best-practices"],
    },
    ("nextjs", "playwright"): {
        "hub": ["currents-dev/playwright-best-practices-skill/playwright-best-practices"],
    },
}

_TRUST_RANK = {"builtin": 3, "trusted": 2, "community": 1}


@dataclass
class DetectedProject:
    root: Path
    technologies: set[str]
    combos: set[str]
    available_project_skills: list[dict[str, Any]]


def _normalize_path(path: str | Path | None) -> Path:
    if path is None:
        candidate = Path.cwd()
    else:
        candidate = Path(path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    return candidate.resolve()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_package_json(root: Path) -> dict[str, Any]:
    package_json = root / "package.json"
    if not package_json.exists():
        return {}
    try:
        return json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dep_names(package_json: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        block = package_json.get(key)
        if isinstance(block, dict):
            names.update(str(name) for name in block.keys())
    return names


def _detect_technologies(root: Path) -> set[str]:
    techs: set[str] = set()
    package_json = _load_package_json(root)
    deps = _dep_names(package_json)

    if "next" in deps or any((root / name).exists() for name in ("next.config.js", "next.config.mjs", "next.config.ts")):
        techs.add("nextjs")
    if {"react", "react-dom"} & deps:
        techs.add("react")
    if "typescript" in deps or (root / "tsconfig.json").exists():
        techs.add("typescript")
    if {"tailwindcss", "@tailwindcss/vite"} & deps or any((root / name).exists() for name in ("tailwind.config.js", "tailwind.config.ts", "tailwind.config.cjs")):
        techs.add("tailwind")
    if (root / "components.json").exists():
        techs.add("shadcn")
    if {"playwright", "@playwright/test"} & deps or any((root / name).exists() for name in ("playwright.config.ts", "playwright.config.js")):
        techs.add("playwright")
    if {"@supabase/supabase-js", "@supabase/ssr"} & deps:
        techs.add("supabase")
    if "ai" in deps or any(name.startswith("@ai-sdk/") for name in deps):
        techs.add("vercel-ai")

    if any((root / name).exists() for name in ("requirements.txt", "pyproject.toml", "setup.py", "Pipfile")):
        techs.add("python")
        requirements = _read_text(root / "requirements.txt")
        pyproject = _read_text(root / "pyproject.toml")
        combined = f"{requirements}\n{pyproject}".lower()
        if "fastapi" in combined:
            techs.add("fastapi")
        if "django" in combined:
            techs.add("django")

    if (root / "Dockerfile").exists():
        techs.add("docker")

    return techs


def _detect_combos(technologies: set[str]) -> set[str]:
    combos: set[str] = set()
    for combo in COMBO_RECOMMENDATION_MAP:
        if all(item in technologies for item in combo):
            combos.add("+".join(combo))
    return combos


def _classify_project_skill_dir(scan_dir: Path) -> tuple[str, str]:
    resolved = scan_dir.resolve()
    if resolved.name == "skills" and resolved.parent.name == ".hermes":
        return "project-local-hermes", "project-local"
    if resolved.name == "skills" and resolved.parent.name == ".agents":
        return "project-local-agents", "project-local"
    return "project-local", "project-local"


def _discover_project_skills(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scan_dir in get_project_local_skills_dirs(root):
        source, scope = _classify_project_skill_dir(scan_dir)
        for skill_md in sorted(scan_dir.rglob("SKILL.md")):
            if any(part in {".git", ".github", ".hub"} for part in skill_md.parts):
                continue
            content = _read_text(skill_md)
            if not content:
                continue
            frontmatter, body = parse_frontmatter(content)
            name = str(frontmatter.get("name") or skill_md.parent.name)
            if name in seen:
                continue
            seen.add(name)
            description = str(frontmatter.get("description") or "").strip()
            if not description:
                for line in body.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        description = line
                        break
            results.append(
                {
                    "name": name,
                    "description": description,
                    "source": source,
                    "scope": scope,
                    "source_dir": str(scan_dir.resolve()),
                    "skill_dir": str(skill_md.parent.resolve()),
                }
            )
    return results


def detect_project(path: str | Path | None = None) -> DetectedProject:
    start = _normalize_path(path)
    root = find_git_root(start) or start
    technologies = _detect_technologies(root)
    combos = _detect_combos(technologies)
    available_project_skills = _discover_project_skills(root)
    return DetectedProject(
        root=root.resolve(),
        technologies=technologies,
        combos=combos,
        available_project_skills=available_project_skills,
    )


def _resolve_meta(identifier: str, sources: Iterable[Any]) -> dict[str, Any]:
    for source in sources:
        try:
            meta = source.inspect(identifier)
        except Exception:
            meta = None
        if meta:
            return {
                "name": getattr(meta, "name", identifier.rsplit("/", 1)[-1]),
                "description": getattr(meta, "description", ""),
                "source": getattr(meta, "source", "official" if identifier.startswith("official/") else "github"),
                "trust_level": getattr(meta, "trust_level", "builtin" if identifier.startswith("official/") else "community"),
            }
    return {
        "name": identifier.rsplit("/", 1)[-1],
        "description": "",
        "source": "official" if identifier.startswith("official/") else "github",
        "trust_level": "builtin" if identifier.startswith("official/") else "community",
    }


def _add_recommendation(store: dict[str, dict[str, Any]], identifier: str, bucket: str, matched_on: str, score: int):
    entry = store.get(identifier)
    if entry is None:
        store[identifier] = {
            "identifier": identifier,
            "bucket": bucket,
            "matched_on": [matched_on],
            "score": score,
        }
        return
    if matched_on not in entry["matched_on"]:
        entry["matched_on"].append(matched_on)
    entry["score"] = max(entry["score"], score)


def recommend_skills(path: str | Path | None = None, source_filter: str = "all", limit: int = 20) -> dict[str, Any]:
    detected = detect_project(path)
    remote: dict[str, dict[str, Any]] = {}

    for tech in sorted(detected.technologies):
        mapping = RECOMMENDATION_MAP.get(tech, {})
        if source_filter in {"all", "official"}:
            for identifier in mapping.get("official", []):
                _add_recommendation(remote, identifier, "official", tech, 50)
        if source_filter == "all":
            for identifier in mapping.get("hub", []):
                _add_recommendation(remote, identifier, "third_party", tech, 40)

    for combo in sorted(detected.combos):
        key = tuple(combo.split("+"))
        mapping = COMBO_RECOMMENDATION_MAP.get(key, {})
        if source_filter in {"all", "official"}:
            for identifier in mapping.get("official", []):
                _add_recommendation(remote, identifier, "official", combo, 80)
        if source_filter == "all":
            for identifier in mapping.get("hub", []):
                _add_recommendation(remote, identifier, "third_party", combo, 70)

    sources = create_source_router() if remote else []
    official: list[dict[str, Any]] = []
    third_party: list[dict[str, Any]] = []

    for identifier, entry in remote.items():
        meta = _resolve_meta(identifier, sources)
        row = {
            "identifier": identifier,
            "name": meta["name"],
            "description": meta["description"],
            "source": meta["source"],
            "trust_level": meta["trust_level"],
            "state": "installable",
            "matched_on": sorted(entry["matched_on"]),
            "reason": f"Matched {', '.join(sorted(entry['matched_on']))}.",
            "score": entry["score"],
        }
        if entry["bucket"] == "official":
            official.append(row)
        else:
            third_party.append(row)

    def _sort_key(item: dict[str, Any]):
        return (-item["score"], -_TRUST_RANK.get(item.get("trust_level", "community"), 0), item["name"])

    official.sort(key=_sort_key)
    third_party.sort(key=_sort_key)

    available = [
        {
            **skill,
            "identifier": skill["name"],
            "state": "available",
            "matched_on": ["project-local"],
            "reason": "Already available in this project.",
            "score": 100,
        }
        for skill in detected.available_project_skills
    ]

    return {
        "detected": {
            "root": str(detected.root),
            "technologies": sorted(detected.technologies),
            "combos": sorted(detected.combos),
        },
        "available": available[:limit],
        "official": official[:limit],
        "third_party": third_party[:limit],
    }
