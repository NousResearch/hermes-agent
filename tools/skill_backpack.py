from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from tools.registry import registry


FRONTMATTER_RE = re.compile(r"^---\n(?P<body>.*?)\n---\n", re.DOTALL)
FIELD_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.+?)\s*$")


def _skills_config() -> dict[str, Any]:
    from hermes_cli.config import load_config

    return load_config().get("skills", {})


def _library_root() -> Path:
    configured = _skills_config().get("skill_backpack_root")
    if configured:
        return Path(configured).expanduser().resolve()
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return (hermes_home / "skill-library").resolve()


def _library_roots() -> list[Path]:
    root = _library_root()
    return [root] if root.exists() else []


def _response(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _frontmatter_fields(text: str) -> dict[str, str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field = FIELD_RE.match(line)
        if field:
            fields[field.group("key").strip()] = field.group("value").strip().strip('"\'')
    return fields


def _safe_skill_path(root: Path, relative_path: str) -> Path | None:
    raw_path = Path(relative_path)
    if not relative_path or raw_path.is_absolute() or raw_path.name != "SKILL.md":
        return None
    resolved_root = root.resolve()
    candidate = resolved_root / raw_path
    root_relative_parents = [
        parent for parent in candidate.parents if parent == resolved_root or resolved_root in parent.parents
    ]
    if candidate.is_symlink() or any(parent.is_symlink() for parent in root_relative_parents):
        return None
    path = candidate.resolve()
    if resolved_root != path and resolved_root not in path.parents:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


def _manifest_skill_paths(root: Path) -> list[tuple[str, Path]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    paths = []
    for module_id, module in manifest.get("modules", {}).items():
        if module.get("status") != "enabled":
            continue
        path = _safe_skill_path(root, str(module.get("path") or ""))
        if path is not None:
            paths.append((str(module_id), path))
    return paths


def _discovered_skill_paths(root: Path) -> list[tuple[str, Path]]:
    candidates = []
    search_roots = [root]
    modules_root = root / "modules"
    if modules_root.is_dir() and not modules_root.is_symlink():
        search_roots.insert(0, modules_root)
    for search_root in search_roots:
        if search_root.is_symlink():
            continue
        try:
            skill_files = sorted(path for path in search_root.rglob("SKILL.md") if path.is_file())
        except OSError:
            continue
        for skill_path in skill_files:
            if skill_path.is_symlink() or any(parent.is_symlink() for parent in skill_path.parents):
                continue
            try:
                relative_path = str(skill_path.relative_to(root.resolve()))
            except ValueError:
                relative_path = ""
            path = _safe_skill_path(root, relative_path)
            if path is not None:
                candidates.append((path.parent.name, path))
    seen = set()
    unique = []
    for name, path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append((name, path))
    return unique


def _skill_entries(root: Path) -> list[dict[str, Any]]:
    paths = _manifest_skill_paths(root) if (root / "manifest.json").exists() else _discovered_skill_paths(root)
    entries = []
    for fallback_name, path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        fields = _frontmatter_fields(content)
        name = fields.get("name") or fallback_name
        description = fields.get("description") or "No description."
        entries.append({"name": name, "description": description, "path": path, "content": content})
    return sorted(entries, key=lambda entry: entry["name"])


def _all_skill_entries() -> list[dict[str, Any]]:
    entries_by_name: dict[str, dict[str, Any]] = {}
    for root in _library_roots():
        for entry in _skill_entries(root):
            entries_by_name.setdefault(entry["name"], entry)
    return sorted(entries_by_name.values(), key=lambda entry: entry["name"])


def _index_response(root: Path) -> str:
    skills = [
        [index, entry["name"], entry["description"]]
        for index, entry in enumerate(_all_skill_entries(), start=1)
    ]
    return _response(status="ok", decision="skill_index", d="index", skills=skills, next="select <number|skill-name>")


def _select_response(root: Path, request: str) -> str:
    match = re.match(r"^\s*select\s+(.+?)\s*$", request.strip(), re.IGNORECASE)
    if not match:
        return _response(status="error", message="request must be index or select <number|skill-name>")
    selector = match.group(1).strip()
    entries = _all_skill_entries()
    entry = None
    if selector.isdigit():
        selected_number = int(selector)
        if 1 <= selected_number <= len(entries):
            entry = entries[selected_number - 1]
    else:
        lowered = selector.lower()
        entry = next((candidate for candidate in entries if candidate["name"].lower() == lowered), None)
    if entry is None:
        return _response(status="error", message="No skill matched this selection.")
    return _response(status="ok", decision="select_skill", d="loaded", skill=entry["name"], content=entry["content"])


def skill_backpack(args: dict[str, Any], **_kwargs: Any) -> str:
    request = args.get("request")
    if not isinstance(request, str) or not request.strip():
        request = "index"
    root = _library_root()
    if request.strip().lower() in {"index", "list", "show skills", "skills"}:
        return _index_response(root)
    return _select_response(root, request)


SKILL_BACKPACK_SCHEMA = {
    "name": "skill_backpack",
    "description": "Skill gateway.",
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "Use select <skill-name> directly when the skill name is known. Use index only when discovering unknown skills, then select <number|skill-name>.",
            }
        },
        "required": ["request"],
        "additionalProperties": False,
    },
}


def check_skill_backpack_requirements() -> bool:
    return _skills_config().get("skill_backpack_enabled") is True and bool(_library_roots())


registry.register(
    name="skill_backpack",
    toolset="skill_backpack",
    schema=SKILL_BACKPACK_SCHEMA,
    handler=skill_backpack,
    check_fn=check_skill_backpack_requirements,
    emoji="🎒",
    max_result_size_chars=100_000,
)
