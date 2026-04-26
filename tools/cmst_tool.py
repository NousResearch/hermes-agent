#!/usr/bin/env python3
"""Catmaster Skill Tree native Hermes tools."""

import hashlib
import json
import os
import re
from pathlib import Path

from tools.registry import registry, tool_error


REF_RE = re.compile(r"^cmst://(?P<tree>[A-Za-z0-9_-]+)/(?P<module>[A-Za-z0-9_-]+)@sha256:(?P<sha>[a-f0-9]{64})$")
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
FRONTMATTER_DESCRIPTION_RE = re.compile(r"^description:\s*(?P<description>.+?)\s*$", re.MULTILINE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "set",
    "the",
    "to",
    "use",
    "when",
    "with",
}
GENERIC_TASK_WORDS = {
    "add",
    "build",
    "change",
    "create",
    "fix",
    "help",
    "implement",
    "make",
    "modify",
    "task",
    "update",
}
DEBUG_TASK_WORDS = {
    "bug",
    "bugfix",
    "bugs",
    "debug",
    "debugging",
    "error",
    "errors",
    "exception",
    "exceptions",
    "fail",
    "failed",
    "failing",
    "failure",
    "failures",
    "pytest",
    "regression",
    "traceback",
    "unexpected",
}
DEBUG_TASK_PHRASES = {"test failure", "test failures", "unexpected behavior", "unexpected behaviour"}
PDF_TASK_WORDS = {"document", "documents", "pdf", "pdfs"}
PDF_ACTION_WORDS = {"edit", "fix", "modify", "typo", "typos", "update"}
TDD_TASK_WORDS = {"bugfix", "feature", "implementation", "test", "testing", "tests"}
TDD_ACTION_WORDS = {"add", "build", "fix", "implement", "write"}
DEFAULT_INDEX_LIMIT = 8


def _tree_root() -> Path:
    from hermes_cli.config import load_config

    skills_cfg = load_config().get("skills", {})
    configured = skills_cfg.get("cmst_tree_root")
    if configured:
        return Path(configured).expanduser().resolve()
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return (hermes_home / "cmst-skill-tree").resolve()


def _read_manifest(tree_root: Path) -> dict:
    with (tree_root / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _module_ref(tree: str, module_id: str, module: dict) -> str:
    return f"cmst://{tree}/{module_id}@sha256:{module['sha256']}"


def _route_match(task: str, module_id: str, module: dict) -> tuple[int, list[str]]:
    task_text = task.lower()
    task_words = set(WORD_RE.findall(task_text))
    score = 0
    signals = []
    for keyword in module.get("keywords", []):
        normalized = str(keyword).lower().strip()
        if not normalized or normalized in STOPWORDS:
            continue
        if " " in normalized:
            if normalized in task_text:
                score += 3
                signals.append(normalized)
        elif normalized in task_words:
            if normalized in GENERIC_TASK_WORDS:
                score += 1
            else:
                score += 2
            signals.append(normalized)
    for part in module_id.lower().split("-"):
        if part and part not in STOPWORDS and part in task_words:
            score += 1
            signals.append(part)
    if module_id == "systematic-debugging" and (task_words & DEBUG_TASK_WORDS or any(phrase in task_text for phrase in DEBUG_TASK_PHRASES)):
        score += 4
        signals.append("debug-task")
    if module_id == "test-driven-development" and (task_words & TDD_TASK_WORDS or ({"bug"} <= task_words and task_words & TDD_ACTION_WORDS)):
        score += 3
        signals.append("test-first-implementation")
    if module_id == "nano-pdf" and task_words & PDF_TASK_WORDS and task_words & PDF_ACTION_WORDS:
        score += 4
        signals.append("pdf-edit")
    return score, sorted(set(signals))


def _route_score(task: str, module_id: str, module: dict) -> int:
    score, _ = _route_match(task, module_id, module)
    return score


def _enabled_modules(manifest: dict) -> list[tuple[str, dict]]:
    return [(module_id, module) for module_id, module in manifest.get("modules", {}).items() if module.get("status") == "enabled"]


def _summary_from_skill_file(tree_root: Path, module: dict) -> str | None:
    relative_path = module.get("path")
    if not relative_path:
        return None
    try:
        path = _safe_module_path(tree_root, relative_path)
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    match = FRONTMATTER_DESCRIPTION_RE.search(text)
    if not match:
        return None
    description = match.group("description").strip().strip('"\'')
    return description or None


def _module_description(tree_root: Path, module_id: str, module: dict) -> str:
    explicit = module.get("summary") or module.get("description")
    if explicit:
        return str(explicit).strip()
    from_file = _summary_from_skill_file(tree_root, module)
    if from_file:
        return from_file
    keywords = [str(keyword).strip() for keyword in module.get("keywords", []) if str(keyword).strip() and str(keyword).strip() != module_id]
    if keywords:
        return "Signals: " + ", ".join(keywords[:8])
    return "No summary available. Inspect the skill before using it."


def _candidate_index(task: str, tree_root: Path, manifest: dict, limit: int = DEFAULT_INDEX_LIMIT) -> list[dict]:
    enabled = _enabled_modules(manifest)
    scored = []
    for module_id, module in enabled:
        score, signals = _route_match(task, module_id, module)
        scored.append((score, module_id, module, signals))
    scored.sort(key=lambda item: (-item[0], item[1]))
    positive = [item for item in scored if item[0] > 0]
    selected = positive[:limit] if positive else scored[:limit]
    candidates = []
    for index, (score, module_id, module, signals) in enumerate(selected, start=1):
        candidates.append(
            {
                "number": index,
                "description": _module_description(tree_root, module_id, module),
                "ref": _module_ref(manifest["tree"], module_id, module),
            }
        )
    return candidates


def _auto_route(task: str, manifest: dict) -> tuple[str, dict] | None:
    enabled = _enabled_modules(manifest)
    scored = [(_route_score(task, module_id, module), module_id, module) for module_id, module in enabled]
    scored = [item for item in scored if item[0] > 0]
    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        _, module_id, module = scored[0]
        return module_id, module
    if enabled:
        return enabled[0]
    return None


def _safe_module_path(tree_root: Path, relative_path: str) -> Path:
    root = tree_root.resolve()
    path = (root / relative_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError("module path escapes tree root")
    if path.is_symlink():
        raise ValueError("module path is a symlink")
    return path


def cmst_route(task: str, mode: str = "index", limit: int = DEFAULT_INDEX_LIMIT) -> str:
    if not task or not task.strip():
        return tool_error("task is required")
    if mode not in {"index", "auto"}:
        return tool_error("mode must be 'index' or 'auto'")
    try:
        tree_root = _tree_root()
        manifest = _read_manifest(tree_root)
        if mode == "index":
            candidates = _candidate_index(task, tree_root, manifest, max(1, int(limit)))
            if not candidates:
                return tool_error("no enabled modules")
            return json.dumps(
                {
                    "mode": "index",
                    "candidates": candidates,
                    "instruction": "Choose by number using the descriptions, then call cmst_load with the chosen ref.",
                },
                indent=2,
                sort_keys=True,
            )
        selected = _auto_route(task, manifest)
        if not selected:
            return tool_error("no enabled modules")
        module_id, module = selected
        return json.dumps({"mode": "runtime", "module": {"id": module_id, "ref": _module_ref(manifest["tree"], module_id, module)}}, indent=2, sort_keys=True)
    except Exception as exc:
        return tool_error(str(exc))


def cmst_load(ref: str) -> str:
    match = REF_RE.match(ref or "")
    if not match:
        return tool_error("invalid cmst ref")
    try:
        tree_root = _tree_root()
        manifest = _read_manifest(tree_root)
        tree = match.group("tree")
        module_id = match.group("module")
        expected_sha = match.group("sha")
        if tree != manifest.get("tree"):
            return tool_error("tree does not match manifest")
        module = manifest.get("modules", {}).get(module_id)
        if module is None:
            return tool_error("module not found")
        if module.get("status") != "enabled":
            return tool_error("module is not enabled")
        if module.get("sha256") != expected_sha:
            return tool_error("module hash mismatch")
        path = _safe_module_path(tree_root, module["path"])
        content = path.read_bytes()
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            return tool_error("module content hash mismatch")
        return json.dumps({"id": module_id, "ref": ref, "content": content.decode("utf-8")}, indent=2, sort_keys=True)
    except Exception as exc:
        return tool_error(str(exc))


CMST_ROUTE_SCHEMA = {
    "name": "cmst_route",
    "description": "Show a numbered Catmaster Skill Tree skill index for a task. Read the descriptions, choose the appropriate candidate number, then call cmst_load with that candidate's ref.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Current user task or task stage to route."},
            "limit": {"type": "integer", "description": "Maximum number of candidate cards to return in index mode."},
        },
        "required": ["task"],
    },
}


CMST_LOAD_SCHEMA = {
    "name": "cmst_load",
    "description": "Load verified content for a Catmaster Skill Tree module ref returned by cmst_route.",
    "parameters": {
        "type": "object",
        "properties": {"ref": {"type": "string", "description": "cmst:// tree module ref from the selected cmst_route candidate."}},
        "required": ["ref"],
    },
}


def check_cmst_requirements() -> bool:
    return (_tree_root() / "manifest.json").exists()


registry.register(
    name="cmst_route",
    toolset="cmst",
    schema=CMST_ROUTE_SCHEMA,
    handler=lambda args, **kw: cmst_route(args.get("task", ""), args.get("mode", "index"), args.get("limit", DEFAULT_INDEX_LIMIT)),
    check_fn=check_cmst_requirements,
    emoji="🌲",
)

registry.register(
    name="cmst_load",
    toolset="cmst",
    schema=CMST_LOAD_SCHEMA,
    handler=lambda args, **kw: cmst_load(args.get("ref", "")),
    check_fn=check_cmst_requirements,
    emoji="🌲",
)
