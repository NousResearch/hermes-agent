#!/usr/bin/env python3
"""Path-scoped publication tool for verified Scout research artifacts."""

import json
import re
from pathlib import Path
from typing import Any

import yaml

from tools import file_state
from tools.file_tools import write_file_tool
from tools.registry import registry, tool_error


VAULT_ROOT = Path(
    "/Users/felipelamartine/Documents/hermes-obsidian-long-term-memory"
)
_PACKET_MODES = {
    "fact_check",
    "decision_brief",
    "literature_review",
    "osint",
    "monitoring_candidate",
}
_EVIDENCE_STATUSES = {"verified", "mixed", "provisional"}
_URL_CITATION_RE = re.compile(r"\[[^\]\n]+\]\(https?://[^)\s]+\)")


def _frontmatter(text: str, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(text, str) or not text.startswith("---\n"):
        return None, f"{label} must start with YAML frontmatter."
    closing = text.find("\n---", 4)
    if closing < 0:
        return None, f"{label} YAML frontmatter is not closed."
    try:
        values = yaml.safe_load(text[4:closing]) or {}
    except yaml.YAMLError as exc:
        return None, f"{label} YAML frontmatter is invalid: {exc}"
    if not isinstance(values, dict):
        return None, f"{label} YAML frontmatter must be a mapping."
    return values, None


def _validate_evidence_packet(packet: str) -> str | None:
    metadata, error = _frontmatter(packet, "Evidence packet")
    if error:
        return error
    required = ("question", "mode", "created", "as_of", "collector", "publication_status")
    missing = [field for field in required if not metadata.get(field)]
    if metadata.get("packet_type") != "scout_evidence_packet" or missing:
        suffix = f" Missing fields: {', '.join(missing)}." if missing else ""
        return "Evidence packet metadata is incomplete or has an invalid packet_type." + suffix
    if metadata["mode"] not in _PACKET_MODES:
        return "Evidence packet mode is not an approved Scout research mode."
    if metadata["collector"] != "scout":
        return "Evidence packet collector must be 'scout'."
    for heading in ("## Claim Ledger", "## Source Register"):
        if heading not in packet:
            return f"Evidence packet is missing required section '{heading}'."
    if not re.search(r"https?://\S+", packet):
        return "Evidence packet source register must include a stable source URL."
    return None


def _validate_resource(content: str) -> str | None:
    metadata, error = _frontmatter(content, "Published resource")
    if error:
        return error
    required = (
        "created",
        "updated",
        "type",
        "source_agent",
        "evidence_packet_date",
        "evidence_status",
        "as_of",
        "related",
    )
    missing = [field for field in required if not metadata.get(field)]
    if missing:
        return f"Published resource metadata is missing: {', '.join(missing)}."
    if metadata["type"] != "resource" or metadata["source_agent"] != "Scout":
        return "Published resource must declare type 'resource' and source_agent 'Scout'."
    if metadata["evidence_status"] not in _EVIDENCE_STATUSES:
        return "Published resource evidence_status must be verified, mixed, or provisional."
    for heading in ("## Findings", "## Evidence Quality", "## Sources"):
        if heading not in content:
            return f"Published resource is missing required section '{heading}'."
    findings = _section_content(content, "## Findings")
    if not findings or not _URL_CITATION_RE.search(findings):
        return "Published resource findings must contain at least one inline Markdown source citation."
    return None


def _section_content(content: str, heading: str) -> str:
    start = content.find(heading)
    if start < 0:
        return ""
    start += len(heading)
    next_heading = content.find("\n## ", start)
    return content[start:] if next_heading < 0 else content[start:next_heading]


def _symlink_component(path: Path, root: Path) -> bool:
    current = root
    if current.is_symlink():
        return True
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validate_target(path: str, artifact_type: str) -> tuple[Path | None, str | None]:
    if not isinstance(path, str) or not path.strip():
        return None, "Publication path is required."
    target = Path(path).expanduser()
    if not target.is_absolute():
        return None, "Publication path must be absolute."
    if ".." in target.parts:
        return None, "Publication path traversal using '..' is not allowed."

    root = VAULT_ROOT.expanduser()
    try:
        target.relative_to(root)
    except ValueError:
        return None, "Publication target is not an approved vault path."
    if _symlink_component(target, root):
        return None, "Publication target contains a symlink and is not allowed."

    resolved_root = root.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    resources = (resolved_root / "05 Resources").resolve(strict=False)
    operations_log = resolved_root / "09 System/Operations Log.md"
    if artifact_type == "resource":
        try:
            resolved_target.relative_to(resources)
        except ValueError:
            return None, "Research resources may be written only under the approved vault Resources path."
        if resolved_target.suffix.lower() != ".md":
            return None, "Published research resources must be Markdown files."
        return resolved_target, None
    if artifact_type != "operations_log" or resolved_target != operations_log:
        return None, "Publication target is not an approved vault path for this artifact type."
    return resolved_target, None


def _append_operations_log(target: Path, content: str, task_id: str) -> str:
    if not target.exists():
        return tool_error("Operations log must already exist before an entry can be appended.")
    if not content.lstrip().startswith("- ") or "[[" not in content:
        return tool_error("Operations log entries must be bullet entries linking the published resource.")
    try:
        with file_state.lock_path(str(target)):
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
            file_state.note_write(task_id, str(target))
        return json.dumps(
            {"status": "ok", "path": str(target), "appended": True},
            ensure_ascii=False,
        )
    except Exception as exc:
        return tool_error(str(exc))


def publish_research_artifact(
    path: str,
    content: str,
    evidence_packet: str,
    artifact_type: str = "resource",
    task_id: str = "default",
) -> str:
    """Write an evidence-backed research artifact to the limited vault surface."""
    target, error = _validate_target(path, artifact_type)
    if error:
        return tool_error(error)
    packet_error = _validate_evidence_packet(evidence_packet)
    if packet_error:
        return tool_error(packet_error)
    if not isinstance(content, str) or not content.strip():
        return tool_error("Published artifact content is required.")
    if artifact_type == "resource":
        resource_error = _validate_resource(content)
        if resource_error:
            return tool_error(resource_error)
    if artifact_type == "operations_log":
        return _append_operations_log(target, content, task_id)
    return write_file_tool(str(target), content, task_id=task_id)


PUBLISH_RESEARCH_ARTIFACT_SCHEMA = {
    "name": "publish_research_artifact",
    "description": (
        "Publish an evidence-backed Scout research artifact to an approved Obsidian vault path. "
        "This tool rejects paths outside its publication allowlist, traversal or symlink paths, "
        "and research resources lacking evidence metadata and Markdown citations. It cannot "
        "delete, move, or modify profile/system files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute approved vault path for the artifact.",
            },
            "content": {
                "type": "string",
                "description": "Complete Markdown artifact content to publish.",
            },
            "evidence_packet": {
                "type": "string",
                "description": "Completed Scout evidence packet supporting this publication.",
            },
            "artifact_type": {
                "type": "string",
                "enum": ["resource", "operations_log"],
                "description": "Approved artifact class matching the target path.",
                "default": "resource",
            },
        },
        "required": ["path", "content", "evidence_packet"],
    },
}


def _handle_publish_research_artifact(args, **kw):
    return publish_research_artifact(
        path=args.get("path", ""),
        content=args.get("content", ""),
        evidence_packet=args.get("evidence_packet", ""),
        artifact_type=args.get("artifact_type", "resource"),
        task_id=kw.get("task_id") or "default",
    )


registry.register(
    name="publish_research_artifact",
    toolset="vault-publish",
    schema=PUBLISH_RESEARCH_ARTIFACT_SCHEMA,
    handler=_handle_publish_research_artifact,
    emoji="📑",
    max_result_size_chars=100_000,
)
