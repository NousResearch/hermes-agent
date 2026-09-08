"""Pure evidence-frontmatter merge helpers for skill_manage."""

from __future__ import annotations

import hashlib
from typing import Any

import yaml


_ALLOWED_EVIDENCE = {"success_count", "fail_count", "steps", "evolution", "version", "updated"}
_STEP_KEYS = {"name", "ok", "fail"}
_EVOLUTION_KEYS = {"from", "to", "date", "reason"}


class EvidenceMergeError(ValueError):
    """Raised when evidence or a merge delta is malformed."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise EvidenceMergeError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _split_document(content: str) -> tuple[dict[str, Any], str]:
    if not content.startswith("---\n"):
        raise EvidenceMergeError("SKILL.md must start with YAML frontmatter")
    marker = content.find("\n---\n", 4)
    if marker < 0:
        raise EvidenceMergeError("SKILL.md frontmatter is not closed")
    raw = content[4:marker]
    body = content[marker + len("\n---\n"):]
    try:
        data = yaml.load(raw, Loader=_UniqueLoader)  # nosec B506 — SafeLoader subclass rejects duplicate keys
    except EvidenceMergeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EvidenceMergeError(f"malformed YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceMergeError("frontmatter must be a mapping")
    return data, body


def _int(value: Any, label: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceMergeError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise EvidenceMergeError(f"{label} must be non-negative")
    return value


def _validate_evidence(value: Any) -> dict[str, Any]:
    if value is None:
        raise EvidenceMergeError("evidence must not be null")
    if not isinstance(value, dict):
        raise EvidenceMergeError("evidence must be a mapping")
    unknown = set(value) - _ALLOWED_EVIDENCE
    if unknown:
        raise EvidenceMergeError(f"unknown evidence fields: {sorted(unknown)}")
    result = dict(value)
    result["success_count"] = _int(result.get("success_count", 0), "success_count")
    result["fail_count"] = _int(result.get("fail_count", 0), "fail_count")
    steps = result.get("steps", [])
    if not isinstance(steps, list):
        raise EvidenceMergeError("steps must be a list")
    seen = set()
    normalized_steps = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != _STEP_KEYS:
            raise EvidenceMergeError("each step must contain exactly name, ok, fail")
        name = step.get("name")
        if not isinstance(name, str) or not name:
            raise EvidenceMergeError("step name must be a non-empty string")
        if name in seen:
            raise EvidenceMergeError(f"duplicate step name: {name}")
        seen.add(name)
        normalized_steps.append({"name": name, "ok": _int(step["ok"], f"step {name} ok"),
                                 "fail": _int(step["fail"], f"step {name} fail")})
    result["steps"] = normalized_steps
    evolution = result.get("evolution", [])
    if not isinstance(evolution, list):
        raise EvidenceMergeError("evolution must be a list")
    normalized_evolution = []
    prior = 0
    for item in evolution:
        if not isinstance(item, dict) or set(item) != _EVOLUTION_KEYS:
            raise EvidenceMergeError("each evolution item has exactly from, to, date, reason")
        frm = _int(item["from"], "evolution.from")
        to = _int(item["to"], "evolution.to")
        if frm != prior or to <= frm:
            raise EvidenceMergeError("evolution must be contiguous and increasing")
        if not isinstance(item["date"], str) or not item["date"]:
            raise EvidenceMergeError("evolution.date must be a non-empty string")
        if not isinstance(item["reason"], str) or not item["reason"]:
            raise EvidenceMergeError("evolution.reason must be a non-empty string")
        normalized_evolution.append(dict(item))
        prior = to
    result["evolution"] = normalized_evolution
    has_version = "version" in result
    has_updated = "updated" in result
    if has_version != has_updated:
        raise EvidenceMergeError("version and updated must appear together")
    if has_version:
        result["version"] = _int(result["version"], "version")
        if result["version"] != prior:
            raise EvidenceMergeError("version must equal the end of evolution history")
        if not isinstance(result["updated"], str) or not result["updated"]:
            raise EvidenceMergeError("updated must be a non-empty string")
    return result


def merge_evidence(content: str, delta: dict[str, Any]) -> str:
    """Return a merged SKILL.md or raise EvidenceMergeError; never writes files."""
    if not isinstance(delta, dict) or not delta:
        raise EvidenceMergeError("evidence_merge must be a non-empty mapping")
    if set(delta) - _ALLOWED_EVIDENCE:
        raise EvidenceMergeError(f"unknown merge fields: {sorted(set(delta) - _ALLOWED_EVIDENCE)}")
    frontmatter, body = _split_document(content)
    current = _validate_evidence(frontmatter.get("evidence", {}))
    for field in ("success_count", "fail_count"):
        if field in delta:
            current[field] += _int(delta[field], field)
    if "steps" in delta:
        additions = delta["steps"]
        if not isinstance(additions, list):
            raise EvidenceMergeError("merge steps must be a list")
        by_name = {s["name"]: s for s in current["steps"]}
        delta_names = set()
        for step in additions:
            checked = _validate_evidence({"steps": [step]})["steps"][0]
            if checked["name"] in delta_names:
                raise EvidenceMergeError(f"duplicate merge step name: {checked['name']}")
            delta_names.add(checked["name"])
            existing = by_name.get(checked["name"])
            if existing:
                existing["ok"] += checked["ok"]
                existing["fail"] += checked["fail"]
            else:
                current["steps"].append(checked)
    if "evolution" in delta:
        additions = delta["evolution"]
        if not isinstance(additions, list):
            raise EvidenceMergeError("merge evolution must be a list")
        for item in additions:
            if not isinstance(item, dict) or set(item) != _EVOLUTION_KEYS:
                raise EvidenceMergeError("each merge evolution item has exactly from, to, date, reason")
            frm = _int(item["from"], "merge evolution.from")
            to = _int(item["to"], "merge evolution.to")
            if not isinstance(item["date"], str) or not item["date"]:
                raise EvidenceMergeError("merge evolution.date must be a non-empty string")
            if not isinstance(item["reason"], str) or not item["reason"]:
                raise EvidenceMergeError("merge evolution.reason must be a non-empty string")
            expected = current.get("version", 0)
            if frm != expected:
                raise EvidenceMergeError("merge evolution.from must equal current version")
            if to <= frm:
                raise EvidenceMergeError("merge evolution.to must increase")
            current["evolution"].append(dict(item))
            current["version"] = to
            current["updated"] = item["date"]
    if "version" in delta or "updated" in delta:
        if set(delta) & {"version", "updated"} != {"version", "updated"}:
            raise EvidenceMergeError("version and updated must be supplied together")
        if "evolution" not in delta:
            raise EvidenceMergeError("version metadata requires evolution")
    frontmatter["evidence"] = _validate_evidence(current)
    try:
        rendered = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip("\n")
    except Exception as exc:  # noqa: BLE001
        raise EvidenceMergeError(f"could not render frontmatter: {exc}") from exc
    return f"---\n{rendered}\n---\n{body}"


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
