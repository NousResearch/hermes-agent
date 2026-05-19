"""Jira issue loading tools.

The tool intentionally keeps scope read-only. It accepts either a Jira issue key
(e.g. CPG-123) or a full issue URL and returns normalized ticket content for the
agent to use when implementing work.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tools.registry import registry, tool_error, tool_result

ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
JIRA_URL_RE = re.compile(r"^https?://[^\s/]+/(?:browse|jira/(?:software|core)/c/projects/[^/]+/issues)/([A-Z][A-Z0-9]+-\d+)", re.I)

FIELD_ALLOWLIST = {
    "summary",
    "description",
    "issuetype",
    "status",
    "priority",
    "assignee",
    "reporter",
    "labels",
    "components",
    "fixVersions",
    "versions",
    "parent",
    "issuelinks",
    "subtasks",
    "attachment",
    "project",
}

CUSTOM_FIELD_NAME_HINTS = (
    "acceptance",
    "criteria",
    "component",
    "repository",
    "repo",
    "team",
    "epic",
    "story points",
    "environment",
    "labels",
)


class JiraConfigurationError(RuntimeError):
    pass


class JiraRequestError(RuntimeError):
    pass


def _has_env(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())


def _jira_requirements_met() -> bool:
    return bool((_has_env("JIRA_API_TOKEN") and _has_env("JIRA_EMAIL")) or _has_env("JIRA_BEARER_TOKEN"))


def _normalize_site(site: str | None) -> str | None:
    if not site:
        return None
    site = site.strip().rstrip("/")
    if not site:
        return None
    if not site.startswith(("http://", "https://")):
        site = "https://" + site
    return site


def _site_from_reference(reference: str) -> str | None:
    parsed = urllib.parse.urlparse(reference)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return None


def extract_issue_key(reference: str) -> str:
    """Extract a Jira issue key from a URL or key-like reference."""
    reference = (reference or "").strip()
    if not reference:
        raise ValueError("Provide a Jira issue key or URL.")
    match = JIRA_URL_RE.search(reference)
    if match:
        return match.group(1).upper()
    match = ISSUE_KEY_RE.search(reference)
    if match:
        return match.group(0).upper()
    raise ValueError(f"Could not find a Jira issue key in {reference!r}.")


def _resolve_site(reference: str, site_url: str | None = None) -> str:
    site = _normalize_site(site_url) or _site_from_reference(reference) or _normalize_site(os.getenv("JIRA_SITE_URL")) or _normalize_site(os.getenv("ATLASSIAN_SITE_URL"))
    if not site:
        raise JiraConfigurationError("Set JIRA_SITE_URL (or ATLASSIAN_SITE_URL), or pass a full Jira issue URL.")
    return site


def _auth_header() -> str:
    bearer = (os.getenv("JIRA_BEARER_TOKEN") or "").strip()
    if bearer:
        return f"Bearer {bearer}"
    email = (os.getenv("JIRA_EMAIL") or "").strip()
    token = (os.getenv("JIRA_API_TOKEN") or "").strip()
    if not email or not token:
        raise JiraConfigurationError("Set JIRA_EMAIL + JIRA_API_TOKEN, or JIRA_BEARER_TOKEN.")
    encoded = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def _request_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": _auth_header(),
            "Accept": "application/json",
            "User-Agent": "hermes-agent-jira-tool/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise JiraRequestError(f"Jira returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise JiraRequestError(f"Could not reach Jira: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise JiraRequestError("Jira returned invalid JSON.") from exc


def _adf_to_text(node: Any) -> str:
    """Convert Atlassian Document Format fragments to readable plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(part for part in (_adf_to_text(item).strip() for item in node) if part)
    if not isinstance(node, dict):
        return str(node)

    node_type = node.get("type")
    if node_type == "text":
        return str(node.get("text") or "")
    if node_type == "hardBreak":
        return "\n"
    if node_type in {"mention", "emoji", "inlineCard"}:
        attrs = node.get("attrs") or {}
        return str(attrs.get("text") or attrs.get("url") or attrs.get("shortName") or "")

    children = [_adf_to_text(child) for child in node.get("content") or []]
    text = "".join(children) if node_type in {"paragraph", "heading", "listItem"} else "\n".join(part.strip() for part in children if part.strip())

    if node_type == "bulletList":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(f"- {line}" for line in lines)
    if node_type == "orderedList":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(f"{idx}. {line}" for idx, line in enumerate(lines, 1))
    return text.strip()


def _name(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("displayName") or value.get("name") or value.get("value") or value.get("key") or value.get("id")
    return value


def _names(values: Any) -> list[Any]:
    if not isinstance(values, list):
        return []
    return [item for item in (_name(value) for value in values) if item not in (None, "")]


def _simplify_link(link: dict[str, Any]) -> dict[str, Any]:
    link_type = (link.get("type") or {}).get("name")
    outward = link.get("outwardIssue") or {}
    inward = link.get("inwardIssue") or {}
    other = outward or inward
    direction = "outward" if outward else "inward" if inward else None
    fields = other.get("fields") or {}
    return {
        "type": link_type,
        "direction": direction,
        "key": other.get("key"),
        "summary": fields.get("summary"),
        "status": _name(fields.get("status")),
    }


def _compact_field(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        compacted = [_compact_field(item) for item in value]
        return [item for item in compacted if item not in (None, "", [], {})] or None
    if isinstance(value, dict):
        if value.get("type") == "doc" or "content" in value:
            text = _adf_to_text(value)
            return text or None
        named = _name(value)
        if named and named != value:
            return named
        compacted = {k: _compact_field(v) for k, v in value.items() if k not in {"avatarUrls", "self", "iconUrl"}}
        return {k: v for k, v in compacted.items() if v not in (None, "", [], {})} or None
    return str(value)


def normalize_issue(issue: dict[str, Any], *, site_url: str) -> dict[str, Any]:
    fields = issue.get("fields") or {}
    names = issue.get("names") or {}
    custom_fields: dict[str, Any] = {}

    for field_id, value in fields.items():
        if field_id in FIELD_ALLOWLIST or not field_id.startswith("customfield_"):
            continue
        display_name = str(names.get(field_id) or field_id)
        lowered = display_name.lower()
        if any(hint in lowered for hint in CUSTOM_FIELD_NAME_HINTS):
            compact = _compact_field(value)
            if compact not in (None, "", [], {}):
                custom_fields[display_name] = compact

    key = issue.get("key")
    normalized = {
        "success": True,
        "key": key,
        "url": f"{site_url}/browse/{key}" if key else issue.get("self"),
        "summary": fields.get("summary"),
        "description": _adf_to_text(fields.get("description")),
        "issue_type": _name(fields.get("issuetype")),
        "status": _name(fields.get("status")),
        "priority": _name(fields.get("priority")),
        "assignee": _name(fields.get("assignee")),
        "reporter": _name(fields.get("reporter")),
        "labels": fields.get("labels") or [],
        "components": _names(fields.get("components")),
        "fix_versions": _names(fields.get("fixVersions")),
        "affected_versions": _names(fields.get("versions")),
        "project": _name(fields.get("project")),
        "parent": _compact_field(fields.get("parent")),
        "subtasks": [
            {"key": item.get("key"), "summary": (item.get("fields") or {}).get("summary"), "status": _name((item.get("fields") or {}).get("status"))}
            for item in fields.get("subtasks") or []
        ],
        "issue_links": [_simplify_link(link) for link in fields.get("issuelinks") or []],
        "attachments": [
            {"filename": item.get("filename"), "mimeType": item.get("mimeType"), "size": item.get("size"), "content": item.get("content")}
            for item in fields.get("attachment") or []
        ],
        "custom_fields": custom_fields,
    }
    return {k: v for k, v in normalized.items() if v not in (None, "", [], {})}


def jira_get_issue(reference: str, site_url: str | None = None) -> str:
    """Load and normalize a Jira issue by URL or issue key."""
    try:
        issue_key = extract_issue_key(reference)
        site = _resolve_site(reference, site_url)
        query = urllib.parse.urlencode({"fields": "*all", "expand": "names"})
        issue = _request_json(f"{site}/rest/api/3/issue/{urllib.parse.quote(issue_key)}?{query}")
        return tool_result(normalize_issue(issue, site_url=site))
    except (ValueError, JiraConfigurationError, JiraRequestError) as exc:
        return tool_error(str(exc), success=False)


JIRA_GET_ISSUE_SCHEMA = {
    "name": "jira_get_issue",
    "description": (
        "Read a Jira issue by full issue URL or issue key (for example CPG-123). "
        "Use this before implementing work requested from a Jira ticket. Returns summary, "
        "description, acceptance-criteria-like custom fields, labels, components, links, subtasks, and attachments. "
        "Read-only; does not transition or comment on issues."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reference": {
                "type": "string",
                "description": "Jira issue key or full Jira issue URL, e.g. CPG-123 or https://example.atlassian.net/browse/CPG-123.",
            },
            "site_url": {
                "type": "string",
                "description": "Optional Jira site base URL. Required for bare issue keys unless JIRA_SITE_URL or ATLASSIAN_SITE_URL is configured.",
            },
        },
        "required": ["reference"],
    },
}


registry.register(
    name="jira_get_issue",
    toolset="jira",
    schema=JIRA_GET_ISSUE_SCHEMA,
    handler=lambda args, **kw: jira_get_issue(args.get("reference", ""), site_url=args.get("site_url")),
    check_fn=_jira_requirements_met,
    requires_env=["JIRA_EMAIL", "JIRA_API_TOKEN"],
    emoji="🎫",
    max_result_size_chars=100_000,
)
