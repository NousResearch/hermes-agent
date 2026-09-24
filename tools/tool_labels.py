from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.connectors.gateway.names import parse_connector_name
from tools.mcp_tool_schema import MCP_TOOL_NAME_PREFIX

logger = logging.getLogger("tools.tool_labels")

CONNECTOR_EMOJI = "🔗"
MCP_EMOJI = "🔌"
SEARCH_EMOJI = "🔎"
DESCRIBE_EMOJI = "📖"
DEFAULT_TOOL_EMOJI = "⚡"

_APP_TITLES = {
    "discord": "Discord",
    "figma": "Figma",
    "github": "GitHub",
    "gmail": "Gmail",
    "googlecalendar": "Google Calendar",
    "googledocs": "Google Docs",
    "googledrive": "Google Drive",
    "jira": "Jira",
    "linear": "Linear",
    "notion": "Notion",
    "outlook": "Outlook",
    "slack": "Slack",
    "stripe_mcp": "Stripe",
    "todoist": "Todoist",
}

BRIDGE_TOOL_NAMES = frozenset({"tool_call", "tool_search", "tool_describe"})

__all__ = [
    "BRIDGE_TOOL_NAMES", "CONNECTOR_EMOJI", "DEFAULT_TOOL_EMOJI", "MCP_EMOJI", "ToolLabel",
    "app_title", "label_for_tool_name", "labels_for_call"]


@dataclass(frozen=True)
class ToolLabel:

    kind: str
    app: str
    action: str
    emoji: str
    name: str
    preview: str = ""

    @property
    def text(self) -> str:
        return f"{self.app} · {self.action}" if self.action else self.app

    def as_payload(self) -> Dict[str, str]:
        return {"kind": self.kind, "app": self.app, "action": self.action, "emoji": self.emoji,
                "text": self.text, "name": self.name, "preview": self.preview}


def app_title(slug: str) -> str:
    key = (slug or "").strip()
    known = _APP_TITLES.get(key.lower())
    if known:
        return known
    words = [word for word in key.replace("-", " ").replace("_", " ").split() if word]
    return " ".join(word[:1].upper() + word[1:] for word in words) or key


def _action_phrase(tool: str, app_slug: str = "") -> str:
    text = (tool or "").strip()
    prefix = f"{app_slug.replace('-', '_').upper()}_"
    if app_slug and text.upper().startswith(prefix):
        text = text[len(prefix):]
    words = [word for word in text.replace("-", " ").replace("_", " ").split() if word]
    return " ".join(words).lower()


def _registry_emoji(name: str) -> str:
    try:
        from tools.registry import registry
        return registry.get_emoji(name, default="") or DEFAULT_TOOL_EMOJI
    except Exception:
        return DEFAULT_TOOL_EMOJI


def _arg_preview(name: str, arguments: Any) -> str:
    if not isinstance(arguments, dict) or not arguments:
        return ""
    try:
        from agent.display import build_tool_preview
        return build_tool_preview(name, arguments) or ""
    except Exception:
        logger.debug("argument preview failed for %s", name, exc_info=True)
        return ""


def _local_tool_label(name: str, arguments: Any) -> ToolLabel:
    verb = ""
    try:
        from agent.display import get_tool_verb
        verb = get_tool_verb(name) or ""
    except Exception:
        logger.debug("friendly verb lookup failed for %s", name, exc_info=True)
    return ToolLabel(kind="tool", app=verb or app_title(name), action="",
                     emoji=_registry_emoji(name), name=name, preview=_arg_preview(name, arguments))


def _unnamed_label(arguments: Any) -> ToolLabel:
    return ToolLabel(kind="tool", app="Unnamed call", action="", emoji=DEFAULT_TOOL_EMOJI,
                     name="", preview=_arg_preview("", arguments))


def label_for_tool_name(name: str, arguments: Any = None) -> Optional[ToolLabel]:
    if not isinstance(name, str) or not name.strip():
        return None
    connector = parse_connector_name(name)
    if connector is not None:
        return ToolLabel(kind="connector", app=app_title(connector.connector),
                         action=_action_phrase(connector.tool, connector.connector),
                         emoji=CONNECTOR_EMOJI, name=name, preview=_arg_preview(name, arguments))
    if name.startswith(MCP_TOOL_NAME_PREFIX):
        server, _, tool = name[len(MCP_TOOL_NAME_PREFIX):].partition("__")
        if tool:
            return ToolLabel(kind="mcp", app=app_title(server), action=_action_phrase(tool),
                             emoji=MCP_EMOJI, name=name, preview=_arg_preview(name, arguments))
    return _local_tool_label(name, arguments)


def _bridge_labels(function_name: str, args: Dict[str, Any]) -> List[ToolLabel]:
    if function_name == "tool_search":
        queries = args.get("queries")
        queries = [queries] if isinstance(queries, str) else queries
        first = next((str(q) for q in queries if str(q or "").strip()), "") if isinstance(queries, list) else ""
        return [ToolLabel(kind="tool", app="Searching tools", action=first,
                          emoji=SEARCH_EMOJI, name=function_name)]
    names = args.get("names")
    names = [names] if isinstance(names, str) else names
    count = len(names) if isinstance(names, list) else 0
    return [ToolLabel(kind="tool", app="Reading tool details",
                      action=f"{count} tool" if count == 1 else f"{count} tools",
                      emoji=DESCRIBE_EMOJI, name=function_name)]


def labels_for_call(function_name: str, function_args: Any) -> List[ToolLabel]:
    args = function_args if isinstance(function_args, dict) else {}
    if function_name in ("tool_search", "tool_describe"):
        return _bridge_labels(function_name, args)
    if function_name != "tool_call":
        label = label_for_tool_name(function_name, args)
        return [label] if label is not None and label.kind != "tool" else []
    entries = args.get("calls")
    entries = entries if isinstance(entries, list) else [args]
    labels = []
    for entry in entries:
        entry = entry if isinstance(entry, dict) else {}
        arguments = entry.get("arguments")
        label = label_for_tool_name(entry.get("name"), arguments)
        labels.append(label if label is not None else _unnamed_label(arguments))
    return labels
