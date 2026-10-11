"""Model-facing copy must name tools by their offered names, not the pre-rename aliases.

The 2026-08 renames (process -> process_manage, cronjob -> cronjob_manage, tour -> gui_tour,
...) kept the old names as dispatch aliases, but prompts, schemas, notices and error strings
that still say ``process(action=...)`` or ``process(submit)`` steer the model to a name that
is never offered. Hermes' own dispatcher maps the alias back, yet an OpenAI-compatible server
that filters unoffered tool names while streaming drops the call first, and the turn ends on a
reasoning-only stop (#124583). Docstrings are developer prose and are not scanned; every other
string literal is, because prompts, tool results, notices and schema descriptions are all
built from them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from model_tools import _LEGACY_TOOL_ALIASES
from tools.registry import registry

REPO = Path(__file__).resolve().parents[2]
SCANNED_DIRS = ("agent", "cron", "gateway", "tools", "tui_gateway")


def _action_enum(tool_name: str) -> list[str]:
    schema = registry.get_schema(tool_name) or {}
    action = schema.get("parameters", {}).get("properties", {}).get("action", {})
    return [a for a in action.get("enum", []) if isinstance(a, str)]


def _legacy_patterns() -> list[re.Pattern[str]]:
    # ``process(action=...)``: keyword-call copy. ``(?<![\w.])`` skips ``registry.process(``
    # and ``process_manage(`` itself.
    aliases = "|".join(map(re.escape, sorted(_LEGACY_TOOL_ALIASES)))
    patterns = [re.compile(r"(?<![\w.])(%s)\(\s*\w+\s*=" % aliases)]
    # ``process(submit)``: shorthand limited to the offered tool's real action names, so
    # ordinary prose such as "process(es)" is not flagged.
    for alias, offered in sorted(_LEGACY_TOOL_ALIASES.items()):
        actions = _action_enum(offered)
        if actions:
            patterns.append(re.compile(
                r"(?<![\w.])(%s)\(\s*['\"]?(?:%s)\b"
                % (re.escape(alias), "|".join(map(re.escape, actions)))
            ))
    return patterns


def _docstring_nodes(tree: ast.AST) -> set[int]:
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, owners)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }


def _legacy_call_copy() -> list[str]:
    patterns = _legacy_patterns()
    hits = []
    for top in SCANNED_DIRS:
        for path in sorted((REPO / top).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            docstrings = _docstring_nodes(tree)
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                ):
                    continue
                for pattern in patterns:
                    for match in pattern.finditer(node.value):
                        hits.append(
                            f"{path.relative_to(REPO)}:{node.lineno}: {match.group(0)} "
                            f"-> {_LEGACY_TOOL_ALIASES[match.group(1)]}("
                        )
    return hits


def test_model_facing_strings_use_offered_tool_names():
    hits = _legacy_call_copy()
    assert not hits, "legacy tool names in model-facing strings:\n" + "\n".join(hits)
