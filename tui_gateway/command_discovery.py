"""Shared slash discovery data; no server import or session execution runtime.

Transport wrappers own authentication and profile binding before calling these
builders. Keep the legacy catalog and completion semantics identical here.
"""
import contextlib
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

_TUI_HIDDEN: frozenset[str] = frozenset({"sethome", "set-home", "commands", "approve", "deny"})

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/density", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    ("/mouse", "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]", "TUI"),
    ("/sessions", "Switch between live TUI sessions", "TUI"),
]

_SLASH_EXTRAS = [
    ("/density", "Toggle compact display mode"), ("/details", "Control agent detail visibility"),
    ("/logs", "Show recent gateway log lines"),
    ("/mouse", "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]")]


def _item(text: str, meta: str, display: str | None = None) -> dict:
    return {"text": text, "display": display if display is not None else text, "meta": meta}


def _skill_usage_lookup():
    """``(usage, origin)`` callables for the skill catalog: activity count (use + view + patch) and
    "hub" / "bundled" / "local" (``/api/skills`` ``provenance``, "local" spelled "agent"). Failure → 0 / "local"."""
    try:
        from tools.skill_usage import (
            _read_bundled_manifest_names, _read_hub_installed_names, activity_count, load_usage)
        records, bundled, hub = load_usage(), _read_bundled_manifest_names(), _read_hub_installed_names()
    except Exception as e:
        logger.debug("skill usage lookup unavailable: %s", e)
        return (lambda _name: 0), (lambda _name: "local")

    def usage(name: str) -> int:
        with contextlib.suppress(Exception):
            return activity_count(records.get(name) or {})
        return 0

    def origin(name: str) -> str:
        return "hub" if name in hub else "bundled" if name in bundled else "local"
    return usage, origin


_SLASH_COMPLETION_LIMIT = 30


def _rank_slash_completions(items: list[dict], usage, origin_of, *, browsing: bool, score_of=None) -> list[dict]:
    """Registry commands keep their order; only skills reorder: fuzzy ``score_of`` first, then most-used, then
    A-Z. The limit is spent PER KIND (a flat cut on a large install offered no skill at all). ``browsing``
    (bare ``/``) drops never-used bundled skills as noise; a typed query is SEARCHING — nothing pruned, only reordered."""
    def name_of(item: dict) -> str:
        return str(item.get("text", "")).strip().lstrip("/").lower()
    commands = [item for item in items if item.get("kind") != "skill"]
    skills = [item for item in items if item.get("kind") == "skill"]
    if browsing:
        skills = [item for item in skills if origin_of(name_of(item)) != "bundled" or usage(name_of(item)) > 0]
    skills.sort(key=lambda item: (
        *(() if score_of is None else (score_of(item),)), -usage(name_of(item)), name_of(item)))
    return commands[:_SLASH_COMPLETION_LIMIT] + skills[:_SLASH_COMPLETION_LIMIT]


_DETAILS_SECTIONS = ("thinking", "tools", "subagents", "activity")
_DETAILS_MODES = ("hidden", "collapsed", "expanded")


def _details_root_meta(candidate: str) -> str:
    if candidate in _DETAILS_SECTIONS:
        return "section override"
    return "cycle global mode" if candidate == "cycle" else "global mode"


def _details_completions(text: str) -> list[dict] | None:
    """Argument completions for ``/details [section] [mode]``; None when ``text`` is not that command."""
    if not text.lower().startswith("/details"):
        return None
    stripped = text.strip()
    if stripped and not "/details".startswith(stripped.lower().split()[0]):
        return None
    body = text[len("/details") :].removeprefix(" ")
    parts = body.split()
    trailing = text.endswith(" ")
    root_candidates = (*_DETAILS_MODES, "cycle", *_DETAILS_SECTIONS)
    if not body or (not parts and trailing):
        lead = "" if trailing else " "
        return [_item(f"{lead}{c}", _details_root_meta(c)) for c in root_candidates]
    if len(parts) == 1 and not trailing:
        prefix = parts[0].lower()
        return [_item(c, _details_root_meta(c)) for c in root_candidates if c.startswith(prefix) and c != prefix]
    section = parts[0].lower() if parts else ""
    if section not in _DETAILS_SECTIONS:
        return []

    def section_meta(candidate: str) -> str:
        return f"clear {section} override" if candidate == "reset" else f"set {section}"
    mode_candidates = (*_DETAILS_MODES, "reset")
    if len(parts) == 1:  # trailing space after the section
        return [_item(c, section_meta(c)) for c in mode_candidates]
    if len(parts) == 2 and not trailing:
        prefix = parts[1].lower()
        return [_item(c, section_meta(c)) for c in mode_candidates if c.startswith(prefix) and c != prefix]
    return []


class _Catalog:
    """Accumulator for commands.catalog: ``pairs`` (every [key, desc]), ``canon`` (lowercase
    key/alias → canonical key), ``commands`` (key → desktop meta) and ordered categories."""

    def __init__(self) -> None:
        self.pairs: list[list[str]] = []
        self.canon: dict[str, str] = {}
        self.commands: dict[str, dict[str, str | None]] = {}
        self.cat_map: dict[str, list[list[str]]] = {}  # insertion order = category order

    def add(self, key: str, desc: str, cat: str) -> None:
        self.canon[key.lower()] = key
        self.pairs.append([key, desc])
        self.cat_map.setdefault(cat, []).append([key, desc])


def _catalog_registry(cat: _Catalog, module_loader) -> None:
    commands = module_loader("hermes_cli.commands")
    for cmd in commands.COMMAND_REGISTRY:
        meta = commands.command_desktop_meta(cmd)
        cat.commands.update({f"/{key}": dict(meta) for key in (cmd.name, *cmd.aliases)})
        if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
            continue
        cat.add(f"/{cmd.name}", commands._build_description(cmd), cmd.category)
        for a in cmd.aliases:
            cat.canon[f"/{a}".lower()] = f"/{cmd.name}"
    for name, desc, category in _TUI_EXTRA:
        # Registry command/alias wins over a colliding TUI extra (e.g. /compact, /sessions).
        if name.lower() not in cat.canon:
            cat.add(name, desc, category)


def _catalog_quick_commands(cat: _Catalog, load_cfg) -> None:
    qcmds = load_cfg().get("quick_commands", {}) or {}
    if not (isinstance(qcmds, dict) and qcmds):
        return
    cat.cat_map.setdefault("User commands", [])  # category exists even when every entry is malformed
    for qname, qc in sorted(qcmds.items()):
        if not isinstance(qc, dict):
            continue
        qtype = qc.get("type", "")
        default_desc = {"exec": f"exec: {qc.get('command', '')}", "alias": f"alias → {qc.get('target', '')}"}
        desc = str(qc.get("description") or default_desc.get(qtype, qtype or "quick command"))
        cat.add(f"/{qname}", desc, "User commands")


def _catalog_plugin_commands(cat: _Catalog, module_loader) -> None:
    plugin_cmds = module_loader("hermes_cli.plugins").get_plugin_commands() or {}
    if plugin_cmds:
        cat.cat_map.setdefault("Plugin commands", [])
    for pname, info in sorted(plugin_cmds.items()):
        key = f"/{pname}"
        if not isinstance(info, dict) or key.lower() in cat.canon:
            continue
        cat.add(key, str(info.get("description") or "Plugin command"), "Plugin commands")
        mode = info.get("argument_mode")
        if mode not in {"options", "text", "mixed"}:
            mode = "text" if str(info.get("args_hint") or "").strip() else None
        cat.commands[key] = {"argument_mode": mode, "desktop": None}


def _catalog_skills(cat: _Catalog, skills: dict[str, dict], module_loader) -> None:
    """Append skill pairs and fill ``skills`` = ``{key: {usage, origin}}`` (every consumer ranks by them)."""
    usage, origin_of = _skill_usage_lookup()
    for k, info in sorted(module_loader("agent.skill_commands").scan_skill_commands().items()):
        cat.pairs.append([k, str(info.get("description", "Skill"))])
        name = str(info.get("name") or k.lstrip("/"))
        skills[k] = {"usage": usage(name), "origin": origin_of(name)}


def command_catalog(load_cfg=None, module_loader=import_module) -> dict:
    """Registry-backed slash metadata, categorized, no aliases. Discovery failures land in ``warning``
    (skills' message wins, then quick commands', then plugins')."""
    if load_cfg is None:
        from hermes_cli.config import load_config_readonly
        load_cfg = load_config_readonly
    cat = _Catalog()
    _catalog_registry(cat, module_loader)
    warning = ""
    try:
        _catalog_quick_commands(cat, load_cfg)
    except Exception as e:
        warning = f"quick_commands discovery unavailable: {e}"
    try:
        _catalog_plugin_commands(cat, module_loader)
    except Exception as e:
        warning = warning or f"plugin command discovery unavailable: {e}"
    skills: dict[str, dict] = {}
    try:
        _catalog_skills(cat, skills, module_loader)
    except Exception as e:
        warning = f"skill discovery unavailable: {e}"
    return {
        "pairs": cat.pairs, "sub": {k: v[:] for k, v in module_loader("hermes_cli.commands").SUBCOMMANDS.items()},
        "canon": cat.canon,
        "commands": cat.commands,
        "categories": [{"name": c, "pairs": rows} for c, rows in cat.cat_map.items()],
        "skills": skills, "skill_count": len(skills), "warning": warning}


def slash_completions(text: str = "") -> dict:
    if not text.startswith("/"):
        return {"items": []}
    from hermes_cli.commands_completion import SlashCommandCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import to_plain_text
    from agent.skill_commands import get_skill_commands
    from agent.skill_bundles import get_skill_bundles
    completer = SlashCommandCompleter(
        skill_commands_provider=lambda: get_skill_commands(), skill_bundles_provider=lambda: get_skill_bundles())
    # `kind` reaches the TUI as data (from the providers, not sniffed from ⚡/▣ glyphs):
    # skills/bundles are the only completions for an inline `/skill` typed mid-message.
    skill_names = {key.lstrip("/").lower() for key in (*get_skill_commands(), *get_skill_bundles())}

    def to_items(doc: Document) -> list[dict]:
        # display/display_meta are FormattedText; the TUI contract is a plain string
        # (the raw list trips Ink's row layout into 1-char truncation).
        return [
            {
                "text": c.text, "display": to_plain_text(c.display) if c.display else c.text,
                "meta": to_plain_text(c.display_meta) if c.display_meta else "",
                "kind": "skill" if c.text.strip().lstrip("/").lower() in skill_names else "command"}
            for c in completer.get_completions(doc, None)]
    items = to_items(Document(text, len(text)))
    # Rank + bound while a `/token` is under the cursor (the one stage skills are
    # offered at); an argument stage (`/personality `) keeps its command's order.
    if text.rsplit(" ", 1)[-1].startswith("/"):
        score_of = None
        # Command-token stage: the completer only emits name-prefix matches, so merge in
        # catalog entries whose name SUBSTRING or DESCRIPTION words match (name outranks description).
        if " " not in text and len(text) > 1:
            from tui_gateway.slash_fuzzy import fuzzy_rank_slash_items, normalize_slash_search_query
            items, score_of = fuzzy_rank_slash_items(
                items, to_items(Document("/", 1)), normalize_slash_search_query(text))
        usage, origin_of = _skill_usage_lookup()
        items = _rank_slash_completions(items, usage, origin_of, browsing=text == "/", score_of=score_of)
    else:
        items = items[:_SLASH_COMPLETION_LIMIT]
    text_lower = text.lower()
    for extra_text, extra_meta in _SLASH_EXTRAS:
        if extra_text.startswith(text_lower) and not any(item["text"] == extra_text for item in items):
            items.append({**_item(extra_text, extra_meta), "kind": "command"})
    if (details_items := _details_completions(text)) is not None:
        return {"items": details_items, "replace_from": text.rfind(" ") + 1 if " " in text else len(text)}
    return {"items": items, "replace_from": text.rfind(" ") + 1 if " " in text else 1}


