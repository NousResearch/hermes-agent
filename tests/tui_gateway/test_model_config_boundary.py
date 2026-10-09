"""TUI config/picker/profile consumers must not import CLI provider/model authority."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    "tui_gateway/entry.py",
    "tui_gateway/methods_config.py",
    "tui_gateway/methods_profiles.py",
    "tui_gateway/methods_session_model_guard.py",
)
FORBIDDEN = (
    "hermes_cli.models", "application_provider_discovery",
    "hermes_cli.model_selection_guards", "hermes_cli.models_validate",
)


def test_tui_presentation_uses_canonical_model_and_provider_queries():
    bad = []
    for name in TARGETS:
        path = ROOT / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("hermes_cli."):
                    modules.append(node.value)
            bad.extend((name, mod) for mod in modules if any(
                mod == old or mod.startswith(old + ".") for old in FORBIDDEN
            ))
    assert bad == []


def test_config_provider_payload_reads_shared_application_projection(monkeypatch):
    import application_provider_listing
    from tui_gateway import server
    monkeypatch.setattr(server, "_resolve_model", lambda: "anthropic/claude-sonnet-5")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "anthropic"}})
    monkeypatch.setattr(
        application_provider_listing, "list_available_providers",
        lambda config: [{"id": "plugin-late", "label": "Plugin", "aliases": [], "authenticated": False}],
    )
    payload = server._cfg_get_provider({})
    assert payload["model"] == "anthropic/claude-sonnet-5"
    assert payload["provider"] == "anthropic"
    assert payload["providers"][0]["id"] == "plugin-late"

def test_final_shared_owners_do_not_import_old_cli_model_authority():
    owners = (
        "application_picker_prewarm.py",
        "application_provider_listing.py",
        "models/selection_conflict.py",
    )
    forbidden = (
        "hermes_cli.models", "hermes_cli.model_switch",
        "application_provider_discovery", "hermes_cli.models_validate",
    )
    for name in owners:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        modules = [
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        modules += [
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not [
            module for module in modules
            if any(module == old or module.startswith(old + ".") for old in forbidden)
        ], name
