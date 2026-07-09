"""Regression guard for the `wiki` memory provider (llmwiki_hermes recovery).

The wiki provider is a thin wrapper (`plugins/memory/wiki/`) over the
vendored top-level `llmwiki_hermes` package. A profile with
`memory.provider: wiki` is dead in the water if that package goes missing
or the wrapper stops registering `WikiMemoryProvider` — exactly the outage
this recovery fixed. These tests fail loudly if that regresses, so CI
catches it before a `provider: wiki` profile does at runtime.
"""

from __future__ import annotations

import importlib

import pytest


def test_llmwiki_hermes_package_importable():
    """The vendored backing package must import (no ModuleNotFoundError)."""
    mod = importlib.import_module("llmwiki_hermes")
    assert mod is not None


def test_wiki_provider_class_conforms_to_memory_provider_abc():
    """WikiMemoryProvider must instantiate as the current MemoryProvider ABC."""
    from agent.memory_provider import MemoryProvider
    from llmwiki_hermes.provider.plugin import WikiMemoryProvider

    provider = WikiMemoryProvider()
    assert isinstance(provider, MemoryProvider)
    assert provider.name == "wiki"
    # Lifecycle hooks the plugin manifest declares must exist.
    assert hasattr(provider, "on_session_end")
    assert hasattr(provider, "on_pre_compress")


def test_wiki_plugin_wrapper_registers_provider():
    """`register(ctx)` must hand a WikiMemoryProvider to the plugin context."""
    from plugins.memory.wiki import register

    captured = []

    class _Ctx:
        def register_memory_provider(self, provider):
            captured.append(provider)

    register(_Ctx())
    assert len(captured) == 1
    assert type(captured[0]).__name__ == "WikiMemoryProvider"


def test_wiki_discoverable_and_loadable_by_hermes():
    """Hermes-native discovery/load must surface and construct `wiki`."""
    from plugins.memory import discover_memory_providers, load_memory_provider

    names = {row[0] for row in discover_memory_providers()}
    assert "wiki" in names, f"'wiki' not discovered; found {sorted(names)}"

    provider = load_memory_provider("wiki")
    assert provider is not None
    assert type(provider).__name__ == "WikiMemoryProvider"


def test_wiki_provider_cli_exposes_expected_commands():
    """The provider CLI must expose the subcommands the wrapper dispatches."""
    from llmwiki_hermes.provider import cli as provider_cli

    for command in ("init", "ingest", "reindex", "recall", "doctor", "compact"):
        assert hasattr(provider_cli, command), f"provider CLI missing {command!r}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
