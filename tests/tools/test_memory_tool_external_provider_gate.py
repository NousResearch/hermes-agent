"""Regression tests for #57793: legacy memory tool gated when external memory
provider is configured.

When ``memory.provider`` is set in config (e.g., ``memory.provider: mnemosyne``),
the legacy flat-file ``memory`` tool should NOT be exposed to the agent — the
external provider handles all memory operations and the agent reflexively using
the legacy tool bypasses the configured provider (silent data loss).

Acceptance: ``check_memory_requirements()`` returns ``False`` when
``memory.provider`` is set, ``True`` when it is not. Tool is gated via
``check_fn`` so the registry's ``get_definitions()`` excludes it.
"""

from unittest.mock import patch

import pytest

from tools.memory_tool import check_memory_requirements


class TestMemoryToolGate:
    """Lines around check_memory_requirements() in tools/memory_tool.py."""

    def setup_method(self):
        """Reset check_fn cache between tests."""
        from tools.registry import invalidate_check_fn_cache
        try:
            invalidate_check_fn_cache()
        except Exception:
            pass

    def test_returns_true_when_no_provider_configured(self):
        """No memory.provider set → tool is available (legacy behavior)."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   return_value=""):
            assert check_memory_requirements() is True

    def test_returns_false_when_provider_is_mnemosyne(self):
        """memory.provider: mnemosyne → tool should be gated."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   return_value="mnemosyne"):
            assert check_memory_requirements() is False

    def test_returns_false_when_provider_is_honcho(self):
        """memory.provider: honcho → tool should be gated."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   return_value="honcho"):
            assert check_memory_requirements() is False

    def test_returns_false_when_provider_is_mem0(self):
        """memory.provider: mem0 → tool should be gated."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   return_value="mem0"):
            assert check_memory_requirements() is False

    def test_returns_false_when_provider_set_to_anything_truthy(self):
        """Any non-empty provider name → tool should be gated."""
        for provider_name in ["mnemosyne", "honcho", "mem0", "custom-provider", "x"]:
            with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                       return_value=provider_name):
                assert check_memory_requirements() is False, (
                    f"Provider {provider_name!r} should gate the legacy tool"
                )

    def test_falls_back_to_true_on_import_error(self):
        """If the config loader is unavailable (e.g., during tests with broken
        imports), fall back to True (legacy behavior) rather than break the tool."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   side_effect=ImportError("config not available")):
            # Should not raise; should return True (safe fallback)
            assert check_memory_requirements() is True

    def test_falls_back_to_true_on_any_exception(self):
        """Any unexpected exception from the config loader falls back to True."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   side_effect=RuntimeError("config file corrupted")):
            # Should not raise; should return True (safe fallback)
            assert check_memory_requirements() is True


class TestMemoryToolGateIntegrated:
    """End-to-end: with provider configured, the memory tool is excluded from
    ``get_definitions()`` (the registry-level integration)."""

    def setup_method(self):
        from tools.registry import invalidate_check_fn_cache
        try:
            invalidate_check_fn_cache()
        except Exception:
            pass

    def _extract_names(self, definitions):
        """Helper: extract tool name from OpenAI-format schema.

        Schema format: {"type": "function", "function": {"name": ..., ...}}.
        """
        names = []
        for d in definitions:
            if isinstance(d, dict):
                func = d.get("function") or {}
                name = func.get("name") if isinstance(func, dict) else None
                if not name:
                    name = d.get("name")
                names.append(name)
        return names

    def test_memory_tool_excluded_from_definitions_when_provider_set(self):
        """When provider is set, the memory tool schema does NOT appear in
        get_definitions() output."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   return_value="mnemosyne"):
            from tools.registry import registry, invalidate_check_fn_cache
            invalidate_check_fn_cache()
            # Make sure the memory tool is registered
            registry.get_entry("memory")
            entries_by_name = {e.name: e for e in registry._snapshot_entries()}

            # The entry exists, but check_fn returns False
            assert "memory" in entries_by_name
            entry = entries_by_name["memory"]
            assert entry.check_fn is not None

            # get_definitions with check_fn=False should exclude
            definitions = registry.get_definitions(tool_names={"memory"})
            names = self._extract_names(definitions)
            assert "memory" not in names

    def test_memory_tool_included_when_no_provider_set(self):
        """Without provider, the memory tool IS exposed (legacy behavior)."""
        with patch("hermes_cli.plugins_cmd._get_current_memory_provider",
                   return_value=""):
            from tools.registry import registry, invalidate_check_fn_cache
            invalidate_check_fn_cache()
            registry.get_entry("memory")
            definitions = registry.get_definitions(tool_names={"memory"})
            names = self._extract_names(definitions)
            assert "memory" in names
