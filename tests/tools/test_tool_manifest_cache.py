"""Tests for the mtime-based tool discovery manifest cache.

The manifest cache avoids re-parsing 31 Python files with AST on every
process start when none of them have changed since the last run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest


def _load_registry_module():
    """Import tools/registry.py fresh so module-level state is clean."""
    spec = importlib.util.spec_from_file_location(
        "_registry_under_test",
        Path(__file__).resolve().parents[2] / "tools" / "registry.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tool_dir(tmp_path):
    """Create a minimal tools directory with two self-registering files."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_text("", encoding="utf-8")
    (tools / "registry.py").write_text("# stub\n", encoding="utf-8")
    (tools / "alpha.py").write_text(
        "from tools.registry import registry\n"
        "registry.register(name='alpha', toolset='a', schema={}, handler=lambda *_a, **_k: '{}')\n",
        encoding="utf-8",
    )
    (tools / "beta.py").write_text(
        "from tools.registry import registry\n"
        "registry.register(name='beta', toolset='b', schema={}, handler=lambda *_a, **_k: '{}')\n",
        encoding="utf-8",
    )
    (tools / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    return tools


@pytest.fixture()
def cache_path(tmp_path):
    return tmp_path / "cache" / "tool_manifest.json"


class TestManifestCache:
    def test_save_and_load_roundtrip(self, tool_dir, cache_path):
        module = _load_registry_module()
        module._save_manifest_cache(cache_path, tools_dir=tool_dir, module_names=["tools.alpha", "tools.beta"])
        assert cache_path.exists()
        loaded = module._load_manifest_cache(cache_path, tools_dir=tool_dir)
        assert loaded is not None
        assert loaded == ["tools.alpha", "tools.beta"]

    def test_cache_miss_when_file_missing(self, tool_dir, cache_path):
        module = _load_registry_module()
        assert module._load_manifest_cache(cache_path, tools_dir=tool_dir) is None

    def test_cache_miss_when_mtime_changes(self, tool_dir, cache_path):
        module = _load_registry_module()
        module._save_manifest_cache(cache_path, tools_dir=tool_dir, module_names=["tools.alpha", "tools.beta"])
        time.sleep(0.05)
        (tool_dir / "alpha.py").touch()
        assert module._load_manifest_cache(cache_path, tools_dir=tool_dir) is None

    def test_cache_miss_when_file_added(self, tool_dir, cache_path):
        module = _load_registry_module()
        module._save_manifest_cache(cache_path, tools_dir=tool_dir, module_names=["tools.alpha", "tools.beta"])
        (tool_dir / "gamma.py").write_text(
            "from tools.registry import registry\n"
            "registry.register(name='gamma', toolset='g', schema={}, handler=lambda *_a, **_k: '{}')\n",
            encoding="utf-8",
        )
        assert module._load_manifest_cache(cache_path, tools_dir=tool_dir) is None

    def test_cache_miss_when_file_removed(self, tool_dir, cache_path):
        module = _load_registry_module()
        module._save_manifest_cache(cache_path, tools_dir=tool_dir, module_names=["tools.alpha", "tools.beta"])
        (tool_dir / "beta.py").unlink()
        assert module._load_manifest_cache(cache_path, tools_dir=tool_dir) is None

    def test_cache_miss_when_corrupted_json(self, tool_dir, cache_path):
        module = _load_registry_module()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("not json", encoding="utf-8")
        assert module._load_manifest_cache(cache_path, tools_dir=tool_dir) is None

    def test_discover_uses_cache_to_skip_ast_scan(self, tool_dir, cache_path, monkeypatch):
        import unittest.mock

        module = _load_registry_module()
        module._save_manifest_cache(cache_path, tools_dir=tool_dir, module_names=["tools.alpha", "tools.beta"])
        ast_calls = {"count": 0}
        original_scan = module._module_registers_tools

        def counting_scan(path):
            ast_calls["count"] += 1
            return original_scan(path)

        monkeypatch.setattr(module, "_module_registers_tools", counting_scan)
        with unittest.mock.patch.object(module.importlib, "import_module", side_effect=lambda name: None):
            result = module.discover_builtin_tools(tools_dir=tool_dir, manifest_cache_path=cache_path)
        assert sorted(result) == ["tools.alpha", "tools.beta"]
        assert ast_calls["count"] == 0, "AST scan should be skipped when cache is valid"

    def test_discover_falls_back_to_scan_when_cache_invalid(self, tool_dir, cache_path, monkeypatch):
        import unittest.mock

        module = _load_registry_module()
        ast_calls = {"count": 0}
        original_scan = module._module_registers_tools

        def counting_scan(path):
            ast_calls["count"] += 1
            return original_scan(path)

        monkeypatch.setattr(module, "_module_registers_tools", counting_scan)
        with unittest.mock.patch.object(module.importlib, "import_module", side_effect=lambda name: None):
            result = module.discover_builtin_tools(tools_dir=tool_dir, manifest_cache_path=cache_path)
        assert sorted(result) == ["tools.alpha", "tools.beta"]
        assert ast_calls["count"] > 0, "AST scan should run when cache is invalid"
        assert cache_path.exists(), "Cache should be written after fallback scan"

    def test_discover_skips_cache_when_env_disabled(self, tool_dir, cache_path, monkeypatch):
        import unittest.mock

        module = _load_registry_module()
        module._save_manifest_cache(cache_path, tools_dir=tool_dir, module_names=["tools.alpha", "tools.beta"])
        monkeypatch.setenv("HERMES_NO_TOOL_CACHE", "1")
        ast_calls = {"count": 0}
        original_scan = module._module_registers_tools

        def counting_scan(path):
            ast_calls["count"] += 1
            return original_scan(path)

        monkeypatch.setattr(module, "_module_registers_tools", counting_scan)
        with unittest.mock.patch.object(module.importlib, "import_module", side_effect=lambda name: None):
            result = module.discover_builtin_tools(tools_dir=tool_dir, manifest_cache_path=cache_path)
        assert sorted(result) == ["tools.alpha", "tools.beta"]
        assert ast_calls["count"] > 0, "AST scan should run when cache is disabled via env"
