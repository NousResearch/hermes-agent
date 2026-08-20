#!/usr/bin/env python3
"""
Unit tests for the Project Context tool module in hermes-agent (ported from Cortex Agent).
"""

import pytest
from pathlib import Path
from tools.project_context import (
    project_context_tool,
    find_project_context_file,
    check_project_context_requirements,
)
from tools.registry import registry


def test_project_context_requirements():
    assert check_project_context_requirements() is True


def test_project_context_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = project_context_tool(action="init", project_name="TestApp")
    assert "Successfully initialized" in res
    assert (tmp_path / "HERMES.md").exists()

    # Re-init fails gracefully
    res_duplicate = project_context_tool(action="init")
    assert "already exists" in res_duplicate


def test_project_context_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "HERMES.md").write_text("# Custom Context\nRule 1: Always test code", encoding="utf-8")
    
    res = project_context_tool(action="read")
    assert "Custom Context" in res
    assert "Rule 1: Always test code" in res


def test_project_context_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "HERMES.md").write_text("# Project Context\n\n## Learned Architectural Insights\n", encoding="utf-8")
    
    res = project_context_tool(action="update", insight="Use async Playwright for browser tasks")
    assert "Successfully appended insight" in res
    
    content = (tmp_path / "HERMES.md").read_text(encoding="utf-8")
    assert "Use async Playwright for browser tasks" in content


def test_find_project_context_file_traversal(tmp_path):
    sub_dir = tmp_path / "src" / "components"
    sub_dir.mkdir(parents=True)
    (tmp_path / "CORTEX.md").write_text("# Cortex Root File", encoding="utf-8")

    found = find_project_context_file(start_path=sub_dir)
    assert found is not None
    assert found.name == "CORTEX.md"


def test_project_context_registered_in_registry():
    entry = registry.get_entry("project_context")
    assert entry is not None
    assert entry.name == "project_context"
    assert entry.toolset == "project_context"
