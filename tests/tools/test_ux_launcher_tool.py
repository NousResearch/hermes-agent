"""Tests for the UX launcher/shortcut tool.

Asserts:
- success envelope shape
- category filtering behavior
- empty-filter default behavior
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


UX_LAUNCHER_PATH = os.path.join(TOOLS_DIR, "ux_launcher_tool.py")
if not os.path.exists(UX_LAUNCHER_PATH):
    pytest.fail(f"Missing tool module: {UX_LAUNCHER_PATH}")

ux_launcher_tool = _load_module("ux_launcher_tool", UX_LAUNCHER_PATH)
run = ux_launcher_tool.run
check_fn = ux_launcher_tool.check_fn
MENU_ITEMS = ux_launcher_tool.MENU_ITEMS
NAME = ux_launcher_tool.NAME


def test_check_fn_always_returns_true():
    assert check_fn({}) is True
    assert check_fn({"filter": "workspace"}) is True
    assert check_fn({"category": [], "foo": "bar"}) is True


def test_success_envelope_shape_empty_filter():
    envelope = run({})
    assert isinstance(envelope, dict)
    assert envelope.get("kind") == "success"
    assert "data" in envelope
    data = envelope["data"]
    assert data["tool"] == NAME
    assert data["total_entries"] == len(MENU_ITEMS)
    assert data["returned_entries"] == len(MENU_ITEMS)
    assert isinstance(data["menu"], list)
    assert len(data["menu"]) == len(MENU_ITEMS)
    assert isinstance(data["full_menu"], list)
    assert len(data["full_menu"]) == len(MENU_ITEMS)
    assert data["filter_keywords"] == []
    for item in data["menu"]:
        assert "id" in item
        assert "command" in item
        assert "category" in item
        assert "guidance" in item


def test_category_filtering_limits_returned_menu():
    envelope = run({"category": "ops"})
    assert envelope["kind"] == "success"
    data = envelope["data"]
    assert data["tool"] == NAME
    assert len(data["filter_keywords"]) == 1
    assert "ops" in data["filter_keywords"]
    assert data["returned_entries"] == len(data["menu"])
    for item in data["menu"]:
        assert item["category"] == "ops" or item["id"].startswith("or depends on filter contract")
        assert item["category"] == "ops" or item["id"] == "cron_list" or item["id"] == "docker_health"


def test_category_filtering_only_dev_entries():
    envelope = run({"filter": "dev"})
    assert envelope["kind"] == "success"
    data = envelope["data"]
    assert data["filter_keywords"] == ["dev"]
    ids = {item["id"] for item in data["menu"]}
    assert ids == {"hermes_tools", "hermes_tools_list"}
    assert all(item["category"] == "dev" for item in data["menu"])


def test_multiple_filters_union_behavior():
    envelope = run({"filters": ["workspace", "ops"]})
    assert envelope["kind"] == "success"
    data = envelope["data"]
    assert set(data["filter_keywords"]) == {"workspace", "ops"}
    ids = [item["id"] for item in data["menu"]]
    for identifier in ids:
        assert identifier in {
            "cron_list",
            "repo_health",
            "test_health",
            "docker_health",
            "todo_list_status",
            "hermes_tools",
            "hermes_tools_list",
        }


def test_empty_string_filter_defaults_to_full_menu():
    envelope = run({"category": ""})
    assert envelope["kind"] == "success"
    data = envelope["data"]
    assert data["filter_keywords"] == []
    assert data["returned_entries"] == len(MENU_ITEMS)
    assert len(data["menu"]) == len(MENU_ITEMS)


def test_invalid_filter_keywords_are_ignored():
    envelope = run({"filter": "not-a-real-category,dev"})
    assert envelope["kind"] == "success"
    data = envelope["data"]
    assert data["filter_keywords"] == ["dev"]
    assert all(item["category"] == "dev" for item in data["menu"])
