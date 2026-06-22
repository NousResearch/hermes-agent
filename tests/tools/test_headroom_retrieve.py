"""Tests for tools/headroom_retrieve.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import agent.tool_output_compressor as comp
from tools.headroom_retrieve import headroom_retrieve


@pytest.fixture(autouse=True)
def _reset():
    comp._reset_state()
    yield
    comp._reset_state()


def test_retrieve_found():
    """Successfully retrieve a cached original."""
    with comp._cache_lock:
        comp._cache["abc123def456"] = "hello world"
    result = headroom_retrieve(hash="abc123def456")
    import json
    data = json.loads(result)
    assert data["found"] is True
    assert data["content"] == "hello world"


def test_retrieve_not_found():
    """Missing hash returns not found."""
    result = headroom_retrieve(hash="nonexistent")
    import json
    data = json.loads(result)
    assert data["found"] is False


def test_check_requirements_respects_config():
    """Tool is only available when headroom is enabled."""
    from tools.headroom_retrieve import check_requirements
    with patch("hermes_cli.config.load_config_readonly", return_value={
        "context": {"headroom": {"enabled": True}}
    }):
        assert check_requirements() is True
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert check_requirements() is False
