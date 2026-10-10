"""Tests for MCP JSON-RPC wire-frame logging tap and CLI diagnostics (#90115)."""

import asyncio
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest

from tools.mcp_wire_log import (
    get_mcp_wire_log_path,
    is_wire_log_enabled,
    log_wire_frame,
    tap_streams,
    _WireTapReceiveStream,
    _WireTapSendStream,
)
from hermes_cli.mcp_config import cmd_mcp_log


class DummyStream:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def send(self, item):
        self.sent.append(item)

    def send_nowait(self, item):
        self.sent.append(item)

    def close(self):
        self.closed = True


class DummyRecvStream:
    def __init__(self, items):
        self.items = list(items)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def __aiter__(self):
        return self

    async def receive(self):
        if not self.items:
            raise anyio.EndOfStream()
        return self.items.pop(0)

    async def __anext__(self):
        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None

    def close(self):
        self.closed = True


def test_is_wire_log_enabled_default():
    with patch.dict(os.environ, {}, clear=True):
        assert is_wire_log_enabled("test_server") is False
        assert is_wire_log_enabled("test_server", {}) is False


def test_is_wire_log_enabled_env_var():
    with patch.dict(os.environ, {"HERMES_MCP_WIRE_LOG": "1"}):
        assert is_wire_log_enabled("test_server") is True

    with patch.dict(os.environ, {"HERMES_MCP_WIRE_LOG": "true"}):
        assert is_wire_log_enabled("test_server") is True

    with patch.dict(os.environ, {"HERMES_MCP_WIRE_LOG": "0"}):
        assert is_wire_log_enabled("test_server", {"wire_log": True}) is False


def test_is_wire_log_enabled_config():
    with patch.dict(os.environ, {}, clear=True):
        assert is_wire_log_enabled("srv", {"wire_log": True}) is True
        assert is_wire_log_enabled("srv", {"wire_log": False}) is False

    with patch.dict(os.environ, {}, clear=True):
        with patch("hermes_cli.config.load_config_readonly", return_value={"mcp": {"wire_log": True}}):
            assert is_wire_log_enabled("srv") is True

        with patch("hermes_cli.config.load_config_readonly", return_value={"mcp_servers": {"srv": {"wire_log": True}}}):
            assert is_wire_log_enabled("srv") is True


def test_get_mcp_wire_log_path(tmp_path):
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        path = get_mcp_wire_log_path("github-server")
        assert path.name == "github-server.jsonl"
        assert path.parent == tmp_path / "logs" / "mcp"
        assert path.parent.is_dir()

        # Check sanitization of special characters
        path2 = get_mcp_wire_log_path("srv/sub:test")
        assert "/" not in path2.name
        assert ":" not in path2.name


def test_log_wire_frame(tmp_path):
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        log_wire_frame("test_srv", "send", {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        log_wire_frame("test_srv", "recv", {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}})

        path = get_mcp_wire_log_path("test_srv")
        assert path.exists()

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["server"] == "test_srv"
        assert entry1["direction"] == "send"
        assert entry1["frame"]["method"] == "initialize"
        assert "timestamp" in entry1

        entry2 = json.loads(lines[1])
        assert entry2["server"] == "test_srv"
        assert entry2["direction"] == "recv"
        assert entry2["frame"]["result"]["protocolVersion"] == "2024-11-05"


@pytest.mark.asyncio
async def test_tap_streams_disabled():
    read_s = DummyRecvStream([])
    write_s = DummyStream()

    with patch("tools.mcp_wire_log.is_wire_log_enabled", return_value=False):
        r, w = tap_streams("test_srv", read_s, write_s)
        # Identity check: zero proxy overhead when disabled
        assert r is read_s
        assert w is write_s


@pytest.mark.asyncio
async def test_tap_streams_enabled(tmp_path):
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        with patch("tools.mcp_wire_log.is_wire_log_enabled", return_value=True):
            send_underlying, recv_underlying = anyio.create_memory_object_stream(10)

            tap_r, tap_w = tap_streams("test_srv", recv_underlying, send_underlying)
            assert isinstance(tap_r, _WireTapReceiveStream)
            assert isinstance(tap_w, _WireTapSendStream)

            # Send through tap
            req = {"jsonrpc": "2.0", "id": 100, "method": "tools/list"}
            await tap_w.send(req)

            # Receive through tap
            received_item = await tap_r.receive()
            assert received_item == req

            log_path = get_mcp_wire_log_path("test_srv")
            assert log_path.exists()
            lines = [json.loads(l) for l in log_path.read_text(encoding="utf-8").strip().split("\n")]
            assert len(lines) == 2
            assert lines[0]["direction"] == "send"
            assert lines[0]["frame"]["id"] == 100
            assert lines[1]["direction"] == "recv"
            assert lines[1]["frame"]["id"] == 100


def test_cmd_mcp_log_not_found(tmp_path, capsys):
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        args = Namespace(name="nonexistent", lines=50, follow=False, raw=False)
        cmd_mcp_log(args)
        out = capsys.readouterr().out
        assert "No wire-frame log found" in out


def test_cmd_mcp_log_display(tmp_path, capsys):
    with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
        log_wire_frame("my_server", "send", {"jsonrpc": "2.0", "id": 42, "method": "tools/call", "params": {"name": "read_file"}})
        log_wire_frame("my_server", "recv", {"jsonrpc": "2.0", "id": 42, "result": {"content": "ok"}})

        args = Namespace(name="my_server", lines=50, follow=False, raw=False)
        cmd_mcp_log(args)
        out = capsys.readouterr().out
        assert "SEND" in out
        assert "RECV" in out
        assert "tools/call" in out

        # Test raw flag
        args_raw = Namespace(name="my_server", lines=50, follow=False, raw=True)
        cmd_mcp_log(args_raw)
        out_raw = capsys.readouterr().out
        assert '"method": "tools/call"' in out_raw
