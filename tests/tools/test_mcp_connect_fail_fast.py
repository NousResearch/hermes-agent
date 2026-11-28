"""A construction-time MCP failure must reach the caller as that error (#146924).

``MCPServerRunMixin.run`` is a detached task: an exception escaping ``_prepare_run`` (a TLS
context that cannot be built, an unreadable client cert) was only logged as "Task exception
was never retrieved", while ``start()`` kept waiting on ``_ready`` and the caller burned the
whole ``connect_timeout`` and reported an ambiguous timeout instead of the real cause. These
tests pin the fail-fast contract: ``start()`` raises the original error immediately, and a
``_prepare_run`` that reports "do not start" still exits quietly.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tools.mcp_tool import MCPServerTask


def test_start_raises_the_construction_error_instead_of_waiting_for_the_timeout(monkeypatch):
    task = MCPServerTask("broken")
    connect_timeout = 30.0

    async def _boom(config):
        raise FileNotFoundError(2, "No such file or directory", "/dead/certifi/cacert.pem")

    monkeypatch.setattr(task, "_prepare_run", _boom)

    async def _drive():
        return await asyncio.wait_for(
            task.start({"url": "https://mcp.example.test/mcp", "connect_timeout": connect_timeout}),
            timeout=5,
        )

    started = time.monotonic()
    with pytest.raises(FileNotFoundError) as caught:
        asyncio.run(_drive())
    elapsed = time.monotonic() - started

    assert "certifi" in str(caught.value), "the original error text must survive, not a timeout"
    assert elapsed < connect_timeout / 2, f"caller waited {elapsed:.1f}s — fail-fast did not fire"


def test_run_exits_quietly_when_prepare_run_declines(monkeypatch):
    """A ``_prepare_run`` that reports "do not start" leaves the retry loop unentered.

    Pinned on ``run()`` itself, not on ``start()``: every real decline publishes its error
    through ``_publish_error`` (which fires ``_ready``), so what this guards is that the
    branch returns instead of falling into the reconnect ladder.
    """
    task = MCPServerTask("declined")

    async def _decline(config):
        return False

    monkeypatch.setattr(task, "_prepare_run", _decline)

    asyncio.run(asyncio.wait_for(task.run({"url": "https://mcp.example.test/mcp"}), timeout=5))
    assert not task._ready.is_set(), "run() must not claim readiness on a decline"
