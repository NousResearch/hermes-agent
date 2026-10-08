"""``machine.facts`` answers the host summary through the registered handler and its contract, and
answers the same for every profile one backend serves."""

from __future__ import annotations

import pytest

import tui_gateway.server as server
from hermes_platform.host import facts, summary


@pytest.fixture
def n1x(monkeypatch):
    pinned = {"os_family": "win32", "native_arch": "arm64", "gpu_class": "unknown",
              "cpu_model": "NVIDIA RTX Spark N1X", "cpu_vendor": "NVIDIA", "ram_total_bytes": 128 * 2**30}
    for name, value in pinned.items():
        monkeypatch.setattr(facts, name, lambda value=value: value)
    monkeypatch.setattr(summary, "_account", lambda: ("ada", "Ada Lovelace"))
    summary.summary.cache_clear()
    yield
    summary.summary.cache_clear()


def _call(params: dict) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": 7, "method": "machine.facts", "params": params})


def test_machine_facts_over_the_wire(n1x):
    result = _call({})["result"]

    assert result["is_spark"] is True
    assert result["machine_kind"] == "Spark"
    assert result["has_nvidia_gpu"] is False
    assert result["full_name"] == "Ada Lovelace"
    assert result["machine"]["native_arch"] == "arm64"
    assert result["machine"]["ram_gb"] == 128


def test_machine_facts_takes_no_profile(n1x):
    # Host-level: a profile param is a client bug, not a scope.
    assert _call({"profile": "work"})["error"]["code"] == 4000
