"""Idle hello needs no conditional workers; first use must remain shutdown-safe."""
from __future__ import annotations

import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tui_gateway import compute_host, host_conditional, server


def test_idle_host_hello_does_not_construct_conditional_runtime(monkeypatch):
    def unexpected(_host):
        raise AssertionError("idle hello must not construct conditional workers")

    monkeypatch.setattr(host_conditional, "HostConditionalProtocol", unexpected)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setenv("HERMES_COMPUTE_HOST_HEARTBEAT_SECS", "0")
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(compute_host.signal, "signal", lambda *_args: None)
    out = io.StringIO()
    compute_host.run_host(stdin=io.StringIO(), stdout=out)
    hello = json.loads(out.getvalue())
    assert hello["type"] == "hello"
    assert hello["host_pid"] == os.getpid()
    assert hello["boot_id"]


@pytest.mark.parametrize("boundary", ["concurrent_use", "close"])
def test_paused_first_use_initializes_once_and_cannot_outlive_close(monkeypatch, boundary):
    host = compute_host.ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    entered, resume, second_started = threading.Event(), threading.Event(), threading.Event()
    original = host_conditional.HostConditionalProtocol
    created = []

    def construct(current):
        entered.set()
        assert resume.wait(5)
        protocol = original(current)
        created.append(protocol)
        return protocol

    def close_protocol(protocol):
        # Closing takes protocol locks; the initializer's lock must already be released.
        assert host._conditional_lock.acquire(blocking=False)
        host._conditional_lock.release()
        original_close(protocol)

    def second_use():
        second_started.set()
        return host._conditional_protocol()

    original_close = original.close
    monkeypatch.setattr(original, "close", close_protocol)
    monkeypatch.setattr(host_conditional, "HostConditionalProtocol", construct)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(host._conditional_protocol)
            try:
                assert entered.wait(5)
                if boundary == "close":
                    second = executor.submit(host.close)
                    assert host._closed.wait(5)
                else:
                    second = executor.submit(second_use)
                    assert second_started.wait(5)
            finally:
                resume.set()
            first_result = first.result(timeout=5)
            result = second.result(timeout=5)
            assert len(created) == 1
            protocol = created[0]
            if boundary == "concurrent_use":
                assert first_result is result is protocol
                host.close()
            else:
                assert first_result is None
        assert protocol._membership.stop.is_set()
        assert host._conditional_protocol() is None
        host.handle_frame({"type": "conditional", "boot_id": host._boot_id,
                           "action": "prepare", "request_id": "after-close"})
        assert json.loads(host._stdout.getvalue())["error"] == 4007
        assert created == [protocol]
    finally:
        resume.set()
        host.close()
