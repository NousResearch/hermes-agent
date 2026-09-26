"""Resource admission for single-process local model endpoints."""
from types import SimpleNamespace
from threading import Event, Thread

from tools.delegate_tool_dispatch import (
    _acquire_resource_gate,
    _release_resource_gate,
)


def _child(url):
    return SimpleNamespace(base_url=url)


def test_same_local_endpoint_is_serialized_and_released():
    first = _acquire_resource_gate(_child("http://127.0.0.1:8080/v1"))
    acquired = Event()
    release_second = Event()
    second_gate = []

    def wait_for_second():
        gate = _acquire_resource_gate(_child("http://localhost:8080/v1"))
        second_gate.append(gate)
        acquired.set()
        release_second.wait(2)
        _release_resource_gate(gate)

    thread = Thread(target=wait_for_second)
    thread.start()
    assert not acquired.wait(0.05)

    _release_resource_gate(first)
    assert acquired.wait(1)
    release_second.set()
    thread.join(1)
    assert not thread.is_alive()


def test_different_local_endpoints_can_run_concurrently():
    first = _acquire_resource_gate(_child("http://127.0.0.1:8080/v1"))
    acquired = Event()
    second_gate = []

    def acquire_other():
        gate = _acquire_resource_gate(_child("http://127.0.0.1:11435/v1"))
        second_gate.append(gate)
        acquired.set()

    thread = Thread(target=acquire_other)
    thread.start()
    assert acquired.wait(1)
    thread.join(1)
    assert second_gate

    _release_resource_gate(first)
    _release_resource_gate(second_gate[0])


def test_remote_endpoints_are_not_serialized_by_local_gate():
    first = _acquire_resource_gate(_child("https://api.example.test/v1"))
    second = _acquire_resource_gate(_child("https://api.example.test/v1"))
    assert first is None
    assert second is None


def test_nested_child_skips_local_gate_to_avoid_orchestrator_deadlock():
    child = _child("http://127.0.0.1:8080/v1")
    child._delegate_depth = 2
    assert _acquire_resource_gate(child) is None
