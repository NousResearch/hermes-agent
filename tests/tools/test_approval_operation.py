"""Remembered-command identity must not follow redacted display text or another context."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.approval_operation import approval_operation, approval_operation_key


def key(command="rm build", *, cwd="/workspace", backend="local", config=None):
    with approval_operation(cwd=cwd, backend=backend, config=config or {}):
        return approval_operation_key(command, ["dangerous-command"])


def test_requires_an_explicit_execution_context():
    assert approval_operation_key("rm build", ["dangerous-command"]) == ""
    assert key()
    assert approval_operation_key("rm build", ["dangerous-command"]) == ""


def test_stable_identity_and_display_redaction_collision():
    assert key() == key()
    assert key("curl -H 'Authorization: Bearer secret-one' example.test") != key(
        "curl -H 'Authorization: Bearer secret-two' example.test")
    assert key() != key(cwd="/another-workspace")
    assert key() != key("rm other-build")


def test_ssh_destination_changes_identity():
    first = {"ssh_host": "host-a", "ssh_port": "22", "ssh_user": "worker"}
    second = {**first, "ssh_host": "host-b"}
    assert key(backend="ssh", config=first) != key(backend="ssh", config=second)
    assert key(backend="ssh", config=first) != key(backend="ssh", config={**first, "ssh_user": "root"})


@pytest.mark.parametrize(("backend", "cwd", "config"), [
    ("unknown", "/workspace", {}), ("ssh", "/workspace", {}), ("local", "relative", {}),
])
def test_unknown_execution_identity_is_not_rememberable(backend, cwd, config):
    assert key(backend=backend, cwd=cwd, config=config) == ""


def test_disabled_context_cannot_reuse_outer_context():
    with approval_operation(cwd="/one", backend="local", config={}) as outer:
        before = approval_operation_key("rm build", ["a"])
        assert outer.requested
        with approval_operation(cwd="/two", backend="local", config={}, enabled=False):
            assert approval_operation_key("rm build", ["a"]) == ""
        assert approval_operation_key("rm build", ["a"]) == before


def test_threads_do_not_share_operation_context():
    with approval_operation(cwd="/one", backend="local", config={}):
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(approval_operation_key, "rm build", ["a"]).result() == ""


def test_exact_unicode_context_and_pattern_order():
    with approval_operation(cwd="/work/été", backend="local", config={}):
        assert approval_operation_key("rm 'résumé'", ["b", "a"]) == approval_operation_key(
            "rm 'résumé'", ["a", "b", "a"])
    assert key(cwd=r"C:\work")


def test_invalid_or_unbounded_identity_is_not_emitted():
    with approval_operation(cwd="/work", backend="local", config={}) as context:
        assert approval_operation_key("", ["a"]) == ""
        assert approval_operation_key("x" * 65537, ["a"]) == ""
        assert approval_operation_key("x" * 513, ["a"]) == ""
        assert approval_operation_key("echo one\nrm build", ["a"]) == ""
        assert approval_operation_key("rm build", []) == ""
        assert not context.requested
