"""Focused tests for cli.exec structured stdout/stderr payload fields."""

from types import SimpleNamespace

import tui_gateway.methods_tools as m


def _proc(stdout, stderr, returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_structured_streams_split_both():
    payload = m._cli_exec_output(_proc("out\n", "warn\n"))
    assert payload["output"] == "out\n\nwarn"
    assert payload["stdout"] == "out\n"
    assert payload["stderr"] == "warn\n"


def test_empty_and_none_streams_yield_placeholders():
    for stdout, stderr in (("", ""), (None, None)):
        payload = m._cli_exec_output(_proc(stdout, stderr))
        assert payload["output"] == "(no output)"
        assert payload["stdout"] == ""
        assert payload["stderr"] == ""


def test_bounds_and_default_limit_identity():
    limited = m._cli_exec_output(_proc("abcdefghijk", "1234567890"), limit=8)
    assert limited["stdout"] == "abcdefgh"
    assert limited["stderr"] == "12345678"
    assert limited["output"] == "abcdefgh"

    payload = m._cli_exec_output(_proc("a" * 60_000, "b" * 60_000))
    assert len(payload["stdout"]) == 48_000
    assert len(payload["stderr"]) == 48_000
    assert payload["output"] == "a" * 48_000


def test_payload_key_set_is_exact():
    payload = m._cli_exec_output(_proc("out", "warn"))
    assert set(payload) == {"output", "stdout", "stderr"}
