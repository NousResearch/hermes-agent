"""Regression tests: paste.collapse must reject non-string ``text``.

``paste.collapse`` stored ``params.get("text", "")`` and passed it straight
to ``Path.write_text``. A list or dict for ``text`` passes the truthiness
check and then makes ``write_text`` raise ``TypeError`` (and
``text.count("\\n")`` would raise first for types without ``.count``). The
entry loop turns that into a generic -32000 handler error and a crash-log
entry, which reads like a gateway fault instead of a bad request.

The fix returns a 4000 invalid-params error for non-string ``text`` before
any disk write.
"""

import pytest

from tui_gateway import server


BAD_TEXT_VALUES = [
    ["line"],
    {"a": 1},
    123,
    1.5,
    True,   # bool passes truthiness, write_text(True) would TypeError
    None,   # not a string, so the guard returns 4000 before the empty check
]


def _request(params):
    return {"jsonrpc": "2.0", "id": "r1", "method": "paste.collapse", "params": params}


@pytest.mark.parametrize("bad", BAD_TEXT_VALUES)
def test_non_string_text_returns_error_not_crash(bad):
    resp = server.handle_request(_request({"text": bad}))
    assert isinstance(resp, dict)
    assert "error" in resp
    assert resp["error"]["code"] == 4000


def test_empty_text_still_returns_4004():
    resp = server.handle_request(_request({"text": ""}))
    assert "error" in resp
    assert resp["error"]["code"] == 4004


def test_valid_text_still_writes_paste():
    resp = server.handle_request(_request({"text": "hello\nworld"}))
    assert "error" not in resp
    assert resp["result"]["lines"] == 2
