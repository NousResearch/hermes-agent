"""Regression coverage for the harness timeout adapter, without a browser."""
import sys
import types

import pytest

from tools.browser_use_cli import _ipc_response_timeout_preamble


@pytest.fixture
def harness(monkeypatch):
    helpers = types.ModuleType("browser_harness.helpers")
    exec("""
def _send(req, response_timeout=5.0):
    return response_timeout

def cdp(method, session_id=None, _response_timeout=5.0, **params):
    return _send({'method': method}, response_timeout=_response_timeout)

def page_info():
    return [_send({'meta': 'pending_dialog'}), cdp('Runtime.evaluate')]

def drain_events():
    return _send({'meta': 'drain_events'})
""", helpers.__dict__)
    package = types.ModuleType("browser_harness")
    setattr(package, "helpers", helpers)
    monkeypatch.setitem(sys.modules, "browser_harness", package)
    monkeypatch.setitem(sys.modules, "browser_harness.helpers", helpers)
    return helpers


def test_direct_and_cdp_replies_share_configured_timeout(harness):
    alias = harness.cdp
    exec(_ipc_response_timeout_preamble(20), {})
    assert harness.page_info() == [20.0, 20.0]
    assert harness.drain_events() == 20.0
    assert alias('Runtime.evaluate') == 20.0
    assert alias('Page.captureScreenshot', _response_timeout=60.0) == 60.0
    assert harness._send({}, response_timeout=60.0) == 60.0


@pytest.mark.parametrize("signature", [
    "method, session_id=None, _response_timeout=5.0, extra='kept', **params",
    "method, session_id=None, *, _response_timeout=5.0, extra='kept', **params",
])
def test_timeout_is_resolved_by_name(harness, signature):
    exec(f"def cdp({signature}): return _response_timeout, extra", harness.__dict__)
    alias = harness.cdp
    exec(_ipc_response_timeout_preamble(20), {})
    assert alias('test') == (20.0, 'kept')
    assert alias('test', _response_timeout=60.0) == (60.0, 'kept')


@pytest.mark.parametrize("replacement", [
    "def cdp(method, session_id=None): return session_id",
    "def cdp(method, _response_timeout): return _response_timeout",
    "cdp = None",
])
def test_incompatible_harness_warns_and_keeps_other_helper(harness, replacement, capsys):
    exec(replacement, harness.__dict__)
    exec(_ipc_response_timeout_preamble(20), {})
    assert 'cdp._response_timeout' in capsys.readouterr().err
    assert harness.drain_events() == 20.0


def test_missing_harness_warns(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'browser_harness', None)
    exec(_ipc_response_timeout_preamble(20), {})
    assert 'browser_harness' in capsys.readouterr().err
