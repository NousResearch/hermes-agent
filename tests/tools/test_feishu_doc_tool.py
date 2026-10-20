"""Tests for the ``feishu_doc_read`` tool handler and its plugin-compat shim.

Exercises the handler in isolation with a fake lark client injected through the
module's thread-local ``set_client`` — no real ``lark_oapi`` SDK required.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from tools import feishu_doc_tool as fdt


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Isolate the handler from the environment:

    - ``lark_oapi`` is an optional dependency and may not be installed in the test
      environment, so stub ``build_request`` to a sentinel — tests then exercise the
      response-handling logic regardless. (The "lark not installed" test overrides this
      to raise ``ImportError`` on purpose.)
    - The lark client lives in a thread-local; clear it around each test so a client
      leaked by one test can't mask the client-is-None branch in another.
    """
    monkeypatch.setattr(fdt, "build_request", lambda *a, **k: object())
    fdt.set_client(None)
    yield
    fdt.set_client(None)


class _FakeClient:
    """Stand-in for the lark client: returns a canned response, records the request."""

    def __init__(self, response):
        self._response = response
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        return self._response


def _resp(**kw):
    kw.setdefault("raw", None)
    kw.setdefault("data", None)
    return SimpleNamespace(**kw)


def test_missing_doc_token_is_rejected():
    assert "doc_token is required" in json.loads(fdt._handle_feishu_doc_read({}))["error"]
    # whitespace-only collapses to empty after .strip() and is rejected the same way
    assert "doc_token is required" in json.loads(fdt._handle_feishu_doc_read({"doc_token": "   "}))["error"]


def test_no_client_means_no_feishu_context():
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert "not available" in result["error"]


def test_lark_not_installed_is_surfaced(monkeypatch):
    fdt.set_client(_FakeClient(_resp(code=0)))

    def _raise(*a, **k):
        raise ImportError("no lark_oapi")

    # build_request is imported into this module's namespace, so patch it here.
    monkeypatch.setattr(fdt, "build_request", _raise)
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert "lark_oapi not installed" in result["error"]


def test_success_reads_content_from_raw_body():
    raw = SimpleNamespace(content=json.dumps({"data": {"content": "Hello doc"}}))
    fdt.set_client(_FakeClient(_resp(code=0, raw=raw)))
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert result == {"success": True, "content": "Hello doc"}


def test_api_error_code_is_reported():
    fdt.set_client(_FakeClient(_resp(code=1, msg="boom")))
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert "code=1" in result["error"]
    assert "boom" in result["error"]


def test_falls_back_to_typed_data_dict_when_raw_body_absent():
    # raw is None -> raw_body() returns None -> handler uses the typed .data dict
    fdt.set_client(_FakeClient(_resp(code=0, raw=None, data={"content": "typed"})))
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert result == {"success": True, "content": "typed"}


def test_falls_back_to_typed_data_object_without_dict():
    # non-dict .data goes through the getattr(data, "content", ...) branch
    fdt.set_client(_FakeClient(_resp(code=0, raw=None, data=SimpleNamespace(content="obj"))))
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert result == {"success": True, "content": "obj"}


def test_no_content_anywhere_is_an_error():
    fdt.set_client(_FakeClient(_resp(code=0, raw=None, data=None)))
    result = json.loads(fdt._handle_feishu_doc_read({"doc_token": "d1"}))
    assert "No content returned" in result["error"]


def test_compat_shim_resolves_a_known_pointer():
    # PEP 562 __getattr__ maps the historical ``logger`` name to tools.approval.logger.
    assert isinstance(fdt.logger, logging.Logger)


def test_compat_shim_rejects_unknown_attribute():
    with pytest.raises(AttributeError):
        fdt.nonexistent_attribute
