"""Tests for Feishu tool client fallback outside comment contexts."""

import types

from tools import feishu_doc_tool, feishu_drive_tool


class _FakeBuilder:
    def __init__(self, built_client):
        self.built_client = built_client
        self.calls = []

    def app_id(self, value):
        self.calls.append(("app_id", value))
        return self

    def app_secret(self, value):
        self.calls.append(("app_secret", value))
        return self

    def log_level(self, value):
        self.calls.append(("log_level", value))
        return self

    def build(self):
        self.calls.append(("build", None))
        return self.built_client


class _FakeClientFactory:
    def __init__(self, builder):
        self._builder = builder

    def builder(self):
        return self._builder


def _install_fake_lark(monkeypatch, built_client="env-client"):
    builder = _FakeBuilder(built_client)
    fake_lark = types.SimpleNamespace(
        Client=_FakeClientFactory(builder),
        LogLevel=types.SimpleNamespace(ERROR="ERROR"),
    )
    monkeypatch.setitem(__import__("sys").modules, "lark_oapi", fake_lark)
    return builder, built_client


def _clear_thread_client(module):
    if hasattr(module._local, "client"):
        delattr(module._local, "client")


def test_doc_tool_builds_client_from_feishu_env_when_no_comment_client(monkeypatch):
    _clear_thread_client(feishu_doc_tool)
    builder, built_client = _install_fake_lark(monkeypatch, built_client="doc-client")
    monkeypatch.setenv("FEISHU_APP_ID", "app-id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "app-secret")
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)

    assert feishu_doc_tool.get_client() == built_client
    assert ("app_id", "app-id") in builder.calls
    assert ("app_secret", "app-secret") in builder.calls
    assert ("log_level", "ERROR") in builder.calls
    assert ("build", None) in builder.calls


def test_drive_tool_builds_client_from_lark_env_when_no_comment_client(monkeypatch):
    _clear_thread_client(feishu_drive_tool)
    builder, built_client = _install_fake_lark(monkeypatch, built_client="drive-client")
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    monkeypatch.setenv("LARK_APP_ID", "lark-id")
    monkeypatch.setenv("LARK_APP_SECRET", "lark-secret")

    assert feishu_drive_tool.get_client() == built_client
    assert ("app_id", "lark-id") in builder.calls
    assert ("app_secret", "lark-secret") in builder.calls
    assert ("build", None) in builder.calls


def test_existing_comment_client_takes_precedence_over_env(monkeypatch):
    builder, _ = _install_fake_lark(monkeypatch, built_client="env-client")
    monkeypatch.setenv("FEISHU_APP_ID", "app-id")
    monkeypatch.setenv("FEISHU_APP_SECRET", "app-secret")

    feishu_doc_tool.set_client("comment-client")

    try:
        assert feishu_doc_tool.get_client() == "comment-client"
        assert builder.calls == []
    finally:
        _clear_thread_client(feishu_doc_tool)


def test_no_env_credentials_keeps_client_unavailable(monkeypatch):
    _clear_thread_client(feishu_doc_tool)
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    monkeypatch.delenv("LARK_APP_ID", raising=False)
    monkeypatch.delenv("LARK_APP_SECRET", raising=False)

    assert feishu_doc_tool.get_client() is None
