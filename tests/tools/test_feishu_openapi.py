import json
import sys
import types

import pytest

from tools.feishu_openapi import (
    FeishuOpenAPIError,
    _normalize_queries,
    check_feishu_openapi_requirements,
    load_settings,
)


def test_load_settings_reads_feishu_env(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setenv("FEISHU_DOMAIN", "lark")

    settings = load_settings()

    assert settings.app_id == "app"
    assert settings.app_secret == "secret"
    assert settings.domain_name == "lark"
    assert settings.base_url == "https://open.larksuite.com"


def test_check_requirements_false_without_credentials(monkeypatch):
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)

    assert check_feishu_openapi_requirements() is False


def test_check_requirements_true_with_fake_lark(monkeypatch):
    monkeypatch.setenv("FEISHU_APP_ID", "app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret")
    monkeypatch.setitem(sys.modules, "lark_oapi", types.SimpleNamespace())

    assert check_feishu_openapi_requirements() is True


def test_normalize_queries_drops_empty_and_stringifies_bool():
    assert _normalize_queries({"a": 1, "b": True, "c": None, "d": ""}) == [("a", "1"), ("b", "true")]


def test_feishu_openapi_error_carries_code_and_data():
    err = FeishuOpenAPIError("bad", code=999, data={"msg": "bad"})

    assert str(err) == "bad"
    assert err.code == 999
    assert err.data == {"msg": "bad"}