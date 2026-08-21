"""Regression tests for delegation.request_overrides on the direct-endpoint branch.

The direct base_url branch of _resolve_delegation_credentials (delegation.base_url
set, provider=custom) used to drop delegation.request_overrides on the floor —
the named-provider branch forwards runtime request_overrides (delegate_tool.py
"request_overrides": dict(runtime.get("request_overrides") or {})), but the
direct branch returned no key. That made it impossible to give delegation
children OpenRouter routing hints (extra_body.provider = {"sort": "throughput"})
when delegating straight to openrouter.ai/api/v1 via base_url+api_key.
"""

import pytest

from tools.delegate_tool import _resolve_delegation_credentials


def _cfg(**overrides):
    cfg = {
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key-1234567890",
    }
    cfg.update(overrides)
    return cfg


def test_direct_branch_forwards_request_overrides():
    """delegation.request_overrides flows through the direct-endpoint branch."""
    cfg = _cfg(
        request_overrides={
            "extra_body": {"provider": {"sort": "throughput"}},
        }
    )
    creds = _resolve_delegation_credentials(cfg, parent_agent=None)
    assert creds["request_overrides"] == {
        "extra_body": {"provider": {"sort": "throughput"}},
    }


def test_direct_branch_absent_request_overrides_stays_none():
    """No delegation.request_overrides → None, preserving the old contract."""
    creds = _resolve_delegation_credentials(_cfg(), parent_agent=None)
    assert creds["request_overrides"] is None


def test_direct_branch_non_dict_request_overrides_stays_none():
    """Garbage in config (string/list) must not crash or forward junk."""
    for bad in ("throughput", ["extra_body"], 42):
        creds = _resolve_delegation_credentials(
            _cfg(request_overrides=bad), parent_agent=None
        )
        assert creds["request_overrides"] is None
