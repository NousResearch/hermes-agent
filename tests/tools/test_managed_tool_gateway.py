import os
import json
from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "managed_tool_gateway.py"
MODULE_SPEC = spec_from_file_location("managed_tool_gateway_test_module", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
managed_tool_gateway = module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = managed_tool_gateway
MODULE_SPEC.loader.exec_module(managed_tool_gateway)
is_managed_tool_gateway_ready = managed_tool_gateway.is_managed_tool_gateway_ready
resolve_managed_tool_gateway = managed_tool_gateway.resolve_managed_tool_gateway


def test_resolve_managed_tool_gateway_derives_vendor_origin_from_shared_domain():
    with patch.dict(
        os.environ,
        {
            "TOOL_GATEWAY_DOMAIN": "nousresearch.com",
        },
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        result = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=lambda: "nous-token",
        )

    assert result is not None
    assert result.gateway_origin == "https://firecrawl-gateway.nousresearch.com"
    assert result.nous_user_token == "nous-token"
    assert result.managed_mode is True


def test_is_managed_tool_gateway_ready_skips_refresh_for_expired_cached_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {
            "nous": {
                "access_token": "expired-token",
                "refresh_token": "refresh-token",
                "expires_at": expired_at,
            }
        }
    }))
    refresh_calls = []

    def _record_refresh(*, refresh_skew_seconds=120, **_kwargs):
        refresh_calls.append(refresh_skew_seconds)
        return "fresh-token"

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_nous_access_token",
        _record_refresh,
    )

    with patch.dict(
        os.environ,
        {"TOOL_GATEWAY_DOMAIN": "nousresearch.com"},
        clear=False,
    ), patch.object(managed_tool_gateway, "managed_nous_tools_enabled", return_value=True):
        assert is_managed_tool_gateway_ready("modal") is True

    assert refresh_calls == []
