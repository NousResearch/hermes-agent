"""RFC 8628 login invariants through the production CLI and local HTTP."""
from pathlib import Path

import pytest

from evals.mcp_device_flow import DEVICE_GRANT, run_cli


@pytest.mark.parametrize("mode", ["success", "preregistered"])
def test_device_login_registers_authorizes_and_persists(mode):
    result = run_cli(Path(__file__).resolve().parents[2], mode)
    assert result["token_persisted"], result
    assert "TEST-CODE" in result["output"]
    assert "Authenticated" in result["output"]
    tokens = [row for row in result["wire"] if row["path"] == "/token"]
    polls = [row for row in tokens if row["data"]["grant_type"] != "refresh_token"]
    assert any(row["data"]["grant_type"] == "refresh_token" for row in tokens), result
    assert "Connected" in result["refresh_output"], result
    assert all(row["data"]["grant_type"] == DEVICE_GRANT for row in polls)
    assert all(row["data"]["resource"].endswith("/mcp") for row in polls)
    if mode == "success":
        assert polls[2]["at"] - polls[1]["at"] >= 5
    else:
        assert not any(row["path"] == "/register" for row in result["wire"])


@pytest.mark.parametrize("mode", ["denied", "expiry", "unsupported", "issuer", "resource", "malformed", "persistence"])
def test_device_login_failure_does_not_persist_or_disclose_credentials(mode):
    result = run_cli(Path(__file__).resolve().parents[2], mode)
    assert "unrecognized arguments" not in result["output"], result
    assert result["state_preserved"], result
    assert result["token_persisted"] == (mode == "persistence"), result
    assert "Authentication failed" in result["output"], result
    assert "fixture-device-secret" not in result["output"], result
    assert "Authenticated" not in result["output"], result
