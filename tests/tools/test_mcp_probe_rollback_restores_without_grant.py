"""Rollback of the dashboard OAuth probe when no durable grant landed on disk.

``probe_with_rollback`` deletes the pre-attempt tokens, then demotes an exception to a
discovery error whenever the MCP handshake reached initialize. A probe that initialized
but raised before any token file was written must not keep that half-state: the attempt
failed, so the snapshot goes back and the error propagates instead of a tokenless commit.
"""

import pytest

from tools.connectors import mcp_oauth as connectors_mcp_oauth


def _write_tokens(tmp_path, name: str, body: str) -> None:
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage(name, hermes_home=str(tmp_path))
    path = storage._tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _probe_side_effect(details):
    """Probe double: initialize happens, then tool discovery raises; no token is written."""

    def _probe(server_name, cfg, connect_timeout=None, *, details=None):
        if details is not None:
            details["initialized"] = True
        raise RuntimeError("tools/list failed after initialize")

    return _probe


class _RecordingManager:
    """The single call site's manager double: remove/restore of a provider entry."""

    def __init__(self):
        self.removed = []
        self.restored = []

    def remove(self, server_name, hermes_home=None):
        self.removed.append(server_name)
        return object()  # a provider entry to put back on rollback

    def restore_entry(self, server_name, entry, hermes_home=None):
        self.restored.append((server_name, entry))


def _run_probe(monkeypatch, tmp_path, probe, *, tokens_body='{"access_token":"old"}'):
    _write_tokens(tmp_path, "reports", tokens_body)
    manager = _RecordingManager()
    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", lambda: manager)
    monkeypatch.setattr(
        "tools.mcp_oauth.login_connect_timeout", lambda cfg: 30.0, raising=False
    )
    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", probe)
    monkeypatch.setattr(
        "hermes_cli.mcp_config._oauth_tokens_present", lambda name: False
    )
    monkeypatch.setattr(connectors_mcp_oauth, "_ACTIVE", {})

    commits = []
    monkeypatch.setattr(
        connectors_mcp_oauth,
        "_commit",
        lambda server_name, cfg, on_commit, flow=None: commits.append(server_name),
    )

    probe_with_rollback = connectors_mcp_oauth.probe_with_rollback
    pytest.raises(
        RuntimeError,
        probe_with_rollback,
        "reports",
        {"url": "https://mcp.example"},
        str(tmp_path),
        None,
        False,
    )
    return manager, commits


def test_probe_failure_without_durable_grant_restores_snapshot(monkeypatch, tmp_path):
    """Initialize-then-raise with no token file: undo runs, the error propagates, nothing commits."""
    manager, commits = _run_probe(monkeypatch, tmp_path, _probe_side_effect(None))

    assert manager.removed == ["reports"]
    assert len(manager.restored) == 1
    assert commits == []
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("reports", hermes_home=str(tmp_path))
    assert (
        storage._tokens_path().read_text(encoding="utf-8") == '{"access_token":"old"}'
    )


def test_probe_failure_with_durable_grant_still_demotes_to_discovery_error(
    monkeypatch, tmp_path
):
    """A probe that raised but did land a token keeps today's tolerant behaviour: no undo, commit runs,
    the failure is recorded as a discovery error instead of destroying the fresh grant."""
    calls = {"undo": 0}

    from tools.mcp_oauth import HermesTokenStorage

    def _probe(server_name, cfg, connect_timeout=None, *, details=None):
        if details is not None:
            details["initialized"] = True
        _write_tokens(tmp_path, "reports", '{"access_token":"fresh"}')
        raise RuntimeError("tools/list failed after initialize")

    _write_tokens(tmp_path, "reports", '{"access_token":"old"}')
    manager = _RecordingManager()
    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", lambda: manager)
    monkeypatch.setattr(
        "tools.mcp_oauth.login_connect_timeout", lambda cfg: 30.0, raising=False
    )
    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", _probe)
    monkeypatch.setattr(
        "hermes_cli.mcp_config._oauth_tokens_present", lambda name: True
    )
    monkeypatch.setattr(connectors_mcp_oauth, "_ACTIVE", {})

    commits = []
    monkeypatch.setattr(
        connectors_mcp_oauth,
        "_commit",
        lambda server_name, cfg, on_commit, flow=None: commits.append(server_name),
    )

    connectors_mcp_oauth.probe_with_rollback(
        "reports", {"url": "https://mcp.example"}, str(tmp_path), None, False
    )

    assert manager.removed == ["reports"]
    assert manager.restored == []
    assert commits == ["reports"]
    storage = HermesTokenStorage("reports", hermes_home=str(tmp_path))
    assert (
        storage._tokens_path().read_text(encoding="utf-8") == '{"access_token":"fresh"}'
    )
