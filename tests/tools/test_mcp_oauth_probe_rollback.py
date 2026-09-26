"""``probe_with_rollback`` must restore the token snapshot whenever the probe ends with no token
on disk (#118323): ``details['initialized']`` only proves the MCP handshake returned a result —
the task is claimed before ``server.start()`` runs OAuth — so it must never gate the rollback
alone."""

from __future__ import annotations

import pytest


class _Flow:
    def __init__(self) -> None:
        self.tools = None
        self.discovery_error = None
        self.backup = None
        self.approved = False

    def mark_approved(self) -> None:
        self.approved = True


class _Storage:
    last: "_Storage | None" = None

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.restored: list = []
        _Storage.last = self

    def snapshot(self) -> dict:
        return {"mcp-tokens-srv.json": b"previous-grant"}

    def restore(self, backup: dict) -> None:
        self.restored.append(backup)


class _Manager:
    def __init__(self) -> None:
        self.removed: list = []
        self.restored_entries: list = []

    def remove(self, server_name: str, hermes_home: str | None = None) -> dict:
        self.removed.append(server_name)
        return {"url": "https://srv/mcp"}

    def restore_entry(
        self, server_name: str, entry, hermes_home: str | None = None
    ) -> None:
        self.restored_entries.append(server_name)


def _wire(monkeypatch, probe, tokens_present) -> tuple:
    from hermes_cli import mcp_config
    from tools import mcp_oauth as oauth_mod
    from tools import mcp_dashboard_oauth as dashboard_mod
    from tools import mcp_oauth_manager as manager_mod

    manager = _Manager()
    saved: list = []

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)
    monkeypatch.setattr(
        mcp_config, "_oauth_tokens_present", lambda name: tokens_present
    )
    monkeypatch.setattr(
        mcp_config, "_save_mcp_server", lambda name, cfg: saved.append(name) or True
    )
    monkeypatch.setattr(oauth_mod, "HermesTokenStorage", _Storage)
    monkeypatch.setattr(oauth_mod, "login_connect_timeout", lambda cfg: 30.0)
    monkeypatch.setattr(manager_mod, "get_manager", lambda: manager)
    monkeypatch.setattr(dashboard_mod, "exception_message", lambda exc: str(exc))
    return manager, saved


def test_probe_exception_with_no_token_restores_snapshot_and_never_commits(monkeypatch):
    """The handshook server answered ``initialize`` but the grant never landed on disk: the
    deleted previous token must come back and the failure must surface, not commit."""
    from tools.connectors.mcp_oauth import probe_with_rollback

    def probe(name, cfg, connect_timeout=None, details=None):
        details["initialized"] = True  # MCP initialize answered…
        raise RuntimeError(
            "token request failed after initialize"
        )  # …but OAuth never completed

    manager, saved = _wire(monkeypatch, probe, tokens_present=False)
    flow = _Flow()
    with pytest.raises(RuntimeError, match="token request failed"):
        probe_with_rollback("srv", {"url": "https://srv/mcp"}, "home", flow, False)
    assert manager.removed == ["srv"]
    assert _Storage.last.restored == [{"mcp-tokens-srv.json": b"previous-grant"}]
    assert manager.restored_entries == ["srv"]
    assert saved == [], "an unauthorized attempt must not commit the server config"
    assert flow.approved is False


def test_probe_exception_with_new_token_keeps_the_grant_and_commits(monkeypatch):
    """Discovery failed after a grant that did land on disk: that authorization is real work and
    must be kept — the error is recorded as a discovery error and the config commits."""
    from tools.connectors.mcp_oauth import probe_with_rollback

    def probe(name, cfg, connect_timeout=None, details=None):
        details["initialized"] = True
        raise RuntimeError(
            "tool listing failed"
        )  # grant succeeded, discovery after it failed

    manager, saved = _wire(monkeypatch, probe, tokens_present=True)
    flow = _Flow()
    probe_with_rollback("srv", {"url": "https://srv/mcp"}, "home", flow, False)
    assert _Storage.last.restored == [], "a written grant must not be rolled back"
    assert manager.restored_entries == []
    assert saved == ["srv"]
    assert flow.approved is True
    assert flow.discovery_error == "tool listing failed"


def test_probe_returning_without_token_still_restores_snapshot(monkeypatch):
    """The pre-existing protection: a server that answers the probe yet writes no token is
    rejected with the snapshot restored."""
    from tools.connectors.mcp_oauth import probe_with_rollback

    def probe(name, cfg, connect_timeout=None, details=None):
        details["initialized"] = True
        return []  # answered cleanly, no exception — but also no token

    manager, saved = _wire(monkeypatch, probe, tokens_present=False)
    flow = _Flow()
    with pytest.raises(RuntimeError, match="no OAuth token"):
        probe_with_rollback("srv", {"url": "https://srv/mcp"}, "home", flow, False)
    assert _Storage.last.restored == [{"mcp-tokens-srv.json": b"previous-grant"}]
    assert manager.restored_entries == ["srv"]
    assert saved == []
    assert flow.approved is False
