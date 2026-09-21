"""Rollback of ``probe_with_rollback`` must never destroy a newer grant it did not create.

``undo()`` reverts the token snapshot after a failed probe. The probe removed the pre-attempt
state first, so a grant written while the probe was in flight (a gateway-path OAuth flow in
another process) is newer than the snapshot: rolling back over it would silently destroy the
user's just-completed login. The dashboard path already restores with ``only_if_absent`` for
exactly this reason; the probe path must match.
"""

import pytest

from tools.connectors import mcp_oauth as connectors_mcp_oauth


def _write_tokens(home, name: str, body: str) -> None:
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage(name, hermes_home=str(home))
    path = storage._tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


class _RecordingManager:
    """Manager double: remove/restore of a provider entry."""

    def __init__(self):
        self.restored = []

    def remove(self, server_name, hermes_home=None):
        return object()

    def restore_entry(self, server_name, entry, hermes_home=None):
        self.restored.append(server_name)


def _patch_harness(monkeypatch, home, probe, *, tokens_present):
    # The probe builds its storage without an explicit home, so the global home must point at
    # the isolated tmp dir for the whole call — otherwise it reads the real token dir.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    manager = _RecordingManager()
    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", lambda: manager)
    monkeypatch.setattr(
        "tools.mcp_oauth.login_connect_timeout", lambda cfg: 30.0, raising=False
    )
    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", probe)
    monkeypatch.setattr(
        "hermes_cli.mcp_config._oauth_tokens_present", lambda name: tokens_present
    )
    monkeypatch.setattr(connectors_mcp_oauth, "_ACTIVE", {})
    monkeypatch.setattr(
        connectors_mcp_oauth,
        "_commit",
        lambda server_name, cfg, on_commit, flow=None: None,
    )
    return manager


def _run_probe(home):
    return connectors_mcp_oauth.probe_with_rollback(
        "reports", {"url": "https://mcp.example"}, str(home), None, False
    )


def test_undo_skips_restore_when_newer_grant_landed(monkeypatch, tmp_path):
    """A concurrent flow's grant written mid-probe survives the probe's rollback."""
    _write_tokens(tmp_path, "reports", '{"access_token":"old"}')

    def _probe(server_name, cfg, connect_timeout=None, *, details=None):
        if details is not None:
            details["initialized"] = False
        _write_tokens(tmp_path, "reports", '{"access_token":"fresh-from-gateway"}')
        raise RuntimeError("probe failed after a concurrent flow wrote its grant")

    manager = _patch_harness(monkeypatch, tmp_path, _probe, tokens_present=False)

    with pytest.raises(RuntimeError):
        _run_probe(tmp_path)

    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("reports", hermes_home=str(tmp_path))
    assert (
        storage._tokens_path().read_text(encoding="utf-8")
        == '{"access_token":"fresh-from-gateway"}'
    )


def test_undo_still_restores_snapshot_when_nothing_newer_exists(monkeypatch, tmp_path):
    """Nothing on disk at rollback time: the pre-probe snapshot goes back as before."""
    _write_tokens(tmp_path, "reports", '{"access_token":"old"}')

    def _probe(server_name, cfg, connect_timeout=None, *, details=None):
        if details is not None:
            details["initialized"] = False
        raise RuntimeError("probe failed without writing anything")

    manager = _patch_harness(monkeypatch, tmp_path, _probe, tokens_present=False)

    with pytest.raises(RuntimeError):
        _run_probe(tmp_path)

    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("reports", hermes_home=str(tmp_path))
    assert (
        storage._tokens_path().read_text(encoding="utf-8") == '{"access_token":"old"}'
    )
    assert manager.restored == ["reports"]


def test_undo_runs_when_commit_is_canceled_with_newer_grant(monkeypatch, tmp_path):
    """The commit-cancellation rollback obeys the same rule: a newer grant survives, an older
    attempt never overwrites it (the ``_commit`` path raises ``AttemptCanceled``)."""
    from tools.connectors.mcp_oauth import AttemptCanceled

    _write_tokens(tmp_path, "reports", '{"access_token":"old"}')

    def _probe(server_name, cfg, connect_timeout=None, *, details=None):
        if details is not None:
            details["initialized"] = True
        _write_tokens(tmp_path, "reports", '{"access_token":"fresh-mid-flight"}')

    def _commit(server_name, cfg, on_commit, flow=None):
        raise AttemptCanceled()

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "tools.mcp_oauth_manager.get_manager", lambda: _RecordingManager()
    )
    monkeypatch.setattr(
        "tools.mcp_oauth.login_connect_timeout", lambda cfg: 30.0, raising=False
    )
    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", _probe)
    monkeypatch.setattr(
        "hermes_cli.mcp_config._oauth_tokens_present", lambda name: True
    )
    monkeypatch.setattr(connectors_mcp_oauth, "_ACTIVE", {})
    monkeypatch.setattr(connectors_mcp_oauth, "_commit", _commit)

    with pytest.raises(AttemptCanceled):
        connectors_mcp_oauth.probe_with_rollback(
            "reports", {"url": "https://mcp.example"}, str(tmp_path), None, False
        )

    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("reports", hermes_home=str(tmp_path))
    assert (
        storage._tokens_path().read_text(encoding="utf-8")
        == '{"access_token":"fresh-mid-flight"}'
    )
