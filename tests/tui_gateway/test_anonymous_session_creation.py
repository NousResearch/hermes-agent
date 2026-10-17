"""Anonymous creation does not probe authenticated provenance; owner binding stays scoped."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from tui_gateway import server
from tests.tui_gateway import test_session_creation_binding as creation


@pytest.mark.parametrize("profile", [None, "secondary"])
def test_anonymous_creation_does_not_resolve_authenticated_store(monkeypatch, tmp_path, profile):
    creation.prepare(monkeypatch, tmp_path)
    resolve = Path.resolve

    def refuse_metadata(path, *args, **kwargs):
        if path.name == "state.db":
            raise AssertionError("anonymous creation resolved authenticated store metadata")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", refuse_metadata)
    result = creation.create(creation.LoginPeer(None), **({"profile": profile} if profile else {}))
    assert "creation_binding" not in result
    record = server._sessions[result["session_id"]]
    assert record["auth_user_id"] is None
    assert record["creation_binding"] is None


@pytest.mark.parametrize("profile", [None, "secondary"])
def test_authenticated_creation_retains_profile_store_and_owner_binding(monkeypatch, tmp_path, profile):
    home, secondary = creation.prepare(monkeypatch, tmp_path)
    expected_store = str(((secondary if profile else home) / "state.db").resolve())
    resolve = Path.resolve
    probes = Mock()

    def observe_metadata(path, *args, **kwargs):
        if path.name == "state.db":
            probes(str(path))
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", observe_metadata)
    params = {"idempotency_key": "owner-bound", **({"profile": profile} if profile else {})}
    result = creation.create(creation.LoginPeer(), **params)
    origin = server._sessions[result["session_id"]]["creation_binding"]
    assert origin.store_path == expected_store
    assert origin.authenticated_owner == "basic:alice"
    assert result["creation_binding"]["authenticated_owner"] == "basic:alice"
    probes.assert_any_call(expected_store)
    for foreign in (creation.LoginPeer(None), creation.LoginPeer("bob"), creation.LoginPeer(provider="oidc")):
        assert "creation_binding" not in creation.create(foreign, **params)
    assert creation.create(creation.LoginPeer(), **params)["creation_binding"] == result["creation_binding"]


@pytest.mark.parametrize("profile", [None, "secondary"])
def test_anonymous_retry_cannot_probe_authenticated_origin(monkeypatch, tmp_path, profile):
    creation.prepare(monkeypatch, tmp_path)
    params = {"idempotency_key": "authenticated-origin", **({"profile": profile} if profile else {})}
    first = creation.create(creation.LoginPeer(), **params)
    origin = server._sessions[first["session_id"]]["creation_binding"]
    resolve = Path.resolve

    def refuse_metadata(path, *args, **kwargs):
        if path.name == "state.db":
            raise AssertionError("anonymous retry resolved authenticated store metadata")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", refuse_metadata)
    retry = creation.create(creation.LoginPeer(None), **params)
    assert retry["session_id"] == first["session_id"]
    assert "creation_binding" not in retry
    assert server._sessions[first["session_id"]]["creation_binding"] is origin
    assert origin.authenticated_owner == "basic:alice"
