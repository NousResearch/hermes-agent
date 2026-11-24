"""Non-finite floats must not reach the wire (#159335).

``Infinity``/``NaN`` are Python ``json`` extensions, not JSON. A YAML 1.1 float that overflows — a
session id such as ``20260819_045052_92e993`` resolves to ``inf`` — made every strict client reject
the whole ``profiles.list`` frame and reconnect in a loop. Both boundaries now map them to ``null``.
"""

from __future__ import annotations

import json
import logging

import pytest

import hermes_yaml
import tui_gateway.server as srv
from tui_gateway.transport import serialize_frame

_log = logging.getLogger("test.serialize_frame")


def _strict_loads(text: str):
    """``json.loads`` that rejects the Infinity/NaN extensions a strict client would reject."""

    def reject(token):
        raise AssertionError(f"bare {token} on the wire")

    return json.loads(text, parse_constant=reject)


def test_non_finite_floats_become_null_not_bare_tokens():
    frame = {"jsonrpc": "2.0", "id": 7, "result": {"a": float("inf"), "b": -float("inf"), "c": float("nan")}}
    assert _strict_loads(serialize_frame(frame, "test", _log))["result"] == {"a": None, "b": None, "c": None}


def test_finite_floats_and_shape_survive():
    frame = {"jsonrpc": "2.0", "id": 7, "result": {"rows": [{"n": 1.5, "s": "x", "t": (1, 2)}]}}
    assert _strict_loads(serialize_frame(frame, "test", _log))["result"] == {"rows": [{"n": 1.5, "s": "x", "t": [1, 2]}]}


def test_unserializable_payload_still_becomes_an_error_frame():
    """The #92506 contract survives: a client gets its id back, it does not wait forever."""
    frame = {"jsonrpc": "2.0", "id": 9, "result": {"bad": object()}}
    out = json.loads(serialize_frame(frame, "test", _log))
    assert (out["id"], out["error"]["code"]) == (9, -32603)


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def test_overflowing_yaml_id_in_ui_meta_lists_as_null(home):
    """The reported chain end to end: YAML 1.1 resolves the id to ``inf``, the roster row and the
    serialized frame carry ``null``, and a strict parser accepts the response."""
    raw = "ui_meta:\n  hermes-bots:\n    chat: 20260819_045052_92e993\n"
    assert hermes_yaml.safe_load(raw)["ui_meta"]["hermes-bots"]["chat"] == float("inf")
    (home / "profile.yaml").write_text(raw, encoding="utf-8")  # the default profile's dir IS the home

    response = srv._methods["profiles.list"]("list", {"include_sessions": False})
    row = next(r for r in response["result"]["profiles"] if r["name"] == "default")
    assert row["ui_meta"] == {"hermes-bots": {"chat": None}}
    _strict_loads(serialize_frame(response, "test", _log))


def test_rest_twin_keeps_the_document_instead_of_dropping_it(home):
    """``read_profile_meta`` stored ui_meta through a strict dump that returned ``{}`` on ``inf``, so the
    REST roster lost the whole document. It now holds the row with ``null``."""
    from hermes_cli.profiles import list_profiles, read_profile_meta
    from hermes_cli.web_routers.profiles import _profile_to_dict

    (home / "profile.yaml").write_text("ui_meta:\n  hermes-bots:\n    chat: 20260819_045052_92e993\n",
                                       encoding="utf-8")
    info = list_profiles()[0]
    assert read_profile_meta(info.path)["ui_meta"] == {"hermes-bots": {"chat": None}}
    assert _profile_to_dict(info)["ui_meta"] == {"hermes-bots": {"chat": None}}
