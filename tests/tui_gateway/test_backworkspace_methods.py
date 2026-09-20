"""backworkspace.* JSON-RPC handlers (tui_gateway/methods_backworkspace.py).

The Desktop's flip-side page. Contracts:
- a page saved without an id gets a session-shaped id, and saving with that id rewrites the
  same file instead of starting another page;
- a client-supplied id can never name a file outside ``<HERMES_HOME>/backworkspace``;
- the ``path`` a result carries is the file that was written, so an agent can open it;
- ``params.profile`` decides whose home a page lives in (launch -> worker -> launch).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

import tui_gateway.server as server
from hermes_state_ids import SESSION_ID_PATTERN


def _call(method: str, params: dict) -> dict:
    return server._methods[method]("rid", params)


def _result(envelope: dict) -> dict:
    assert "error" not in envelope, envelope
    return envelope["result"]


def _base64_of_size(size: int) -> str:
    return base64.b64encode(b"\0" * size).decode("ascii")


def test_save_then_open_round_trips_one_page(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _result(_call("backworkspace.open", {})) == {"page": None}

    page_id = _result(_call("backworkspace.save", {"content": "first"}))["id"]
    again = _result(_call("backworkspace.save", {"id": page_id, "content": "second"}))

    assert SESSION_ID_PATTERN.match(page_id)
    assert again["id"] == page_id
    assert [p.name for p in (tmp_path / "backworkspace").iterdir()] == [f"{page_id}.md"]
    assert again["path"] == str(tmp_path / "backworkspace" / f"{page_id}.md")
    assert Path(again["path"]).read_text(encoding="utf-8") == "second"
    assert _result(_call("backworkspace.open", {}))["page"] == {
        "id": page_id,
        "content": "second",
        "path": again["path"],
    }


def test_attach_stores_the_image_beside_the_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    attached = _result(_call("backworkspace.attach", {"name": "clipboard.png", "data": "aGVsbG8="}))

    page_path = Path(_result(_call("backworkspace.save", {"content": "a note"}))["path"])

    assert Path(attached["path"]).read_bytes() == b"hello"
    # The contract the page is written against: the link it carries is relative to the page's
    # own file, so page and pictures stay one folder that can be moved or synced together.
    assert page_path.parent / attached["href"] == Path(attached["path"])
    # A document that can carry script is not a picture, whatever the clipboard says.
    assert "error" in _call("backworkspace.attach", {"name": "drawing.svg", "data": "aGVsbG8="})
    # Bigger than the Desktop will read back as a data URL is stored by nobody.
    assert "error" in _call(
        "backworkspace.attach",
        {"name": "huge.png", "data": _base64_of_size(17 * 1024 * 1024)},
    )


@pytest.mark.parametrize("page_id", ["../config", "20260920_101010_abcdef\n"])
def test_crafted_page_id_is_rejected(tmp_path, monkeypatch, page_id):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    envelope = _call("backworkspace.save", {"id": page_id, "content": "x"})

    assert "error" in envelope
    assert not list(tmp_path.rglob("*.md"))


def test_profile_param_keeps_pages_per_profile(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    worker = tmp_path / "profiles" / "code"
    launch.mkdir()
    worker.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(
        server,
        "_profile_home",
        lambda name: worker if (name or "").strip() == "code" else None,
    )

    launch_id = _result(_call("backworkspace.save", {"content": "launch page"}))["id"]
    worker_id = _result(
        _call("backworkspace.save", {"content": "worker page", "profile": "code"})
    )["id"]

    assert (launch / "backworkspace" / f"{launch_id}.md").read_text(
        encoding="utf-8"
    ) == "launch page"
    assert (worker / "backworkspace" / f"{worker_id}.md").read_text(
        encoding="utf-8"
    ) == "worker page"
    assert _result(_call("backworkspace.open", {}))["page"]["content"] == "launch page"
    assert (
        _result(_call("backworkspace.open", {"profile": "code"}))["page"]["content"]
        == "worker page"
    )
    assert _result(_call("backworkspace.open", {}))["page"]["content"] == "launch page"
