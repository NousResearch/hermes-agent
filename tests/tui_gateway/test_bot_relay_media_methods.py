"""Handler-level tests: bot_relay.deliver / bot_relay.reply carry media (D2).

The deliver RPC is the Desktop's door onto the TARGET gateway: it must
stage inline rows durably before the turn and rewrite ``MEDIA:`` refs to
paths that resolve on THIS gateway (never the sender's disk). The reply
RPC persists path rows only — inline payloads never touch the reply file.
"""

from __future__ import annotations

import json

import pytest

import tui_gateway.server as srv
from tools import bot_relay

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def test_deliver_stages_inline_media_and_rewrites_media_refs(home, monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "got the chart"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["query"] = open(argv[argv.index("--query-file") + 1], encoding="utf-8").read()
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    media = [{"path": "/sender-host/fig.png", "kind": "image", "mime": "image/png",
              "data_url": f"data:image/png;base64,{_PNG_B64}"}]
    out = _result(
        srv._methods["bot_relay.deliver"](
            1, {"profile": "ops", "message": "ping", "id": "a" * 32, "media": media}
        )
    )
    assert out["reply"] == "got the chart"
    # The turn saw a MEDIA: ref pointing INSIDE this gateway's staged media
    # dir — never the sender-host path.
    assert "/sender-host/fig.png" not in calls["query"]
    assert "MEDIA:" in calls["query"]
    ref = [ln for ln in calls["query"].splitlines() if ln.startswith("MEDIA:")][0]
    staged_dir = bot_relay.relay_root(home) / bot_relay.MEDIA_DIR / ("a" * 32)
    assert ref.startswith("MEDIA:" + str(staged_dir))
    from pathlib import Path

    assert Path(ref[len("MEDIA:"):]).read_bytes()[:4] == b"\x89PNG"


def test_deliver_metadata_only_media_becomes_visible_note(home, monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["query"] = open(argv[argv.index("--query-file") + 1], encoding="utf-8").read()
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)
    media = [{"path": "/sender-host/huge.mp4", "kind": "video", "size": 900_000_000}]
    _result(
        srv._methods["bot_relay.deliver"](
            1, {"profile": "ops", "message": "ping", "id": "b" * 32, "media": media}
        )
    )
    # Nothing silently vanishes: the turn is told the media exists but could
    # not be transferred inline (fallback-card path client-side).
    assert "huge.mp4" in calls["query"]
    assert "MEDIA:" not in calls["query"]


def test_reply_persists_media_paths_and_strips_payloads(home):
    media = [{"path": "/ops/back.png", "kind": "image",
              "data_url": f"data:image/png;base64,{_PNG_B64}"}]
    _result(
        srv._methods["bot_relay.reply"](
            1, {"id": "c" * 32, "reply": "here you go", "media": media}
        )
    )
    data = json.loads(
        (bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / ("c" * 32 + ".json")).read_text(
            encoding="utf-8"
        )
    )
    assert data["media"][0]["path"] == "/ops/back.png"
    assert "data_url" not in data["media"][0]
    assert "base64" not in json.dumps(data)
