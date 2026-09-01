"""Tests: bot-relay media envelopes (D2) — tools/bot_relay.py.

The relay was text-only (H1, S3): a cross-connection DM carrying generated
media left the recipient with a path that did not exist on their machine and
the Desktop with nothing fetchable on the delivering connection. Contracts
pinned here:

- envelopes carry an optional ``media[]`` of validated rows (same shape the
  D1 gateway events use: path/name/kind/mime/size);
- ``stage_media`` materializes inline ``data_url`` rows (mirroring the
  ``image.generate`` 8MB data-URL precedent) into a durable per-envelope
  media dir on the receiving gateway; oversize rows are refused per-row with
  a typed error, never fatal;
- replies persist staged rows (paths, not payloads) and the sender-side
  waiter prints them as ``MEDIA:`` refs so the reply wakes the sender with
  renderable, locally-fetchable paths;
- stale media dirs are swept with the other relay artifacts.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time

import pytest

from tools import bot_relay


@pytest.fixture()
def root(tmp_path):
    return tmp_path


def _target(conn="cloud-1", profile="scout", handle="scout"):
    return {"profile": profile, "handle": handle, "connection_id": conn,
            "connection_label": "", "title": "", "description": ""}


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _png_data_url():
    return f"data:image/png;base64,{_PNG_B64}"


# ── row validation ───────────────────────────────────────────────────────────


def test_normalize_media_rows_accepts_and_drops(root):
    rows = [
        {"path": "/tmp/chart.png", "kind": "image", "mime": "image/png", "size": 12},
        {"path": "~/out/report.pdf", "name": "report.pdf"},
        {"path": "relative/nope.png"},              # not absolute → dropped
        {"path": "/x/\n evil.png"},                 # control char → dropped
        {"path": "/ok.png", "kind": "hologram"},    # unknown kind → kind omitted
        {"path": "/ok.png", "mime": "not a mime"},  # bad mime → mime omitted
        {"path": "/ok.png", "size": -1},            # negative size → size omitted
        {"path": "/ok.png", "size": "big"},         # non-int size → size omitted
        "not-a-dict",                               # dropped
    ]
    out = bot_relay.normalize_media_rows(rows)
    assert [r["path"] for r in out] == [
        "/tmp/chart.png", "~/out/report.pdf",
        "/ok.png", "/ok.png", "/ok.png", "/ok.png",
    ]
    # invalid optional fields are omitted, never carried through
    for r in out[2:]:
        assert set(r) == {"path"}


def test_normalize_media_rows_caps_count_and_path_length(root):
    rows = [{"path": f"/d/f{i}.png"} for i in range(bot_relay.MEDIA_MAX_ROWS + 3)]
    assert len(bot_relay.normalize_media_rows(rows)) == bot_relay.MEDIA_MAX_ROWS
    long = {"path": "/" + "a" * 2000 + ".png"}
    assert bot_relay.normalize_media_rows([long]) == []


def test_normalize_media_rows_data_url_shape(root):
    ok = bot_relay.normalize_media_rows([{"path": "/a.png", "data_url": _png_data_url()}])
    assert ok and ok[0]["data_url"].startswith("data:image/png;base64,")
    # non-base64 / non-data payloads are dropped (data_url field omitted)
    for bad in ("http://x/y.png", "data:text/plain,hello", "not a url"):
        out = bot_relay.normalize_media_rows([{"path": "/a.png", "data_url": bad}])
        assert out and "data_url" not in out[0]


# ── envelope transport ───────────────────────────────────────────────────────


def test_envelope_carries_media_rows(root):
    media = [{"path": "/tmp/chart.png", "kind": "image", "mime": "image/png", "size": 3}]
    env = bot_relay.enqueue_envelope(
        root, target=_target(), message="see attachment",
        sender_profile="default", sender_handle="hermes", media=media,
    )
    claimed = bot_relay.claim_pending_envelopes(root)
    assert claimed[0]["id"] == env["id"]
    assert claimed[0]["media"] == media


def test_envelope_without_media_has_no_media_key(root):
    bot_relay.enqueue_envelope(
        root, target=_target(), message="plain",
        sender_profile="default", sender_handle="hermes",
    )
    claimed = bot_relay.claim_pending_envelopes(root)
    assert "media" not in claimed[0]


# ── staging (inline data URL under cap — image.generate precedent) ──────────


def test_stage_media_writes_data_url_rows(root):
    rows = bot_relay.normalize_media_rows(
        [{"path": "/sender/odd name!.png", "data_url": _png_data_url(), "kind": "image"}]
    )
    out = bot_relay.stage_media(root, rows, envelope_id="a" * 32)
    row = out["rows"][0]
    assert row["local_path"]
    staged = out["rows"][0]["local_path"]
    assert staged.startswith(str(bot_relay.relay_root(root) / bot_relay.MEDIA_DIR))
    assert staged.endswith(".png")
    from pathlib import Path

    assert Path(staged).read_bytes()[:4] == b"\x89PNG"
    # hostile name characters were sanitized into the filename
    assert "!" not in Path(staged).name and " " not in Path(staged).name


def test_stage_media_oversize_row_refused_not_fatal(root, monkeypatch):
    big = "data:image/png;base64," + "A" * 64
    rows = bot_relay.normalize_media_rows([{"path": "/big.png", "data_url": big}])
    monkeypatch.setattr(bot_relay, "MEDIA_INLINE_CAP_BYTES", 16)
    out = bot_relay.stage_media(root, rows, envelope_id="b" * 32)
    assert out["rows"][0].get("error") == "too_large"
    assert not out["rows"][0].get("local_path")


def test_stage_media_metadata_only_rows_pass_through(root):
    rows = bot_relay.normalize_media_rows(
        [{"path": "/sender/huge.mp4", "kind": "video", "size": 900_000_000}]
    )
    out = bot_relay.stage_media(root, rows, envelope_id="c" * 32)
    assert out["rows"][0]["path"] == "/sender/huge.mp4"
    assert not out["rows"][0].get("local_path")
    # metadata-only rows never fabricate a local MEDIA: tag
    assert "MEDIA:" not in out["note"]


def test_stage_media_invalid_envelope_id_generates_one(root):
    rows = bot_relay.normalize_media_rows([{"path": "/a.png", "data_url": _png_data_url()}])
    out = bot_relay.stage_media(root, rows, envelope_id="../evil")
    import re

    assert re.match(r"^[0-9a-f]{32}$", out["envelope_id"])


# ── reply carries media; waiter prints staged refs ──────────────────────────


def test_write_reply_stages_media_and_persists_paths(root):
    rows = bot_relay.normalize_media_rows([{"path": "/t/fig.png", "data_url": _png_data_url()}])
    out = bot_relay.stage_media(root, rows, envelope_id="d" * 32)
    path = bot_relay.write_reply(root, "d" * 32, reply="here", media=out["rows"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["media"][0]["local_path"].endswith(".png")
    # payloads are never persisted — paths only
    assert "data_url" not in data["media"][0]
    assert "base64" not in json.dumps(data)


def test_waiter_prints_media_refs_from_reply(root):
    rows = bot_relay.normalize_media_rows([{"path": "/t/fig.png", "data_url": _png_data_url()}])
    out = bot_relay.stage_media(root, rows, envelope_id="e" * 32)
    bot_relay.write_reply(root, "e" * 32, reply="done", media=out["rows"])
    cmd = bot_relay.waiter_command(
        root, {"id": "e" * 32, "target_handle": "scout", "target_connection": "cloud-1"}
    )
    code = shlex.split(cmd)[shlex.split(cmd).index("-c") + 1]
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0
    assert "MEDIA:" in proc.stdout
    staged = out["rows"][0]["local_path"]
    assert staged in proc.stdout


def test_waiter_mentions_metadata_only_media(root):
    bot_relay.write_reply(
        root, "f" * 32, reply="done",
        media=[{"path": "/t/huge.mp4", "name": "huge.mp4", "kind": "video", "size": 123}],
    )
    cmd = bot_relay.waiter_command(
        root, {"id": "f" * 32, "target_handle": "scout", "target_connection": "cloud-1"}
    )
    code = shlex.split(cmd)[shlex.split(cmd).index("-c") + 1]
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0
    assert "huge.mp4" in proc.stdout and "MEDIA:" not in proc.stdout.split("Reply from")[1].split("Media:")[0]


# ── full Bot-Chat round trip: sender → drain → deliver → reply → waiter ─────


def test_relay_media_round_trip_across_roots(root, tmp_path):
    """H1 repro, green path: media survives every hop of the relay.

    sender root: envelope queued with an inline-able image row.
    target root: deliver stages the bytes durably and appends a MEDIA: ref
    the target turn can resolve; the reply comes back with media.
    sender root: the reply persists staged paths; the waiter prints them.
    """
    sender = root
    target = tmp_path / "target-install"
    target.mkdir()

    media = bot_relay.normalize_media_rows(
        [{"path": "/sender/fig.png", "data_url": _png_data_url(), "kind": "image", "size": 100}]
    )
    env = bot_relay.enqueue_envelope(
        sender, target=_target(), message="chart for you",
        sender_profile="default", sender_handle="hermes", media=media,
    )
    claimed = bot_relay.claim_pending_envelopes(sender)

    # Desktop fetches the bytes from the owner connection and hands them to
    # the target gateway inside the deliver params (owner-connection case).
    staged = bot_relay.stage_media(
        target, bot_relay.normalize_media_rows(claimed[0]["media"]),
        envelope_id=env["id"],
    )
    assert staged["rows"][0]["local_path"]

    # delivery turn ran; reply comes back carrying media (cross-connection
    # inline under cap), persisted on the sender root.
    reply_rows = bot_relay.normalize_media_rows(
        [{"path": "/target/back.png", "data_url": _png_data_url(), "kind": "image"}]
    )
    staged_reply = bot_relay.stage_media(
        sender, reply_rows, envelope_id=env["id"]
    )
    reply_path = bot_relay.write_reply(
        sender, env["id"], reply="got it", media=staged_reply["rows"]
    )
    data = json.loads(reply_path.read_text(encoding="utf-8"))
    assert data["media"][0]["local_path"]

    # sender-side waiter resolves with renderable refs
    cmd = bot_relay.waiter_command(
        sender, {"id": env["id"], "target_handle": "scout", "target_connection": "cloud-1"}
    )
    code = shlex.split(cmd)[shlex.split(cmd).index("-c") + 1]
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert staged_reply["rows"][0]["local_path"] in proc.stdout


# ── housekeeping: stale media dirs are swept ─────────────────────────────────


def test_sweep_removes_stale_media_dirs(root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(root))
    rows = bot_relay.normalize_media_rows([{"path": "/a.png", "data_url": _png_data_url()}])
    out = bot_relay.stage_media(root, rows, envelope_id="1" * 32)
    media_root = bot_relay.relay_root(root) / bot_relay.MEDIA_DIR
    assert (media_root / ("1" * 32)).is_dir()
    old = time.time() - bot_relay.STALE_AFTER_SECONDS - 1
    import os

    os.utime(media_root / ("1" * 32), (old, old))
    removed = bot_relay.cleanup_bot_relay_artifacts()
    assert removed >= 1
    assert not out["rows"][0]["local_path"] or not __import__("pathlib").Path(
        out["rows"][0]["local_path"]
    ).exists()
