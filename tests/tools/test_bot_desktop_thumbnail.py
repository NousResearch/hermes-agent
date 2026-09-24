"""``thumbnail_data_url`` swaps the process-wide XAUTHORITY around the grab; two profiles grabbed on
worker threads at once must each see their own cookie file, not the other's."""

from __future__ import annotations

import os
import threading

from PIL import Image, ImageGrab

from tools.bot_desktop import runtime, thumbnail


def test_concurrent_grabs_each_see_their_own_xauthority(monkeypatch):
    envs = {"a": {"DISPLAY": ":91", "XAUTHORITY": "/tmp/xauth-a"},
            "b": {"DISPLAY": ":92", "XAUTHORITY": "/tmp/xauth-b"}}
    local = threading.local()
    monkeypatch.setattr(runtime, "published_env", lambda: envs[local.profile])
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: 4242)
    monkeypatch.delenv("XAUTHORITY", raising=False)
    seen = {}
    start = threading.Barrier(2)

    def fake_grab(xdisplay=None):
        seen[xdisplay] = os.environ.get("XAUTHORITY")
        threading.Event().wait(0.05)  # hold the env long enough for the other thread to collide
        return Image.new("RGB", (8, 8))
    monkeypatch.setattr(ImageGrab, "grab", fake_grab)

    def worker(profile):
        local.profile = profile
        start.wait()
        thumbnail.thumbnail_data_url()

    threads = [threading.Thread(target=worker, args=(p,)) for p in envs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert seen == {":91": "/tmp/xauth-a", ":92": "/tmp/xauth-b"}
    assert "XAUTHORITY" not in os.environ
