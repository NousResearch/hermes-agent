"""shared_metrics.status / shared_metrics.set: the Desktop consent path keeps the CLI wizard's
invariants (sending needs collection, every change reconciles the consent windows) and lands in the
profile the request names, never the launch profile."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import hermes_yaml as yaml

import tui_gateway.server as server


def _bind_homes(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "code"
    for home in (launch, worker):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "nous"}}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if (name or "").strip() == "code" else None)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    return launch, worker


def _call(method: str, params: dict) -> dict:
    return server._methods[method]("rid", params)["result"]


def _shared_metrics(home: Path) -> dict:
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    return (cfg.get("telemetry") or {}).get("shared_metrics") or {}


def _windows(home: Path) -> list[tuple]:
    db = home / "telemetry" / "shared_metrics" / "metrics.sqlite3"
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT opened_at, closed_at FROM send_consent_windows ORDER BY rowid").fetchall()


def test_send_without_collection_is_normalized_off_and_closes_the_consent_window(tmp_path, monkeypatch):
    launch, worker = _bind_homes(monkeypatch, tmp_path)

    assert _call("shared_metrics.set", {"profile": "code", "enabled": True, "send": True}) == {
        "enabled": True, "send": True, "decided": True}
    [(opened, closed)] = _windows(worker)
    assert opened and closed is None

    # send=true alone is refused: collection off withdraws send consent, in config AND in the store.
    assert _call("shared_metrics.set", {"profile": "code", "enabled": False, "send": True}) == {
        "enabled": False, "send": False, "decided": True}
    assert _shared_metrics(worker) == {"enabled": False, "send": False}
    [(_, closed)] = _windows(worker)
    assert closed is not None

    assert _shared_metrics(launch) == {}
    assert not (launch / "telemetry").exists()


def test_status_is_undecided_until_a_key_is_written(tmp_path, monkeypatch):
    _launch, worker = _bind_homes(monkeypatch, tmp_path)

    assert _call("shared_metrics.status", {"profile": "code"}) == {"enabled": False, "send": False, "decided": False}

    # An answer given in `hermes setup` (only `enabled: false` written) counts as decided.
    (worker / "config.yaml").write_text(
        yaml.safe_dump({"telemetry": {"shared_metrics": {"enabled": False}}}), encoding="utf-8")
    assert _call("shared_metrics.status", {"profile": "code"}) == {"enabled": False, "send": False, "decided": True}


def test_only_the_first_run_answer_records_desktop_setup_completed(tmp_path, monkeypatch):
    _launch, _worker = _bind_homes(monkeypatch, tmp_path)
    import hermes_cli.observability.shared_metrics_events as events

    calls: list[dict] = []
    monkeypatch.setattr(events, "record_setup_completed", lambda **kw: calls.append(kw))

    _call("shared_metrics.set", {"profile": "code", "enabled": True, "send": False, "first_run": True})
    _call("shared_metrics.set", {"profile": "code", "enabled": True, "send": True})

    assert calls == [{"surface": "desktop", "provider": "nous"}]


def test_unknown_named_consent_target_fails_closed(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    worker = launch / "profiles" / "code"
    for home in (launch, worker):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    before = {home: (home / "config.yaml").read_bytes() for home in (launch, worker)}
    for profile in ("missing", "custom"):
        for method in ("shared_metrics.status", "shared_metrics.set"):
            response = server.handle_request({
                "jsonrpc": "2.0", "id": "rid", "method": method,
                "params": {"profile": profile, **({"enabled": True, "send": True} if method.endswith(".set") else {})},
            })
            assert response["error"]["code"] == 4064
    for home in (launch, worker):
        assert (home / "config.yaml").read_bytes() == before[home]
        assert not (home / "telemetry").exists()


def test_custom_home_omitted_profile_consent_stays_on_owning_launch_home(tmp_path, monkeypatch):
    launch, worker = _bind_homes(monkeypatch, tmp_path)
    worker_before = (worker / "config.yaml").read_bytes()
    assert _call("shared_metrics.status", {}) == {"enabled": False, "send": False, "decided": False}
    assert _call("shared_metrics.set", {"enabled": True, "send": False}) == {
        "enabled": True, "send": False, "decided": True}
    assert _shared_metrics(launch) == {"enabled": True, "send": False}
    assert (worker / "config.yaml").read_bytes() == worker_before
    assert not (worker / "telemetry").exists()
