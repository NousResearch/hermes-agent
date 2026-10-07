"""shared_metrics.status / shared_metrics.set: the Desktop consent path keeps the CLI wizard's
collection-only shape and lands in the profile the request names, never the launch profile."""

from __future__ import annotations

from pathlib import Path

import rabbit_yaml as yaml

import tui_gateway.server as server


def _bind_homes(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "code"
    for home in (launch, worker):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "nous"}}), encoding="utf-8")
    monkeypatch.setattr(server, "_rabbit_home", launch)
    monkeypatch.setenv("RABBIT_HOME", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if (name or "").strip() == "code" else None)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    return launch, worker


def _call(method: str, params: dict) -> dict:
    return server._methods[method]("rid", params)["result"]


def _shared_metrics(home: Path) -> dict:
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
    return (cfg.get("telemetry") or {}).get("shared_metrics") or {}


def test_set_lands_in_the_named_profile(tmp_path, monkeypatch):
    launch, worker = _bind_homes(monkeypatch, tmp_path)

    assert _call("shared_metrics.set", {"profile": "code", "enabled": True}) == {
        "enabled": True, "decided": True}
    assert _shared_metrics(worker) == {"enabled": True}

    assert _call("shared_metrics.set", {"profile": "code", "enabled": False}) == {
        "enabled": False, "decided": True}
    assert _shared_metrics(worker) == {"enabled": False}

    assert _shared_metrics(launch) == {}
    assert not (launch / "telemetry").exists()


def test_status_is_undecided_until_a_key_is_written(tmp_path, monkeypatch):
    _launch, worker = _bind_homes(monkeypatch, tmp_path)

    assert _call("shared_metrics.status", {"profile": "code"}) == {"enabled": False, "decided": False}

    # An answer given in `rabbit setup` (only `enabled: false` written) counts as decided.
    (worker / "config.yaml").write_text(
        yaml.safe_dump({"telemetry": {"shared_metrics": {"enabled": False}}}), encoding="utf-8")
    assert _call("shared_metrics.status", {"profile": "code"}) == {"enabled": False, "decided": True}


def test_only_the_first_run_answer_records_desktop_setup_completed(tmp_path, monkeypatch):
    _launch, _worker = _bind_homes(monkeypatch, tmp_path)
    import rabbit_cli.observability.shared_metrics_events as events

    calls: list[dict] = []
    monkeypatch.setattr(events, "record_setup_completed", lambda **kw: calls.append(kw))

    _call("shared_metrics.set", {"profile": "code", "enabled": True, "first_run": True})
    _call("shared_metrics.set", {"profile": "code", "enabled": False})

    assert calls == [{"surface": "desktop", "provider": "nous"}]
