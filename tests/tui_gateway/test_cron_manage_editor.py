"""Desktop plugin cron editors use the existing scoped store and mutation path (#126278)."""

import json

from tui_gateway import server


def _rpc(**params):
    return server.handle_request({"id": "editor", "method": "cron.manage", "params": params})


def _homes(tmp_path, monkeypatch):
    import hermes_cli.profiles as profiles

    homes = {name: tmp_path / name for name in ("alpha", "beta")}
    for name, home in homes.items():
        (home / "cron").mkdir(parents=True)
        job = {
            "id": "same-id", "name": "routine", "prompt": f"{name} instructions " * 30,
            "enabled": False, "state": "paused", "paused_reason": "fixture",
            "schedule": {"kind": "interval", "minutes": 60}, "schedule_display": "every 1h",
            "repeat": {"times": 5, "completed": 2}, "deliver": "local",
        }
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [job]}), encoding="utf-8")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: homes.get(name, tmp_path / "missing"))
    return homes


def test_editor_get_preserves_full_record_and_profile_isolation(tmp_path, monkeypatch):
    homes = _homes(tmp_path, monkeypatch)
    from hermes_constants import get_hermes_home_override

    for profile in ("alpha", "beta", "alpha"):
        response = _rpc(action="get", job_id="same-id", profile=profile)
        assert "result" in response, response
        result = response["result"]
        assert result["success"] is True
        stored = json.loads((homes[profile] / "cron" / "jobs.json").read_text())["jobs"][0]
        assert result["job"]["prompt"] == stored["prompt"]
        assert result["job"]["schedule"] == stored["schedule"]
        assert result["job"]["repeat"] == stored["repeat"]
        assert get_hermes_home_override() is None
    assert _rpc(action="get", job_id="absent", profile="alpha")["result"]["success"] is False
    assert "error" in _rpc(action="get", job_id="same-id", profile="missing")
    assert _rpc(action="get", name="routine", profile="alpha")["result"]["job"]["id"] == "same-id"
    listing = _rpc(action="list", profile="alpha", include_disabled=True)["result"]
    assert len(listing["jobs"]) == 1
    assert "prompt" not in listing["jobs"][0]
    assert len(listing["jobs"][0]["prompt_preview"]) < len(stored["prompt"])
    assert _rpc(action="pause", name="same-id", profile="alpha")["result"]["success"] is True
    assert _rpc(action="remove", name="same-id", profile="alpha")["result"]["success"] is True
    assert _rpc(action="list", profile="alpha", include_disabled=True)["result"]["jobs"] == []


def test_editor_get_errors_are_read_only(tmp_path, monkeypatch):
    homes = _homes(tmp_path, monkeypatch)
    path = homes["alpha"] / "cron" / "jobs.json"
    data = json.loads(path.read_text())
    data["jobs"].append({**data["jobs"][0], "id": "other-id"})
    path.write_text(json.dumps(data), encoding="utf-8")
    before = {name: (home / "cron" / "jobs.json").read_bytes() for name, home in homes.items()}
    assert _rpc(action="get", name="routine", profile="alpha")["result"]["success"] is False
    assert _rpc(action="get", profile="alpha")["result"]["success"] is False
    assert _rpc(action="get", job_id="same-id", profile="alpha")["result"]["job"]["id"] == "same-id"
    assert {name: (home / "cron" / "jobs.json").read_bytes() for name, home in homes.items()} == before
