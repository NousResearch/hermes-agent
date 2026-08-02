from __future__ import annotations

import threading
from pathlib import Path

import yaml

from hermes_cli import plugins
from tui_gateway import server


def _write_hook_plugin(home: Path, marker: str) -> None:
    plugin_dir = home / "plugins" / "marker"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "marker", "version": "1"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    ctx.register_hook('pre_llm_call', lambda **kw: {marker!r})\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["marker"]}}), encoding="utf-8"
    )


def test_selected_profile_manager_is_discovered_bound_and_frozen(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "selected"
    _write_hook_plugin(home, "selected")
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    monkeypatch.setattr(plugins, "_plugin_managers", {})
    session = {"profile_home": str(home)}

    token = server._bind_session_plugin_manager(session)
    try:
        snapshot = session["plugin_manager"]
        assert snapshot.profile_home == home.resolve()
        assert snapshot._discovered is True
        assert plugins.get_plugin_manager() is snapshot
        assert plugins.invoke_hook("pre_llm_call") == ["selected"]
    finally:
        server._reset_session_plugin_manager(token)

    assert plugins.get_bound_plugin_manager() is None

    replacement = plugins.discover_plugins(profile_home=home, force=True)
    assert replacement is not snapshot
    assert session["plugin_manager"] is snapshot

    token = server._bind_session_plugin_manager(session)
    try:
        assert plugins.get_plugin_manager() is snapshot
    finally:
        server._reset_session_plugin_manager(token)


def test_concurrent_session_plugin_scopes_do_not_cross_profiles(tmp_path, monkeypatch):
    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    _write_hook_plugin(home_a, "a")
    _write_hook_plugin(home_b, "b")
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    monkeypatch.setattr(plugins, "_plugin_managers", {})

    barrier = threading.Barrier(2)
    results: dict[str, tuple[str, list[str], str | None]] = {}

    def run(marker: str, home: Path) -> None:
        session = {"profile_home": str(home)}
        token = server._bind_session_plugin_manager(session)
        try:
            barrier.wait()
            manager = plugins.get_plugin_manager()
            results[marker] = (
                manager.profile_key,
                plugins.invoke_hook("pre_llm_call"),
                str(manager.profile_home),
            )
        finally:
            server._reset_session_plugin_manager(token)

    threads = [
        threading.Thread(target=run, args=("a", home_a)),
        threading.Thread(target=run, args=("b", home_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert results["a"][1:] == (["a"], str(home_a.resolve()))
    assert results["b"][1:] == (["b"], str(home_b.resolve()))
    assert results["a"][0] != results["b"][0]
    assert plugins.get_bound_plugin_manager() is None


def test_deferred_record_has_snapshot_slot_for_selected_profile(tmp_path):
    home = tmp_path / "profiles" / "selected"
    record = server._deferred_session_record(
        "session-key",
        cols=80,
        cwd=str(tmp_path),
        history=[],
        lease=None,
        profile_home=home,
    )

    assert record["profile_home"] == str(home)
    assert record["plugin_manager"] is None
