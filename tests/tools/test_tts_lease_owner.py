"""TTS engine leases follow the profile that ACQUIRED them, not the one that releases.

``tui:voice-tts`` and ``desktop:read-aloud`` are process-wide holder names, and ``voice.toggle`` /
``/api/audio/tts-lease`` run in the caller's profile. So profile B can acquire and profile A (or the
keep-warm timer thread) can drop the last hold. The keep-warm window, the user provider's release
hook and the secrets that hook forwards must still come from B.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from hermes_constants import (
    get_hermes_home, hermes_home_key, reset_hermes_home_override, set_hermes_home_override)
from tools import tts_command_provider, tts_tool, tts_tool_lifecycle


pytestmark = pytest.mark.usefixtures("tts_lifecycle_reset")


@pytest.fixture
def timers(tts_keep_warm_timers):
    """Fire every pending keep-warm timer; returns their intervals."""
    def fire():
        pending = [t for t in tts_keep_warm_timers if t.pending]
        for t in pending:
            t.fire()
        return [t.interval for t in pending]
    return fire


def _join_hook_threads():
    for t in list(threading.enumerate()):
        if t is not threading.current_thread() and t.name.startswith("tts-"):
            t.join(10)
            assert not t.is_alive(), t.name


def _srv_config(tag: str, keep_warm: float) -> dict:
    return {"provider": "srv", "keep_warm_seconds": keep_warm, "providers": {"srv": {
        "command": "true {input_path} {output_path}", "warm_command": f"warm-{tag}",
        "release_command": f"release-{tag}", "env_passthrough": ["SRV_TOKEN"]}}}


@pytest.fixture
def hooks(monkeypatch):
    """Record each command hook with the home and SRV_TOKEN it runs under (nothing executes)."""
    from agent.secret_scope import get_secret
    seen: list = []

    def run(command, timeout, env_passthrough=None):
        seen.append((command, get_hermes_home().name, get_secret("SRV_TOKEN")))
    monkeypatch.setattr(tts_command_provider, "run_command_provider", run)
    return seen


def test_voice_toggle_release_by_another_profile_follows_the_acquirer(
        tmp_path, monkeypatch, timers, hooks):
    """Real ``voice.toggle``: ``tts`` on from a profile-B session, off from the launch profile A."""
    from hermes_cli.config import atomic_config_write
    from tui_gateway import server
    launch, home_b = get_hermes_home(), tmp_path / "profiles" / "b"
    home_b.mkdir(parents=True)
    atomic_config_write(launch / "config.yaml", {"tts": _srv_config("A", 0)})
    atomic_config_write(home_b / "config.yaml", {"tts": _srv_config("B", 30)})
    (home_b / ".env").write_text("SRV_TOKEN=b-token\n")
    monkeypatch.setenv("SRV_TOKEN", "launch-token")
    monkeypatch.setenv("HERMES_VOICE", "1")
    monkeypatch.setenv("HERMES_VOICE_TTS", "0")
    monkeypatch.setattr(server, "_profile_home", lambda name: home_b if name == "b" else None)

    def toggle(params):
        resp = server.handle_request({"id": "1", "method": "voice.toggle", "params": params})
        for worker in [t for t in threading.enumerate() if t.name == "tts-lease-tui:voice-tts"]:
            worker.join(10)
        _join_hook_threads()
        return resp["result"]["tts"]

    assert toggle({"action": "tts", "profile": "b"}) is True
    assert toggle({"action": "tts"}) is False
    assert timers() == [30.0]  # B's window, not A's inline unload
    _join_hook_threads()
    assert hooks == [("warm-B", "b", "b-token"), ("release-B", "b", "b-token")]
    assert tts_tool_lifecycle.tts_lease_holders() == []


@contextmanager
def _as_home(home):
    """Bind ``home`` by HERMES_HOME override only: no secret scope is hydrated (unlike
    ``bind_profile_scope``), so every profile reads the one process-env SRV_TOKEN."""
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Bare homes a/b -> ``(dirs, as_profile)``; ``as_profile(name)`` binds one with ``_as_home``."""
    dirs = {name: tmp_path / name for name in ("a", "b")}
    for home in dirs.values():
        home.mkdir()
    monkeypatch.setenv("SRV_TOKEN", "token")
    return dirs, lambda name: _as_home(dirs[name])


@pytest.fixture
def profiles(homes, monkeypatch):
    """Homes a/b with their own command-provider config (a: 0s window, b: 30s); returns ``as_profile``."""
    _dirs, as_profile = homes
    configs = {name: _srv_config(name.upper(), keep_warm) for name, keep_warm in (("a", 0), ("b", 30))}
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: configs[get_hermes_home().name])
    return as_profile


def _release_hooks(hooks):
    return sorted(c for c, _home, _tok in hooks if c.startswith("release-"))


def test_every_acquiring_profile_is_released_in_its_own_profile(profiles, timers, hooks):
    """Distinct holders from two profiles, each dropped by the OTHER profile: the window is the
    longest an owner asked for (B's 30s, not releaser B's view of A's 0), then both are released."""
    with profiles("b"):
        tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    with profiles("a"):
        tts_tool_lifecycle.acquire_tts_lease("desktop:read-aloud")
        tts_tool_lifecycle.release_tts_lease("tui:voice-tts")
    with profiles("b"):
        tts_tool_lifecycle.release_tts_lease("desktop:read-aloud")
    assert timers() == [30.0]
    _join_hook_threads()
    released = sorted((c, h) for c, h, _ in hooks if c.startswith("release-"))
    assert released == [("release-A", "a"), ("release-B", "b")]


def test_same_name_reacquired_by_another_profile_releases_both_owners(profiles, timers, hooks):
    """Holder names stay process-wide (either profile's release ends the hold, as before); what
    each profile warmed is still released under that profile, and nothing is retained after."""
    with profiles("b"):
        tts_tool_lifecycle.acquire_tts_lease("desktop:read-aloud")
    with profiles("a"):
        tts_tool_lifecycle.acquire_tts_lease("desktop:read-aloud")
        assert tts_tool_lifecycle.release_tts_lease("desktop:read-aloud")["leases"] == 0
    timers()
    _join_hook_threads()
    assert _release_hooks(hooks) == ["release-A", "release-B"]
    assert tts_tool_lifecycle._tts_owners == {}


class _GatedPlugin:
    """Plugin provider whose ``release()`` can block on a gate; records ordering."""

    def __init__(self):
        self.calls: list = []
        self.in_release, self.gate = threading.Event(), threading.Event()
        self.gate.set()
        self.lock_free_in_release = None

    def warm(self):
        self.calls.append("warm")

    def release(self):
        self.calls.append("release-start")
        got = tts_tool_lifecycle._tts_lease_lock.acquire(timeout=2)
        if got:
            tts_tool_lifecycle._tts_lease_lock.release()
        self.lock_free_in_release = got
        self.in_release.set()
        assert self.gate.wait(5)
        self.calls.append("release-end")


@pytest.fixture
def plugin(monkeypatch):
    provider = _GatedPlugin()
    monkeypatch.setattr(tts_tool_lifecycle, "_lookup_plugin_provider", lambda name: provider)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "custom", "keep_warm_seconds": 10})
    return provider


def test_release_hook_runs_outside_the_lease_lock(plugin, timers):
    tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    tts_tool_lifecycle.release_tts_lease("tui:voice-tts")
    timers()
    assert plugin.lock_free_in_release is True
    assert plugin.calls == ["warm", "release-start", "release-end"]


class _BarrierProbe:
    """The teardown lock, signalling when the reacquiring thread reaches it (then blocks on it)."""

    def __init__(self, lock):
        self.lock, self.reached = lock, threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "reacquire":
            self.reached.set()
        return self.lock.__enter__()

    def __exit__(self, *exc):
        return self.lock.__exit__(*exc)


def test_reacquire_during_a_running_release_warms_after_it(plugin, timers, monkeypatch):
    barrier = _BarrierProbe(tts_tool_lifecycle._tts_teardown_lock)
    monkeypatch.setattr(tts_tool_lifecycle, "_tts_teardown_lock", barrier)
    tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    tts_tool_lifecycle.release_tts_lease("tui:voice-tts")
    plugin.gate.clear()
    teardown = threading.Thread(target=timers)
    teardown.start()
    reacquire = threading.Thread(target=tts_tool_lifecycle.acquire_tts_lease, args=("tui:voice-tts",),
                                 name="reacquire")
    try:
        assert plugin.in_release.wait(5)
        reacquire.start()
        # The reacquire is at the barrier while the running release still holds it (a non-blocking
        # try from here fails), so it cannot have warmed: no timing assumption.
        assert barrier.reached.wait(5), "the reacquire never waited for the running release"
        if barrier.lock.acquire(blocking=False):
            barrier.lock.release()  # never leave the module's real lock held by a failing run
            pytest.fail("the running release did not hold the teardown barrier")
        assert plugin.calls == ["warm", "release-start"]
    finally:
        plugin.gate.set()
        teardown.join(5)
        reacquire.join(5)
    assert not teardown.is_alive() and not reacquire.is_alive()
    assert plugin.calls == ["warm", "release-start", "release-end", "warm"]
    assert tts_tool_lifecycle.tts_lease_holders() == ["tui:voice-tts"]


def test_release_during_warm_still_releases_after_the_warm(monkeypatch, timers):
    """The last holder drops mid-warm and the unload runs then; the warm that finishes afterwards
    left the resource up again, so its owner is restored and released once more."""
    calls: list = []

    class Provider:
        def warm(self):
            calls.append("warm-start")
            tts_tool_lifecycle.release_tts_lease("tui:voice-tts")
            calls.append("warm-end")

        def release(self):
            calls.append("release")

    monkeypatch.setattr(tts_tool_lifecycle, "_lookup_plugin_provider", lambda name: Provider())
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "custom", "keep_warm_seconds": 0})
    tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    assert calls[-2:] == ["warm-end", "release"]
    assert tts_tool_lifecycle._tts_owners == {}


def test_unload_never_enters_an_owner_context_another_thread_is_in(monkeypatch, timers):
    """``Context.run`` refuses a context already entered elsewhere; the owner context must only
    ever be run through a copy (keep-warm read and release hook alike)."""
    seen: list = []
    monkeypatch.setattr(tts_tool_lifecycle, "warm_tts_provider", lambda cfg=None: {"action": "noop"})
    monkeypatch.setattr(tts_tool_lifecycle, "_signal_user_tts_provider",
                        lambda name, cfg, hook: seen.append((hook, name)))
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "srv", "keep_warm_seconds": 0})
    tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    owner = tts_tool_lifecycle._tts_owners[hermes_home_key()]
    inside, leave = threading.Event(), threading.Event()
    holder = threading.Thread(target=owner.run, args=(lambda: (inside.set(), leave.wait(5)),))
    holder.start()
    try:
        assert inside.wait(5)
        assert tts_tool_lifecycle.release_tts_lease("tui:voice-tts") == {"leases": 0, "released": 0}
    finally:
        leave.set()
        holder.join(5)
    assert seen == [("release", "srv")]
    assert tts_tool_lifecycle._tts_owners == {}


@pytest.fixture
def real_profiles(homes):
    """Homes a/b whose ``config.yaml`` the real loader reads, both healthy command providers.
    Returns ``(as_profile, write)``; ``write(name, text)`` replaces a profile's config file (an edit
    made while the lease is held: a broken file already fails that profile's own warm-up)."""
    from hermes_cli.config import atomic_config_write
    dirs, as_profile = homes
    for name, home in dirs.items():
        atomic_config_write(home / "config.yaml", {"tts": _srv_config(name.upper(), 0)})

    def write(name, text):
        (dirs[name] / "config.yaml").write_text(text)
    return as_profile, write


def _acquire_in(as_profile, order):
    for name in order:
        with as_profile(name):
            tts_tool_lifecycle.acquire_tts_lease(f"lease-{name}")


ORDERS = pytest.mark.parametrize("order", [("a", "b"), ("b", "a")], ids=["broken-first", "broken-last"])


@ORDERS
def test_an_unreadable_owner_window_still_schedules_the_healthy_owner(
        real_profiles, timers, hooks, monkeypatch, order):
    """Profile a's ``tts:`` is a truthy non-dict the real loader passes through: its window cannot
    be read, so it counts as the default window, and b is still released in its own profile."""
    from tools import tts_tool_local
    as_profile, write = real_profiles
    _acquire_in(as_profile, order)
    write("a", "tts: broken-shape\n")
    monkeypatch.setitem(tts_tool_local._piper_voice_cache, "voice", object())
    with as_profile("b"):
        for name in order:
            tts_tool_lifecycle.release_tts_lease(f"lease-{name}")
    assert timers() == [tts_tool_lifecycle._DEFAULT_KEEP_WARM_SECONDS]
    _join_hook_threads()
    assert _release_hooks(hooks) == ["release-B"]
    assert tts_tool_local._piper_voice_cache == {}
    assert tts_tool_lifecycle._tts_owners == {}


@ORDERS
def test_a_failing_owner_release_does_not_skip_the_healthy_owner(
        real_profiles, timers, hooks, monkeypatch, order):
    """Profile a's window reads (0) but its release raises (a non-string provider): b is still
    released, the resident model is still unloaded and counted, and nothing escapes the caller."""
    from tools import tts_tool_local
    as_profile, write = real_profiles
    _acquire_in(as_profile, order)
    write("a", "tts:\n  provider: 5\n  keep_warm_seconds: 0\n")
    monkeypatch.setitem(tts_tool_local._piper_voice_cache, "voice", object())
    with as_profile("a"):
        tts_tool_lifecycle.release_tts_lease(f"lease-{order[0]}")
        result = tts_tool_lifecycle.release_tts_lease(f"lease-{order[1]}")
    assert result == {"leases": 0, "released": 1}
    _join_hook_threads()
    assert _release_hooks(hooks) == ["release-B"]
    assert tts_tool_local._piper_voice_cache == {}
    assert tts_tool_lifecycle._tts_owners == {}


def test_a_lone_failing_owner_still_unloads_local_models(real_profiles, timers, hooks, monkeypatch):
    from tools import tts_tool_local
    as_profile, write = real_profiles
    _acquire_in(as_profile, ("a",))
    write("a", "tts:\n  provider: 5\n  keep_warm_seconds: 0\n")
    monkeypatch.setitem(tts_tool_local._piper_voice_cache, "voice", object())
    with as_profile("a"):
        assert tts_tool_lifecycle.release_tts_lease("lease-a") == {"leases": 0, "released": 1}
    assert tts_tool_local._piper_voice_cache == {}


@ORDERS
def test_an_owner_config_backend_outage_does_not_stop_the_healthy_owner(
        real_profiles, timers, hooks, monkeypatch, order):
    """A config-backend outage is a ``SystemExit`` subclass that skips ``except Exception``; it must
    stay inside the failing owner too, for both its window read and its release."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    as_profile, _write = real_profiles
    _acquire_in(as_profile, order)
    real_load = tts_tool._load_tts_config

    def load():
        if get_hermes_home().name == "a":
            raise ConfigBackendUnavailable("backend down")
        return real_load()
    monkeypatch.setattr(tts_tool, "_load_tts_config", load)
    with as_profile("b"):
        for name in order:
            tts_tool_lifecycle.release_tts_lease(f"lease-{name}")
    assert timers() == [tts_tool_lifecycle._DEFAULT_KEEP_WARM_SECONDS]
    _join_hook_threads()
    assert _release_hooks(hooks) == ["release-B"]
    assert tts_tool_lifecycle._tts_owners == {}


@pytest.fixture
def signals(monkeypatch):
    """Record each user-provider lease signal as ``(hook, provider, home name)``; config is ``srv``."""
    seen: list = []
    monkeypatch.setattr(tts_tool_lifecycle, "_signal_user_tts_provider",
                        lambda name, cfg, hook: seen.append((hook, name, get_hermes_home().name)))
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "srv", "keep_warm_seconds": 0})
    return seen


def _resident(**counts):
    from tools import tts_tool_local
    for name, n in counts.items():
        tts_tool_local._LOCAL_TTS_MODEL_CACHES[name].update({f"m{i}": object() for i in range(n)})
    return tts_tool_local._LOCAL_TTS_MODEL_CACHES


@pytest.mark.parametrize("provider", [None, "", "   "], ids=["none", "blank", "whitespace"])
def test_release_provider_default_signals_once_and_drops_every_cache(signals, provider):
    caches = _resident(piper=2, kittentts=1)
    assert tts_tool_lifecycle.release_tts_provider(provider) == {"released": 3}
    assert [s[:2] for s in signals] == [("release", "srv")]
    assert not any(caches.values())


@pytest.mark.parametrize("provider", ["piper", " Piper "])
def test_release_provider_named_drops_only_that_cache_and_signals_nothing(signals, provider):
    caches = _resident(piper=2, kittentts=1)
    assert tts_tool_lifecycle.release_tts_provider(provider) == {"released": 2}
    assert signals == []
    assert (len(caches["piper"]), len(caches["kittentts"])) == (0, 1)


def test_release_provider_unknown_name_drops_nothing(signals):
    caches = _resident(piper=1)
    assert tts_tool_lifecycle.release_tts_provider("no-such-engine") == {"released": 0}
    assert signals == [] and len(caches["piper"]) == 1


def test_unload_signals_each_owner_once_in_its_profile_and_counts_models_once(homes, signals, timers):
    _dirs, as_profile = homes
    for name in ("a", "b"):
        with as_profile(name):
            tts_tool_lifecycle.acquire_tts_lease(f"lease-{name}")
    caches = _resident(piper=2)
    with as_profile("a"):
        tts_tool_lifecycle.release_tts_lease("lease-b")
        assert tts_tool_lifecycle.release_tts_lease("lease-a") == {"leases": 0, "released": 2}
    assert sorted(s for s in signals if s[0] == "release") == [("release", "srv", "a"), ("release", "srv", "b")]
    assert not caches["piper"] and tts_tool_lifecycle._tts_owners == {}


def test_reentrant_reacquire_during_release_keeps_freshly_warmed_local_model(monkeypatch, timers):
    from tools import tts_tool_local
    reacquired = False

    class Provider:
        def warm(self):
            pass

        def release(self):
            nonlocal reacquired
            if not reacquired:
                reacquired = True
                tts_tool_lifecycle.acquire_tts_lease("desktop:read-aloud", {"provider": "piper"})

    def load_piper(_cfg):
        tts_tool_local._piper_voice_cache["reacquired"] = object()
        return tts_tool_local._piper_voice_cache["reacquired"]

    monkeypatch.setattr(tts_tool_lifecycle, "_lookup_plugin_provider", lambda name: Provider())
    monkeypatch.setattr(tts_tool_lifecycle, "_local_tts_warmers",
                        lambda: {"piper": load_piper, "kittentts": lambda _c: None})
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "custom", "keep_warm_seconds": 0})
    tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    assert tts_tool_lifecycle.release_tts_lease("tui:voice-tts") == {"leases": 0, "released": 0}
    assert tts_tool_lifecycle.tts_lease_holders() == ["desktop:read-aloud"]
    assert "reacquired" in tts_tool_local._piper_voice_cache


def test_reentrant_reacquire_then_release_keeps_model_for_new_window(monkeypatch, timers):
    from tools import tts_tool_local
    reacquired = False

    class Provider:
        def warm(self):
            pass

        def release(self):
            nonlocal reacquired
            if not reacquired:
                reacquired = True
                tts_tool_lifecycle.acquire_tts_lease("desktop:read-aloud", {"provider": "piper"})
                tts_tool_lifecycle.release_tts_lease("desktop:read-aloud")

    def load_piper(_cfg):
        tts_tool_local._piper_voice_cache["kept-warm"] = object()
        return tts_tool_local._piper_voice_cache["kept-warm"]

    monkeypatch.setattr(tts_tool_lifecycle, "_lookup_plugin_provider", lambda name: Provider())
    monkeypatch.setattr(tts_tool_lifecycle, "_local_tts_warmers",
                        lambda: {"piper": load_piper, "kittentts": lambda _c: None})
    monkeypatch.setattr(
        tts_tool, "_load_tts_config",
        lambda: {"provider": "piper", "keep_warm_seconds": 30} if reacquired
        else {"provider": "custom", "keep_warm_seconds": 0},
    )
    tts_tool_lifecycle.acquire_tts_lease("tui:voice-tts")
    assert tts_tool_lifecycle.release_tts_lease("tui:voice-tts") == {"leases": 0, "released": 0}
    assert tts_tool_lifecycle.tts_lease_holders() == []
    assert "kept-warm" in tts_tool_local._piper_voice_cache
    assert timers() == [30.0]
    assert tts_tool_local._piper_voice_cache == {}


@pytest.mark.parametrize(
    ("reacquire_as", "expected_events", "expected_owners"),
    [
        ("b", ["A:warm", "B:warm", "A:release", "B:warm"], ["b"]),
        ("a", ["A:warm", "B:warm", "A:release", "A:warm", "B:release"], ["a"]),
    ],
    ids=["reacquire-other-skips-its-stale-release", "reacquire-self-still-releases-other"],
)
def test_reentrant_reacquire_skips_only_superseded_owner(
        homes, monkeypatch, timers, reacquire_as, expected_events, expected_owners):
    dirs, as_profile = homes
    events: list = []
    reacquired = False

    class PluginA:
        def warm(self):
            events.append("A:warm")

        def release(self):
            nonlocal reacquired
            events.append("A:release")
            if not reacquired:
                reacquired = True
                with as_profile(reacquire_as):
                    tts_tool_lifecycle.acquire_tts_lease(f"lease-{reacquire_as}-new")

    class PluginB:
        def warm(self):
            events.append("B:warm")

        def release(self):
            events.append("B:release")

    plugins = {"pa": PluginA(), "pb": PluginB()}
    monkeypatch.setattr(tts_tool_lifecycle, "_lookup_plugin_provider", lambda name: plugins[name])
    monkeypatch.setattr(
        tts_tool, "_load_tts_config",
        lambda: {"provider": "pa" if get_hermes_home().name == "a" else "pb", "keep_warm_seconds": 0},
    )
    with as_profile("a"):
        tts_tool_lifecycle.acquire_tts_lease("lease-a")
    with as_profile("b"):
        tts_tool_lifecycle.acquire_tts_lease("lease-b")
    with as_profile("a"):
        tts_tool_lifecycle.release_tts_lease("lease-a")
        tts_tool_lifecycle.release_tts_lease("lease-b")
    assert events == expected_events
    assert sorted(k for k in tts_tool_lifecycle._tts_owners) == [hermes_home_key(dirs[n]) for n in expected_owners]


def test_reentrant_reacquire_and_zero_window_release_does_not_resignal_stale_owner(homes, monkeypatch, timers):
    """If A's release hook reacquires B and immediately releases B with keep_warm_seconds=0, the
    nested teardown already signalled B under its new context; the outer loop must not signal B a
    second time under the stale acquisition context."""
    from agent import secret_scope as ss

    _dirs, as_profile = homes
    events: list = []
    reacquired = False

    class PluginA:
        def warm(self):
            events.append("A:warm")

        def release(self):
            nonlocal reacquired
            events.append("A:release")
            if not reacquired:
                reacquired = True
                with as_profile("b"):
                    tok = ss.set_secret_scope({"B_VER": "new"})
                    try:
                        tts_tool_lifecycle.acquire_tts_lease("lease-b-new")
                        tts_tool_lifecycle.release_tts_lease("lease-b-new")
                    finally:
                        ss.reset_secret_scope(tok)

    class PluginB:
        def warm(self):
            events.append(("B:warm", ss.get_secret("B_VER")))

        def release(self):
            events.append(("B:release", ss.get_secret("B_VER")))

    plugins = {"pa": PluginA(), "pb": PluginB()}
    monkeypatch.setattr(tts_tool_lifecycle, "_lookup_plugin_provider", lambda name: plugins[name])
    monkeypatch.setattr(
        tts_tool, "_load_tts_config",
        lambda: {"provider": "pa" if get_hermes_home().name == "a" else "pb", "keep_warm_seconds": 0},
    )
    with as_profile("a"):
        tts_tool_lifecycle.acquire_tts_lease("lease-a")
    with as_profile("b"):
        tok = ss.set_secret_scope({"B_VER": "old"})
        try:
            tts_tool_lifecycle.acquire_tts_lease("lease-b")
        finally:
            ss.reset_secret_scope(tok)
    with as_profile("a"):
        tts_tool_lifecycle.release_tts_lease("lease-a")
        tts_tool_lifecycle.release_tts_lease("lease-b")
    assert events == ["A:warm", ("B:warm", "old"), "A:release", ("B:warm", "new"), ("B:release", "new")]
    assert tts_tool_lifecycle._tts_owners == {}
