"""Regression tests for the frequently-polled /api/status platform-config read."""

import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli import web_server


@pytest.fixture(autouse=True)
def clear_platform_cache():
    def reset():
        web_server._CONFIGURED_PLATFORM_CACHE_WAITERS = 0
        web_server._CONFIGURED_PLATFORM_CACHE.clear()
        web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT.clear()
        web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.clear()
        web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION.clear()

    reset()
    yield
    reset()


def test_configured_gateway_platforms_are_cached_within_poll_window(monkeypatch):
    calls = 0

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        return FakeGatewayConfig()

    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 1


def test_configured_gateway_platform_cache_is_profile_scoped(monkeypatch):
    home = "/tmp/hermes-default"
    calls = []

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def load_gateway_config():
        calls.append(home)
        return FakeGatewayConfig("discord" if home.endswith("default") else "telegram")

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: home)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    home = "/tmp/hermes-worker"
    assert web_server._load_configured_gateway_platforms() == {"telegram"}
    home = "/tmp/hermes-default"
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    home = "/tmp/hermes-worker"
    assert web_server._load_configured_gateway_platforms() == {"telegram"}
    assert calls == ["/tmp/hermes-default", "/tmp/hermes-worker"]


def test_configured_gateway_platform_cache_collapses_concurrent_misses(monkeypatch):
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return FakeGatewayConfig()

    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(web_server._load_configured_gateway_platforms) for _ in range(5)]
        assert entered.wait(timeout=2)
        time.sleep(0.05)
        release.set()
        assert [future.result(timeout=2) for future in futures] == [{"discord"}] * 5

    assert calls == 1


def test_configured_gateway_platform_cache_does_not_serialize_profiles(monkeypatch):
    local = threading.local()
    both_loaders_entered = threading.Barrier(2)

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def load_gateway_config():
        both_loaders_entered.wait(timeout=1)
        return FakeGatewayConfig(local.platform)

    def load_for(home, platform):
        local.home = home
        local.platform = platform
        return web_server._load_configured_gateway_platforms()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: local.home)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        default = pool.submit(load_for, "/tmp/hermes-default", "discord")
        worker = pool.submit(load_for, "/tmp/hermes-worker", "telegram")
        assert default.result(timeout=2) == {"discord"}
        assert worker.result(timeout=2) == {"telegram"}


def test_configured_gateway_platform_cache_refreshes_after_ttl(monkeypatch):
    calls = 0

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        return FakeGatewayConfig()

    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    entry = web_server._CONFIGURED_PLATFORM_CACHE[str(web_server.get_hermes_home())]
    entry["ts"] -= web_server._CONFIGURED_PLATFORM_CACHE_TTL + 0.1
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_invalidates_on_source_change(monkeypatch):
    calls = 0
    source_version = (("config.yaml", 1, 1, 10),)

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        return FakeGatewayConfig()

    monkeypatch.setattr(
        web_server,
        "_configured_platform_source_version",
        lambda _home: source_version,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    source_version = (("config.yaml", 1, 2, 11),)
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_does_not_poison_on_error(monkeypatch):
    calls = 0

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary config failure")
        return FakeGatewayConfig()

    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    with pytest.raises(RuntimeError, match="temporary config failure"):
        web_server._load_configured_gateway_platforms()
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_wakes_waiters_after_base_exception(
    monkeypatch, tmp_path
):
    calls = 0
    entered = threading.Event()
    release = threading.Event()
    outcomes = []

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2)
            raise SystemExit("plugin exited during discovery")
        return FakeGatewayConfig()

    def load_platforms():
        try:
            outcomes.append(web_server._load_configured_gateway_platforms())
        except BaseException as exc:
            outcomes.append(exc)

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    builder = threading.Thread(target=load_platforms, daemon=True)
    waiter = threading.Thread(target=load_platforms, daemon=True)
    builder.start()
    assert entered.wait(timeout=2)
    waiter.start()
    time.sleep(0.05)
    release.set()
    builder.join(timeout=2)
    waiter.join(timeout=0.5)
    waiter_was_stuck = waiter.is_alive()

    # Ensure a failing implementation cannot strand a daemon thread after the
    # assertion records the defect.
    for flight in web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT.values():
        event = flight["event"] if isinstance(flight, dict) else flight
        event.set()
    waiter.join(timeout=2)

    assert not waiter_was_stuck
    assert any(isinstance(outcome, SystemExit) for outcome in outcomes)
    assert any(isinstance(outcome, RuntimeError) for outcome in outcomes)
    assert calls == 1
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_wakes_waiters_after_cache_write_failure(
    monkeypatch, tmp_path
):
    calls = 0
    entered = threading.Event()
    release = threading.Event()
    outcomes = []

    class FailOnceCache(dict):
        failed = False

        def __setitem__(self, key, value):
            if not self.failed:
                self.failed = True
                raise MemoryError("simulated cache write failure")
            super().__setitem__(key, value)

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=2)
        return FakeGatewayConfig()

    def load_platforms():
        try:
            outcomes.append(web_server._load_configured_gateway_platforms())
        except BaseException as exc:
            outcomes.append(exc)

    monkeypatch.setattr(web_server, "_CONFIGURED_PLATFORM_CACHE", FailOnceCache())
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    builder = threading.Thread(target=load_platforms, daemon=True)
    waiter = threading.Thread(target=load_platforms, daemon=True)
    builder.start()
    assert entered.wait(timeout=2)
    waiter.start()
    time.sleep(0.05)
    release.set()
    builder.join(timeout=2)
    waiter.join(timeout=0.5)
    waiter_was_stuck = waiter.is_alive()

    for flight in web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT.values():
        event = flight["event"] if isinstance(flight, dict) else flight
        event.set()
    waiter.join(timeout=2)

    assert not waiter_was_stuck
    assert any(isinstance(outcome, MemoryError) for outcome in outcomes)
    assert any(isinstance(outcome, RuntimeError) for outcome in outcomes)
    assert calls == 1
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_ttl_starts_after_slow_load(
    monkeypatch, tmp_path
):
    calls = 0
    now = 100.0

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls, now
        calls += 1
        now += web_server._CONFIGURED_PLATFORM_CACHE_TTL + 1
        return FakeGatewayConfig()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server.time, "monotonic", lambda: now)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 1


def test_configured_gateway_platform_cache_checks_ttl_after_fingerprint(
    monkeypatch, tmp_path
):
    calls = 0
    fingerprint_checks = 0
    now = 100.0

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        return FakeGatewayConfig()

    def source_version(_home):
        nonlocal fingerprint_checks, now
        fingerprint_checks += 1
        if fingerprint_checks == 2:
            now += web_server._CONFIGURED_PLATFORM_CACHE_TTL + 1
        return (("config.yaml", 1, 1, 10),)

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server.time, "monotonic", lambda: now)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_broadcasts_failure_to_waiters(
    monkeypatch, tmp_path
):
    calls = 0
    entered = threading.Event()
    release = threading.Event()
    outcomes = []

    def load_gateway_config():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        raise RuntimeError("persistent config failure")

    def load_platforms():
        try:
            outcomes.append(web_server._load_configured_gateway_platforms())
        except Exception as exc:
            outcomes.append(exc)

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    threads = [threading.Thread(target=load_platforms) for _ in range(4)]
    threads[0].start()
    assert entered.wait(timeout=2)
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.05)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert calls == 1
    assert len(outcomes) == 4
    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)


def test_configured_gateway_platform_cache_waiters_time_out_on_hung_builder(
    monkeypatch, tmp_path
):
    entered = threading.Event()
    release = threading.Event()
    waiter_outcomes = []

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        entered.set()
        assert release.wait(timeout=2)
        return FakeGatewayConfig()

    def wait_for_platforms():
        try:
            waiter_outcomes.append(web_server._load_configured_gateway_platforms())
        except Exception as exc:
            waiter_outcomes.append(exc)

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        web_server,
        "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT",
        0.05,
        raising=False,
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    builder = threading.Thread(target=web_server._load_configured_gateway_platforms)
    waiter = threading.Thread(target=wait_for_platforms)
    builder.start()
    assert entered.wait(timeout=2)
    waiter.start()
    waiter.join(timeout=0.25)
    waiter_was_stuck = waiter.is_alive()
    release.set()
    builder.join(timeout=2)
    waiter.join(timeout=2)

    assert not waiter_was_stuck
    assert len(waiter_outcomes) == 1
    assert isinstance(waiter_outcomes[0], TimeoutError)


def test_configured_gateway_platform_cache_real_source_change_invalidates(
    monkeypatch, tmp_path
):
    calls = 0

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        return FakeGatewayConfig()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    before = web_server._configured_platform_source_version(tmp_path)
    assert before[0] == (str(tmp_path / "config.yaml"), None, None, None)
    assert web_server._load_configured_gateway_platforms() == {"discord"}

    (tmp_path / "config.yaml").write_text("gateway: {}\n")
    after = web_server._configured_platform_source_version(tmp_path)

    assert after != before
    assert after[0][0] == str(tmp_path / "config.yaml")
    assert all(value is not None for value in after[0][1:])
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert calls == 2


def test_configured_gateway_platform_cache_does_not_publish_superseded_build(
    monkeypatch, tmp_path
):
    calls = 0
    version = 1
    first_entered = threading.Event()
    release_first = threading.Event()
    outcomes = []

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        return FakeGatewayConfig()

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_platforms():
        outcomes.append(web_server._load_configured_gateway_platforms())

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    older = threading.Thread(target=load_platforms)
    newer = threading.Thread(target=load_platforms)
    older.start()
    assert first_entered.wait(timeout=2)
    version = 2
    newer.start()
    newer.join(timeout=2)
    assert not newer.is_alive()
    version = 1
    release_first.set()
    older.join(timeout=2)

    assert not older.is_alive()
    assert outcomes == [{"discord"}, {"discord"}]
    entry = web_server._CONFIGURED_PLATFORM_CACHE[str(tmp_path)]
    assert entry["source_version"] == (("config.yaml", 1, 2, 10),)
    assert calls == 2


def test_configured_gateway_platform_cache_suppresses_older_build_while_newer_runs(
    monkeypatch, tmp_path
):
    calls = 0
    version = 1
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    outcomes = {}

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def load_gateway_config():
        nonlocal calls
        calls += 1
        call = calls
        if call == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
            assert release_second.wait(timeout=2)
        return FakeGatewayConfig("discord" if call == 1 else "telegram")

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_platforms(name):
        outcomes[name] = web_server._load_configured_gateway_platforms()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    older = threading.Thread(target=load_platforms, args=("older",))
    newer = threading.Thread(target=load_platforms, args=("newer",))
    older.start()
    assert first_entered.wait(timeout=2)
    version = 2
    newer.start()
    assert second_entered.wait(timeout=2)

    # Return the source fingerprint to v1 while the newer v2 build is still
    # running. The older generation must not publish merely because no newer
    # cache entry exists yet.
    version = 1
    release_first.set()
    older.join(timeout=2)

    assert not older.is_alive()
    assert outcomes["older"] == {"discord"}
    older_published = str(tmp_path) in web_server._CONFIGURED_PLATFORM_CACHE

    version = 2
    release_second.set()
    newer.join(timeout=2)

    assert not newer.is_alive()
    assert not older_published
    assert outcomes["newer"] == {"telegram"}
    entry = web_server._CONFIGURED_PLATFORM_CACHE[str(tmp_path)]
    assert entry["source_version"] == (("config.yaml", 1, 2, 10),)
    assert entry["platforms"] == frozenset({"telegram"})
    assert calls == 2


def test_configured_gateway_platform_cache_stays_bounded(monkeypatch, tmp_path):
    home = tmp_path / "profile-0"

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: home)
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: FakeGatewayConfig(),
    )

    for index in range(web_server._CONFIGURED_PLATFORM_CACHE_MAX_KEYS + 1):
        home = tmp_path / f"profile-{index}"
        assert web_server._load_configured_gateway_platforms() == {"discord"}

    assert (
        len(web_server._CONFIGURED_PLATFORM_CACHE)
        == web_server._CONFIGURED_PLATFORM_CACHE_MAX_KEYS
    )
    assert str(tmp_path / "profile-0") not in web_server._CONFIGURED_PLATFORM_CACHE
    assert (
        len(web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION)
        == web_server._CONFIGURED_PLATFORM_CACHE_MAX_KEYS
    )
    assert (
        str(tmp_path / "profile-0")
        not in web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION
    )


def test_configured_gateway_platform_cache_replaces_persistently_hung_flight(
    monkeypatch, tmp_path
):
    """A wedged builder must not stall every later poll for a full timeout."""
    calls = 0
    calls_lock = threading.Lock()
    first_entered = threading.Event()
    release_first = threading.Event()

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def load_gateway_config():
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_entered.set()
            release_first.wait(timeout=10)
            return FakeGatewayConfig("discord")
        return FakeGatewayConfig("telegram")

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 0.05, raising=False
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    hung = threading.Thread(target=web_server._load_configured_gateway_platforms)
    hung.start()
    try:
        assert first_entered.wait(timeout=10)

        # The first waiter gives up on the hung flight...
        with pytest.raises(TimeoutError):
            web_server._load_configured_gateway_platforms()

        # ...and must have detached it, so the next poll builds a replacement
        # instead of queueing behind the same hang for another full timeout.
        assert web_server._load_configured_gateway_platforms() == {"telegram"}
        assert calls == 2
        assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}
    finally:
        release_first.set()
        hung.join(timeout=10)

    # The late stale builder must not clobber the replacement's entry, remove
    # its bookkeeping, or publish its own superseded result.
    assert not hung.is_alive()
    entry = web_server._CONFIGURED_PLATFORM_CACHE[str(tmp_path)]
    assert entry["platforms"] == frozenset({"telegram"})
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}
    assert web_server._load_configured_gateway_platforms() == {"telegram"}
    assert calls == 2


def test_configured_gateway_platform_cache_abandons_stale_flight_without_waiting(
    monkeypatch, tmp_path
):
    """A caller arriving past the max flight age refuses to queue behind it."""
    calls = 0
    calls_lock = threading.Lock()
    first_entered = threading.Event()
    release_first = threading.Event()

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal calls
        with calls_lock:
            calls += 1
            call = calls
        if call == 1:
            first_entered.set()
            release_first.wait(timeout=10)
        return FakeGatewayConfig()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_FLIGHT_MAX_AGE", 0.05, raising=False
    )
    # Long enough that a caller which parks on the hung flight is unmistakably
    # stuck rather than merely slow.
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 2.0, raising=False
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    hung = threading.Thread(target=web_server._load_configured_gateway_platforms)
    hung.start()
    try:
        assert first_entered.wait(timeout=10)
        time.sleep(0.06)

        started = time.monotonic()
        platforms = web_server._load_configured_gateway_platforms()
        elapsed = time.monotonic() - started
    finally:
        release_first.set()
        hung.join(timeout=10)

    assert platforms == {"discord"}
    assert elapsed < 1.0
    assert calls == 2
    assert not hung.is_alive()


def test_configured_gateway_platform_cache_bounds_builders_under_version_churn(
    monkeypatch, tmp_path
):
    """Churn plus indefinitely-held loaders must not spawn a builder per poll.

    Every poller sees a brand-new fingerprint, so none of them can reuse a cache
    entry or join a same-fingerprint flight -- each one *wants* to build.  The
    loaders never return, so a thread that reaches ``load_gateway_config()``
    stays parked in the worker pool for the rest of the test and the count of
    entries is a direct, race-free measure of how many builders were spawned.
    """
    home = str(tmp_path)
    pollers = 8
    entries = 0
    entries_lock = threading.Lock()
    release = threading.Event()
    version = 0
    version_lock = threading.Lock()

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def source_version(_home):
        nonlocal version
        with version_lock:
            version += 1
            return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
        release.wait(timeout=30)
        return FakeGatewayConfig()

    returned = threading.Semaphore(0)

    def poll():
        try:
            web_server._load_configured_gateway_platforms()
        except BaseException:  # pragma: no cover - being turned away is the point
            pass
        finally:
            returned.release()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 0.05, raising=False
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    threads = [threading.Thread(target=poll) for _ in range(pollers)]
    try:
        for thread in threads:
            thread.start()
        # Drain returns until they stop coming; whatever has not returned is
        # wedged inside the loader.  Cheaper and just as strict as joining the
        # wedged threads, which by construction never finish on their own.
        returns = 0
        while returns < pollers and returned.acquire(timeout=1.0):
            returns += 1
        with entries_lock:
            builders_spawned = entries
        still_running = [thread for thread in threads if thread.is_alive()]

        # The load is unkillable and holds a worker-pool thread, so the bound
        # has to be on how many are ever started -- not just on dict size.
        assert builders_spawned < pollers
        assert len(still_running) < pollers
        # Turned-away pollers must return rather than pile up behind the wedge.
        assert len(still_running) == builders_spawned
        assert builders_spawned == web_server._CONFIGURED_PLATFORM_CACHE_MAX_BUILDERS
        assert (
            web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS[home] == builders_spawned
        )
        # Bookkeeping stays bounded alongside the threads.
        assert len(web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT) <= 1
        assert len(web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION) == 1
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    # Once the wedged loads return, every permit and bookkeeping entry is
    # reclaimed -- nothing leaked by the pollers that were turned away.
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION == {}


def test_configured_gateway_platform_cache_recovers_from_one_wedged_builder(
    monkeypatch, tmp_path
):
    """One permanently wedged builder must not stop the profile refreshing."""
    entries = 0
    entries_lock = threading.Lock()
    first_entered = threading.Event()
    release_first = threading.Event()
    version = 1

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
            entry = entries
        if entry == 1:
            first_entered.set()
            release_first.wait(timeout=30)
            return FakeGatewayConfig("discord")
        return FakeGatewayConfig("telegram")

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 0.05, raising=False
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    wedged = threading.Thread(target=web_server._load_configured_gateway_platforms)
    wedged.start()
    try:
        assert first_entered.wait(timeout=10)
        # Move the fingerprint on so later polls genuinely need a new build
        # rather than joining the wedged same-version flight.
        version = 2

        # The surviving permit must be recycled, not consumed once: repeated
        # refreshes keep succeeding while the first builder stays stuck.
        for _ in range(4):
            web_server._CONFIGURED_PLATFORM_CACHE.clear()
            assert web_server._load_configured_gateway_platforms() == {"telegram"}
            assert (
                web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS[str(tmp_path)] == 1
            ), "the wedged builder should hold exactly one permit"
    finally:
        release_first.set()
        wedged.join(timeout=30)

    assert not wedged.is_alive()
    assert entries == 5
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}


def test_configured_gateway_platform_cache_late_builder_cannot_touch_current_generation(
    monkeypatch, tmp_path
):
    """A late stale builder may neither unregister nor publish over its successor."""
    home = str(tmp_path)
    entries = 0
    entries_lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    version = 1
    outcomes = {}

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
            entry = entries
        if entry == 1:
            first_entered.set()
            release_first.wait(timeout=30)
            return FakeGatewayConfig("discord")
        second_entered.set()
        release_second.wait(timeout=30)
        return FakeGatewayConfig("telegram")

    def load_platforms(name):
        outcomes[name] = web_server._load_configured_gateway_platforms()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    older = threading.Thread(target=load_platforms, args=("older",))
    newer = threading.Thread(target=load_platforms, args=("newer",))
    older.start()
    try:
        assert first_entered.wait(timeout=10)
        version = 2
        newer.start()
        assert second_entered.wait(timeout=10)

        current_flight = web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[home]
        current_generation = web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION[home]
        assert current_flight["source_version"] == (("config.yaml", 1, 2, 10),)
        assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS[home] == 2

        # Restore the older build's fingerprint so its own publish guard passes:
        # only the generation high-water mark can stop it now.
        version = 1
        release_first.set()
        older.join(timeout=30)
        assert not older.is_alive()

        # The late builder left its successor's flight, mark, and permit alone,
        # and published nothing.
        assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[home] is current_flight
        assert (
            web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION[home]
            == current_generation
        )
        assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS[home] == 1
        assert home not in web_server._CONFIGURED_PLATFORM_CACHE
    finally:
        release_first.set()
        version = 2
        release_second.set()
        newer.join(timeout=30)

    assert not newer.is_alive()
    assert outcomes == {"older": {"discord"}, "newer": {"telegram"}}
    entry = web_server._CONFIGURED_PLATFORM_CACHE[home]
    assert entry["platforms"] == frozenset({"telegram"})
    assert entry["source_version"] == (("config.yaml", 1, 2, 10),)
    assert entry["generation"] == current_generation
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}


def test_configured_gateway_platform_cache_serves_last_known_when_saturated(
    monkeypatch, tmp_path
):
    """With every permit wedged, fall back to the profile's last result."""
    home = str(tmp_path)
    entered = threading.Semaphore(0)
    release = threading.Event()
    version = 1

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        if version == 1:
            return FakeGatewayConfig("discord")
        entered.release()
        release.wait(timeout=30)
        return FakeGatewayConfig("telegram")

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 0.05, raising=False
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}

    max_builders = web_server._CONFIGURED_PLATFORM_CACHE_MAX_BUILDERS
    wedged = [
        threading.Thread(target=web_server._load_configured_gateway_platforms)
        for _ in range(max_builders)
    ]
    try:
        for index, thread in enumerate(wedged):
            # A distinct fingerprint per thread, otherwise same-version
            # single-flight would collapse them onto one builder.
            version = 2 + index
            thread.start()
            assert entered.acquire(timeout=10)
        assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS[home] == max_builders

        # First poll parks on the registered flight and gives up on it...
        with pytest.raises(TimeoutError):
            web_server._load_configured_gateway_platforms()
        # ...after which there is no flight and no permit, so the stale entry is
        # served rather than raising or adding another builder.
        assert web_server._load_configured_gateway_platforms() == {"discord"}
        assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS[home] == max_builders
    finally:
        release.set()
        for thread in wedged:
            thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in wedged)
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    # The profile refreshes normally once the wedged loads drain.
    assert web_server._load_configured_gateway_platforms() == {"telegram"}


def test_configured_gateway_platform_cache_bounds_builders_across_profiles(
    monkeypatch, tmp_path
):
    """Many profiles with wedged loaders must not each claim their own quota.

    Profile names reach this code from the request, so a per-profile ceiling is
    not a bound on Starlette's shared worker pool -- only a process-wide one is.
    Every loader here is held for the whole test, so the number of loader
    entries is a direct, race-free count of how many worker threads were
    committed to discovery.
    """
    attempts = 14
    local = threading.local()
    entries = 0
    entries_lock = threading.Lock()
    release = threading.Event()
    returned = threading.Semaphore(0)
    outcomes = []
    outcomes_lock = threading.Lock()

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
        release.wait(timeout=30)
        return FakeGatewayConfig()

    def poll(home):
        local.home = home
        try:
            result = web_server._load_configured_gateway_platforms()
        except BaseException as exc:  # pragma: no cover - being turned away
            result = exc
        with outcomes_lock:
            outcomes.append(result)
        returned.release()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: local.home)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    threads = [
        threading.Thread(target=poll, args=(tmp_path / f"profile-{index}",))
        for index in range(attempts)
    ]
    try:
        started = time.monotonic()
        for thread in threads:
            thread.start()
        # Drain returns until they stop coming; whatever has not returned is
        # wedged inside a loader and by construction never finishes on its own.
        returns = 0
        while returns < attempts and returned.acquire(timeout=1.0):
            returns += 1
        elapsed = time.monotonic() - started
        with entries_lock:
            builders_spawned = entries
        still_running = [thread for thread in threads if thread.is_alive()]

        # The behavioural bound: distinct profiles cannot each start a builder.
        assert builders_spawned < attempts
        assert len(still_running) < attempts

        cap = web_server._CONFIGURED_PLATFORM_CACHE_MAX_TOTAL_BUILDERS
        assert attempts > cap, "raise `attempts` above the process-wide cap"
        assert builders_spawned == cap
        assert len(still_running) == cap
        # Everyone turned away returned rather than parking on a worker thread.
        assert returns == attempts - cap
        with outcomes_lock:
            assert len(outcomes) == attempts - cap
            assert all(isinstance(outcome, TimeoutError) for outcome in outcomes)
        assert elapsed < 3.0

        # Global bookkeeping ceilings, none of which _MAX_KEYS constrains.
        assert len(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS) == cap
        assert (
            sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values()) == cap
        )
        assert len(web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT) == cap
        assert len(web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION) == cap
        assert len(
            web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION
        ) <= web_server._CONFIGURED_PLATFORM_CACHE_MAX_KEYS + cap
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    # Every permit and bookkeeping entry is reclaimed once the loads return.
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}


def test_configured_gateway_platform_cache_global_saturation_serves_or_fails_fast(
    monkeypatch, tmp_path
):
    """Saturated process-wide: serve the profile's stale entry, else fail fast."""
    primed = tmp_path / "primed"
    unknown = tmp_path / "unknown"
    local = threading.local()
    local.home = primed
    entries = 0
    entries_lock = threading.Lock()
    release = threading.Event()
    entered = threading.Semaphore(0)
    version = 1

    class FakeGatewayConfig:
        def __init__(self, platform):
            self.platform = platform

        def get_connected_platforms(self):
            return [SimpleNamespace(value=self.platform)]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
        if local.home == primed:
            return FakeGatewayConfig("discord")
        entered.release()
        release.wait(timeout=30)
        return FakeGatewayConfig("telegram")

    def wedge(home):
        local.home = home
        try:
            web_server._load_configured_gateway_platforms()
        except BaseException:  # pragma: no cover - not expected
            pass

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: local.home)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    assert web_server._load_configured_gateway_platforms() == {"discord"}
    assert str(primed) in web_server._CONFIGURED_PLATFORM_CACHE

    wedged_profiles = 8
    wedged = [
        threading.Thread(target=wedge, args=(tmp_path / f"other-{index}",))
        for index in range(wedged_profiles)
    ]
    try:
        for thread in wedged:
            thread.start()
            assert entered.acquire(timeout=10)
        assert (
            sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values())
            == wedged_profiles
        )
        with entries_lock:
            entries_while_saturated = entries

        # Move the fingerprint on so the primed profile is now a genuine miss.
        version = 2

        # A profile with a prior result is served stale rather than blocking or
        # adding another unkillable thread.
        local.home = primed
        started = time.monotonic()
        assert web_server._load_configured_gateway_platforms() == {"discord"}
        stale_elapsed = time.monotonic() - started
        assert stale_elapsed < 1.0
        with entries_lock:
            # Serving stale must not have started a discovery.
            assert entries == entries_while_saturated

        # A profile with nothing safe to serve fails fast instead of waiting or
        # spawning.  (Checked second: on an unbounded build it would wedge.)
        local.home = unknown
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="process-wide"):
            web_server._load_configured_gateway_platforms()
        failure_elapsed = time.monotonic() - started
        assert failure_elapsed < 1.0
        with entries_lock:
            assert entries == entries_while_saturated

        cap = web_server._CONFIGURED_PLATFORM_CACHE_MAX_TOTAL_BUILDERS
        assert wedged_profiles == cap, "wedged profiles must match the global cap"
        assert sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values()) == cap
        assert str(unknown) not in web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT
    finally:
        release.set()
        for thread in wedged:
            thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in wedged)
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}

    # Normal service resumes for both profiles once the permits are released.
    local.home = primed
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    local.home = unknown
    assert web_server._load_configured_gateway_platforms() == {"telegram"}


def test_configured_gateway_platform_cache_no_parking_on_matching_flight_at_cap(
    monkeypatch, tmp_path
):
    """At the global cap, a *matching* hung flight must not absorb new waiters.

    Parking is only free while the process has spare discovery capacity: a
    parked caller still holds a worker-pool thread for the whole wait budget.
    The wait timeout is left at its real value here, so a caller that parks is
    unmistakably stuck rather than merely slow -- the pollers must come back
    without ever reaching the wait.
    """
    cached = tmp_path / "cached"
    uncached = tmp_path / "uncached"
    pollers = 12
    local = threading.local()
    local.home = cached
    entries = 0
    entries_lock = threading.Lock()
    priming = True
    entered = threading.Semaphore(0)
    release = threading.Event()
    returned = threading.Semaphore(0)
    outcomes = []
    outcomes_lock = threading.Lock()
    version = 1

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
        if priming:
            return FakeGatewayConfig()
        entered.release()
        release.wait(timeout=30)
        return FakeGatewayConfig()

    def wedge(home):
        local.home = home
        try:
            web_server._load_configured_gateway_platforms()
        except BaseException:  # pragma: no cover - not expected
            pass

    def poll(home):
        local.home = home
        try:
            result = web_server._load_configured_gateway_platforms()
        except BaseException as exc:
            result = exc
        with outcomes_lock:
            outcomes.append((home, result))
        returned.release()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: local.home)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    # Give `cached` a result, then move the fingerprint on so its entry is stale
    # and the hung flight registered next genuinely matches later pollers.
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    priming = False
    version = 2

    cap = web_server._CONFIGURED_PLATFORM_CACHE_MAX_TOTAL_BUILDERS
    homes = [cached, uncached] + [tmp_path / f"filler-{i}" for i in range(cap - 2)]
    wedged = [threading.Thread(target=wedge, args=(home,)) for home in homes]
    threads = []
    try:
        for thread in wedged:
            thread.start()
            assert entered.acquire(timeout=10)

        # Both target profiles now have a registered, *matching*, hung flight,
        # and every discovery permit in the process is consumed.
        assert sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values()) == cap
        cached_flight = web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(cached)]
        uncached_flight = web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(uncached)]
        assert cached_flight["source_version"] == (("config.yaml", 1, 2, 10),)
        assert uncached_flight["source_version"] == (("config.yaml", 1, 2, 10),)
        assert str(cached) in web_server._CONFIGURED_PLATFORM_CACHE
        assert str(uncached) not in web_server._CONFIGURED_PLATFORM_CACHE
        with entries_lock:
            entries_at_cap = entries
        in_flight_at_cap = dict(web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT)
        builders_at_cap = dict(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS)
        generations_at_cap = dict(
            web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION
        )

        started = time.monotonic()
        for index in range(pollers):
            home = cached if index % 2 == 0 else uncached
            thread = threading.Thread(target=poll, args=(home,))
            thread.start()
            threads.append(thread)
        returns = 0
        while returns < pollers and returned.acquire(timeout=2.0):
            returns += 1
        elapsed = time.monotonic() - started

        # None of them parked behind the matching hung flights.
        assert returns == pollers
        assert all(not thread.is_alive() for thread in threads)
        assert elapsed < 2.0

        with outcomes_lock:
            results = list(outcomes)
        assert len(results) == pollers
        # Stale-cache callers are served; uncached callers fail fast.
        assert all(
            result == {"discord"}
            for home, result in results
            if home == cached
        )
        assert all(
            isinstance(result, TimeoutError) and "process-wide" in str(result)
            for home, result in results
            if home == uncached
        )

        # And none of them spawned a builder or disturbed the bookkeeping.
        with entries_lock:
            assert entries == entries_at_cap
        assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == builders_at_cap
        assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == in_flight_at_cap
        assert (
            web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION
            == generations_at_cap
        )
        # The hung flights were neither abandoned nor replaced.
        assert (
            web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(cached)]
            is cached_flight
        )
        assert (
            web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(uncached)]
            is uncached_flight
        )
        assert not cached_flight["abandoned"]
        assert not uncached_flight["abandoned"]
    finally:
        release.set()
        for thread in wedged + threads:
            thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in wedged)
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}

    # Normal collapse-onto-a-matching-flight behaviour returns once the process
    # is back under its ceiling.
    local.home = uncached
    assert web_server._load_configured_gateway_platforms() == {"discord"}


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_configured_gateway_platform_cache_bounds_total_worker_occupancy(
    monkeypatch, tmp_path
):
    """One hung flight far below the builder cap must not absorb every poller.

    A parked caller holds a worker-pool thread exactly like a running load does,
    so with a single builder wedged there are still only
    ``_MAX_OCCUPANCY - 1`` slots left before new callers must be turned away.
    The wait timeout is left at its real value, so anything that parks is
    unmistakably stuck rather than merely slow.
    """
    cached = tmp_path / "cached"
    unknown = tmp_path / "unknown"
    pollers = 24
    local = threading.local()
    local.home = cached
    entries = 0
    entries_lock = threading.Lock()
    priming = True
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Semaphore(0)
    outcomes = []
    outcomes_lock = threading.Lock()
    version = 1

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal entries
        with entries_lock:
            entries += 1
        if priming:
            return FakeGatewayConfig()
        entered.set()
        release.wait(timeout=30)
        return FakeGatewayConfig()

    def wedge():
        local.home = cached
        web_server._load_configured_gateway_platforms()

    def poll():
        local.home = cached
        try:
            result = web_server._load_configured_gateway_platforms()
        except BaseException as exc:
            result = exc
        with outcomes_lock:
            outcomes.append(result)
        returned.release()

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: local.home)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    # Give the profile a result, then move the fingerprint on so the hung flight
    # registered next genuinely matches the pollers that follow it.
    assert web_server._load_configured_gateway_platforms() == {"discord"}
    priming = False
    version = 2

    builder = threading.Thread(target=wedge)
    threads = []
    try:
        builder.start()
        assert entered.wait(timeout=10)

        # A single builder -- far below the builder ceiling, which stays free.
        assert sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values()) == 1
        assert 1 < web_server._CONFIGURED_PLATFORM_CACHE_MAX_TOTAL_BUILDERS
        hung_flight = web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(cached)]
        with entries_lock:
            entries_at_cap = entries

        started = time.monotonic()
        for _ in range(pollers):
            thread = threading.Thread(target=poll)
            thread.start()
            threads.append(thread)

        # Drain returns until they stop coming; whatever has not come back is
        # parked behind the one hung flight.
        returns = 0
        while returns < pollers and returned.acquire(timeout=2.0):
            returns += 1
        elapsed = time.monotonic() - started

        # The behavioural bound: a single hung flight must not swallow every
        # poller just because the builder ceiling is nowhere near reached.
        assert returns > 0, "every poller parked behind one hung flight"
        assert elapsed < 10.0

        occupancy_cap = web_server._CONFIGURED_PLATFORM_CACHE_MAX_OCCUPANCY
        expected_waiters = occupancy_cap - 1
        assert pollers > expected_waiters, "raise `pollers` above the occupancy cap"
        assert returns == pollers - expected_waiters
        assert web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == expected_waiters
        assert (
            sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values())
            + web_server._CONFIGURED_PLATFORM_CACHE_WAITERS
            == occupancy_cap
        )
        # Overflow callers were served the profile's stale entry, not blocked.
        with outcomes_lock:
            assert len(outcomes) == returns
            assert all(outcome == {"discord"} for outcome in outcomes)

        # A profile with nothing safe to serve fails fast on the same ceiling,
        # even though six builder permits are still free.
        local.home = unknown
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="occupancy"):
            web_server._load_configured_gateway_platforms()
        assert time.monotonic() - started < 2.0
        local.home = cached

        # None of that started a discovery or disturbed the hung flight.
        with entries_lock:
            assert entries == entries_at_cap
        assert sum(web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.values()) == 1
        assert (
            web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(cached)] is hung_flight
        )
        assert not hung_flight["abandoned"]
        assert str(unknown) not in web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT
    finally:
        release.set()
        builder.join(timeout=30)
        for thread in threads:
            thread.join(timeout=30)

    assert not builder.is_alive()
    assert all(not thread.is_alive() for thread in threads)
    # Everything parked drains and every slot is reclaimed.
    with outcomes_lock:
        assert len(outcomes) == pollers
        assert all(outcome == {"discord"} for outcome in outcomes)
    assert web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == 0
    assert web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS == {}
    assert web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT == {}


def test_configured_gateway_platform_cache_waiter_slot_released_on_every_exit(
    monkeypatch, tmp_path
):
    """A waiter slot must never leak, whatever ends the wait."""
    calls = 0
    calls_lock = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    outcome = {}
    mode = "success"
    version = 1

    class FakeGatewayConfig:
        @staticmethod
        def get_connected_platforms():
            return [SimpleNamespace(value="discord")]

    def source_version(_home):
        return (("config.yaml", 1, version, 10),)

    def load_gateway_config():
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        release.wait(timeout=30)
        if mode == "error":
            raise RuntimeError("config failure")
        return FakeGatewayConfig()

    def wait_for_platforms(name):
        try:
            outcome[name] = web_server._load_configured_gateway_platforms()
        except BaseException as exc:
            outcome[name] = exc

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(web_server, "_configured_platform_source_version", source_version)
    monkeypatch.setattr("gateway.config.load_gateway_config", load_gateway_config)

    def run_phase(name, finish):
        """Park a waiter behind a wedged builder, then end the wait via `finish`."""
        entered.clear()
        release.clear()
        outcome.clear()
        builder = threading.Thread(target=wait_for_platforms, args=("builder",))
        builder.start()
        assert entered.wait(timeout=10)
        waiter = threading.Thread(target=wait_for_platforms, args=("waiter",))
        waiter.start()
        assert _wait_for(
            lambda: web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == 1
        ), f"{name}: waiter never registered"
        try:
            finish()
        finally:
            release.set()
            builder.join(timeout=30)
            waiter.join(timeout=30)
        assert not builder.is_alive() and not waiter.is_alive()
        assert web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == 0, f"{name}: leaked"
        web_server._CONFIGURED_PLATFORM_CACHE.clear()
        web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT.clear()
        web_server._CONFIGURED_PLATFORM_CACHE_BUILDERS.clear()
        web_server._CONFIGURED_PLATFORM_CACHE_LATEST_GENERATION.clear()

    # 1. Success: the builder publishes and the waiter picks the result up.
    run_phase("success", lambda: release.set())
    assert outcome["waiter"] == {"discord"}

    # 2. Builder error: the waiter is woken with a failure.
    mode = "error"
    run_phase("error", lambda: release.set())
    assert isinstance(outcome["waiter"], RuntimeError)
    mode = "success"

    # 3. Wait timeout: the waiter gives up on a builder that has not returned.
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 0.05, raising=False
    )
    run_phase(
        "timeout",
        lambda: _wait_for(lambda: isinstance(outcome.get("waiter"), TimeoutError)),
    )
    assert isinstance(outcome["waiter"], TimeoutError)
    monkeypatch.setattr(
        web_server, "_CONFIGURED_PLATFORM_CACHE_WAIT_TIMEOUT", 30.0, raising=False
    )

    # 4. Abandonment: a superseding caller wakes the waiter, which retries.
    def supersede():
        nonlocal version
        flight = web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(tmp_path)]
        version = 2
        with web_server._CONFIGURED_PLATFORM_CACHE_LOCK:
            web_server._abandon_configured_platform_flight(str(tmp_path), flight)
        # The waiter retries, finds no flight and a free permit, and rebuilds.
        assert _wait_for(lambda: web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == 0)
        assert _wait_for(lambda: calls >= 3)

    run_phase("abandonment", supersede)
    assert outcome["waiter"] == {"discord"}
    version = 1

    # 5. BaseException while parked: swap in an event whose wait() explodes.
    class ExplodingEvent:
        def __init__(self):
            self._set = False

        def wait(self, timeout=None):
            raise KeyboardInterrupt("interrupted while parked")

        def is_set(self):
            return self._set

        def set(self):
            self._set = True

    # Driven directly rather than through `run_phase`: swapping the flight's
    # event would orphan any waiter already parked on the original one.
    entered.clear()
    release.clear()
    outcome.clear()
    builder = threading.Thread(target=wait_for_platforms, args=("builder",))
    builder.start()
    try:
        assert entered.wait(timeout=10)
        web_server._CONFIGURED_PLATFORM_CACHE_IN_FLIGHT[str(tmp_path)][
            "event"
        ] = ExplodingEvent()
        interrupted = threading.Thread(target=wait_for_platforms, args=("interrupted",))
        interrupted.start()
        interrupted.join(timeout=10)
        assert not interrupted.is_alive()
        assert isinstance(outcome["interrupted"], KeyboardInterrupt)
        assert web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == 0, (
            "base-exception: leaked"
        )
    finally:
        release.set()
        builder.join(timeout=30)

    assert not builder.is_alive()
    assert web_server._CONFIGURED_PLATFORM_CACHE_WAITERS == 0
