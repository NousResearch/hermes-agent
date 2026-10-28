"""A check_fn that never answers must not hang the turn waiting for it (#131145).

A check_fn probes external state (a Docker daemon, an MCP server, a browser
control endpoint). The probe used to run inline on the caller's thread, so a
probe that never returned held its caller forever: the boot warm-up never
completed (``Turn-machinery warm-up still running after 20s``), and the first
inbound turn then stopped dead before its provider request - nothing logged,
``active_agents`` pinned at 1, the messaging platform left in its "typing"
state until a gateway restart.

These tests cover the bound itself, the build that has to keep going when a
probe wedges, and the two properties the bound must not break: a probe that
raises is still reported as raised, and the profile secret scope and the
probe's own contextvar publication still reach the caller.
"""

import json
import logging
import threading
import time

import pytest

from tools import registry as registry_module
from tools.registry import ToolRegistry, _check_fn_cached, invalidate_check_fn_cache

# The wedged probes park on this until the test ends, so a parked probe thread
# never outlives the test that created it.
_RELEASE = threading.Event()


def _schema(name="probe_tool"):
    return {
        "name": name,
        "description": f"A {name}",
        "parameters": {"type": "object", "properties": {}},
    }


def _handler(args, **kwargs):
    return json.dumps({"ok": True})


def _wedged_probe():
    """A check_fn that never answers until the test releases it."""

    def probe():
        _RELEASE.wait(timeout=30)
        return True

    return probe


def _clear_parked_probes():
    probes = getattr(registry_module, "_check_fn_probes", None)
    lock = getattr(registry_module, "_check_fn_probe_lock", None)
    wedged = getattr(registry_module, "_check_fn_wedged_probes", None)
    if probes is None or lock is None:
        return
    with lock:
        probes.clear()
        if wedged is not None:
            wedged.clear()


def _probe_threads():
    return [t for t in threading.enumerate() if t.name == "check-fn-probe"]


def _call_within(fn, seconds):
    """Call *fn* on a worker thread; report whether it came back in time.

    A regression here is a hang, so the call is made from a thread the test
    can abandon: the test fails on the bound instead of blocking the suite.
    """
    outcome = {}

    def _run():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # pragma: no cover - reported, not raised
            outcome["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(seconds)
    return worker.is_alive(), outcome


@pytest.fixture(autouse=True)
def _bounded_probes(monkeypatch):
    """A short probe budget and a clean cache around every test.

    The budget is set with ``raising=False`` so this file also runs against a
    tree without the bound, where the wedged-probe cases fail on the hang
    rather than on a missing attribute.
    """
    monkeypatch.setattr(
        registry_module, "_CHECK_FN_PROBE_TIMEOUT_SECONDS", 0.25, raising=False
    )
    _RELEASE.clear()
    invalidate_check_fn_cache()
    _clear_parked_probes()
    yield
    _RELEASE.set()
    invalidate_check_fn_cache()
    _clear_parked_probes()


def test_a_wedged_check_fn_answers_unavailable_instead_of_hanging(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.registry"):
        stuck, outcome = _call_within(lambda: _check_fn_cached(_wedged_probe()), 5.0)

    assert not stuck, "a wedged check_fn must not hang its caller"
    assert outcome.get("value") is False
    assert any("did not answer" in record.getMessage() for record in caplog.records)


def test_a_wedged_check_fn_does_not_hold_the_tool_schema_build_open():
    """The turn's hot path: the build that names the agent's tools must finish."""
    reg = ToolRegistry()
    reg.register(
        name="wedged",
        toolset="probe",
        schema=_schema("wedged"),
        handler=_handler,
        check_fn=_wedged_probe(),
    )
    reg.register(
        name="fine",
        toolset="probe",
        schema=_schema("fine"),
        handler=_handler,
        check_fn=lambda: True,
    )

    stuck, outcome = _call_within(
        lambda: reg.get_definitions({"wedged", "fine"}, quiet=True), 5.0
    )

    assert not stuck, "one wedged probe must not hold the build open"
    assert [d["function"]["name"] for d in outcome["value"]] == ["fine"]


def test_a_wedged_probe_keeps_the_rest_of_the_tool_surface():
    """One wedged probe costs its own budget; every other tool is still offered."""
    reg = ToolRegistry()
    reg.register(
        name="wedged",
        toolset="probe",
        schema=_schema("wedged"),
        handler=_handler,
        check_fn=_wedged_probe(),
    )
    for index in range(3):
        reg.register(
            name=f"fine{index}",
            toolset="probe",
            schema=_schema(f"fine{index}"),
            handler=_handler,
            check_fn=lambda: True,
        )

    stuck, outcome = _call_within(
        lambda: reg.get_definitions(
            {"wedged", "fine0", "fine1", "fine2"}, quiet=True
        ),
        5.0,
    )

    assert not stuck
    assert [d["function"]["name"] for d in outcome["value"]] == [
        "fine0",
        "fine1",
        "fine2",
    ]


def test_a_build_with_several_wedged_probes_stops_probing(monkeypatch):
    """An environment where every probe wedges must not cost the whole build."""
    monkeypatch.setattr(
        registry_module, "_CHECK_FN_MAX_UNANSWERED_PROBES", 2, raising=False
    )
    reg = ToolRegistry()
    for index in range(4):
        reg.register(
            name=f"wedged{index}",
            toolset="probe",
            schema=_schema(f"wedged{index}"),
            handler=_handler,
            check_fn=_wedged_probe(),
        )

    started = time.monotonic()
    stuck, outcome = _call_within(
        lambda: reg.get_definitions(
            {"wedged0", "wedged1", "wedged2", "wedged3"}, quiet=True
        ),
        5.0,
    )
    elapsed = time.monotonic() - started

    assert not stuck
    assert outcome["value"] == []
    assert elapsed < 1.5, "the build must stop probing once the probes keep failing"


def test_a_wedged_probe_is_not_restarted_for_every_sweep(monkeypatch):
    """A parked probe is shared, so a wedged probe leaks no thread per TTL window."""
    monkeypatch.setattr(
        registry_module, "_CHECK_FN_PROBE_TIMEOUT_SECONDS", 2.0, raising=False
    )
    probe = _wedged_probe()
    before = len(_probe_threads())

    stuck, outcome = _call_within(lambda: _check_fn_cached(probe), 5.0)
    assert not stuck
    assert outcome.get("value") is False
    invalidate_check_fn_cache()  # the TTL verdict expired; the probe itself is still parked

    started = time.monotonic()
    stuck, outcome = _call_within(lambda: _check_fn_cached(probe), 5.0)
    elapsed = time.monotonic() - started

    assert not stuck
    assert outcome.get("value") is False
    assert elapsed < 1.0, "a known-wedged probe must not be waited on again"
    assert len(_probe_threads()) - before <= 1


def test_a_probe_the_sweep_never_reached_is_not_cached(monkeypatch):
    """A tool the exhausted sweep skipped was never asked, so its False verdict must not be
    cached for the TTL - the next build has to probe it again (#142062)."""
    monkeypatch.setattr(registry_module, "_CHECK_FN_MAX_UNANSWERED_PROBES", 1, raising=False)
    monkeypatch.setattr(registry_module, "_CHECK_FN_WEDGED_TTL_SECONDS", 0.0, raising=False)
    reg = ToolRegistry()
    for name in ("answered", "skipped"):
        reg.register(name=name, toolset="probe", schema=_schema(name), handler=_handler,
                     check_fn=_wedged_probe())

    assert reg.get_definitions({"answered", "skipped"}, quiet=True) == []  # one poisons the sweep

    _RELEASE.set()  # both would answer now
    built = reg.get_definitions({"answered", "skipped"}, quiet=True)
    assert [entry["function"]["name"] for entry in built] == ["skipped"]


def test_a_wedged_probe_is_not_restarted_for_another_session(monkeypatch):
    """The request identity scopes the verdict, not the failure: a probe that already went silent
    will not answer for another session either, so it is not waited on again (#142062)."""
    # A full second, so paying the budget again is unmistakable against the fast path.
    monkeypatch.setattr(registry_module, "_CHECK_FN_PROBE_TIMEOUT_SECONDS", 1.0, raising=False)
    probe = _wedged_probe()
    # Every invocation differs (the identity is read once for the cache scope and once for the
    # key), so a fresh key would mean a second probe.
    counter = iter(range(1, 1000))
    monkeypatch.setattr(registry_module, "_probe_identity", lambda: f"s{next(counter)}", raising=False)

    assert _check_fn_cached(probe) is False
    invalidate_check_fn_cache()  # the verdict cache is scope-keyed; this is about the probe itself
    threads_before = len(_probe_threads())
    assert _check_fn_cached(probe) is False
    # No second worker: the identity is not part of the wedged key, so the failure is reused.
    assert len(_probe_threads()) == threads_before, "another session started a second probe"


def test_the_wedged_marker_expires(monkeypatch):
    """The fast path is not permanent: once the marker is stale, the probe is asked again."""
    monkeypatch.setattr(registry_module, "_CHECK_FN_WEDGED_TTL_SECONDS", 0.0, raising=False)
    probe = _wedged_probe()
    counter = iter(range(1, 1000))
    monkeypatch.setattr(registry_module, "_probe_identity", lambda: f"s{next(counter)}", raising=False)

    assert _check_fn_cached(probe) is False  # times out, marks the (fn, profile)
    _RELEASE.set()
    invalidate_check_fn_cache()  # the TTL verdict is not what this test is about
    assert _check_fn_cached(probe) is True  # marker window lapsed: probed, and it answers


def test_a_raising_check_fn_still_reports_raised(caplog):
    def boom():
        raise RuntimeError("probe exploded")

    with caplog.at_level(logging.WARNING, logger="tools.registry"):
        assert _check_fn_cached(boom) is False

    assert any("raised" in record.getMessage() for record in caplog.records)


def test_a_probe_that_answers_keeps_its_verdict():
    assert _check_fn_cached(lambda: True) is True
    assert _check_fn_cached(lambda: False) is False


def test_the_profile_secret_scope_reaches_the_probe_thread():
    """The probe runs on a worker thread; the profile secret scope is a
    contextvar, so it has to travel with the probe or every scoped probe would
    fail closed and strip its toolset (#110635)."""
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope

    seen = {}

    def scoped_probe():
        seen["value"] = get_secret("PROBE_ONLY_KEY")
        return True

    token = set_secret_scope({"PROBE_ONLY_KEY": "s3cret"})
    try:
        assert _check_fn_cached(scoped_probe) is True
    finally:
        reset_secret_scope(token)

    assert seen["value"] == "s3cret"


def test_the_probes_contextvar_publication_reaches_the_caller():
    """A check_fn can publish into a contextvar for the schema build that
    follows it (the memory tool's availability snapshot: one definition pass
    must use ONE config decision). The probe runs on a worker thread, so the
    publication has to travel back to the caller."""
    from contextvars import ContextVar

    published: ContextVar = ContextVar("probe_published", default=None)

    def publishing_probe():
        published.set(("snapshot",))
        return True

    assert _check_fn_cached(publishing_probe) is True
    assert published.get() == ("snapshot",)
