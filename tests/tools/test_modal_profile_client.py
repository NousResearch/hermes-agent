"""Direct Modal sandboxes run on the serving profile's own Modal account.

Without an explicit ``client=``, ``modal.App.lookup`` and ``modal.Sandbox.create`` use the SDK's
process-wide ``Client.from_env()``: one client built from ``os.environ`` (the launch profile's
``.env``) on first use and kept for the process. On a multiplexed gateway every routed profile's
sandbox was then created, billed and run on the launch profile's Modal account.
"""

import asyncio
import json
import sys
import threading
import types
from concurrent.futures import Future

import pytest

from tests.waiting import wait_until


async def _noop(*_args, **_kwargs):
    return None


@pytest.fixture
def modal_sdk(monkeypatch):
    """A ``modal`` stand-in recording which token pair each client was built from and which client
    every App lookup and Sandbox create ran on (``None``: the SDK's own process-wide client)."""
    from tools.environments import modal as modal_env

    calls: dict[str, list] = {"clients": [], "lookups": [], "creates": []}
    from_credentials = _FromCredentials(calls["clients"])

    async def lookup(_name, *, create_if_missing=False, client=None):
        calls["lookups"].append(getattr(client, "token_id", None))
        return types.SimpleNamespace(name=_name)

    async def create(*_args, image=None, app=None, timeout=None, client=None, **_kwargs):
        calls["creates"].append(getattr(client, "token_id", None))
        return types.SimpleNamespace(terminate=types.SimpleNamespace(aio=_noop))

    aio = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(
        Client=aio(from_credentials=from_credentials),
        App=aio(lookup=aio(aio=lookup)),
        Sandbox=aio(create=aio(aio=create)),
        Image=aio(from_registry=lambda *_a, **_k: "image", from_id=lambda image_id: image_id)))
    monkeypatch.setattr(modal_env, "ensure_lazy_dep", lambda extra: None)
    monkeypatch.setattr(modal_env, "_profile_clients", {})
    monkeypatch.setattr(modal_env, "FileSyncManager", lambda **_kw: types.SimpleNamespace(
        sync=lambda force=False: None, sync_back=lambda: None))
    monkeypatch.setattr(modal_env.ModalEnvironment, "init_session", lambda self: None)
    return modal_env, calls


class _FromCredentials:
    """``modal.Client.from_credentials``: the sync call and its ``.aio`` form both build a client.
    A token's next build waits for its ``gates`` future (held open until the test resolves it) and
    then raises the next of its ``failures``."""

    def __init__(self, built: list):
        self.built = built
        self.gates: dict[str, Future] = {}
        self.failures: dict[str, list[Exception]] = {}

    def _finish(self, token_id, token_secret):
        if self.failures.get(token_id):
            raise self.failures[token_id].pop(0)
        return types.SimpleNamespace(token_id=token_id, token_secret=token_secret)

    def __call__(self, token_id, token_secret):
        self.built.append(token_id)
        if (gate := self.gates.pop(token_id, None)) is not None:
            gate.result(timeout=30)
        return self._finish(token_id, token_secret)

    async def aio(self, token_id, token_secret):
        self.built.append(token_id)
        if (gate := self.gates.pop(token_id, None)) is not None:
            await asyncio.wrap_future(gate)
        return self._finish(token_id, token_secret)


def _fake_from_credentials() -> _FromCredentials:
    """The ``modal_sdk`` fixture's installed ``modal.Client.from_credentials`` stand-in."""
    return sys.modules["modal"].Client.from_credentials


def _sandbox(modal_env):
    env = modal_env.ModalEnvironment(image="python:3.11", persistent_filesystem=False)
    env.cleanup()


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Launch profile A, its pair exported into the process env as a gateway start does, and a
    routed profile B; multiplexing on."""
    from agent import secret_scope

    root = tmp_path / ".hermes"
    homes = {"A": root, "B": root / "profiles" / "b"}
    for label, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-A")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-A")
    was_active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)

    def home(label, dotenv=None):
        token = label.split("-")[0]
        (homes[token] / ".env").write_text(
            f"MODAL_TOKEN_ID=ak-{label}\nMODAL_TOKEN_SECRET=as-{label}\n" if dotenv is None else dotenv)
        return homes[token]

    try:
        yield home
    finally:
        secret_scope.set_multiplex_active(was_active)


def test_routed_sandboxes_run_on_their_own_profiles_account(modal_sdk, profiles):
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    for label in ("B", "A", "B"):
        with bind_profile_scope(profiles(label), hydrate_secrets=False):
            _sandbox(modal_env)

    assert calls["lookups"] == ["ak-B", "ak-A", "ak-B"]
    assert calls["creates"] == ["ak-B", "ak-A", "ak-B"]
    # One client per profile, reused by its later sandboxes; a rotated pair replaces it in place.
    assert calls["clients"] == ["ak-B", "ak-A"]
    with bind_profile_scope(profiles("B-rotated"), hydrate_secrets=False):
        _sandbox(modal_env)
    assert calls["creates"][-1] == "ak-B-rotated"
    assert calls["clients"] == ["ak-B", "ak-A", "ak-B-rotated"]
    homes = []
    for label in ("B", "A"):
        with bind_profile_scope(profiles(label), hydrate_secrets=False):
            homes.append(modal_env.hermes_home_key())
    assert sorted(modal_env._profile_clients) == sorted(homes)  # keyed by the serving profile's home


@pytest.mark.parametrize("dotenv", ["OTHER=1\n", "MODAL_TOKEN_ID=ak-B\n", "MODAL_TOKEN_ID=ak-B\nMODAL_TOKEN_SECRET=\n"],
                         ids=["absent", "id-only", "empty-secret"])
def test_routed_profile_without_its_own_pair_never_borrows_the_launch_account(
        modal_sdk, profiles, tmp_path, monkeypatch, dotenv):
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # Windows reads USERPROFILE, not HOME
    (tmp_path / ".modal.toml").write_text('[default]\ntoken_id = "ak-cli"\ntoken_secret = "as-cli"\n')
    with bind_profile_scope(profiles("B", dotenv), hydrate_secrets=False), \
            pytest.raises(ValueError, match="MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"):
        modal_env.ModalEnvironment(image="python:3.11", persistent_filesystem=False)
    assert calls == {"clients": [], "lookups": [], "creates": []}


def test_single_profile_keeps_the_sdks_own_credentials(modal_sdk, tmp_path, monkeypatch):
    """Outside multiplexing the SDK keeps resolving the process env or the ``modal`` CLI login."""
    modal_env, calls = modal_sdk
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-shell")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-shell")

    _sandbox(modal_env)
    assert calls == {"clients": [], "lookups": [None], "creates": [None]}


@pytest.mark.parametrize(("dotenv", "usable"), [
    (None, True), ("OTHER=1\n", False), ("MODAL_TOKEN_ID=ak-B\n", False)],
    ids=["full-pair", "absent", "id-only"])
def test_backend_selection_agrees_with_the_constructor_for_a_routed_profile(
        modal_sdk, profiles, tmp_path, monkeypatch, dotenv, usable):
    """A host ``~/.modal.toml`` login never makes direct Modal look available for a routed profile
    the constructor would then refuse: selection, status and construction share one answer."""
    from agent.runtime_scope import bind_profile_scope
    from tools.terminal_tool_backends import _get_modal_backend_state
    from tools.tool_backend_helpers import has_direct_modal_credentials

    modal_env, calls = modal_sdk
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # Windows reads USERPROFILE, not HOME
    (tmp_path / ".modal.toml").write_text('[default]\ntoken_id = "ak-cli"\ntoken_secret = "as-cli"\n')
    with bind_profile_scope(profiles("B", dotenv), hydrate_secrets=False):
        assert has_direct_modal_credentials() is usable
        assert (_get_modal_backend_state("direct")["selected_backend"] == "direct") is usable
        if usable:
            _sandbox(modal_env)
        else:
            with pytest.raises(ValueError, match="MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"):
                modal_env.ModalEnvironment(image="python:3.11", persistent_filesystem=False)
    assert calls["creates"] == (["ak-B"] if usable else [])


def test_routed_direct_mode_refusal_names_the_profile_tokens(profiles, tmp_path, monkeypatch):
    from agent.runtime_scope import bind_profile_scope
    from tools.terminal_tool_backends import _get_modal_backend_state, _modal_unavailable_reason

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # Windows reads USERPROFILE, not HOME
    (tmp_path / ".modal.toml").write_text('[default]\ntoken_id = "ak-cli"\ntoken_secret = "as-cli"\n')
    with bind_profile_scope(profiles("B", "OTHER=1\n"), hydrate_secrets=False):
        log_reason, error = _modal_unavailable_reason(_get_modal_backend_state("direct"))
    for reason in (log_reason, error):
        assert "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET" in reason
        assert "single-profile" in reason


def test_single_profile_direct_mode_refusal_is_unchanged(tmp_path, monkeypatch):
    from tools.terminal_tool_backends import _get_modal_backend_state, _modal_unavailable_reason

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # Windows reads USERPROFILE, not HOME
    _, error = _modal_unavailable_reason(_get_modal_backend_state("direct"))
    assert error == "Modal backend is configured for direct mode, but no direct Modal credentials/config were found."


def test_single_profile_host_login_still_selects_direct(tmp_path, monkeypatch):
    from tools.tool_backend_helpers import has_direct_modal_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # Windows reads USERPROFILE, not HOME
    (tmp_path / ".modal.toml").write_text('[default]\ntoken_id = "ak-cli"\ntoken_secret = "as-cli"\n')
    assert has_direct_modal_credentials() is True


def test_unbound_multiplexed_modal_preflight_does_not_offer_launch_tokens(profiles):
    from tools.terminal_tool_backends import _get_modal_backend_state

    assert _get_modal_backend_state("direct")["selected_backend"] is None


class _JoinRecorder(dict):
    """``_client_flights`` stand-in: notes when a sandbox found a client build already in progress."""

    def __init__(self):
        super().__init__()
        self.joined = threading.Event()

    def get(self, key, default=None):
        flight = super().get(key, default)
        if flight is not None:
            self.joined.set()
        return flight


def _start(target, *args):
    errors: list[BaseException] = []

    def run():
        try:
            target(*args)
        except BaseException as exc:  # surfaced through ``errors``
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def _routed_sandbox(modal_env, home):
    from agent.runtime_scope import bind_profile_scope

    with bind_profile_scope(home, hydrate_secrets=False):
        _sandbox(modal_env)


def test_concurrent_first_sandboxes_of_one_profile_build_one_client(modal_sdk, profiles, monkeypatch):
    """Two first sandboxes of the same profile at once, each on its own environment's loop: the SDK
    keeps every opened client until process exit, so only one may be built. Another profile is not
    held up meanwhile."""
    modal_env, calls = modal_sdk
    flights = _JoinRecorder()
    monkeypatch.setattr(modal_env, "_client_flights", flights)
    home_b, home_a = profiles("B"), profiles("A")
    gate = _fake_from_credentials().gates["ak-B"] = Future()

    first, first_errors = _start(_routed_sandbox, modal_env, home_b)
    second, second_errors = _start(_routed_sandbox, modal_env, home_b)
    assert flights.joined.wait(10), "the second sandbox did not join the build in progress"
    other, other_errors = _start(_routed_sandbox, modal_env, home_a)
    other.join(10)
    assert not other.is_alive(), "another profile's sandbox waited on B's client build"
    gate.set_result(None)
    for thread in (first, second):
        thread.join(10)
        assert not thread.is_alive()
    assert first_errors == second_errors == other_errors == []
    assert sorted(calls["clients"]) == ["ak-A", "ak-B"]
    assert sorted(calls["creates"]) == ["ak-A", "ak-B", "ak-B"]
    assert flights == {}


def test_failed_client_build_is_not_cached(modal_sdk, profiles):
    modal_env, calls = modal_sdk
    _fake_from_credentials().failures["ak-B"] = [RuntimeError("boom")]
    home = profiles("B")
    with pytest.raises(RuntimeError, match="boom"):
        _routed_sandbox(modal_env, home)
    _routed_sandbox(modal_env, home)
    assert calls["clients"] == ["ak-B", "ak-B"]
    assert calls["creates"] == ["ak-B"]


def test_client_build_runs_inside_the_one_startup_window(modal_sdk, profiles, monkeypatch):
    """The client build counts against the startup's single 300 s window (the window is shortened
    here; nothing waits 300 s): a build that never finishes times the startup out instead of
    hanging it, and the next sandbox builds afresh."""
    modal_env, calls = modal_sdk
    requested: list[int] = []
    run_coroutine = modal_env._AsyncWorker.run_coroutine

    def shortened(self, coro, timeout=600):
        requested.append(timeout)  # the first window is cut to 2 s; the retry may take up to 15 s
        return run_coroutine(self, coro, timeout=min(timeout, 2 if len(requested) == 1 else 15))

    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", shortened)
    home = profiles("B")
    gate = _fake_from_credentials().gates["ak-B"] = Future()
    starting, errors = _start(_routed_sandbox, modal_env, home)
    starting.join(10)
    hung = starting.is_alive()
    gate.set_result(None)
    starting.join(10)
    assert not hung, "the client build ran outside the startup deadline"
    assert requested == [300]
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert modal_env._client_flights == {}
    _routed_sandbox(modal_env, home)
    assert calls["clients"] == ["ak-B", "ak-B"]
    assert calls["creates"] == ["ak-B"]


def test_timed_out_startup_releases_the_sandboxes_waiting_on_its_build(modal_sdk, profiles, monkeypatch):
    """The startup running a build times out and stops its loop: a sandbox waiting on that build
    fails at once instead of waiting on a stopped loop, and a later sandbox builds afresh."""
    modal_env, calls = modal_sdk
    flights = _JoinRecorder()
    monkeypatch.setattr(modal_env, "_client_flights", flights)
    run_coroutine = modal_env._AsyncWorker.run_coroutine
    creator_waits: list[bool] = [True]

    def expiring(self, coro, timeout=600):
        if not creator_waits:
            return run_coroutine(self, coro, timeout=min(timeout, 15))
        creator_waits.clear()  # the first wait is the creating sandbox's startup
        # Its own wait expires while the build is held open, as 300 s would.
        startup = asyncio.run_coroutine_threadsafe(coro, self._loop)
        assert flights.joined.wait(10), "the second sandbox did not join the build in progress"
        return startup.result(timeout=2)

    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", expiring)
    home = profiles("B")
    gate = _fake_from_credentials().gates["ak-B"] = Future()
    first, first_errors = _start(_routed_sandbox, modal_env, home)
    wait_until(lambda: calls["clients"] == ["ak-B"], timeout=15, what="the first sandbox started its build")
    second, second_errors = _start(_routed_sandbox, modal_env, home)
    first.join(10)
    second.join(10)
    released = not second.is_alive()
    gate.set_result(None)
    second.join(20)
    assert released, "a sandbox kept waiting on a build whose startup had stopped"
    assert len(first_errors) == 1 and isinstance(first_errors[0], TimeoutError)
    assert len(second_errors) == 1 and "abandoned" in str(second_errors[0])
    assert flights == {}
    _routed_sandbox(modal_env, home)
    assert calls["clients"] == ["ak-B", "ak-B"]
    assert calls["creates"] == ["ak-B"]


def test_timed_out_startup_leaves_another_profiles_build_running(modal_sdk, profiles, monkeypatch):
    """Only the builds a timed-out startup ran itself are abandoned: another profile's build in
    progress on its own sandbox's loop still completes for that sandbox."""
    modal_env, calls = modal_sdk
    run_coroutine = modal_env._AsyncWorker.run_coroutine
    expiring = threading.local()

    def maybe_expiring(self, coro, timeout=600):
        return run_coroutine(self, coro, timeout=2 if getattr(expiring, "on", False) else min(timeout, 15))

    def expiring_sandbox(home):
        expiring.on = True
        _routed_sandbox(modal_env, home)

    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", maybe_expiring)
    gates = _fake_from_credentials().gates
    gate_b, gate_a = gates["ak-B"], gates["ak-A"] = Future(), Future()
    other, other_errors = _start(_routed_sandbox, modal_env, profiles("B"))
    wait_until(lambda: calls["clients"] == ["ak-B"], timeout=15, what="B's build started")
    timed_out, timed_out_errors = _start(expiring_sandbox, profiles("A"))
    timed_out.join(10)
    assert not timed_out.is_alive()
    gate_b.set_result(None)
    other.join(10)
    gate_a.set_result(None)
    assert not other.is_alive()
    assert len(timed_out_errors) == 1 and isinstance(timed_out_errors[0], TimeoutError)
    assert other_errors == []
    assert calls["creates"] == ["ak-B"]


def test_cancelled_waiter_leaves_the_shared_build_running(modal_sdk, profiles, monkeypatch):
    """A waiting sandbox cancelled on its own loop: the build the others wait on keeps running."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    flights = _JoinRecorder()
    monkeypatch.setattr(modal_env, "_client_flights", flights)
    home = profiles("B")
    gate = _fake_from_credentials().gates["ak-B"] = Future()
    creator, creator_errors = _start(_routed_sandbox, modal_env, home)
    wait_until(lambda: calls["clients"] == ["ak-B"], timeout=15, what="the first sandbox started its build")
    with bind_profile_scope(home, hydrate_secrets=False):
        home_key = modal_env.hermes_home_key()

    async def cancelled_waiter():
        waiter = asyncio.ensure_future(
            modal_env._profile_client(sys.modules["modal"], home_key, ("ak-B", "as-B")))
        await asyncio.sleep(0)  # the waiter runs up to its wait on the build in progress
        assert flights.joined.is_set()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    asyncio.run(cancelled_waiter())
    gate.set_result(None)
    creator.join(10)
    assert not creator.is_alive()
    assert creator_errors == []
    assert calls["clients"] == ["ak-B"]
    assert calls["creates"] == ["ak-B"]
    assert flights == {}


def test_an_older_pair_finishing_late_never_replaces_the_newer_one(modal_sdk, profiles):
    """Rotation while the old pair's build is in progress: each sandbox runs on its own pair's
    client, and the old build finishing last leaves the newer client cached."""
    modal_env, calls = modal_sdk
    gate = _fake_from_credentials().gates["ak-B"] = Future()
    old, old_errors = _start(_routed_sandbox, modal_env, profiles("B"))
    wait_until(lambda: calls["clients"] == ["ak-B"], timeout=15, what="the old pair's build started")
    _routed_sandbox(modal_env, profiles("B-rotated"))
    gate.set_result(None)
    old.join(10)
    assert not old.is_alive()
    assert old_errors == []
    assert sorted(calls["creates"]) == ["ak-B", "ak-B-rotated"]
    (cached,) = modal_env._profile_clients.values()
    assert cached[0] == ("ak-B-rotated", "as-B-rotated")
    _routed_sandbox(modal_env, profiles("B-rotated"))
    assert calls["clients"] == ["ak-B", "ak-B-rotated"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_startup_abandons_its_client_build(modal_sdk, profiles, monkeypatch, interruption):
    """An interrupted constructor must not leave the new shared flight poisoning later starts."""
    modal_env, calls = modal_sdk
    monkeypatch.setattr(modal_env, "_client_flights", {})
    run_coroutine = modal_env._AsyncWorker.run_coroutine
    start = modal_env._AsyncWorker.start
    workers = []
    interrupt_once = [True]

    def tracking_start(self):
        workers.append(self)
        return start(self)

    def interrupted(self, coro, timeout=600):
        if interrupt_once:
            interrupt_once.clear()
            asyncio.run_coroutine_threadsafe(coro, self._loop)
            wait_until(lambda: calls["clients"] == ["ak-B"], timeout=15, what="the client build started")
            raise interruption()
        return run_coroutine(self, coro, timeout=min(timeout, 15))

    monkeypatch.setattr(modal_env._AsyncWorker, "start", tracking_start)
    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", interrupted)
    gate = _fake_from_credentials().gates["ak-B"] = Future()
    home = profiles("B")
    try:
        with pytest.raises(interruption):
            _routed_sandbox(modal_env, home)
        assert modal_env._client_flights == {}, "interrupted startup retained a poisoned shared build"
        assert not workers[0]._thread.is_alive(), "interrupted startup left its worker running"
        gate.set_result(None)
        _routed_sandbox(modal_env, home)
        assert calls["clients"] == ["ak-B", "ak-B"]
        assert calls["creates"] == ["ak-B"]
    finally:
        if not gate.done():
            gate.set_result(None)
        for worker in workers:
            worker.stop()


class _AdmissionOrder:
    """``_profile_clients_lock`` stand-in pinning one interleaving: the startup's worker reaches its
    client admission, the startup is interrupted, and only once its abandonment has run does the
    worker take the lock, before the worker is stopped. ``refusal`` is what the worker's admission
    raised (None: it was admitted)."""

    def __init__(self, caller):
        self._lock = threading.Lock()
        self.caller = caller
        self.worker: threading.Thread | None = None
        self.at_admission = threading.Event()
        self.abandoned = threading.Event()
        self.admission_ran = threading.Event()
        self.refusal: BaseException | None = None

    def __enter__(self):
        me = threading.current_thread()
        if me is not self.caller and self.worker is None:
            self.worker = me
            self.at_admission.set()
            assert self.abandoned.wait(15), "the interrupted startup never abandoned its builds"
        self._lock.acquire()
        return self

    def __exit__(self, _type, error, _tb):
        self._lock.release()
        me = threading.current_thread()
        if me is self.worker and not self.admission_ran.is_set():
            self.refusal = error
            self.admission_ran.set()
        elif me is self.caller and not self.abandoned.is_set():
            self.abandoned.set()  # the abandonment's critical section is over: let the worker in
            assert self.admission_ran.wait(15), "the worker never attempted its admission"
        return False


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_startup_interrupted_before_its_build_registers_admits_no_late_build(
        modal_sdk, profiles, monkeypatch, interruption):
    """Interrupted after the startup was scheduled but before its client build registered: the
    worker reaching the client cache only after the abandonment (and before it is stopped) must not
    register a build nobody will settle, which every later start would join and time out on."""
    modal_env, calls = modal_sdk
    monkeypatch.setattr(modal_env, "_client_flights", {})
    order = _AdmissionOrder(threading.current_thread())
    monkeypatch.setattr(modal_env, "_profile_clients_lock", order)
    run_coroutine = modal_env._AsyncWorker.run_coroutine
    start = modal_env._AsyncWorker.start
    workers = []
    interrupt_once = [True]

    def tracking_start(self):
        workers.append(self)
        return start(self)

    def interrupted(self, coro, timeout=600):
        if interrupt_once:
            interrupt_once.clear()
            asyncio.run_coroutine_threadsafe(coro, self._loop)
            assert order.at_admission.wait(15), "the startup never reached its client admission"
            raise interruption()
        return run_coroutine(self, coro, timeout=min(timeout, 15))

    monkeypatch.setattr(modal_env._AsyncWorker, "start", tracking_start)
    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", interrupted)
    # Held open: a build admitted late must still be pending when the worker stops.
    gate = _fake_from_credentials().gates["ak-B"] = Future()
    home = profiles("B")
    try:
        with pytest.raises(interruption):
            _routed_sandbox(modal_env, home)
        assert order.admission_ran.is_set()
        assert modal_env._client_flights == {}, "a build registered after its startup was abandoned"
        assert isinstance(order.refusal, RuntimeError) and "abandoned" in str(order.refusal)
        assert not workers[0]._thread.is_alive(), "interrupted startup left its worker running"
        gate.set_result(None)
        _routed_sandbox(modal_env, home)  # a fresh startup, on a fresh loop, is admitted
        assert calls["clients"] == ["ak-B"]
        assert calls["creates"] == ["ak-B"]
        assert modal_env._client_flights == {}
    finally:
        if not gate.done():
            gate.set_result(None)
        for worker in workers:
            worker.stop()


def test_client_build_timeout_preserves_snapshot_and_does_not_retry(modal_sdk, profiles, monkeypatch):
    """A client-build timeout on a persistent sandbox must not delete the saved snapshot or
    re-enter _create on the same stuck flight for a second 300 s wait."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    flights = _JoinRecorder()
    monkeypatch.setattr(modal_env, "_client_flights", flights)
    home = profiles("B")
    snap_file = home / "modal_snapshots.json"
    snap_file.write_text(json.dumps({"direct:task-1": "im-snap-1"}))

    run_coroutine = modal_env._AsyncWorker.run_coroutine
    creator_attempts: list[int] = []
    creator_worker: list[object] = []

    def timeout_first(self, coro, timeout=600):
        if not creator_worker:
            creator_worker.append(self)
        if self is creator_worker[0]:
            creator_attempts.append(timeout)
            if len(creator_attempts) == 1:
                asyncio.run_coroutine_threadsafe(coro, self._loop)
                assert flights.joined.wait(10), "follower did not join the build in progress"
                raise TimeoutError()
        return run_coroutine(self, coro, timeout=min(timeout, 15))

    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", timeout_first)
    gate = _fake_from_credentials().gates["ak-B"] = Future()

    def persistent_sandbox():
        with bind_profile_scope(home, hydrate_secrets=False):
            env = modal_env.ModalEnvironment(
                image="python:3.11", persistent_filesystem=True, task_id="task-1")
            env.cleanup()

    try:
        creator, creator_errors = _start(persistent_sandbox)
        wait_until(lambda: calls["clients"] == ["ak-B"], timeout=15, what="the client build started")
        follower, follower_errors = _start(_routed_sandbox, modal_env, home)
        creator.join(10)
        follower.join(10)
        assert not creator.is_alive() and not follower.is_alive()
        assert creator_attempts == [300]
        assert len(creator_errors) == 1 and isinstance(creator_errors[0], TimeoutError)
        assert len(follower_errors) == 1 and "abandoned" in str(follower_errors[0])
        assert json.loads(snap_file.read_text()) == {"direct:task-1": "im-snap-1"}
        assert flights == {}
        gate.set_result(None)
        persistent_sandbox()
        assert calls["clients"] == ["ak-B", "ak-B"]
        assert calls["creates"] == ["ak-B"]
    finally:
        if not gate.done():
            gate.set_result(None)


def test_client_build_error_preserves_snapshot_and_does_not_retry(modal_sdk, profiles):
    """A routed client-build exception must preserve the saved snapshot and skip fallback retry."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    home = profiles("B")
    snap_file = home / "modal_snapshots.json"
    snap_file.write_text(json.dumps({"direct:task-1": "im-snap-1"}))
    _fake_from_credentials().failures["ak-B"] = [RuntimeError("auth boom")]

    with bind_profile_scope(home, hydrate_secrets=False), pytest.raises(RuntimeError, match="auth boom"):
        modal_env.ModalEnvironment(image="python:3.11", persistent_filesystem=True, task_id="task-1")

    assert json.loads(snap_file.read_text()) == {"direct:task-1": "im-snap-1"}
    assert calls["clients"] == ["ak-B"]
    assert calls["creates"] == []


def test_caller_timeout_after_late_client_finish_still_preserves_snapshot(modal_sdk, profiles, monkeypatch):
    """Even if _profile_client finishes right as the caller's 300 s wait expires, the caller-side
    TimeoutError must not delete the healthy snapshot or start a second 300 s attempt."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    home = profiles("B")
    snap_file = home / "modal_snapshots.json"
    snap_file.write_text(json.dumps({"direct:task-1": "im-snap-1"}))

    lookup_entered = threading.Event()
    release_lookup = threading.Event()
    orig_lookup = sys.modules["modal"].App.lookup.aio

    async def slow_lookup(*args, **kwargs):
        lookup_entered.set()
        await asyncio.to_thread(release_lookup.wait, 10)
        return await orig_lookup(*args, **kwargs)

    monkeypatch.setattr(sys.modules["modal"].App.lookup, "aio", slow_lookup)
    attempts: list[int] = []
    run_coroutine = modal_env._AsyncWorker.run_coroutine

    def expire_after_client(self, coro, timeout=600):
        attempts.append(timeout)
        if len(attempts) == 1:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
            assert lookup_entered.wait(10)
            raise TimeoutError()
        return run_coroutine(self, coro, timeout=min(timeout, 15))

    monkeypatch.setattr(modal_env._AsyncWorker, "run_coroutine", expire_after_client)
    try:
        with bind_profile_scope(home, hydrate_secrets=False), pytest.raises(TimeoutError):
            modal_env.ModalEnvironment(image="python:3.11", persistent_filesystem=True, task_id="task-1")
        assert attempts == [300]
        assert json.loads(snap_file.read_text()) == {"direct:task-1": "im-snap-1"}
        assert calls["creates"] == []
    finally:
        release_lookup.set()


@pytest.mark.parametrize("fail_at", ["from_id", "create"])
def test_snapshot_restore_failure_after_client_build_still_falls_back(
        modal_sdk, profiles, monkeypatch, fail_at):
    """Genuine snapshot-restore failures (image lookup or Sandbox.create after client acquisition)
    still clear the stale snapshot and fall back to the base image."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    home = profiles("B")
    snap_file = home / "modal_snapshots.json"
    snap_file.write_text(json.dumps({"direct:task-1": "im-bad"}))

    seen_images: list[object] = []
    if fail_at == "from_id":
        def bad_from_id(image_id):
            raise RuntimeError(f"bad snapshot {image_id}")
        monkeypatch.setattr(sys.modules["modal"].Image, "from_id", bad_from_id)
    else:
        orig_create = sys.modules["modal"].Sandbox.create.aio

        async def create_reject_snapshot(*args, image=None, **kwargs):
            seen_images.append(image)
            if image == "im-bad":
                raise RuntimeError("snapshot unavailable")
            return await orig_create(*args, image=image, **kwargs)

        monkeypatch.setattr(sys.modules["modal"].Sandbox.create, "aio", create_reject_snapshot)

    with bind_profile_scope(home, hydrate_secrets=False):
        env = modal_env.ModalEnvironment(
            image="python:3.11", persistent_filesystem=True, task_id="task-1")
        env._persistent = False
        env.cleanup()

    assert json.loads(snap_file.read_text()) == {}
    assert calls["clients"] == ["ak-B"]
    assert calls["creates"] == ["ak-B"]
    if fail_at == "create":
        assert seen_images == ["im-bad", "image"]


def test_base_image_retry_client_build_failure_raises_the_original_error(modal_sdk, profiles, monkeypatch):
    """A bad snapshot falls back to the base image; when that retry's client build then fails, the
    caller sees the client's own error, never the module's private wrapper."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    home = profiles("B")
    (home / "modal_snapshots.json").write_text(json.dumps({"direct:task-1": "im-bad"}))

    def bad_from_id(image_id):
        raise RuntimeError(f"bad snapshot {image_id}")

    monkeypatch.setattr(sys.modules["modal"].Image, "from_id", bad_from_id)
    _fake_from_credentials().failures["ak-B"] = [PermissionError("auth boom")]
    with bind_profile_scope(home, hydrate_secrets=False), pytest.raises(PermissionError) as raised:
        modal_env.ModalEnvironment(image="python:3.11", persistent_filesystem=True, task_id="task-1")
    assert type(raised.value) is PermissionError and str(raised.value) == "auth boom"
    assert calls["clients"] == ["ak-B"]
    assert calls["creates"] == []
    assert modal_env._client_flights == {}
