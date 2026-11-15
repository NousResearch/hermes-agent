"""Direct Modal sandboxes run on the serving profile's own Modal account.

Without an explicit ``client=``, ``modal.App.lookup`` and ``modal.Sandbox.create`` use the SDK's
process-wide ``Client.from_env()``: one client built from ``os.environ`` (the launch profile's
``.env``) on first use and kept for the process. On a multiplexed gateway every routed profile's
sandbox was then created, billed and run on the launch profile's Modal account.
"""

import sys
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest


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
    ``on_build`` lets a test hold a build open or make it fail."""

    def __init__(self, built: list):
        self.built = built
        self.on_build: Callable[[str], None] = lambda _token_id: None

    def __call__(self, token_id, token_secret):
        self.built.append(token_id)
        self.on_build(token_id)
        return types.SimpleNamespace(token_id=token_id, token_secret=token_secret)

    async def aio(self, token_id, token_secret):
        return self(token_id, token_secret)


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
    assert len(modal_env._profile_clients) == 2


@pytest.mark.parametrize("dotenv", ["OTHER=1\n", "MODAL_TOKEN_ID=ak-B\n", "MODAL_TOKEN_ID=ak-B\nMODAL_TOKEN_SECRET=\n"],
                         ids=["absent", "id-only", "empty-secret"])
def test_routed_profile_without_its_own_pair_never_borrows_the_launch_account(
        modal_sdk, profiles, tmp_path, monkeypatch, dotenv):
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    monkeypatch.setenv("HOME", str(tmp_path))
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


def test_single_profile_host_login_still_selects_direct(tmp_path, monkeypatch):
    from tools.tool_backend_helpers import has_direct_modal_credentials

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".modal.toml").write_text('[default]\ntoken_id = "ak-cli"\ntoken_secret = "as-cli"\n')
    assert has_direct_modal_credentials() is True


@dataclass
class _RaceState:
    acquirers: set[int | None] = field(default_factory=set)
    b_building: bool = False
    release: bool = False


class _RecordingLock:
    """Stand-in for ``_profile_clients_lock`` that records which threads took it."""

    def __init__(self, on_acquire):
        self._lock, self._on_acquire = threading.Lock(), on_acquire

    def __enter__(self):
        self._lock.acquire()
        self._on_acquire(threading.get_ident())
        return self

    def __exit__(self, *exc):
        self._lock.release()


def test_concurrent_first_sandboxes_of_one_profile_build_one_client(modal_sdk, profiles, monkeypatch):
    """Two first sandboxes of the same profile at once: the SDK opens a client that lives until
    process exit, so only one may be built. Another profile is not held up meanwhile."""
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    home_b, home_a = profiles("B"), profiles("A")
    state = threading.Condition()
    seen = _RaceState()

    def on_acquire(ident):
        with state:
            seen.acquirers.add(ident)
            state.notify_all()

    def hold_first_b_build(token_id):
        if token_id != "ak-B":
            return
        with state:
            seen.b_building = True
            state.notify_all()
            assert state.wait_for(lambda: seen.release, timeout=10)

    monkeypatch.setattr(modal_env, "_profile_clients_lock", _RecordingLock(on_acquire))
    monkeypatch.setattr(_fake_from_credentials(), "on_build", hold_first_b_build)
    errors = []

    def run(home):
        try:
            with bind_profile_scope(home, hydrate_secrets=False):
                _sandbox(modal_env)
        except BaseException as exc:  # surfaced through ``errors`` below
            errors.append(exc)

    first, second, other = (threading.Thread(target=run, args=(h,)) for h in (home_b, home_b, home_a))
    first.start()
    with state:
        assert state.wait_for(lambda: seen.b_building, timeout=10)
    second.start()
    with state:
        # The second B sandbox reached the client cache (or, without the per-home build lock,
        # started a second build of its own).
        assert state.wait_for(
            lambda: second.ident in seen.acquirers or calls["clients"].count("ak-B") > 1, timeout=10)
    other.start()
    other.join(timeout=10)
    assert not other.is_alive(), "another profile's sandbox waited on B's client build"
    with state:
        seen.release = True
        state.notify_all()
    for thread in (first, second):
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert errors == []
    assert sorted(calls["clients"]) == ["ak-A", "ak-B"]
    assert sorted(calls["creates"]) == ["ak-A", "ak-B", "ak-B"]


def test_failed_client_build_is_not_cached(modal_sdk, profiles, monkeypatch):
    from agent.runtime_scope import bind_profile_scope

    modal_env, calls = modal_sdk
    failures = ["boom"]

    def fail_once(_token_id):
        if failures:
            raise RuntimeError(failures.pop())

    monkeypatch.setattr(_fake_from_credentials(), "on_build", fail_once)
    with bind_profile_scope(profiles("B"), hydrate_secrets=False):
        with pytest.raises(RuntimeError, match="boom"):
            _sandbox(modal_env)
        _sandbox(modal_env)
    assert calls["clients"] == ["ak-B", "ak-B"]
    assert calls["creates"] == ["ak-B"]
