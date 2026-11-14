"""Direct Modal sandboxes run on the serving profile's own Modal account.

Without an explicit ``client=``, ``modal.App.lookup`` and ``modal.Sandbox.create`` use the SDK's
process-wide ``Client.from_env()``: one client built from ``os.environ`` (the launch profile's
``.env``) on first use and kept for the process. On a multiplexed gateway every routed profile's
sandbox was then created, billed and run on the launch profile's Modal account.
"""

import sys
import types

import pytest


async def _noop(*_args, **_kwargs):
    return None


@pytest.fixture
def modal_sdk(monkeypatch):
    """A ``modal`` stand-in recording which token pair each client was built from and which client
    every App lookup and Sandbox create ran on (``None``: the SDK's own process-wide client)."""
    from tools.environments import modal as modal_env

    calls = {"clients": [], "lookups": [], "creates": []}

    async def from_credentials(token_id, token_secret):
        calls["clients"].append(token_id)
        return types.SimpleNamespace(token_id=token_id, token_secret=token_secret)

    async def lookup(_name, *, create_if_missing=False, client=None):
        calls["lookups"].append(getattr(client, "token_id", None))
        return types.SimpleNamespace(name=_name)

    async def create(*_args, image=None, app=None, timeout=None, client=None, **_kwargs):
        calls["creates"].append(getattr(client, "token_id", None))
        return types.SimpleNamespace(terminate=types.SimpleNamespace(aio=_noop))

    aio = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(
        Client=aio(from_credentials=aio(aio=from_credentials)),
        App=aio(lookup=aio(aio=lookup)),
        Sandbox=aio(create=aio(aio=create)),
        Image=aio(from_registry=lambda *_a, **_k: "image", from_id=lambda image_id: image_id)))
    monkeypatch.setattr(modal_env, "ensure_lazy_dep", lambda extra: None)
    monkeypatch.setattr(modal_env, "_profile_clients", {}, raising=False)
    monkeypatch.setattr(modal_env, "FileSyncManager", lambda **_kw: types.SimpleNamespace(
        sync=lambda force=False: None, sync_back=lambda: None))
    monkeypatch.setattr(modal_env.ModalEnvironment, "init_session", lambda self: None)
    return modal_env, calls


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
