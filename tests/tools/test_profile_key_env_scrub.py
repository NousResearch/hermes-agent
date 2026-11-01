"""Profile-declared provider key holders never reach a child or the snapshot (#147329).

A custom endpoint's API key lives under a name that no registry can enumerate: Hermes mints it
itself (``custom_endpoint_key_env`` -> ``HERMES_CUSTOM_<SLUG>_API_KEY``), or the operator picks it
in ``providers:`` / ``model:`` / ``fallback_providers:`` via ``key_env``. Both reached every terminal
child and were written in plaintext into the persisted session snapshot, while built-in provider keys
such as COHERE_API_KEY stayed out. Expectations here come from independent config declarations and
literal policy examples, never from the finished blocklist.
"""

import json
import os
from pathlib import Path

import pytest

from hermes_cli.config import custom_endpoint_key_env
from tests.tools._child_env_fixtures import child_env as _isolated_child_env
from tests.tools._child_env_fixtures import observe_child, observe_terminal
from tools.environments import local
from tools.environments.local_env_policy import _is_provider_env_blocklisted


@pytest.fixture
def isolated_home(_isolated_child_env):  # health: allow F811 -- pytest fixture name doubles as the test parameter, the shape every file in tests/tools uses
    """The shared isolated-home fixture under a name no test parameter shadows.

    ``child_env`` imported directly is flagged as a redefinition by the code-health F811
    rule once a test takes it as a parameter, which is why this wrapper exists.
    """
    return _isolated_child_env

SECRET = "sk-profi...14729"

# Names Hermes mints per endpoint: two endpoints on one host get separate slots.
MINTED = [custom_endpoint_key_env("my endpoint"), custom_endpoint_key_env("127.0.0.1:1234")]
# Names an operator chose themselves — the half the minted-name shape cannot cover.
DECLARED = ["THETA_PROXY_KEY", "CORP_LLM_TOKEN"]
# Operator vars that only resemble either form: not Hermes credentials, so they must pass.
OPERATOR_VARS = ["HERMES_CUSTOM_BANNER", "MY_CUSTOM_API_KEY", "THETA_PROXY_URL", "CORP_TEAM_ID"]


def _write_profile_config(home: Path, monkeypatch) -> None:
    """Point HERMES_HOME at a profile whose config declares DECLARED as credential holders."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: theta\n"
        f"  key_env: {DECLARED[0]}\n"
        "providers:\n"
        f"  theta:\n"
        f"    key_env: {DECLARED[0]}\n"
        "  corp:\n"
        f"    api_key_env: {DECLARED[1]}\n"
        "fallback_providers:\n"
        f"  - provider: corp\n"
        f"    key_env: {DECLARED[1]}\n"
        "custom_providers:\n"
        f"  - name: legacy\n"
        f"    key_env: {DECLARED[1]}\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))


BLOCKED = MINTED + DECLARED


@pytest.mark.parametrize("builder", ["foreground", "background", "nonterminal"])
def test_declared_and_minted_keys_never_reach_children(isolated_home, monkeypatch, builder):
    _write_profile_config(isolated_home / "hermes", monkeypatch)
    for name in [*BLOCKED, *OPERATOR_VARS]:
        monkeypatch.setenv(name, SECRET)
    before = dict(os.environ)
    if builder == "foreground":
        observed = observe_terminal(local.LocalEnvironment(cwd=str(isolated_home), timeout=30), BLOCKED + OPERATOR_VARS)
    elif builder == "background":
        observed = observe_child(local._make_run_env(dict(os.environ)), BLOCKED + OPERATOR_VARS)
    else:
        observed = observe_child(local._sanitize_subprocess_env(dict(os.environ)), BLOCKED + OPERATOR_VARS)
    assert observed == {**dict.fromkeys(BLOCKED), **{k: SECRET for k in OPERATOR_VARS}}
    assert dict(os.environ) == before  # scrubbing is per-child, never a mutation of the parent


def test_snapshot_holds_no_plaintext_key(isolated_home, monkeypatch):
    _write_profile_config(isolated_home / "hermes", monkeypatch)
    for name in BLOCKED:
        monkeypatch.setenv(name, SECRET)
    env = None
    try:
        env = local.LocalEnvironment(cwd=str(isolated_home), timeout=30)
        snapshot = Path(env._snapshot_path)
        assert snapshot.exists()
        text = snapshot.read_text(encoding="utf-8")
        for name in BLOCKED:
            assert name not in text, f"{name} leaked into the session snapshot"
        assert SECRET not in text
    finally:
        if env is not None:
            env.cleanup()


def test_inheriting_child_keeps_the_key(isolated_home, monkeypatch):
    """Provider tier, not internal: a credentialed child that drives the endpoint still has it."""
    _write_profile_config(isolated_home / "hermes", monkeypatch)
    for name in BLOCKED:
        monkeypatch.setenv(name, SECRET)
    inherited = local.hermes_subprocess_env(inherit_credentials=True)
    assert observe_child(inherited, BLOCKED) == {k: SECRET for k in BLOCKED}


def test_passthrough_cannot_re_allow_a_provider_key(isolated_home, monkeypatch):
    from tools.env_passthrough import register_env_passthrough
    _write_profile_config(isolated_home / "hermes", monkeypatch)
    for name in BLOCKED:
        monkeypatch.setenv(name, SECRET)
    register_env_passthrough(BLOCKED)
    result = local._sanitize_subprocess_env(dict(os.environ))
    assert not (set(BLOCKED) & set(result))


def test_policy_predicates_on_names_alone():
    """Pure-name checks, no config read: the minted shape is denied, look-alikes are not."""
    assert all(_is_provider_env_blocklisted(n) for n in MINTED)
    assert not any(_is_provider_env_blocklisted(n) for n in OPERATOR_VARS)
