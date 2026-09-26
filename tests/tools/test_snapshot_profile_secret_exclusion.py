"""Profile credentials must never be persisted into the bash env snapshot.

The terminal backend captures the login shell into a snapshot file
(``BaseEnvironment.init_session``) built with ``export -p``, filtered through
``_snapshot_excluded_passthrough_names()``. That exclusion list was computed
only ``if is_multiplex_active()``, because it was introduced for *cross-profile*
passthrough leakage.

The same list, however, is the only thing keeping *credentials* out of the
snapshot. On a single-profile host the list was empty, so ``export -p`` wrote
every secret the profile loader had injected into ``os.environ`` — vault access
token, provider API keys, SSH private key, dashboard secrets — to disk in
plaintext, rewritten on essentially every command.

These tests pin the contract that credential NAMES are excluded regardless of
multiplex state, that values are never read to do so, and that ordinary shell
state still survives.
"""

from __future__ import annotations

import os
import sys

import pytest

from tools.environments.local import LocalEnvironment


_SECRET_NAME = "PROBE_VAULT_TOKEN"
_SECRET_VALUE = "s3cr3t-probe-value-must-not-reach-disk"


@pytest.fixture
def profile_secret_env(tmp_path, monkeypatch):
    """A profile whose secret scope reports one credential name."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import agent.secret_scope as ss

    # Names only — the scope maps name -> value, and the code under test must
    # consume the keys without ever touching the values.
    monkeypatch.setattr(
        ss, "build_profile_secret_scope", lambda _home: {_SECRET_NAME: _SECRET_VALUE}
    )
    return home


@pytest.mark.parametrize("multiplex", [False, True], ids=["single-profile", "multiplex"])
def test_credential_names_excluded_regardless_of_multiplex(profile_secret_env, monkeypatch, multiplex):
    """The single-profile case is the bug: exclusions must not be gated on multiplex."""
    import agent.secret_scope as ss

    monkeypatch.setattr(ss, "is_multiplex_active", lambda: multiplex)

    env = LocalEnvironment(cwd=str(profile_secret_env), timeout=10)
    env._profile_scoped_passthrough = True

    excluded = env._snapshot_excluded_passthrough_names()

    assert _SECRET_NAME in excluded


def test_enumerating_secret_names_does_not_expose_values(profile_secret_env):
    """Names are load-bearing; values must never enter the exclusion machinery."""
    env = LocalEnvironment(cwd=str(profile_secret_env), timeout=10)
    env._profile_scoped_passthrough = True

    names = env._profile_secret_env_names()

    assert _SECRET_NAME in names
    assert _SECRET_VALUE not in names


def test_bootstrap_token_names_are_excluded_though_absent_from_the_scope(profile_secret_env, monkeypatch):
    """A source's bootstrap credential never appears in the resolved scope.

    `load_hermes_dotenv` seeds it from a gitignored `.op.env` or the
    launcher/systemd environment, so it reaches `os.environ` — and therefore
    `export -p` — while `build_profile_secret_scope()` does not report it.
    Excluding only resolved values leaves the token that unlocks the vault on
    disk.
    """
    import agent.secret_scope as ss

    monkeypatch.setattr(ss, "is_multiplex_active", lambda: False)

    env = LocalEnvironment(cwd=str(profile_secret_env), timeout=10)
    env._profile_scoped_passthrough = True

    excluded = set(env._snapshot_excluded_passthrough_names())

    # Declared by the built-in sources via protected_env_vars().
    assert "OP_SERVICE_ACCOUNT_TOKEN" in excluded
    assert "BWS_ACCESS_TOKEN" in excluded
    # ...and the resolved-scope population is still covered.
    assert _SECRET_NAME in excluded


def test_bootstrap_names_follow_a_configured_token_env_override(profile_secret_env, monkeypatch):
    """The name is config-driven, so a renamed token env must still be excluded.

    Pinning the contract against `protected_env_vars()` rather than a hardcoded
    list is what makes a plugin-provided source covered without editing the
    snapshot code.
    """
    import agent.secret_scope as ss

    monkeypatch.setattr(ss, "is_multiplex_active", lambda: False)

    import hermes_cli.env_loader as el

    monkeypatch.setattr(
        el,
        "_load_secrets_config",
        lambda _home: {"bitwarden": {"enabled": True, "access_token_env": "CUSTOM_BWS_ENV"}},
    )

    env = LocalEnvironment(cwd=str(profile_secret_env), timeout=10)
    env._profile_scoped_passthrough = True

    excluded = set(env._snapshot_excluded_passthrough_names())

    assert "CUSTOM_BWS_ENV" in excluded


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_snapshot_file_never_contains_credential_bytes(profile_secret_env, monkeypatch):
    """E2E: a real snapshot on a single-profile host carries PATH but no secret."""
    import agent.secret_scope as ss

    monkeypatch.setattr(ss, "is_multiplex_active", lambda: False)
    monkeypatch.setenv(_SECRET_NAME, _SECRET_VALUE)

    env = LocalEnvironment(cwd=str(profile_secret_env), timeout=30)
    env._profile_scoped_passthrough = True
    env.init_session()
    try:
        env.execute("printf ready")

        snap = env._snapshot_path
        if not os.path.exists(snap):
            pytest.skip("no snapshot produced on this host")
        contents = open(snap).read()

        # The value itself is what leaks — assert on bytes, not the name.
        assert _SECRET_VALUE not in contents
        # ...while the user's ordinary shell state must still persist.
        assert "PATH" in contents

        # And the command still sees the credential, re-injected per-Popen.
        result = env.execute(f'printf "[$%s]"' % _SECRET_NAME)
        assert _SECRET_VALUE in result.get("output", "")
    finally:
        env.cleanup()
