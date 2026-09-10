"""Regression: a default/launch agent build under an ACTIVE multiplexer must not raise
UnscopedSecretError when it reads a profile credential.

A named-profile turn flips multiplex on process-wide (F1). After that, building a default session's
agent used to read ANTHROPIC_TOKEN with no secret scope installed -> UnscopedSecretError. Both build
paths (server._bind_build_profile_scopes for the deferred build, and compute_host's inline default
branch) must bind the launch home's EXEMPT scope so os.environ credential reads keep working.
"""
from __future__ import annotations

import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _multiplex_on(monkeypatch):
    was = ss.is_multiplex_active()
    ss.set_multiplex_active(True)  # a named-profile turn already flipped this on
    yield
    ss.set_multiplex_active(was)


def test_default_build_scope_allows_env_credential_read(monkeypatch):
    # No profile scope installed + multiplex on => bare get_secret raises (the reported bug).
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-launch-value")
    with pytest.raises(ss.UnscopedSecretError):
        ss.get_secret("ANTHROPIC_TOKEN")

    # _bind_build_profile_scopes("") binds the launch home's EXEMPT scope, so the read succeeds.
    from tui_gateway import server
    scopes = server._bind_build_profile_scopes("")
    try:
        assert ss.get_secret("ANTHROPIC_TOKEN") == "sk-launch-value"
        assert scopes.env_fallback is not None  # exempt marker installed for the default build
    finally:
        server._release_build_profile_scopes(scopes)

    # After release the exempt scope is gone -> bare read raises again (no scope leak past the build).
    with pytest.raises(ss.UnscopedSecretError):
        ss.get_secret("ANTHROPIC_TOKEN")


def test_named_build_scope_stays_fail_closed(tmp_path):
    # A named profile build binds a fully fail-closed scope (no env fallback marker): a miss returns
    # the default, never a peer's os.environ value.
    from tui_gateway import server
    prof = tmp_path / "felix"
    prof.mkdir()
    scopes = server._bind_build_profile_scopes(str(prof))
    try:
        assert scopes.env_fallback is None  # named profile is never exempt
    finally:
        server._release_build_profile_scopes(scopes)
