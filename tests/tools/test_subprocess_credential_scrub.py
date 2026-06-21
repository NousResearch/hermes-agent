"""Tests for credential scrubbing of subprocess environments.

Covers the OpenShell-inspired hardening: the encryption passphrase is always
stripped, and credential-shaped vars are swept (config-gated) on top of the
existing provider blocklist.
"""

from __future__ import annotations

from hermes_constants import get_hermes_home


def _set_scrub(enabled: bool) -> None:
    (get_hermes_home() / "config.yaml").write_text(
        f"security:\n  credential_broker:\n    scrub_subprocess_env: {str(enabled).lower()}\n",
        encoding="utf-8",
    )


# ── Terminal backend — tools/environments/local.py ──────────────────────────


def test_encryption_passphrase_always_stripped_from_terminal_env():
    from tools.environments.local import _make_run_env, _sanitize_subprocess_env

    env = {"HERMES_ENCRYPTION_PASSPHRASE": "topsecret", "PATH": "/usr/bin"}
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in _sanitize_subprocess_env(env)
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in _make_run_env(env)


def test_passphrase_stripped_even_with_scrub_disabled():
    # The unconditional strip must ignore the config flag entirely.
    _set_scrub(False)
    from tools.environments.local import _sanitize_subprocess_env

    out = _sanitize_subprocess_env({"HERMES_ENCRYPTION_PASSPHRASE": "x", "PATH": "/usr/bin"})
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in out
    assert out.get("PATH") == "/usr/bin"


def test_shape_sweep_strips_unlisted_credential_var():
    _set_scrub(True)
    from tools.environments.local import _sanitize_subprocess_env

    out = _sanitize_subprocess_env({"ACME_API_KEY": "ak-1", "EDITOR": "vim"})
    assert "ACME_API_KEY" not in out  # caught by the credential-shape sweep
    assert out.get("EDITOR") == "vim"  # ordinary vars preserved


def test_shape_sweep_off_when_disabled():
    _set_scrub(False)
    from tools.environments.local import _sanitize_subprocess_env

    out = _sanitize_subprocess_env({"ACME_API_KEY": "ak-1"})
    # With the sweep off, a var not in the explicit blocklist passes through.
    assert out.get("ACME_API_KEY") == "ak-1"


def test_shape_sweep_off_by_default_when_unset():
    # M3: the default for scrub_subprocess_env was flipped True -> False to
    # honor the PR's opt-in / off-by-default contract. With NO scrub key in
    # config the sweep must be OFF, while the always-on passphrase strip and
    # provider blocklist remain in force. We deliberately do NOT call
    # _set_scrub() here so cfg_get falls through to the new False default.
    (get_hermes_home() / "config.yaml").write_text(
        "security:\n  credential_broker:\n    enabled: false\n",
        encoding="utf-8",
    )
    from tools.environments.local import _sanitize_subprocess_env

    out = _sanitize_subprocess_env({
        "ACME_API_KEY": "ak-1",                 # shaped + unlisted
        "HERMES_ENCRYPTION_PASSPHRASE": "x",    # always-stripped
        "EDITOR": "vim",
    })
    assert out.get("ACME_API_KEY") == "ak-1"          # sweep OFF by default
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in out   # always-on strip survives
    assert out.get("EDITOR") == "vim"


def test_passthrough_allowlist_still_passes_a_shaped_var():
    _set_scrub(True)
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
    from tools.environments.local import _sanitize_subprocess_env

    register_env_passthrough(["ACME_API_KEY"])
    try:
        out = _sanitize_subprocess_env({"ACME_API_KEY": "ak-1"})
        assert out.get("ACME_API_KEY") == "ak-1"
    finally:
        clear_env_passthrough()


# ── Code-execution sandbox — tools/code_execution_tool.py ───────────────────


def test_encryption_passphrase_stripped_from_execute_code_child():
    from tools.code_execution_tool import _scrub_child_env

    child = _scrub_child_env(
        {"HERMES_ENCRYPTION_PASSPHRASE": "topsecret", "PATH": "/usr/bin", "HOME": "/h"},
        is_passthrough=lambda _: False,
        is_windows=False,
    )
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in child
    assert child.get("PATH") and child.get("HOME")


def test_passphrase_stripped_from_execute_code_even_when_passthrough_allowlisted():
    # M1: the unconditional always-strip guard runs BEFORE the passthrough
    # check, so a malicious/careless skill or operator config that allowlists
    # HERMES_ENCRYPTION_PASSPHRASE can no longer resurrect it into the child.
    from tools.code_execution_tool import _scrub_child_env

    child = _scrub_child_env(
        {"HERMES_ENCRYPTION_PASSPHRASE": "topsecret", "MY_OK_VAR": "v", "PATH": "/usr/bin"},
        is_passthrough=lambda k: k in {"HERMES_ENCRYPTION_PASSPHRASE", "MY_OK_VAR"},
        is_windows=False,
    )
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in child  # always-strip beats passthrough
    assert child.get("MY_OK_VAR") == "v"  # ordinary passthrough still honored
    assert child.get("PATH") == "/usr/bin"
