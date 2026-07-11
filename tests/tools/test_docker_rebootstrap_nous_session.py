"""Unit tests for scripts/docker_rebootstrap_nous_session.py.

The boot-time re-seed is the load-bearing "does not clobber a healthy session"
guard: it may overwrite the on-disk Nous provider entry when that entry is
provably terminal (quarantine marker + no usable tokens), or when an
orchestrator seed is demonstrably newer. Older/incomparable seeds must no-op.
These are pure-stdlib tmp_path tests (no container build).
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

# Import the stdlib-only boot helper by path (it lives under scripts/, not an
# installed package) — mirrors the repo's other scripts/-helper tests.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docker_rebootstrap_nous_session.py"
_spec = importlib.util.spec_from_file_location("docker_rebootstrap_nous_session", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _terminal_nous_state():
    """On-disk shape after a terminal quarantine: tokens cleared, marker set."""
    return {
        "portal_base_url": "https://portal.example.com",
        "client_id": "hermes-cli-vps",
        "last_auth_error": {
            "provider": "nous",
            "code": "invalid_grant",
            "relogin_required": True,
        },
    }


def _healthy_nous_state():
    return {
        "portal_base_url": "https://portal.example.com",
        "client_id": "hermes-cli-vps",
        "access_token": "live-at",
        "refresh_token": "live-rt",
    }


def _write_auth(tmp_path: Path, providers: dict) -> str:
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"version": 1, "providers": providers}))
    return str(p)


_FRESH_SEED = json.dumps({
    "version": 1,
    "providers": {
        "nous": {
            "portal_base_url": "https://portal.example.com",
            "client_id": "hermes-cli-vps",
            "access_token": "FRESH-at",
            "refresh_token": "FRESH-rt",
        }
    },
})


def test_reseeds_terminal_entry(tmp_path):
    """Terminal on-disk entry + valid seed → providers.nous replaced."""
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    result = mod.reseed_if_terminal(auth, _FRESH_SEED)
    assert result == "reseeded"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["refresh_token"] == "FRESH-rt"
    assert "last_auth_error" not in store["providers"]["nous"]


def test_does_not_clobber_healthy_entry(tmp_path):
    """LOAD-BEARING: a healthy (live-token) entry must never be overwritten."""
    auth = _write_auth(tmp_path, {"nous": _healthy_nous_state()})
    result = mod.reseed_if_terminal(auth, _FRESH_SEED)
    assert result == "not_terminal"
    store = json.loads(Path(auth).read_text())
    # Untouched — still the live tokens, not the seed.
    assert store["providers"]["nous"]["refresh_token"] == "live-rt"


def test_marker_but_live_token_is_not_terminal(tmp_path):
    """Stale marker + a live token present → NOT terminal (don't clobber)."""
    state = _terminal_nous_state()
    state["refresh_token"] = "somehow-live"
    auth = _write_auth(tmp_path, {"nous": state})
    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "not_terminal"


def test_reseeds_newer_orchestrator_session_over_healthy_stale_entry(tmp_path):
    """A newer orchestrator-issued session replaces the healthy local session.

    NAS revokes the old session before restarting a hosted agent.  Refusing the
    re-seed merely because the local entry still has tokens leaves that revoked
    session in place and guarantees ``invalid_grant`` on its next refresh.
    """
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00+00:00",
    }})
    seed = json.dumps({
        "version": 1,
        "providers": {
            "nous": {
                "portal_base_url": "https://portal.example.com",
                "client_id": "hermes-cli-vps",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "2026-07-14T19:05:00+00:00",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "reseeded_newer"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["refresh_token"] == "FRESH-rt"


def test_does_not_replace_healthy_entry_with_older_seed(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:05:00+00:00",
    }})
    seed = json.dumps({
        "version": 1,
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "access_token": "STALE-at",
                "refresh_token": "STALE-rt",
                "obtained_at": "2026-07-14T19:00:00+00:00",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "not_terminal"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["refresh_token"] == "live-rt"


def test_timezone_less_local_timestamp_is_incomparable(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "2026-07-14T19:05:00Z",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "not_terminal"


def test_malformed_timestamp_does_not_clobber_healthy_entry(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "not-a-time",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "2026-07-14T19:05:00Z",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "not_terminal"


def test_newer_seed_without_tokens_does_not_clobber_healthy_entry(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00Z",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "obtained_at": "2026-07-14T19:05:00Z",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "bad_seed"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["refresh_token"] == "live-rt"


def test_newer_seed_for_non_bootstrap_client_does_not_clobber_healthy_entry(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00Z",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "2026-07-14T19:05:00Z",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "bad_seed"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["refresh_token"] == "live-rt"


def test_timezone_less_seed_timestamp_is_incomparable(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00Z",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "2026-07-14T19:05:00",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "not_terminal"


def test_extreme_timestamp_is_incomparable(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00Z",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "0001-01-01T00:00:00+23:59",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "not_terminal"


def test_equal_instants_with_different_offsets_do_not_reseed(tmp_path):
    auth = _write_auth(tmp_path, {"nous": {
        **_healthy_nous_state(),
        "obtained_at": "2026-07-14T19:00:00Z",
    }})
    seed = json.dumps({
        "providers": {
            "nous": {
                "client_id": "hermes-cli-vps",
                "access_token": "FRESH-at",
                "refresh_token": "FRESH-rt",
                "obtained_at": "2026-07-14T20:00:00+01:00",
            }
        },
    })

    assert mod.reseed_if_terminal(auth, seed) == "not_terminal"


def test_preserves_other_providers(tmp_path):
    """Re-seed swaps ONLY providers.nous; other providers survive intact."""
    auth = _write_auth(tmp_path, {
        "nous": _terminal_nous_state(),
        "openai-codex": {"tokens": {"access_token": "codex-at"}},
    })
    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "reseeded"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["openai-codex"]["tokens"]["access_token"] == "codex-at"
    assert store["providers"]["nous"]["refresh_token"] == "FRESH-rt"


def test_no_seed_is_noop(tmp_path):
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    assert mod.reseed_if_terminal(auth, "") == "no_seed"


def test_bad_seed_is_noop(tmp_path):
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    assert mod.reseed_if_terminal(auth, "}{not json") == "bad_seed"
    # Original terminal entry left untouched.
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["last_auth_error"]["relogin_required"] is True


def test_seed_without_nous_entry_is_noop(tmp_path):
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    seed = json.dumps({"version": 1, "providers": {"openai-codex": {}}})
    assert mod.reseed_if_terminal(auth, seed) == "bad_seed"


def test_absent_auth_file_defers_to_bootstrap(tmp_path):
    """No auth.json → blank volume; the normal *_BOOTSTRAP path handles it."""
    auth = str(tmp_path / "auth.json")
    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "no_auth_file"


def test_unreadable_auth_file_is_left_alone(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text("}{ corrupt")
    assert mod.reseed_if_terminal(str(p), _FRESH_SEED) == "auth_unreadable"
    # Not overwritten.
    assert p.read_text() == "}{ corrupt"


def test_terminal_entry_missing_marker_is_not_terminal(tmp_path):
    """No last_auth_error at all (e.g. a merely-expired but not-quarantined
    entry) → not terminal, no re-seed."""
    auth = _write_auth(tmp_path, {"nous": {"client_id": "hermes-cli-vps"}})
    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "not_terminal"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode semantics")
def test_reseeded_file_is_0600_even_when_chmod_fails(tmp_path, monkeypatch):
    """The re-seeded auth.json must be 0o600 even if the post-write chmod
    raises. Before the fix the file was created with a plain open() at the
    process umask (0o666 under umask 0) and only tightened by a chmod whose
    OSError was silently swallowed — so a chmod failure left the fresh Nous
    refresh_token permanently world-readable."""
    old_umask = os.umask(0)
    try:
        auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})

        def _no_chmod(*_a, **_k):
            raise OSError("chmod unsupported on this volume")

        monkeypatch.setattr(mod.os, "chmod", _no_chmod)

        assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "reseeded"
        mode = stat.S_IMODE(os.stat(auth).st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    finally:
        os.umask(old_umask)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode semantics")
def test_reseeded_temp_is_never_created_group_or_world_readable(tmp_path, monkeypatch):
    """The tokens must never touch disk at a loose mode, even transiently.

    Spy on os.replace to capture the temp file's mode at rename time (i.e.
    the mode the secrets were written at). Before the fix the temp file was
    created via plain open() at the process umask, so this window was 0o666
    under umask 0."""
    old_umask = os.umask(0)
    try:
        auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})

        real_replace = mod.os.replace
        captured = {}

        def _spy_replace(src, dst, *a, **k):
            captured["mode"] = stat.S_IMODE(os.stat(src).st_mode)
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(mod.os, "replace", _spy_replace)

        assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "reseeded"
        assert captured["mode"] == 0o600, (
            f"tokens written to a temp file at {oct(captured['mode'])} — "
            "must be 0o600 with no group/world bits"
        )
    finally:
        os.umask(old_umask)
