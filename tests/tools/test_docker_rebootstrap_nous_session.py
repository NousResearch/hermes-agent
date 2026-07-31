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
import subprocess
import sys
from pathlib import Path, PureWindowsPath

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


def test_auth_layout_matches_default_profile_and_path_equal_mapping(tmp_path):
    runtime = tmp_path / "runtime"
    residence = tmp_path / "residence"
    profile = runtime / "profiles" / "work"

    assert mod.resolve_auth_layout(str(runtime), False) == (
        runtime.resolve(),
        runtime.resolve(),
    )
    assert mod.resolve_auth_layout(str(runtime), True, str(residence)) == (
        residence.resolve(),
        residence.resolve(),
    )
    assert mod.resolve_auth_layout(str(profile), True, str(residence)) == (
        residence.resolve(),
        residence.resolve() / "profiles" / "work",
    )
    assert mod.resolve_auth_layout(str(profile), True, str(profile)) == (
        profile.resolve(),
        profile.resolve(),
    )


def test_auth_layout_rejects_os_home_and_its_ancestor(monkeypatch, tmp_path):
    operator_home = tmp_path / "operator" / "home"
    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HOME", str(operator_home))

    for residence in (operator_home, operator_home.parent):
        with pytest.raises(ValueError, match="OS user home"):
            mod.resolve_auth_layout(
                str(runtime_home),
                True,
                str(residence),
            )


def test_auth_layout_uses_real_home_when_home_is_profile_scoped(
    monkeypatch, tmp_path
):
    runtime_home = tmp_path / "runtime"
    profile_home = runtime_home / "home"
    profile_home.mkdir(parents=True)
    operator_home = tmp_path / "operator"
    monkeypatch.setenv("HOME", str(profile_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(operator_home))

    with pytest.raises(ValueError, match="OS user home"):
        mod.resolve_auth_layout(
            str(runtime_home),
            True,
            str(operator_home),
        )


def test_auth_layout_rejects_runtime_ancestor(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_home = runtime_root / "sessions" / "work"
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))

    with pytest.raises(ValueError, match="must not contain HERMES_HOME"):
        mod.resolve_auth_layout(
            str(runtime_home),
            True,
            str(runtime_root),
        )


def test_auth_layout_boundary_validation_preserves_supported_layouts(
    monkeypatch, tmp_path
):
    operator_home = tmp_path / "operator"
    runtime_home = tmp_path / "runtime"
    monkeypatch.setenv("HOME", str(operator_home))

    assert mod.resolve_auth_layout(
        str(operator_home),
        True,
        str(operator_home),
    ) == (operator_home.resolve(), operator_home.resolve())

    for residence in (operator_home / ".hermes-auth", tmp_path / "auth-residence"):
        assert mod.resolve_auth_layout(
            str(runtime_home),
            True,
            str(residence),
        ) == (residence.resolve(), residence.resolve())

    profile_home = runtime_home / "profiles" / "work"
    assert mod.resolve_auth_layout(
        str(profile_home),
        True,
        str(runtime_home),
    ) == (
        runtime_home.resolve(),
        profile_home.resolve(),
    )

    residence = tmp_path / "profile-auth-residence"
    assert mod.resolve_auth_layout(
        str(profile_home),
        True,
        str(residence),
    ) == (
        residence.resolve(),
        residence.resolve() / "profiles" / "work",
    )


@pytest.mark.parametrize("named_profile", (False, True))
@pytest.mark.parametrize(
    "override",
    (
        None,
        "",
        "   ",
        "relative/auth",
        "~/auth",
        " /absolute/with-leading-space",
        "/absolute/with-trailing-space ",
        "bad\npath",
        "/",
    ),
)
def test_docker_layout_matches_python_strict_resolution(
    monkeypatch, tmp_path, named_profile, override
):
    from hermes_constants import get_hermes_auth_home_for

    runtime = tmp_path / "runtime"
    if named_profile:
        runtime = runtime / "profiles" / "work"
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    if override is None:
        monkeypatch.delenv("HERMES_AUTH_HOME", raising=False)
    else:
        monkeypatch.setenv("HERMES_AUTH_HOME", override)

    python_error = None
    helper_error = None
    try:
        python_path = get_hermes_auth_home_for(runtime)
    except ValueError as exc:
        python_error = exc
        python_path = None
    try:
        _root, helper_path = mod.resolve_auth_layout(
            str(runtime),
            override is not None,
            override or "",
        )
    except ValueError as exc:
        helper_error = exc
        helper_path = None

    assert bool(helper_error) == bool(python_error)
    if python_error is None:
        assert helper_path == python_path


def test_auth_layout_canonicalizes_symlinked_residence_root(tmp_path):
    runtime = tmp_path / "runtime" / "profiles" / "work"
    target = tmp_path / "residence-target"
    target.mkdir()
    link = tmp_path / "residence-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    residence, auth_dir = mod.resolve_auth_layout(str(runtime), True, str(link))

    assert residence == target.resolve()
    assert auth_dir == target.resolve() / "profiles" / "work"


@pytest.mark.parametrize(
    "value",
    (
        "",
        "   ",
        "relative/auth",
        "~/auth",
        " /absolute/with-padding",
        "bad\npath",
        "/",
    ),
)
def test_auth_layout_rejects_invalid_explicit_override(tmp_path, value):
    with pytest.raises(ValueError):
        mod.resolve_auth_layout(str(tmp_path / "runtime"), True, value)


def test_filesystem_anchor_predicate_covers_windows_drive_and_unc_roots():
    import hermes_constants

    roots = (
        PureWindowsPath("C:/"),
        PureWindowsPath("//server/share/"),
    )
    ordinary_directories = (
        PureWindowsPath("C:/auth"),
        PureWindowsPath("//server/share/auth"),
    )

    for path in roots:
        assert hermes_constants._is_filesystem_anchor(path)
        assert mod._is_filesystem_anchor(path)
    for path in ordinary_directories:
        assert not hermes_constants._is_filesystem_anchor(path)
        assert not mod._is_filesystem_anchor(path)


def test_auth_layout_rejects_existing_non_directory_and_symlink_loop(tmp_path):
    runtime = tmp_path / "runtime"
    not_directory = tmp_path / "auth-file"
    not_directory.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        mod.resolve_auth_layout(str(runtime), True, str(not_directory))

    loop = tmp_path / "loop"
    try:
        loop.symlink_to(loop)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(ValueError):
        mod.resolve_auth_layout(str(runtime), True, str(loop))


def test_layout_subprocess_maps_named_profile_for_stage2(tmp_path):
    runtime = tmp_path / "runtime" / "profiles" / "work"
    residence = tmp_path / "residence"
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(runtime),
            "HERMES_AUTH_HOME": str(residence),
        }
    )

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--resolve-auth-layout"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(residence.resolve()),
        str(residence.resolve() / "profiles" / "work"),
        "distinct",
        str(residence.resolve() / "shared"),
    ]


@pytest.mark.parametrize("override_kind", ("unset", "profile", "root"))
def test_layout_subprocess_keeps_shared_store_at_runtime_root_for_no_op(
    tmp_path, override_kind
):
    runtime_root = tmp_path / "runtime"
    runtime = runtime_root / "profiles" / "work"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(runtime)
    if override_kind == "profile":
        env["HERMES_AUTH_HOME"] = str(runtime)
    elif override_kind == "root":
        env["HERMES_AUTH_HOME"] = str(runtime_root)
    else:
        env.pop("HERMES_AUTH_HOME", None)

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--resolve-auth-layout"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected_residence = runtime_root if override_kind == "root" else runtime
    assert result.stdout.splitlines() == [
        str(expected_residence),
        str(runtime),
        "same",
        str(runtime_root / "shared"),
    ]


def test_invalid_override_never_rebootstraps_runtime_fallback(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    auth_path = Path(_write_auth(runtime, {"nous": _terminal_nous_state()}))
    before = auth_path.read_bytes()
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(runtime),
            "HERMES_AUTH_HOME": "relative/auth",
            mod.REBOOTSTRAP_ENV: _FRESH_SEED,
        }
    )

    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid auth residence" in result.stderr
    assert auth_path.read_bytes() == before


def test_reseeds_terminal_entry(tmp_path):
    """Terminal on-disk entry + valid seed → providers.nous replaced."""
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    result = mod.reseed_if_terminal(auth, _FRESH_SEED)
    assert result == "reseeded"
    store = json.loads(Path(auth).read_text())
    assert store["providers"]["nous"]["refresh_token"] == "FRESH-rt"
    assert "last_auth_error" not in store["providers"]["nous"]


def test_reseed_ignores_preplanted_legacy_temp_symlink(tmp_path):
    """The removed fixed temp name must never be opened, replaced, or cleaned."""
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    victim = tmp_path / "unrelated-runtime-file"
    victim.write_text("SAFE", encoding="utf-8")
    legacy_temp = tmp_path / "auth.json.rebootstrap.tmp"
    try:
        legacy_temp.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "reseeded"

    assert victim.read_text(encoding="utf-8") == "SAFE"
    assert legacy_temp.is_symlink()
    assert not Path(auth).is_symlink()
    assert not list(tmp_path.glob("auth.json.tmp.*"))


def test_reseed_temp_is_exclusive_owner_only_and_retries_collision(
    monkeypatch, tmp_path
):
    """A preplanted current-spelling symlink is not followed on collision."""
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    victim = tmp_path / "unrelated-runtime-file"
    victim.write_text("SAFE", encoding="utf-8")
    collision = tmp_path / f"auth.json.tmp.{os.getpid()}.collision"
    try:
        collision.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    ids = iter(("collision", "unique"))
    monkeypatch.setattr(
        mod.uuid,
        "uuid4",
        lambda: type("_UUID", (), {"hex": next(ids)})(),
    )
    real_replace = mod.os.replace
    observed_mode = None

    def inspect_then_replace(source, target):
        nonlocal observed_mode
        observed_mode = stat.S_IMODE(os.lstat(source).st_mode)
        real_replace(source, target)

    monkeypatch.setattr(mod.os, "replace", inspect_then_replace)

    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "reseeded"

    assert observed_mode == 0o600
    assert victim.read_text(encoding="utf-8") == "SAFE"
    assert collision.is_symlink()
    assert not (tmp_path / f"auth.json.tmp.{os.getpid()}.unique").exists()
    assert stat.S_IMODE(Path(auth).stat().st_mode) == 0o600


def test_reseed_cleans_unique_temp_when_replace_fails(monkeypatch, tmp_path):
    auth = _write_auth(tmp_path, {"nous": _terminal_nous_state()})
    before = Path(auth).read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        mod.reseed_if_terminal(auth, _FRESH_SEED)

    assert Path(auth).read_bytes() == before
    assert not list(tmp_path.glob("auth.json.tmp.*"))


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


def test_terminal_entry_missing_marker_is_not_terminal(tmp_path):
    """No last_auth_error at all (e.g. a merely-expired but not-quarantined
    entry) → not terminal, no re-seed."""
    auth = _write_auth(tmp_path, {"nous": {"client_id": "hermes-cli-vps"}})
    assert mod.reseed_if_terminal(auth, _FRESH_SEED) == "not_terminal"
