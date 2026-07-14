"""Contract test: the s6-overlay stage2 hook seeds gateway_state.json from
HERMES_GATEWAY_BOOTSTRAP_STATE on first boot, so a freshly-provisioned
container can come up with the gateway already running.

Background. On a blank volume there is no gateway_state.json, so the boot
reconciler (cont-init.d/02-reconcile-profiles ->
container_boot.reconcile_profile_gateways) registers the gateway-default s6
slot but leaves it DOWN — it only auto-starts when the last recorded state was
"running". A container provisioned on a fresh volume therefore comes up with
the gateway down until something starts it.

An orchestrator that wants the gateway running from first boot sets
HERMES_GATEWAY_BOOTSTRAP_STATE=running; stage2-hook.sh (installed as
/etc/cont-init.d/01-hermes-setup, which runs lexicographically BEFORE
02-reconcile-profiles) seeds the state file so the reconciler sees
prior_state=running and brings the slot up on the very first boot.

This mirrors the existing HERMES_AUTH_JSON_BOOTSTRAP env-seed pattern: it seeds
the SAME gateway_state.json the reconciler already consults, guarded by
``[ ! -f ]`` so persisted runtime state always wins on subsequent boots (a
deliberately-stopped gateway must stay stopped across restarts).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


@pytest.fixture(scope="module")
def stage2_text() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    return STAGE2_HOOK.read_text()


def _seed_block(text: str) -> str:
    """Extract the gateway_state.json bootstrap block."""
    start = text.index('if [ ! -f "$HERMES_HOME/gateway_state.json" ] && \\')
    end = text.index("\n\n# --- Sync bundled skills ---", start)
    return text[start:end]


def _auth_seed_block(text: str) -> str:
    start = text.index(
        'if [ ! -f "$HERMES_HOME/auth.json" ] && '
        '[ -n "${HERMES_AUTH_JSON_BOOTSTRAP:-}" ]; then'
    )
    end = text.index("\n\n# gateway_state.json:", start)
    return text[start:end]


def _path_guard_functions(text: str) -> str:
    start = text.index("path_has_symlink_component() {")
    end = text.index("\n\nchown_hermes_tree() {", start)
    return text[start:end]


def test_seed_block_present_and_guarded(stage2_text: str) -> None:
    block = _seed_block(stage2_text)
    # Must be a first-boot-only seed (the [ ! -f ] guard) keyed on the env var.
    assert '[ ! -f "$HERMES_HOME/gateway_state.json" ]' in block, (
        "seed must be guarded by [ ! -f ] so persisted state wins on restart"
    )
    assert "HERMES_GATEWAY_BOOTSTRAP_STATE" in block
    assert "gateway_state" in block


def _run_seed(
    text: str, *, env_value: str | None, preexisting: str | None
) -> str | None:
    """Run the extracted seed block in a sandbox $HERMES_HOME.

    ``env_value`` is the HERMES_GATEWAY_BOOTSTRAP_STATE value (None = unset).
    ``preexisting`` is the contents of a gateway_state.json placed before the
    block runs (None = no file). Returns the file's contents afterwards, or
    None if it doesn't exist. ``chown``/``chmod`` are stubbed so the block
    runs without real root.
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _seed_block(text)

    with tempfile.TemporaryDirectory() as d:
        dpath = Path(d)
        home = dpath / "home"
        home.mkdir()
        state_file = home / "gateway_state.json"
        if preexisting is not None:
            state_file.write_text(preexisting)

        env_line = (
            f'export HERMES_GATEWAY_BOOTSTRAP_STATE="{env_value}"\n'
            if env_value is not None
            else "unset HERMES_GATEWAY_BOOTSTRAP_STATE\n"
        )
        script = (
            "set -e\n"
            f'HERMES_HOME="{home}"\n'
            f"{_path_guard_functions(text)}\n"
            # Stub privilege ops — the sandbox isn't root.
            "chown() { :; }\n"
            "chmod() { :; }\n"
            + env_line
            + block
        )
        script_path = dpath / "harness.sh"
        script_path.write_text(script)

        proc = subprocess.run(
            [bash, str(script_path)], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr

        if not state_file.exists():
            return None
        return state_file.read_text()


def test_seeds_running_state_on_blank_volume(stage2_text: str) -> None:
    """env=running + no pre-existing file -> writes a valid running state."""
    out = _run_seed(stage2_text, env_value="running", preexisting=None)
    assert out is not None, "seed must create gateway_state.json"
    assert json.loads(out).get("gateway_state") == "running"


def test_does_not_clobber_existing_state(stage2_text: str) -> None:
    """The [ ! -f ] guard: an existing state file is never overwritten, even
    when the bootstrap env var says running. A deliberately-stopped gateway
    must stay stopped across restarts."""
    existing = json.dumps({"gateway_state": "stopped", "pid": 123})
    out = _run_seed(stage2_text, env_value="running", preexisting=existing)
    assert out == existing, "seed must not clobber a persisted state file"


def test_does_not_seed_gateway_state_through_symlink(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    """A dangling gateway_state.json symlink must not become a host write."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _seed_block(stage2_text)

    home = tmp_path / "home"
    home.mkdir()
    outside_state = tmp_path / "outside-gateway-state.json"
    state_file = home / "gateway_state.json"
    try:
        state_file.symlink_to(outside_state)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")

    script = (
        "set -e\n"
        f'HERMES_HOME="{home}"\n'
        f"{_path_guard_functions(stage2_text)}\n"
        "chown() { :; }\n"
        "chmod() { :; }\n"
        'export HERMES_GATEWAY_BOOTSTRAP_STATE="running"\n'
        + block
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not outside_state.exists()
    assert state_file.is_symlink()
    assert "refusing seed through symlinked path" in proc.stdout


def test_does_not_seed_auth_json_through_symlink(
    stage2_text: str,
    tmp_path: Path,
) -> None:
    """A dangling auth.json symlink must not become a host write."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    block = _auth_seed_block(stage2_text)

    home = tmp_path / "home"
    home.mkdir()
    outside_auth = tmp_path / "outside-auth.json"
    auth_file = home / "auth.json"
    try:
        auth_file.symlink_to(outside_auth)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are not available on this platform")

    script = (
        "set -e\n"
        f'HERMES_HOME="{home}"\n'
        f"{_path_guard_functions(stage2_text)}\n"
        "chown() { :; }\n"
        "chmod() { :; }\n"
        'export HERMES_AUTH_JSON_BOOTSTRAP="{\\"ok\\": true}"\n'
        + block
    )
    script_path = tmp_path / "harness.sh"
    script_path.write_text(script)

    proc = subprocess.run([bash, str(script_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not outside_auth.exists()
    assert auth_file.is_symlink()
    assert "refusing seed through symlinked path" in proc.stdout


def test_no_seed_when_env_unset(stage2_text: str) -> None:
    """No env var -> no file written (preserves the default down-on-first-boot
    behaviour for orchestrators that don't opt in)."""
    out = _run_seed(stage2_text, env_value=None, preexisting=None)
    assert out is None, "seed must not run when HERMES_GATEWAY_BOOTSTRAP_STATE is unset"


def test_non_running_value_ignored(stage2_text: str) -> None:
    """Only a literal "running" is honoured; any other value is ignored so a
    typo can't write a bogus state. (The reconciler's _AUTOSTART_STATES is
    exactly {"running"}.)"""
    for bogus in ("stopped", "Running", "1", "true", "starting"):
        out = _run_seed(stage2_text, env_value=bogus, preexisting=None)
        assert out is None, (
            f"only 'running' should seed a state file, not {bogus!r}"
        )
