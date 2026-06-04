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
    """Extract the ``if [ ! -f "$HERMES_HOME/gateway_state.json" ] && … fi``
    block that seeds the gateway state file from the bootstrap env var."""
    m = re.search(
        r'(if \[ ! -f "\$HERMES_HOME/gateway_state\.json" \] && \\\n'
        r"(?:.*\n)*?fi)",
        text,
    )
    assert m, (
        "stage2-hook.sh must contain the gateway_state.json bootstrap-seed block "
        "guarded on HERMES_GATEWAY_BOOTSTRAP_STATE"
    )
    return m.group(1)


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
