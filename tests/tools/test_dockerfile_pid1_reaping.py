"""Contract tests for the container Dockerfile.

These tests assert invariants about how the Dockerfile composes its runtime —
they deliberately avoid snapshotting specific package versions, line numbers,
or exact flag choices.  What they DO assert is that the Dockerfile maintains
the properties required for correct production behaviour:

- A PID-1 init (tini) is installed and wraps the entrypoint, so that orphaned
  subprocesses (MCP stdio servers, git, bun, browser daemons) get reaped
  instead of accumulating as zombies (#15012).
- Signal forwarding runs through the init so ``docker stop`` triggers
  hermes's own graceful-shutdown path.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present in this checkout")
    return DOCKERFILE.read_text()


def test_dockerfile_installs_an_init_for_zombie_reaping(dockerfile_text):
    """Some init (tini, dumb-init, catatonit) must be installed.

    Without a PID-1 init that handles SIGCHLD, hermes accumulates zombie
    processes from MCP stdio subprocesses, git operations, browser
    daemons, etc.  In long-running Docker deployments this eventually
    exhausts the PID table.
    """
    # Accept any of the common reapers.  The contract is behavioural:
    # something must be installed that reaps orphans.
    known_inits = ("tini", "dumb-init", "catatonit")
    installed = any(name in dockerfile_text for name in known_inits)
    assert installed, (
        "No PID-1 init detected in Dockerfile (looked for: "
        f"{', '.join(known_inits)}). Without an init process to reap "
        "orphaned subprocesses, hermes accumulates zombies in Docker "
        "deployments. See issue #15012."
    )


def test_dockerfile_entrypoint_routes_through_the_init(dockerfile_text):
    """The ENTRYPOINT must invoke the init, not the entrypoint script directly.

    Installing tini is only half the fix — the container must actually run
    with tini as PID 1.  If the ENTRYPOINT executes the shell script
    directly, the shell becomes PID 1 and will ``exec`` into hermes,
    which then runs as PID 1 without any zombie reaping.
    """
    # Find the last uncommented ENTRYPOINT line — Docker honours the final one.
    entrypoint_line = None
    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if line.startswith("ENTRYPOINT"):
            entrypoint_line = line

    assert entrypoint_line is not None, "Dockerfile is missing an ENTRYPOINT directive"

    known_inits = ("tini", "dumb-init", "catatonit")
    routes_through_init = any(name in entrypoint_line for name in known_inits)
    assert routes_through_init, (
        f"ENTRYPOINT does not route through an init: {entrypoint_line!r}. "
        "If tini is only installed but not wired into ENTRYPOINT, hermes "
        "still runs as PID 1 and zombies will accumulate (#15012)."
    )
