"""Contract tests for the container Dockerfile.

These tests assert invariants about how the Dockerfile composes its runtime —
they deliberately avoid snapshotting specific package versions, line numbers,
or exact flag choices.  What they DO assert is that the Dockerfile maintains
the properties required for correct production behaviour:

- A PID-1 init is installed and wraps the entrypoint, so that orphaned
  subprocesses (MCP stdio servers, git, bun, browser daemons) get reaped
  instead of accumulating as zombies (#15012).
- Signal forwarding runs through the init so ``docker stop`` triggers
  hermes's own graceful-shutdown path.

The init can be any reaper-capable PID-1: the historical lineage was
``tini``; the current image uses s6-overlay's ``/init`` (which execs
``s6-svscan`` as PID 1, with the same SIGCHLD-reaping property). The
checks below accept either family — the contract is behavioural, not
nominal.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


# Init-process families this repo accepts as PID 1. ``tini`` /
# ``dumb-init`` / ``catatonit`` are classic minimal reapers; s6-overlay
# ships ``/init`` which execs ``s6-svscan`` as PID 1 (same reaper
# contract, plus supervision of declared services). Either family
# satisfies the zombie-reaping invariant — see issue #15012.
_KNOWN_INIT_TOKENS: tuple[str, ...] = (
    "tini",
    "dumb-init",
    "catatonit",
    "s6-overlay",
    "s6-svscan",
    "/init",
)


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present in this checkout")
    return DOCKERFILE.read_text()


def _dockerfile_instructions(dockerfile_text: str) -> list[str]:
    instructions: list[str] = []
    current = ""

    for raw_line in dockerfile_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        continued = line.removesuffix("\\").strip()
        current = f"{current} {continued}".strip()
        if not line.endswith("\\"):
            instructions.append(current)
            current = ""

    return instructions


def _run_steps(dockerfile_text: str) -> list[str]:
    return [
        instruction
        for instruction in _dockerfile_instructions(dockerfile_text)
        if instruction.startswith("RUN ")
    ]


def _instruction_text(dockerfile_text: str) -> str:
    """Join every non-comment Dockerfile instruction into one searchable
    string. Crucially excludes comments — otherwise the historical
    explanation of "we used to use tini" would silently satisfy a
    substring check long after tini was removed from the build.
    """
    return "\n".join(_dockerfile_instructions(dockerfile_text))


def test_dockerfile_installs_an_init_for_zombie_reaping(dockerfile_text):
    """Some init (tini, dumb-init, catatonit, s6-overlay) must be installed.

    Without a PID-1 init that handles SIGCHLD, hermes accumulates zombie
    processes from MCP stdio subprocesses, git operations, browser
    daemons, etc.  In long-running Docker deployments this eventually
    exhausts the PID table.
    """
    # Accept any of the common reapers.  The contract is behavioural:
    # something must be installed that reaps orphans.
    #
    # Scan instructions only (no comments) so a stale historical mention
    # in a comment can't masquerade as a current install. Without this,
    # removing tini from the actual build but leaving the word in a
    # comment would silently keep the test green.
    instructions = _instruction_text(dockerfile_text)
    installed = any(name in instructions for name in _KNOWN_INIT_TOKENS)
    assert installed, (
        "No PID-1 init detected in Dockerfile instructions (looked for: "
        f"{', '.join(_KNOWN_INIT_TOKENS)}). Without an init process to "
        "reap orphaned subprocesses, hermes accumulates zombies in Docker "
        "deployments. See issue #15012."
    )


def test_dockerfile_entrypoint_routes_through_the_init(dockerfile_text):
    """The ENTRYPOINT must invoke the init, not the entrypoint script directly.

    Installing the init is only half the fix — the container must actually
    run with it as PID 1.  If the ENTRYPOINT executes the shell script
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

    routes_through_init = any(name in entrypoint_line for name in _KNOWN_INIT_TOKENS)
    assert routes_through_init, (
        f"ENTRYPOINT does not route through a PID-1 init: {entrypoint_line!r}. "
        f"Expected one of {_KNOWN_INIT_TOKENS}. If the init is installed but "
        "not wired into ENTRYPOINT, hermes still runs as PID 1 and zombies "
        "will accumulate (#15012)."
    )


def test_dockerfile_installs_tui_dependencies(dockerfile_text):
    # The TUI workspace manifests must be present so ``npm install`` can
    # resolve dependencies. The bundled ``hermes-ink`` workspace package is
    # now COPIED into the image as a whole tree (not just its lockfile)
    # because it's referenced as a ``file:`` workspace dependency from
    # ``ui-tui/package.json`` — copying the tree avoids npm stopping at a
    # bare ``package.json`` shell.
    assert "ui-tui/package.json" in dockerfile_text
    assert "ui-tui/package-lock.json" in dockerfile_text
    assert "ui-tui/packages/hermes-ink/" in dockerfile_text
    assert any(
        "ui-tui" in step and "npm" in step and (" install" in step or " ci" in step)
        for step in _run_steps(dockerfile_text)
    )


def test_dockerfile_preinstalls_gateway_messaging_dependencies(dockerfile_text):
    sync_steps = [
        step for step in _run_steps(dockerfile_text)
        if "uv sync" in step and "--no-install-project" in step
    ]

    assert sync_steps, "Dockerfile must install Python dependencies with uv sync"
    assert any("--extra messaging" in step for step in sync_steps), (
        "Published Docker images must preload the [messaging] extra so "
        "Telegram/Discord gateway adapters do not depend on first-boot "
        "lazy installation (#24698)."
    )


def test_dockerfile_builds_tui_assets(dockerfile_text):
    assert any(
        "ui-tui" in step and "npm" in step and "run build" in step
        for step in _run_steps(dockerfile_text)
    )


def test_dockerfile_materializes_local_tui_ink_package(dockerfile_text):
    # ``hermes-ink`` is a bundled workspace package referenced from
    # ``ui-tui/package.json`` via ``file:`` — not pulled from the npm
    # registry. The contract this test pins is just that the image
    # actually carries the package source so ``await import('@hermes/ink')``
    # can resolve at runtime; the previous, much pickier assertion (manual
    # ``rm -rf`` + ``npm install --omit=dev --prefix node_modules/@hermes/ink``)
    # baked in implementation details of an older materialisation flow that
    # was simplified once npm workspaces handled the resolution natively.
    assert "ui-tui/packages/hermes-ink/" in dockerfile_text, (
        "Dockerfile must COPY the bundled hermes-ink workspace package "
        "so ``await import('@hermes/ink')`` resolves at runtime."
    )


def test_dockerignore_excludes_nested_dependency_dirs():
    if not DOCKERIGNORE.exists():
        pytest.skip(".dockerignore not present in this checkout")

    text = DOCKERIGNORE.read_text()

    assert "**/node_modules" in text
    assert "**/.venv" in text
