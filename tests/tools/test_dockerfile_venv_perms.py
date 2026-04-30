"""Contract tests for the runtime hermes-user's write access to the venv.

Issue #17636: when the container runs as the unprivileged ``hermes`` user
(the default for `docker compose up` and `docker run`), the in-image venv
``/opt/hermes/.venv`` was created by ``uv venv`` while the build was still
``USER root`` and therefore root-owned. ``hermes memory setup``,
``hermes skills install``, and any other path that calls
``uv pip install --python /opt/hermes/.venv/bin/python ...`` then failed
with ``EACCES`` on
``/opt/hermes/.venv/lib/.../site-packages/...``.

These tests assert two invariants needed for plugin installation to work
inside the container, regardless of whether the operator overrides
``HERMES_UID`` at runtime:

1. The Dockerfile transfers ownership of ``/opt/hermes/.venv`` to the
   ``hermes`` user before the image is finalised.
2. ``docker/entrypoint.sh`` re-chowns the venv whenever it remaps the
   ``hermes`` user's UID/GID — otherwise the build-time ownership becomes
   stale and the bug returns under ``HERMES_UID=$(id -u) docker compose
   up``.

The tests deliberately avoid snapshotting line numbers or exact flag
choices (mirroring the style of test_dockerfile_pid1_reaping.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present in this checkout")
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def entrypoint_text() -> str:
    if not ENTRYPOINT.exists():
        pytest.skip("docker/entrypoint.sh not present in this checkout")
    return ENTRYPOINT.read_text()


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


def test_dockerfile_makes_venv_writable_by_hermes_user(dockerfile_text):
    """The image must hand ``/opt/hermes/.venv`` ownership to ``hermes``.

    Either the venv is created while the runtime user owns it, or a
    subsequent ``chown`` step restores ownership. Without one of these,
    ``uv pip install`` from the running container (e.g. ``hermes memory
    setup``) fails with EACCES (#17636).
    """
    venv_owned_by_hermes = any(
        ".venv" in step
        and (
            ("chown" in step and "hermes" in step)
            or ("gosu hermes" in step and "uv venv" in step)
        )
        for step in _run_steps(dockerfile_text)
    )

    assert venv_owned_by_hermes, (
        "Dockerfile leaves /opt/hermes/.venv owned by root: no `chown ... "
        "hermes ... .venv` step and no `gosu hermes uv venv` step found. "
        "Plugin installs (`hermes memory setup`, skills install) fail with "
        "EACCES under the default unprivileged hermes user. See #17636."
    )


def test_entrypoint_rechowns_venv_when_hermes_uid_is_remapped(entrypoint_text):
    """When the entrypoint remaps the hermes UID, the venv must follow.

    ``docker compose up`` defaults to ``HERMES_UID=$(id -u)``. After
    ``usermod`` changes the runtime UID, files owned by the build-time
    UID 10000 are no longer writable by the now-renumbered hermes user.
    The existing ``$HERMES_HOME`` chown handles the data volume; the
    venv lives outside the volume and needs the same treatment.
    """
    chown_targets = [
        line.strip()
        for line in entrypoint_text.splitlines()
        if "chown" in line and "hermes" in line and ".venv" in line
    ]

    assert chown_targets, (
        "docker/entrypoint.sh does not chown the venv to the hermes user. "
        "When HERMES_UID is remapped, /opt/hermes/.venv stays owned by the "
        "stale build-time UID and `hermes memory setup` fails with EACCES "
        "(#17636). Extend the existing $HERMES_HOME chown block to also "
        "chown $INSTALL_DIR/.venv."
    )
