"""Regression checks for user-facing Docker commands in the documentation."""

from pathlib import Path


DOCKER_GUIDE = Path(__file__).resolve().parents[2] / "website/docs/user-guide/docker.md"


def test_health_check_uses_a_fresh_container_for_version_check():
    """The health-check command must not assume a running container has the CLI on PATH."""
    docs = DOCKER_GUIDE.read_text(encoding="utf-8")

    assert "docker run -it --rm nousresearch/hermes-agent:latest version" in docs
    assert "docker exec hermes hermes version" not in docs
