"""Security checks for Docker dashboard documentation."""

from pathlib import Path


DOCKER_DOC = Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "docker.md"


def test_dashboard_examples_publish_dashboard_on_localhost_only():
    """Dashboard docs must not publish the unauthenticated dashboard on all host interfaces."""
    content = DOCKER_DOC.read_text()

    assert "-p 127.0.0.1:9119:9119" in content
    assert '"127.0.0.1:9119:9119"' in content
    assert "-p 9119:9119" not in content
    assert '"9119:9119"' not in content
