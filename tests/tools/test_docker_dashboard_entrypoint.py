import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = PROJECT_ROOT / "docker" / "entrypoint.sh"
DOCKER_DOCS = PROJECT_ROOT / "website" / "docs" / "user-guide" / "docker.md"


def test_dashboard_entrypoint_is_loopback_by_default_and_does_not_auto_insecure():
    """HERMES_DASHBOARD=1 must not silently expose the admin dashboard."""
    text = ENTRYPOINT.read_text()

    assert re.search(r'dash_host="\$\{HERMES_DASHBOARD_HOST:-127\.0\.0\.1\}"', text)
    assert not re.search(r'dash_host="\$\{HERMES_DASHBOARD_HOST:-0\.0\.0\.0\}"', text)
    assert re.search(
        r'case "\$\{HERMES_DASHBOARD_INSECURE:-\}" in\s+'
        r'1\|true\|TRUE\|True\|yes\|YES\|Yes\)\s+'
        r'dash_args\+=\(--insecure\)',
        text,
        re.DOTALL,
    )
    assert "--insecure" not in re.sub(
        r'case "\$\{HERMES_DASHBOARD_INSECURE:-\}" in.*?esac',
        "",
        text,
        flags=re.DOTALL,
    )
    assert 'if [ "$dash_host" != "127.0.0.1" ]' not in text


def test_docker_dashboard_docs_use_loopback_published_port_and_explicit_opt_in():
    """Docs must not recommend publishing the token-authenticated dashboard publicly."""
    text = DOCKER_DOCS.read_text()

    assert "127.0.0.1:9119:9119" in text
    assert "HERMES_DASHBOARD_INSECURE=1" in text
    assert "Do not publish the dashboard on an internet-facing interface." in text
    assert '"9119:9119"' not in text
    assert "-p 9119:9119" not in text
