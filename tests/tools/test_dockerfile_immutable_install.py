"""Contract tests for the Docker image's immutable /opt/hermes install tree."""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text()


def test_dockerfile_makes_opt_hermes_root_owned_and_non_writable() -> None:
    text = _dockerfile_text()

    assert "COPY --chown=hermes:hermes . ." not in text
    assert "COPY . ." in text
    assert "chown -R root:root /opt/hermes" in text
    assert "chmod -R a+rX /opt/hermes" in text
    assert "chmod -R a-w /opt/hermes" in text

    immutable_block = re.search(
        r"RUN mkdir -p /opt/hermes/bin && \\\n"
        r"(?:.*\\\n)+?"
        r"\s+chmod -R a-w /opt/hermes",
        text,
    )
    assert immutable_block, "Dockerfile must lock /opt/hermes after installing code/deps"


def test_dockerfile_keeps_mutable_state_under_opt_data() -> None:
    text = _dockerfile_text()

    assert "ENV HERMES_HOME=/opt/data" in text
    assert "ENV HERMES_WRITE_SAFE_ROOT=/opt/data" in text
    assert 'VOLUME [ "/opt/data" ]' in text


def test_dockerfile_disables_runtime_install_mutations() -> None:
    text = _dockerfile_text()

    assert "ENV PYTHONDONTWRITEBYTECODE=1" in text
    assert "ENV HERMES_DISABLE_LAZY_INSTALLS=1" in text
    assert "HERMES_TUI_DIR=/opt/hermes/ui-tui" in text


def test_dockerfile_does_not_chown_install_trees_to_hermes() -> None:
    text = _dockerfile_text()
    forbidden_patterns = (
        r"chown\s+-R\s+hermes:hermes\s+/opt/hermes/\.venv",
        r"chown\s+-R\s+hermes:hermes\s+/opt/hermes/ui-tui",
        r"chown\s+-R\s+hermes:hermes\s+/opt/hermes/gateway",
        r"chown\s+-R\s+hermes:hermes\s+/opt/hermes/node_modules",
    )
    for pattern in forbidden_patterns:
        assert not re.search(pattern, text), (
            "runtime install trees under /opt/hermes must stay immutable; "
            f"found forbidden pattern {pattern!r}"
        )
