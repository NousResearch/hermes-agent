"""Contract tests for the safe Docker Compose smoke profile.

These tests protect the beginner-friendly Docker trial path: users should be
able to build/run a one-shot container without mounting their live ~/.hermes
profile, starting a gateway, exposing host networking, or creating a persistent
restart loop.  The smoke profile is intentionally boring and reversible.
"""
from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


def test_compose_has_isolated_smoke_profile() -> None:
    services = _compose()["services"]

    assert "smoke" in services, (
        "docker-compose.yml should include a one-shot `smoke` service so users "
        "can validate the image without touching the live gateway profile."
    )

    smoke = services["smoke"]
    assert "smoke" in smoke.get("profiles", [])
    assert smoke.get("command") == ["doctor"]
    assert smoke.get("restart", "no") == "no"
    assert smoke.get("network_mode") != "host"


def test_smoke_profile_mounts_only_local_throwaway_data() -> None:
    smoke = _compose()["services"]["smoke"]
    volumes = smoke.get("volumes", [])
    joined = "\n".join(volumes)

    assert "~/.hermes" not in joined, (
        "the smoke profile must not mount the operator's live ~/.hermes data"
    )
    assert "./runtime/docker-smoke:/opt/data" in joined, (
        "the smoke profile should use a repo-local throwaway data directory"
    )


def test_smoke_profile_does_not_start_live_gateway() -> None:
    smoke = _compose()["services"]["smoke"]
    command = " ".join(smoke.get("command", []))

    assert "gateway run" not in command
    assert not any("API_SERVER_KEY" in item for item in smoke.get("environment", []))
