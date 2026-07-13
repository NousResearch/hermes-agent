"""Dependency consistency for core vision support and its lazy fallback."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tools.lazy_deps import LAZY_DEPS


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_vision_lazy_pin_matches_core_pillow_pin() -> None:
    """The fallback installer must not downgrade the core vision dependency."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    pillow_requirement = next(
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.lower().startswith("pillow==")
    )

    assert LAZY_DEPS["tool.vision"] == (pillow_requirement,)
