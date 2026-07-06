from __future__ import annotations

import os
from pathlib import Path

import yaml


def test_profile_child_timeout_seconds_overrides_global():
    hermes_home = Path(os.environ["HERMES_HOME"])
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "delegation": {
                    "child_timeout_seconds": 120,
                    "profiles": {
                        "quick-review": {
                            "provider": "openai-codex",
                            "model": "gpt-5.5",
                            "child_timeout_seconds": 45,
                        },
                        "no-cap": {
                            "child_timeout_seconds": 0,
                        },
                        "inherits-global": {
                            "provider": "openai-codex",
                            "model": "gpt-5.5",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    from hermes_cli.config import load_config
    from tools.delegate_tool import _get_child_timeout, _merge_delegation_profile

    delegation = load_config()["delegation"]

    quick = _merge_delegation_profile(delegation, "quick-review")
    assert _get_child_timeout(quick) == 45.0

    no_cap = _merge_delegation_profile(delegation, "no-cap")
    assert _get_child_timeout(no_cap) is None

    inherited = _merge_delegation_profile(delegation, "inherits-global")
    assert _get_child_timeout(inherited) == 120.0

    root = _merge_delegation_profile(delegation, None)
    assert _get_child_timeout(root) == 120.0
