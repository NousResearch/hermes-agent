"""Cross-surface approval mode/timeout parity invariant.

The approval mode (``approvals.mode``) and timeout (``approvals.timeout``)
must resolve identically on every surface that consults them:

  - the canonical core: ``tools.approval_context._get_approval_mode`` /
    ``tools.approval_context._get_approval_timeout``
  - the TUI gateway: ``tui_gateway.server._load_approval_mode`` (delegates
    to the core as of the decision-core migration)
  - the codex app-server surface: ``agent/codex_runtime.py`` feeds
    ``auto_approve_*`` from ``tools.approval.is_approval_bypass_active()``,
    which itself reads the core resolver — so parity there reduces to
    ``is_approval_bypass_active() == (mode == "off")`` when no yolo
    source is active.

Historic drift class: tui_gateway re-read config raw and normalized
locally (see commits f9cd577915, 1e652cca7a, bd246db10d — repeated parity
re-alignments). This test pins the invariant so drift regressions fail
loudly instead of silently disagreeing per surface.

There is no per-platform ``approvals.mode`` override in the config schema;
mode/timeout are global, so the synthetic configs below cover global-set,
unset (defaults), and malformed values.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture()
def tui_server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        yield importlib.import_module("tui_gateway.server")


def _write_config(home, yaml_text: str | None) -> None:
    cfg = home / "config.yaml"
    if yaml_text is None:
        if cfg.exists():
            cfg.unlink()
    else:
        cfg.write_text(yaml_text, encoding="utf-8")


# (config yaml, expected mode, expected timeout)
CASES = [
    pytest.param(None, "smart", 300, id="unset-defaults"),
    pytest.param(
        "approvals:\n  mode: manual\n", "manual", 300, id="global-manual"
    ),
    pytest.param(
        "approvals:\n  mode: smart\n  timeout: 120\n",
        "smart",
        120,
        id="global-smart-timeout",
    ),
    pytest.param(
        # YAML 1.1 parses bare OFF as boolean False; the normalizer maps
        # False -> "off". Both surfaces must agree on that quirk.
        "approvals:\n  mode: OFF\n  timeout: 45\n",
        "off",
        45,
        id="yaml-bool-off",
    ),
    pytest.param(
        "approvals:\n  mode: bogus-value\n  timeout: not-a-number\n",
        "manual",
        300,
        id="malformed-values",
    ),
    pytest.param(
        "approvals:\n  mode: '  Smart  '\n", "smart", 300, id="whitespace-case"
    ),
]


def _approval_module():
    """Resolve tools.approval via sys.modules, not the package attribute.

    The ``tui_server`` fixture's ``patch.dict("sys.modules", ...)`` purges
    modules imported during its block at teardown; ``from tools import
    approval`` can then hand back a stale attribute cached on the ``tools``
    package while the server re-imports a fresh module object. Going
    through ``importlib.import_module`` keeps the test and the server on
    the same sys.modules entry.
    """
    return importlib.import_module("tools.approval")


@pytest.mark.parametrize("yaml_text,expected_mode,expected_timeout", CASES)
def test_mode_and_timeout_parity_across_surfaces(
    hermes_home, tui_server, yaml_text, expected_mode, expected_timeout
):
    approval_mod = _approval_module()

    _write_config(hermes_home, yaml_text)

    ctx = importlib.import_module("tools.approval_context")
    core_mode = ctx._get_approval_mode()
    core_timeout = ctx._get_approval_timeout()
    tui_mode = tui_server._load_approval_mode()

    # Canonical resolver matches expectations.
    assert core_mode == expected_mode
    assert core_timeout == expected_timeout

    # TUI surface returns the identical mode (delegation invariant).
    assert tui_mode == core_mode

    # Codex surface: auto-approve routing is derived from
    # is_approval_bypass_active(), which must equal (mode == "off")
    # whenever no yolo source is active in this process.
    if not approval_mod._YOLO_MODE_FROZEN:
        with patch.object(
            approval_mod, "is_current_session_yolo_enabled", return_value=False
        ):
            assert approval_mod.is_approval_bypass_active() == (
                core_mode == "off"
            )


def test_tui_loader_delegates_to_core(hermes_home, tui_server):
    """The TUI must not re-resolve mode itself — it delegates to the core.

    Pin the delegation seam directly: patching the core resolver changes
    what the TUI reports, proving there is no independent config read left.
    """
    approval_context = importlib.import_module("tools.approval_context")

    with patch.object(approval_context, "_get_approval_mode", return_value="smart"):
        assert tui_server._load_approval_mode() == "smart"
    with patch.object(approval_context, "_get_approval_mode", return_value="off"):
        assert tui_server._load_approval_mode() == "off"
    # Defensive clamp: an out-of-vocabulary value from the core is coerced
    # to manual rather than leaking an unknown mode to the TUI client.
    with patch.object(
        approval_context, "_get_approval_mode", return_value="weird"
    ):
        assert tui_server._load_approval_mode() == "manual"
