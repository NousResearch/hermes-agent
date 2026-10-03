"""Doc-fact contract: the ``approvals.mode`` default must agree on both sides.

The published docs (security.md, configuration.md) state ``smart`` is the
default approval mode. The effective default comes from
``DEFAULT_CONFIG["approvals"]["mode"]``, which the config loader deep-merges
under the user's ``approvals`` block — so an absent ``mode`` key still
resolves to the default rather than reaching the ``manual`` fail-safe in
``tools.approval_context._normalize_approval_mode``.

Issue #117341 reported the two sides disagreeing. These tests pin the
invariant in both directions so either side drifting fails loudly:

1. the default constant is ``smart`` (and stays a valid mode);
2. an ``approvals`` block that omits ``mode`` resolves to ``smart`` through
   the real merge path — the ``manual`` fail-safe is reserved for explicit
   invalid values;
3. the English docs declare ``smart`` as the default.

Only the English docs are gated; the i18n copies lag by design (precedent:
tests/website/test_slash_commands_doc_parity.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def config_home(tmp_path, monkeypatch):
    """A HERMES_HOME whose approvals block deliberately omits ``mode``."""
    import hermes_cli.config as hc

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\n"
        "approvals:\n  timeout: 15\n  smart_policy: x\n"
        "command_allowlist: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    yield home
    hc._LOAD_CONFIG_CACHE.clear()


def test_default_approval_mode_is_smart():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.approval_context import _VALID_MODES

    mode = DEFAULT_CONFIG["approvals"]["mode"]
    assert mode == "smart"
    assert mode in _VALID_MODES


def test_absent_mode_key_resolves_to_documented_default(config_home):
    """The merge must supply ``smart``; the ``manual`` fail-safe in
    _normalize_approval_mode is only for explicit invalid values."""
    import hermes_cli.config as hc
    from tools.approval_context import _get_approval_config, _get_approval_mode

    merged = hc.load_config_readonly()
    assert merged["approvals"]["mode"] == "smart"
    assert _get_approval_config().get("mode") == "smart"
    assert _get_approval_mode() == "smart"


def test_explicit_invalid_mode_still_fails_safe_to_manual(config_home, monkeypatch):
    import hermes_cli.config as hc
    from tools.approval_context import _get_approval_mode

    (config_home / "config.yaml").write_text(
        "approvals:\n  mode: sometimes\n", encoding="utf-8"
    )
    hc._LOAD_CONFIG_CACHE.clear()
    assert _get_approval_mode() == "manual"


def _mode_table_default(path: Path, valid: tuple) -> str:
    """The ``(default)`` marker of the approval-mode behavior table, e.g.
    ``| **smart** (default) | ...`` in security.md or ``| `smart` (default) | ...``
    in configuration.md. Only rows whose first cell is a real mode name count,
    so an unrelated ``(default)`` row elsewhere in the file can't match."""
    for cell in re.findall(
        r"^\|\s*\**`?(\w+)`?\**\s*\(default\)\s*\|",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    ):
        if cell in valid:
            return cell
    raise AssertionError(f"no approval-mode table row marks a default in {path.name}")


@pytest.mark.parametrize(
    "doc",
    ["website/docs/user-guide/security.md",
     "website/docs/user-guide/configuration.md"],
)
def test_docs_declare_smart_as_default_mode(doc):
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.approval_context import _VALID_MODES

    path = REPO_ROOT / doc
    text = path.read_text(encoding="utf-8")
    default = DEFAULT_CONFIG["approvals"]["mode"]

    # Mode table row marks the default:  | **smart** (default) | ...
    assert _mode_table_default(path, _VALID_MODES) == default
    # security.md also carries the full key table: | `mode` | `smart` | ...
    key_row = re.search(
        r"^\|\s*`mode`\s*\|\s*`([^`]+)`\s*\|", text, re.MULTILINE
    )
    if key_row:
        assert key_row.group(1) == default
    # The example block must not advertise a different mode as the shipped one.
    assert re.search(r"^\s*mode:\s*smart\b", text, re.MULTILINE), (
        f"{doc} example config no longer shows mode: smart"
    )
