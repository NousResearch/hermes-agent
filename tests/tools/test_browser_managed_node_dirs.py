"""The browser PATH candidates must be composed from the canonical managed-node helper.

tests/test_managed_runtime_resolution.py ratchets bare which() lookups; these pin the
sibling property for the directory lists, so the POSIX/Windows managed layout has exactly
one source of truth.
"""

import pytest

from tools import browser_tool_install as bt_install


def test_browser_candidate_path_dirs_uses_canonical_managed_node_dirs(tmp_path, monkeypatch):
    """Managed candidates come from iter_hermes_node_dirs(), plus the Hermes-local node_modules/.bin."""
    managed_root = tmp_path / "managed-node"
    hermes_home = tmp_path / "home"
    monkeypatch.setattr(
        "tools.browser_tool_install.iter_hermes_node_dirs",
        lambda home=None: [managed_root, managed_root / "bin"],
    )
    monkeypatch.setattr("tools.browser_tool_install.get_hermes_home", lambda: hermes_home)

    dirs = bt_install._browser_candidate_path_dirs()

    assert dirs[:3] == [
        str(managed_root),
        str(managed_root / "bin"),
        str(hermes_home / "node_modules" / ".bin"),
    ]


def test_find_agent_browser_recheck_probes_node_modules_bin_first(tmp_path, monkeypatch):
    """The post-install recheck probes <hermes_home>/node_modules/.bin BEFORE the managed node dirs.

    The lazy install puts agent-browser there, so it must outrank the managed runtime's global
    bin — the reverse of _browser_candidate_path_dirs(), and the reason the two sites order the
    same helper differently.
    """
    hermes_home = tmp_path / "home"
    managed_root = tmp_path / "managed-node"
    probed: list = []

    def _fake_which(name, path=None):
        probed.append(path)
        return None

    monkeypatch.setattr("tools.browser_tool_install.iter_hermes_node_dirs", lambda home=None: [managed_root])
    monkeypatch.setattr("tools.browser_tool_install.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("tools.browser_tool_install.shutil.which", _fake_which)
    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", lambda *_a, **_k: True)
    origin = bt_install._origin()
    monkeypatch.setattr(origin, "_agent_browser_resolved", False, raising=False)
    monkeypatch.setattr(origin, "_cached_agent_browser", None, raising=False)

    with pytest.raises(FileNotFoundError):  # nothing resolves — only the probe ORDER is under test
        bt_install._find_agent_browser()

    bin_dir = str(hermes_home / "node_modules" / ".bin")
    assert bin_dir in probed
    assert str(managed_root) in probed
    assert probed.index(bin_dir) < probed.index(str(managed_root))
