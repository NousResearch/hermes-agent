"""Retired runtime installers must not turn agent construction into an update."""
from unittest.mock import Mock

import pytest

from tools.lazy_deps import InstallOutcome, install_specs


@pytest.mark.parametrize("module,name,locals_", [
    ("community_plugin", "cmd_update", ""),
    ("hermes_cli.main", "serve", ""),
    # current updater entrypoints carry the sentinel local
    ("hermes_cli.update_cmd", "_cmd_update_impl",
     "    _hermes_current_updater_frame = True\n"),
])
def test_runtime_lookalikes_do_not_start_updates(module, name, locals_, monkeypatch):
    """Lookalikes degrade with a graceful InstallOutcome; the updater child is never started."""
    from hermes_cli import _old_updater

    child = Mock(return_value=(0, {}))
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)
    namespace = {"__name__": module, "install_specs": install_specs}
    exec(f"def {name}():\n{locals_}    return install_specs(['hindsight-all'])\n", namespace)
    outcome = namespace[name]()
    assert isinstance(outcome, InstallOutcome)
    assert outcome.ok is False
    # actionable: the caller can surface the manual-install hint
    assert "install manually" in outcome.reason
    assert "hindsight-all" in outcome.reason
    child.assert_not_called()


def test_hindsight_style_runtime_probe_degrades_gracefully(monkeypatch):
    """A Hindsight local_embedded style probe survives install_specs (no hard exit, no update)."""
    from hermes_cli import _old_updater

    child = Mock(return_value=(0, {}))
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)

    # The actual caller shape: _ensure_local_runtime() probes its runtime and
    # calls install_specs(["hindsight-all"]) when it is missing; it must be
    # able to degrade (log + fall back) rather than crash or rebuild.
    def ensure_local_runtime() -> tuple[str, bool]:
        spec = ["hindsight-all"]
        outcome = install_specs(spec)
        if not outcome.ok:
            # degrade: surface the manual-install hint, keep the process alive
            return ("hindsight runtime unavailable: " + outcome.reason, False)
        return ("runtime ready", True)

    message, ok = ensure_local_runtime()
    assert ok is False
    assert message.startswith("hindsight runtime unavailable")
    assert "uv pip install" in message  # actionable manual-install hint
    child.assert_not_called()
