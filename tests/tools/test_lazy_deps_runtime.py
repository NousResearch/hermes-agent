"""Retired runtime installers must not turn agent construction into an update."""
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("specs", [[], ["hindsight-all"], ("honcho-ai",)])
@pytest.mark.parametrize("timeout", [0, 120, 300])
def test_install_specs_reports_unavailable_without_update(specs, timeout, monkeypatch):
    from hermes_cli import _old_updater
    from tools.lazy_deps import install_specs

    # Exercise the public API and real handoff code, stopping only at its child
    # boundary so an unfixed checkout cannot run a real install or Desktop build.
    child = Mock(return_value=(0, {}))
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)
    failure = None
    try:
        install_specs(specs, timeout=timeout)
    except BaseException as exc:
        failure = exc
    assert isinstance(failure, ImportError), repr(failure)
    child.assert_not_called()


@pytest.mark.parametrize("module,name", [
    ("community_plugin", "cmd_update"),
    ("hermes_cli.main", "serve"),
    ("hermes_cli.update_cmd", "unrelated"),
])
def test_runtime_lookalikes_do_not_start_updates(module, name, monkeypatch):
    from hermes_cli import _old_updater
    from tools.lazy_deps import install_specs

    child = Mock(return_value=(0, {}))
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)
    namespace = {"__name__": module, "install_specs": install_specs}
    exec(f"def {name}():\n    install_specs(['hindsight-all'])\n", namespace)
    with pytest.raises(ImportError, match="retired"):
        namespace[name]()
    child.assert_not_called()


def test_runtime_caller_can_keep_running_after_retired_installer(monkeypatch):
    from hermes_cli import _old_updater
    from tools.lazy_deps import install_specs

    child = Mock(return_value=(0, {}))
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)
    # The plugin's ordinary exception handler must see failure, not SystemExit.
    try:
        install_specs(["hindsight-all"])
    except Exception:
        available = False
    else:
        available = True
    assert available is False
    child.assert_not_called()
