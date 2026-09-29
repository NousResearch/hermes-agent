"""Retired runtime installers must not turn agent construction into an update."""
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("module,name", [
    ("community_plugin", "cmd_update"),
    ("hermes_cli.main", "serve"),
    ("hermes_cli.update_cmd", "cmd_update"),  # current updater: no pre_update_version local
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
