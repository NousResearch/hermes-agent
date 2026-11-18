"""Retired runtime installers must not turn agent construction into an update."""
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("module,name,locals_", [
    ("community_plugin", "cmd_update", ""),
    ("hermes_cli.main", "serve", ""),
    # current updater entrypoints carry the sentinel local
    ("hermes_cli.update_cmd", "_cmd_update_impl",
     "    _hermes_current_updater_frame = True\n"),
])
def test_runtime_lookalikes_do_not_start_updates(module, name, locals_, monkeypatch):
    from hermes_cli import _old_updater
    from tools.lazy_deps import install_specs

    child = Mock(return_value=(0, {}))
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)
    namespace = {"__name__": module, "install_specs": install_specs}
    exec(f"def {name}():\n{locals_}    install_specs(['hindsight-all'])\n", namespace)
    with pytest.raises(ImportError, match="retired"):
        namespace[name]()
    child.assert_not_called()
