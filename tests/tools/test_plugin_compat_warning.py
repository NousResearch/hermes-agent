"""The plugin-compat layer tells plugin authors where a name went, once, and internal code never trips it.

Kept alongside the compat layer (tests/hermes_cli/test_compat_manifest_targets.py); both are deleted with it.
"""
import importlib
import json
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "compat_manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="compat layer removed (scheduled revert)")


def _a_lazy_entry():
    entries = json.loads(MANIFEST.read_text())["entries"]
    e = next(x for x in entries if x["kind"] == "moved-lazy" and x["facade"] == "tools.web_tools")
    return e["facade"], e["name"]


def test_compat_resolution_warns_once_per_name_with_the_new_location():
    from hermes_cli.plugin_compat import HermesPluginCompatWarning, _seen
    facade, name = _a_lazy_entry()
    _seen.discard((facade, name))
    mod = importlib.import_module(facade)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        getattr(mod, name)
        getattr(mod, name)
    ours = [w for w in rec if issubclass(w.category, HermesPluginCompatWarning)]
    assert len(ours) == 1, [str(w.message) for w in rec]
    msg = str(ours[0].message)
    assert f"{facade}.{name}" in msg


def test_importing_the_facade_itself_does_not_warn():
    """Only RESOLVING a compat name warns; plugins that import the module for its live API stay silent."""
    from hermes_cli.plugin_compat import HermesPluginCompatWarning
    import sys
    sys.modules.pop("tools.web_tools", None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        importlib.import_module("tools.web_tools")
    assert not [w for w in rec if issubclass(w.category, HermesPluginCompatWarning)]


def test_skills_hub_compat_hook_is_idempotent_across_reload_and_reinstall():
    import tools.skills_hub as hub

    hub = importlib.reload(hub)
    assert hub.__getattr__ is not hub._plugin_compat_prev_getattr

    hook = hub.__getattr__
    previous = hub._plugin_compat_prev_getattr
    hub._install_plugin_compat_getattr()
    hub._install_plugin_compat_getattr()

    assert hub.__getattr__ is hook
    assert hub._plugin_compat_prev_getattr is previous
    assert hub.HERMES_HOME.is_dir()
    with pytest.raises(AttributeError):
        getattr(hub, "not_a_real_hub_attribute")

    name, (target_module, target_name) = next(iter(hub._PLUGIN_COMPAT_LAZY.items()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert getattr(hub, name) is getattr(importlib.import_module(target_module), target_name)
