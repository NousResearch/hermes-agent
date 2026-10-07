"""Computer-use provider plugins: ``computer_use.backend`` selects exactly one provider per profile, only that one
is imported, the built-in ``cua`` stays the default, and an unknown name fails loudly instead of falling back."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PLUGIN = '''
from pathlib import Path
from tools.computer_use.backend import ActionResult, ComputerUseBackend, ComputerUseProvider

TAG = Path(__file__).parent.name
CALLS = []

class FakeBackend(ComputerUseBackend):
    def __init__(self, mode): self.mode = mode
    def start(self): CALLS.append(("start", self.mode))
    def stop(self): CALLS.append(("stop", None))
    def is_available(self): return True
    def capture(self, mode="som", app=None, pid=None, window_id=None): raise NotImplementedError
    def click(self, **kw):
        CALLS.append(("click", kw.get("element")))
        return ActionResult(ok=True, action="click")
    def drag(self, **kw): return ActionResult(ok=True, action="drag")
    def scroll(self, **kw): return ActionResult(ok=True, action="scroll")
    def type_text(self, text, **kw): return ActionResult(ok=True, action="type")
    def key(self, keys, **kw): return ActionResult(ok=True, action="key")
    def list_apps(self): return [{"backend": TAG, "file": __file__}]
    def focus_app(self, app, raise_window=False): return ActionResult(ok=True, action="focus_app")
    def launch_app(self, app=None, **kw): return ActionResult(ok=True, action="launch")
    def set_value(self, value, element=None): return ActionResult(ok=True, action="set_value")

class FakeProvider(ComputerUseProvider):
    name = TAG
    def create_backend(self, *, permission_mode): return FakeBackend(permission_mode)

def register(ctx):
    ctx.register_computer_use_provider(FakeProvider())
'''


def _home(home: Path, plugins: tuple, backend: str | None) -> Path:
    for plugin in plugins:
        (plugin_dir := home / "plugins" / plugin).mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": plugin, "description": f"{plugin} driver"}))
        (plugin_dir / "__init__.py").write_text(_PLUGIN)
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"computer_use": {"backend": backend}} if backend else {}))
    return home


def _list_apps(session_id: str = "s1") -> list:
    from tools.computer_use.tool import handle_computer_use
    out = json.loads(handle_computer_use({"action": "list_apps"}, session_id=session_id))
    assert "apps" in out, out
    return out["apps"]


@pytest.fixture(autouse=True)
def _isolate_hermes_bin_dir(monkeypatch):
    from tools.environments import local as env_local
    monkeypatch.setattr(env_local, "_HERMES_BIN_DIR", None)
    monkeypatch.setattr(env_local, "_HERMES_BIN_DIR_IS_PAYLOAD", False)


@pytest.fixture(autouse=True)
def _clean():
    from tools.computer_use.tool import reset_backend_for_tests
    reset_backend_for_tests()
    yield
    reset_backend_for_tests()
    for name in [m for m in sys.modules if m.startswith("_hermes_user_computer_use.")]:
        del sys.modules[name]


def test_selected_plugin_receives_tool_calls_and_unselected_stays_dormant(tmp_path, grant_computer_use_approvals):
    home = _home(tmp_path / "home", ("cu-alpha", "cu-beta"), "cu-beta")
    tok = set_hermes_home_override(str(home))
    try:
        from tools.computer_use.tool import check_computer_use_requirements, handle_computer_use
        assert check_computer_use_requirements() is True
        assert _list_apps()[0]["backend"] == "cu-beta"
        assert json.loads(handle_computer_use({"action": "click", "element": 3}, session_id="s1"))["ok"] is True
        calls = sys.modules["_hermes_user_computer_use.cu-beta"].CALLS
        assert ("start", "standard") in calls and ("click", 3) in calls
        assert "_hermes_user_computer_use.cu-alpha" not in sys.modules  # installed, not selected: never imported
        from hermes_cli.tools_config_providers import _computer_use_provider_rows  # `hermes tools` / Desktop rows
        assert [r["computer_use_backend"] for r in _computer_use_provider_rows()] == ["cu-alpha", "cu-beta"]
        assert "_hermes_user_computer_use.cu-alpha" not in sys.modules  # listing does not import either
    finally:
        reset_hermes_home_override(tok)


def test_unknown_backend_errors_naming_configured_and_available(tmp_path):
    home = _home(tmp_path / "home", (), "nope")
    tok = set_hermes_home_override(str(home))
    try:
        from tools.computer_use.tool import check_computer_use_requirements, handle_computer_use
        out = json.loads(handle_computer_use({"action": "list_apps"}, session_id="s1"))
        assert "'nope'" in out["error"] and "available: cua" in out["error"]
        assert "hint" not in out  # the cua-driver install hint would mislead here
        assert check_computer_use_requirements() is True  # the tool stays visible so the call can say why
    finally:
        reset_hermes_home_override(tok)


@pytest.mark.parametrize("backend", [None, "cua"])
def test_builtin_cua_is_default_and_plugin_never_self_activates(tmp_path, backend):
    home = _home(tmp_path / "home", ("cu-alpha",), backend)  # installed but not selected
    tok = set_hermes_home_override(str(home))
    try:
        from plugins.computer_use import get_active_provider
        from tools.computer_use.cua_backend import CuaDriverBackend
        from tools.computer_use.tool import _new_backend
        assert get_active_provider().name == "cua"
        assert isinstance(_new_backend("standard"), CuaDriverBackend)
        assert "_hermes_user_computer_use.cu-alpha" not in sys.modules
    finally:
        reset_hermes_home_override(tok)


def test_each_profile_gets_its_own_configured_backend(tmp_path):
    home_a = _home(tmp_path / "profiles" / "a", ("cu-alpha",), "cu-alpha")
    home_b = _home(tmp_path / "profiles" / "b", ("cu-beta",), "cu-beta")
    for home, expected in ((home_a, "cu-alpha"), (home_b, "cu-beta"), (home_a, "cu-alpha")):
        tok = set_hermes_home_override(str(home))
        try:
            app = _list_apps(session_id="shared")[0]
            assert app["backend"] == expected and str(home) in app["file"]
        finally:
            reset_hermes_home_override(tok)
