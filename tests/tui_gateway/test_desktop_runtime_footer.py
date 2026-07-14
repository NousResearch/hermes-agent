"""Desktop runtime-footer parity — tui_gateway appends the gateway footer.

Covers the 2026-07-14 /footer desktop enablement:
- _turn_runtime_footer builds a footer from live agent state when
  display.runtime_footer.enabled (or the desktop platform override) is on,
  and returns "" when off / on any failure.
- config.set/get key="footer" round-trips display.runtime_footer.enabled.
"""

from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server


def _agent(model="claude-opus-4-8", provider="claude-apr", prompt_tokens=50_000, ctx=200_000):
    comp = SimpleNamespace(last_prompt_tokens=prompt_tokens, context_length=ctx)
    return SimpleNamespace(model=model, provider=provider, context_compressor=comp, reasoning_config=None)


def _cfg(enabled, platform_override=None):
    cfg = {"display": {"runtime_footer": {"enabled": enabled}}}
    if platform_override is not None:
        cfg["display"]["platforms"] = {"desktop": {"runtime_footer": {"enabled": platform_override}}}
    return cfg


class TestTurnRuntimeFooter:
    def test_disabled_returns_empty(self):
        with patch.object(server, "_load_cfg", return_value=_cfg(False)):
            assert server._turn_runtime_footer(_agent(), None) == ""

    def test_enabled_builds_footer_with_model_and_context(self):
        with patch.object(server, "_load_cfg", return_value=_cfg(True)):
            line = server._turn_runtime_footer(_agent(), None)
        assert "claude-opus-4-8" in line
        assert "50" in line  # 50.0k occupancy rendered

    def test_desktop_platform_override_wins(self):
        # Global off, desktop on → footer renders (platform override wins).
        with patch.object(server, "_load_cfg", return_value=_cfg(False, platform_override=True)):
            assert server._turn_runtime_footer(_agent(), None) != ""
        # Global on, desktop off → suppressed.
        with patch.object(server, "_load_cfg", return_value=_cfg(True, platform_override=False)):
            assert server._turn_runtime_footer(_agent(), None) == ""

    def test_reasoning_none_when_disabled(self):
        agent = _agent()
        agent.reasoning_config = {"enabled": False}
        with patch.object(server, "_load_cfg", return_value=_cfg(True)):
            assert "r:none" in server._turn_runtime_footer(agent, None)

    def test_failure_is_swallowed(self):
        with patch.object(server, "_load_cfg", side_effect=RuntimeError("boom")):
            assert server._turn_runtime_footer(_agent(), None) == ""


class TestFooterConfigKey:
    def _dispatch(self, method):
        return server._methods[method]

    def test_set_and_get_roundtrip(self, tmp_path, monkeypatch):
        saved = {}
        monkeypatch.setattr(server, "_load_cfg", lambda: saved.get("cfg", {}))
        monkeypatch.setattr(server, "_save_cfg", lambda cfg: saved.__setitem__("cfg", cfg))

        set_fn = self._dispatch("config.set")
        get_fn = self._dispatch("config.get")

        res = set_fn(1, {"key": "footer", "value": "on"})
        assert res["result"]["value"] == "on"
        assert saved["cfg"]["display"]["runtime_footer"]["enabled"] is True

        res = get_fn(2, {"key": "footer"})
        assert res["result"]["value"] == "on"

        res = set_fn(3, {"key": "footer", "value": ""})  # bare = toggle
        assert res["result"]["value"] == "off"
        assert saved["cfg"]["display"]["runtime_footer"]["enabled"] is False

        res = set_fn(4, {"key": "footer", "value": "bogus"})
        assert "error" in res

    def test_get_honors_desktop_platform_override(self, monkeypatch):
        # Global off + desktop override on → status must say "on" (what turns
        # actually render), not the bare global flag. Greptile P2 on #333.
        cfg = {
            "display": {
                "runtime_footer": {"enabled": False},
                "platforms": {"desktop": {"runtime_footer": {"enabled": True}}},
            }
        }
        monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
        res = self._dispatch("config.get")(1, {"key": "footer"})
        assert res["result"]["value"] == "on"
