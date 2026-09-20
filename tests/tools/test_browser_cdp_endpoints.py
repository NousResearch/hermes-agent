"""Resolve order for browser.cdp_endpoints (session → CDP URL map).

Does not launch Chrome, Xvfb, or Bot Screen. Config/env only.
"""

from unittest.mock import patch

from tools import browser_tool_cdp as bt_cdp
from tools import browser_use_cli as bu_cli

CAP = "http://127.0.0.1:9222"
LAB2 = "http://127.0.0.1:9223"
LAB3 = "http://127.0.0.1:9224"

ENDPOINTS = {
    "capsolver": CAP,
    "lab2": LAB2,
    "lab3": LAB3,
}

CONFIG = {
    "browser": {
        "backend": "off",
        "cdp_url": CAP,
        "cdp_endpoints": dict(ENDPOINTS),
    }
}


def _cfg(monkeypatch, cfg=CONFIG):
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_ENDPOINT", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: cfg)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda n, d="": __import__("os").environ.get(n, d),
        raising=False,
    )
    # Keep connect-path tests on the configured HTTP URL (no live /json/version).
    monkeypatch.setattr(bt_cdp, "_resolve_cdp_override", lambda url: url)


class TestParseCdpEndpoints:
    def test_drops_invalid_names_and_non_maps(self):
        assert bt_cdp._parse_cdp_endpoints(["lab2", LAB2]) == {}
        assert bt_cdp._parse_cdp_endpoints({
            "lab2": LAB2,
            "": CAP,
            "bad name": CAP,
            "ok": "  ",
            3: LAB3,
        }) == {"lab2": LAB2, "3": LAB3}

    def test_object_form_and_string_form(self):
        raw = {
            "primary": {"url": CAP, "stay_put": True},
            "lab2": LAB2,
            "lab3": {"url": LAB3, "stay_put": "yes"},
            "empty": {"stay_put": True},
            "bad name": {"url": CAP, "stay_put": True},
        }
        assert bt_cdp._parse_cdp_endpoints(raw) == {
            "primary": CAP, "lab2": LAB2, "lab3": LAB3,
        }
        records = bt_cdp._parse_cdp_endpoint_records(raw)
        assert records["primary"]["stay_put"] is True
        assert records["lab2"]["stay_put"] is False
        assert records["lab3"]["stay_put"] is True
        assert "empty" not in records
        assert "bad name" not in records

    def test_expand_connect_target_name_vs_url(self, monkeypatch):
        _cfg(monkeypatch)
        assert bt_cdp.expand_cdp_connect_target("lab2") == LAB2
        assert bt_cdp.expand_cdp_connect_target("http://127.0.0.1:9223") == "http://127.0.0.1:9223"
        assert bt_cdp.expand_cdp_connect_target("127.0.0.1:9224") == "127.0.0.1:9224"
        assert bt_cdp.expand_cdp_connect_target("missing") == "missing"


class TestCdpOverrideResolveOrder:
    def test_env_url_wins_over_named_map_and_cdp_url(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9999")
        monkeypatch.setenv("BROWSER_CDP_ENDPOINT", "lab2")
        assert bt_cdp._get_cdp_override_raw(endpoint="lab3") == "http://127.0.0.1:9999"

    def test_explicit_endpoint_name_beats_cdp_url(self, monkeypatch):
        _cfg(monkeypatch)
        assert bt_cdp._get_cdp_override_raw(endpoint="lab2") == LAB2
        assert bt_cdp._get_cdp_override_raw(endpoint="lab3") == LAB3
        assert bt_cdp._get_cdp_override_raw(endpoint="capsolver") == CAP

    def test_browser_cdp_endpoint_env_name(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("BROWSER_CDP_ENDPOINT", "lab2")
        assert bt_cdp._get_cdp_override_raw() == LAB2

    def test_explicit_endpoint_beats_env_name(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("BROWSER_CDP_ENDPOINT", "lab2")
        assert bt_cdp._get_cdp_override_raw(endpoint="lab3") == LAB3

    def test_unknown_name_falls_back_to_cdp_url(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("BROWSER_CDP_ENDPOINT", "nope")
        assert bt_cdp._get_cdp_override_raw(endpoint="also-missing") == CAP

    def test_unnamed_keeps_cdp_url(self, monkeypatch):
        _cfg(monkeypatch)
        assert bt_cdp._get_cdp_override_raw() == CAP

    def test_hermes_session_id_exact_match(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("HERMES_SESSION_ID", "lab2")
        assert bt_cdp._get_cdp_override_raw() == LAB2

    def test_hermes_session_key_last_segment(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("HERMES_SESSION_KEY", "tui:lab3")
        assert bt_cdp._get_cdp_override_raw() == LAB3

    def test_uuidish_last_segment_does_not_bind(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:deadbeefcafebabe")
        assert bt_cdp._get_cdp_override_raw() == CAP

    def test_empty_map_uses_cdp_url(self, monkeypatch):
        _cfg(monkeypatch, {"browser": {"cdp_url": CAP, "cdp_endpoints": {}}})
        assert bt_cdp._get_cdp_override_raw(endpoint="lab2") == CAP

    def test_map_only_without_cdp_url_unnamed_is_empty(self, monkeypatch):
        _cfg(monkeypatch, {"browser": {"cdp_url": "", "cdp_endpoints": {"lab2": LAB2}}})
        assert bt_cdp._get_cdp_override_raw() == ""
        assert bt_cdp._get_cdp_override_raw(endpoint="lab2") == LAB2


class TestBrowserExecSessionBind:
    def test_named_sidecar_exports_lab_url_and_private_sentinel(self, monkeypatch):
        _cfg(monkeypatch)
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="lab2") is None
        assert env["BU_CDP_URL"] == LAB2
        assert env[bu_cli._PRIVATE_BROWSER_SENTINEL] == "1"

    def test_named_alias_of_default_stays_shared(self, monkeypatch):
        _cfg(monkeypatch)
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="capsolver") is None
        assert env["BU_CDP_URL"] == CAP
        assert bu_cli._PRIVATE_BROWSER_SENTINEL not in env

    def test_unnamed_stays_on_cdp_url(self, monkeypatch):
        _cfg(monkeypatch)
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1") is None
        assert env["BU_CDP_URL"] == CAP
        assert bu_cli._PRIVATE_BROWSER_SENTINEL not in env

    def test_patched_zero_arg_override_still_works(self, monkeypatch):
        """Existing tests stub `_get_cdp_override` with a zero-arg lambda."""
        _cfg(monkeypatch, {"browser": {"cdp_url": CAP, "cdp_endpoints": {}}})
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: CAP)
        env = {}
        assert bu_cli._resolve_backend_cdp(env, "t1", session_name="lab2") is None
        assert env["BU_CDP_URL"] == CAP
        assert bu_cli._PRIVATE_BROWSER_SENTINEL not in env


class TestStayPutProvenance:
    def test_named_object_stay_put_and_string_sidecar(self, monkeypatch):
        _cfg(monkeypatch, {"browser": {
            "cdp_url": CAP,
            "cdp_endpoints": {
                "primary": {"url": CAP, "stay_put": True},
                "lab2": LAB2,
            },
        }})
        assert bt_cdp._cdp_override_is_stay_put(endpoint="primary") is True
        assert bt_cdp._cdp_override_is_stay_put(endpoint="lab2") is False
        # Same URL as the stay-put named entry is fenced even on the unnamed path.
        assert bt_cdp._cdp_override_is_stay_put() is True

    def test_cdp_stay_put_marks_unnamed_only(self, monkeypatch):
        _cfg(monkeypatch, {"browser": {
            "cdp_url": CAP,
            "cdp_stay_put": True,
            "cdp_endpoints": {"lab2": LAB2},
        }})
        assert bt_cdp._cdp_override_is_stay_put() is True
        assert bt_cdp._cdp_override_is_stay_put(endpoint="lab2") is False

    def test_default_is_not_stay_put(self, monkeypatch):
        _cfg(monkeypatch)
        assert bt_cdp._cdp_override_is_stay_put() is False
        assert bt_cdp._cdp_override_is_stay_put(endpoint="lab2") is False

    def test_session_for_key_stamps_stay_put_feature(self, monkeypatch):
        from tools import browser_tool_session as bt_session
        _cfg(monkeypatch, {"browser": {
            "cdp_url": CAP,
            "cdp_endpoints": {
                "primary": {"url": CAP, "stay_put": True},
                "lab2": LAB2,
            },
        }})
        stay = bt_session._create_session_for_key("bu-named-primary", force_local=False)
        assert stay["features"].get("stay_put") is True
        sidecar = bt_session._create_session_for_key("bu-named-lab2", force_local=False)
        assert sidecar["features"].get("stay_put") is not True
        assert sidecar["cdp_url"] == LAB2


class TestExpandDoesNotProbeNetwork:
    def test_get_cdp_override_raw_does_not_call_requests(self, monkeypatch):
        _cfg(monkeypatch)
        monkeypatch.setenv("BROWSER_CDP_ENDPOINT", "lab2")
        with patch("requests.get", side_effect=AssertionError("status path must not HTTP-probe")):
            assert bt_cdp._get_cdp_override_raw() == LAB2
