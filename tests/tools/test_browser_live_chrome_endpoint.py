"""``browser.live_chrome_cdp_url``: live mode attaches to a fixed DevTools endpoint instead of discovering the
user's own Chrome. Contracts: unset = unchanged; set (and live on) = the endpoint is exported on the live
daemon only, no Allow hint, loopback http(s) only, unreachable endpoint reported, a changed value never
reuses the old daemon; overrides still win; set with live off = no effect.
"""
import http.server
import json
import stat
import threading

import pytest

import tools.browser_use_cli as bu_cli
import tools.browser_use_live_chrome as live
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud

ENDPOINT = "http://127.0.0.1:9223"
_PROBE = 'cat > /dev/null\necho "name:${BU_NAME:-default} ws:${BU_CDP_WS:-unset} url:${BU_CDP_URL:-unset}"\n'


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for var in ("BU_NAME", "BU_AUTOSPAWN", "BU_CDP_WS", "BU_CDP_URL", "BROWSER_USE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: None)
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(bs, "get_session_info", lambda key: {
        "cdp_url": f"ws://127.0.0.1:47000/devtools/browser/{key}", "features": {"local": True}})
    bs_cloud.reset_caches()
    yield
    bs_cloud.reset_caches()


@pytest.fixture
def config(monkeypatch):
    cfg: dict = {}
    monkeypatch.setattr(bs_cloud, "browser_section", lambda: cfg)
    return cfg


@pytest.fixture
def reachable(monkeypatch):
    """The endpoint answers; the user's-Chrome discovery preflight must never run."""
    monkeypatch.setattr(live, "endpoint_unreachable", lambda url: None)
    monkeypatch.setattr(live, "preflight", lambda system=None: pytest.fail("discovery preflight ran"))


def _cli(tmp_path, monkeypatch, body=_PROBE):
    script = tmp_path / "browser-use"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(script)])


def _exec(code="print(1)", **kw):
    return json.loads(bu_cli.browser_exec(code, **kw))


class TestUnset:
    def test_live_mode_unchanged(self, config, monkeypatch):
        config["live_chrome"] = True
        monkeypatch.setattr(live, "preflight", lambda system=None: None)
        monkeypatch.setattr(live, "endpoint_unreachable", lambda url: pytest.fail("endpoint probed"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert not bu_cli._has_cdp_env(env)
        assert env[live.LIVE_CHROME_SENTINEL] == "1"

    def test_default_is_empty(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["browser"]["live_chrome_cdp_url"] == ""

    def test_description_still_tells_user_to_click_allow(self, config):
        config["live_chrome"] = True
        assert "the USER clicks Allow" in bu_cli._dynamic_schema_overrides()["description"]


class TestSet:
    @pytest.fixture(autouse=True)
    def _on(self, config, reachable):
        config["live_chrome"] = True
        config["live_chrome_cdp_url"] = ENDPOINT

    def test_exports_endpoint_and_keeps_live_semantics(self):
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_URL"] == ENDPOINT
        assert env[live.LIVE_CHROME_SENTINEL] == "1"
        assert env["BH_TAB_MARKER"] == "0"
        assert "BU_AUTOSPAWN" not in env

    def test_live_daemon_only_and_own_tab_kept(self, tmp_path, monkeypatch):
        _cli(tmp_path, monkeypatch, 'code=$(cat)\necho "name:${BU_NAME} url:${BU_CDP_URL:-unset} ws:${BU_CDP_WS:-unset}"\necho "$code"\n')
        out = _exec(session="r7k2")["output"]
        assert f"url:{ENDPOINT}" in out and "name:_live_r7k2_" in out
        assert "_hermes_live_own_tab" in out

    def test_non_live_exec_unaffected(self, tmp_path, monkeypatch, config):
        config["live_chrome"] = False
        _cli(tmp_path, monkeypatch)
        out = _exec()["output"]
        assert "name:default ws:ws://127.0.0.1:47000/" in out and ENDPOINT not in out

    def test_other_browser_tools_unaffected(self, monkeypatch):
        """The endpoint is not a cdp_url override: every other browser tool's CDP lookup stays empty."""
        import tools.browser_session_cdp as cdp
        monkeypatch.undo()  # drop the autouse stubs: read the real lookup against the config above
        monkeypatch.setattr(bs_cloud, "browser_section", lambda: {"live_chrome": True, "live_chrome_cdp_url": ENDPOINT})
        monkeypatch.setattr(cdp, "live_cdp_override", lambda: "")
        monkeypatch.setattr(cdp, "env_cdp_url", lambda: "")
        assert cdp.get_cdp_override_raw() == ""

    def test_no_allow_hint(self, tmp_path, monkeypatch):
        assert "Allow" not in live.live_chrome_timeout_hint(True, endpoint=ENDPOINT)
        assert "click Allow" in live.live_chrome_timeout_hint(True)
        text = bu_cli._dynamic_schema_overrides()["description"]
        assert "LIVE CHROME" in text and "USER clicks Allow" not in text and ENDPOINT in text
        _cli(tmp_path, monkeypatch, "cat > /dev/null\nsleep 30\n")
        err = _exec(timeout_s=5)["error"]
        assert "timed out" in err and "Allow" not in err

    def test_approval_guard_stays(self):
        assert live.approval_attempt("subprocess.run(['browser-harness','mac-approve'])", True)

    def test_changed_value_never_reuses_old_daemon(self, tmp_path, monkeypatch, config):
        _cli(tmp_path, monkeypatch)
        first = _exec()["output"].splitlines()[0]
        config["live_chrome_cdp_url"] = "http://127.0.0.1:9333"
        second = _exec()["output"].splitlines()[0]
        config["live_chrome_cdp_url"] = ""
        monkeypatch.setattr(live, "preflight", lambda system=None: None)
        third = _exec()["output"].splitlines()[0]
        names = [line.split()[0] for line in (first, second, third)]
        assert len(set(names)) == 3 and all(n.startswith("name:_live_default") for n in names)
        assert names[2] == "name:_live_default"  # unset: the name live mode always had

    def test_live_off_stops_all_live_daemons(self, config, monkeypatch):
        calls = []
        monkeypatch.setattr(bu_cli, "_reload_daemons", lambda d: calls.extend(d))
        monkeypatch.setattr(bu_cli, "hermes_home_key", lambda: "h")
        with bu_cli._driven_daemons_lock:
            bu_cli._driven_daemons.clear()
            bu_cli._driven_daemons[("d", "_live_default_ab12cd34")] = "h"
        config["live_chrome"] = False
        bu_cli._stop_live_daemons_if_off()
        assert calls == [("d", "_live_default_ab12cd34")]


class TestSetLiveOff:
    def test_no_effect(self, config, monkeypatch):
        config["live_chrome_cdp_url"] = ENDPOINT
        monkeypatch.setattr(live, "endpoint_unreachable", lambda url: pytest.fail("endpoint probed"))
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_WS"] == "ws://127.0.0.1:47000/devtools/browser/t1"
        assert ENDPOINT not in env.values()
        assert "USER clicks Allow" not in bu_cli._dynamic_schema_overrides()["description"]


class TestOverridesWin:
    @pytest.fixture(autouse=True)
    def _on(self, config, reachable):
        config["live_chrome"] = True
        config["live_chrome_cdp_url"] = ENDPOINT

    def test_bu_cdp_env(self):
        env = {"BU_CDP_WS": "ws://operator:9222"}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_WS"] == "ws://operator:9222" and "BU_CDP_URL" not in env

    def test_cdp_override(self, monkeypatch):
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: "http://127.0.0.1:9222")
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: "http://127.0.0.1:9222")
        env = {}
        assert bu_cli._route_backend(env, "", "t1", False) is None
        assert env["BU_CDP_URL"] == "http://127.0.0.1:9222"
        assert live.LIVE_CHROME_SENTINEL not in env


class TestValidation:
    @pytest.mark.parametrize("url", ["http://127.0.0.1:9223", "https://localhost:9223", "http://[::1]:9223/"])
    def test_loopback_accepted(self, url):
        assert live.endpoint_format_error(url) is None

    @pytest.mark.parametrize("url", ["http://192.168.1.5:9223", "http://example.com:9223", "ws://127.0.0.1:9223",
                                     "http://127.0.0.1.evil.com:9223", "127.0.0.1:9223", "file:///tmp/x"])
    def test_non_loopback_rejected(self, config, monkeypatch, url):
        assert "loopback" in live.endpoint_format_error(url)
        config["live_chrome"] = True
        config["live_chrome_cdp_url"] = url
        monkeypatch.setattr(live, "endpoint_unreachable", lambda u: pytest.fail("probed a rejected endpoint"))
        env = {}
        assert "loopback" in bu_cli._route_backend(env, "", "t1", False)
        assert not bu_cli._has_cdp_env(env)

    def test_unreachable_message(self, config, monkeypatch):
        monkeypatch.setattr(live, "preflight", lambda system=None: pytest.fail("discovery preflight ran"))
        config["live_chrome"] = True
        config["live_chrome_cdp_url"] = "http://127.0.0.1:1"  # nothing listens
        err = bu_cli._route_backend({}, "", "t1", False)
        assert "127.0.0.1:1" in err and "not reachable" in err and "start" in err
        assert "zchrome" not in err and "Allow" not in err

    def test_probe_does_not_follow_redirects(self):
        hits = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                if self.path == "/json/version":
                    self.send_response(302)
                    self.send_header("Location", "/elsewhere")
                else:
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            assert "not reachable" in live.endpoint_unreachable(f"http://127.0.0.1:{srv.server_port}")
        finally:
            srv.shutdown()
        assert hits == ["/json/version"]

    def test_reachable_endpoint_passes(self):
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/json/version" else 404)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            assert live.endpoint_unreachable(f"http://127.0.0.1:{srv.server_port}") is None
        finally:
            srv.shutdown()
