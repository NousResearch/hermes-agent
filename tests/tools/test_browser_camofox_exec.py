"""Tests for the Camofox exec backend (tools/browser_camofox_exec.py).

Covers the Camofox side of the ``browser_exec`` surface:

* Mode detection — ``browser.backend: camofox`` activates ``browser_exec``
  even when the browser-use CLI is NOT installed; Camofox setups keep the
  built-in tools under every other backend.
* E2E execution — the model's code runs in a real subprocess (Hermes
  interpreter) against a real HTTP fake of the Camofox server: navigation,
  snapshot, evaluate, click/type, screenshot attachment, tab recovery on
  404, and the ``CAMOFOX_TAB_ID`` bookkeeping back into the session cache.
"""

import json
import os
import re
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

import tools.browser_camofox as camofox_mod
import tools.browser_camofox_exec as camofox_exec_mod
import tools.browser_use_cli as bu_cli


# ---------------------------------------------------------------------------
# Fake Camofox server (stdlib http.server, real sockets)
# ---------------------------------------------------------------------------


class _FakeCamofoxHandler(BaseHTTPRequestHandler):
    server_version = "FakeCamofox/1.0"

    def log_message(self, *args):  # silence test output
        pass

    # -- helpers ------------------------------------------------------------

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _tab(self, tab_id: str):
        """Return the tab record or answer like the real server for a dead
        tab: 410 Gone with ``code: browser_restarted`` (the actual Camofox
        server's response for a GC'd tab; the older 404 shape is also
        covered by the runtime's recovery check)."""
        tab = self.server.camofox.tabs.get(tab_id)
        if tab is None or tab.get("dead"):
            self._send_json(
                410,
                {
                    "error": "Tab no longer exists (browser was restarted). "
                    "Create a new tab.",
                    "code": "browser_restarted",
                },
            )
            return None
        return tab

    def _route(self) -> tuple:
        parsed = urlparse(self.path)
        match = re.match(r"^/tabs/([^/]+)/(\w+)$", parsed.path)
        params = parse_qs(parsed.query)
        if match:
            return match.group(1), match.group(2), params
        return None, parsed.path, params

    # -- verbs --------------------------------------------------------------

    def do_GET(self):
        tab_id, path, params = self._route()
        srv = self.server.camofox
        if path == "/health":
            return self._send_json(200, {"ok": True})
        if path == "/tabs" and tab_id is None:
            user_id = (params.get("userId") or [""])[0]
            tabs = [
                {"tabId": tid, "listItemId": t["listItemId"], "userId": t["userId"]}
                for tid, t in srv.tabs.items()
                if not t.get("dead") and t["userId"] == user_id
            ]
            return self._send_json(200, {"tabs": tabs})
        if tab_id is not None and path == "stats":
            srv.stats_requests += 1
            if srv.stats_error_code:
                return self._send_json(srv.stats_error_code, {"error": "forced error"})
            tab = self._tab(tab_id)
            if tab is None:
                return
            return self._send_json(200, {"ok": True, "url": "https://example.com/"})
        if tab_id is not None and path == "snapshot":
            if srv.snapshot_error_code:
                return self._send_json(
                    srv.snapshot_error_code, {"error": "forced snapshot error"}
                )
            tab = self._tab(tab_id)
            if tab is None:
                return
            return self._send_json(
                200, {"snapshot": srv.snapshot, "refsCount": srv.refs_count}
            )
        if tab_id is not None and path == "screenshot":
            tab = self._tab(tab_id)
            if tab is None:
                return
            png = b"\x89PNG\r\n\x1a\n" + b"fakepng"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            return
        if tab_id is not None and path == "links":
            tab = self._tab(tab_id)
            if tab is None:
                return
            links = [{"text": "Example", "href": "https://example.com/"}]
            return self._send_json(200, {"links": links, "count": len(links)})
        self._send_json(404, {"error": f"unknown GET {self.path}"})

    def do_POST(self):
        tab_id, path, _ = self._route()
        srv = self.server.camofox
        body = self._body()
        if path == "/tabs" and tab_id is None:
            # Mirror the real server: an explicit non-http(s) url (e.g.
            # "about:blank") is rejected; omitting the key opens a blank tab.
            url = body.get("url", "")
            if url and not url.startswith(("http://", "https://")):
                return self._send_json(400, {"error": f"Blocked URL scheme: {url}"})
            srv.counter += 1
            tid = f"t{srv.counter}"
            srv.tabs[tid] = {
                "userId": body.get("userId", ""),
                "listItemId": body.get("listItemId", ""),
                "url": body.get("url", ""),
                "dead": False,
            }
            return self._send_json(
                200,
                {"ok": True, "tabId": tid, "url": body.get("url", ""), "title": "Fake"},
            )
        if tab_id is not None and path in (
            "navigate",
            "wait",
            "click",
            "type",
            "press",
            "scroll",
            "evaluate",
        ):
            tab = self._tab(tab_id)
            if tab is None:
                return
            if path == "evaluate":
                if srv.evaluate_error_code:
                    return self._send_json(
                        srv.evaluate_error_code, {"error": "forced evaluate error"}
                    )
                return self._send_json(200, {"ok": True, "result": srv.evaluate_result})
            if path == "wait":
                return self._send_json(200, {"ok": True, "ready": srv.wait_ready})
            if path == "navigate":
                tab["url"] = body.get("url", "")
            return self._send_json(200, {"ok": True, "url": tab.get("url", "")})
        self._send_json(404, {"error": f"unknown POST {self.path}"})

    def do_DELETE(self):
        self._send_json(200, {"ok": True})


class FakeCamofoxServer:
    def __init__(self):
        self.tabs: dict = {}
        self.counter = 0
        self.snapshot = "[heading] Hello Camofox\n[link e1] More information\n"
        self.refs_count = 2
        self.evaluate_result = "eval-result-42"
        self.stats_error_code = None  # when set, /stats answers this status
        self.stats_requests = 0
        self.snapshot_error_code = None
        self.evaluate_error_code = None
        self.wait_ready = True
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCamofoxHandler)
        self.httpd.camofox = self
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def kill_tab(self, tab_id: str):
        self.tabs[tab_id]["dead"] = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("BU_NAME", raising=False)
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    camofox_mod._sessions.clear()
    yield
    camofox_mod._sessions.clear()


@pytest.fixture()
def camofox_server():
    srv = FakeCamofoxServer()
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture()
def camofox_mode(monkeypatch, camofox_server):
    """backend=camofox + a reachable Camofox server (no browser-use CLI)."""
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"backend": "camofox"}},
    )
    monkeypatch.setattr(
        "tools.browser_camofox.get_camofox_url", lambda: camofox_server.url
    )
    return camofox_server


def _exec(code: str, task_id: str = "task-e2e-1") -> dict:
    return json.loads(bu_cli.browser_exec(code, task_id=task_id))


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


class TestModeDetection:
    def test_camofox_backend_enables_exec_without_cli(
        self, monkeypatch, camofox_server
    ):
        """backend=camofox activates browser_exec even with no browser-use CLI."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "camofox"}},
        )
        monkeypatch.setattr(
            "tools.browser_camofox.get_camofox_url", lambda: camofox_server.url
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: None)
        assert bu_cli.is_browser_use_cli_mode() is True

    def test_camofox_backend_without_server_falls_back(self, monkeypatch):
        """backend=camofox but no CAMOFOX_URL: keep the built-in tools."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "camofox"}},
        )
        monkeypatch.setattr("tools.browser_camofox.get_camofox_url", lambda: "")
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_browser_use_backend_with_camofox_stays_builtin(
        self, monkeypatch, camofox_server
    ):
        """Explicit browser-use backend cannot drive Camoufox (no CDP)."""
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "browser-use"}},
        )
        monkeypatch.setattr(
            "tools.browser_camofox.get_camofox_url", lambda: camofox_server.url
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
        assert bu_cli.is_browser_use_cli_mode() is False

    def test_default_with_camofox_keeps_builtin(self, monkeypatch, camofox_server):
        """Backend unset + Camofox configured: built-in tools stay (old default)."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr(
            "tools.browser_camofox.get_camofox_url", lambda: camofox_server.url
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/usr/bin/browser-use"])
        assert bu_cli.is_browser_use_cli_mode() is False


# ---------------------------------------------------------------------------
# E2E execution against the fake server
# ---------------------------------------------------------------------------


class TestExecE2E:
    def test_navigate_and_snapshot(self, camofox_mode):
        result = _exec(
            "new_tab('http://127.0.0.1:8080')\nprint(page_info())", task_id="e2e-nav"
        )
        assert result["success"] is True, result
        assert "Hello Camofox" in result["output"]
        assert "CAMOFOX_TAB_ID=t1" in result["output"]
        # Tab id bookkept back into the in-memory session cache.
        assert camofox_mod._sessions["e2e-nav"]["tab_id"] == "t1"

    def test_js_evaluate(self, camofox_mode):
        result = _exec("print(js('1+1'))", task_id="e2e-js")
        assert result["success"] is True, result
        assert "eval-result-42" in result["output"]

    def test_interaction_helpers(self, camofox_mode):
        result = _exec(
            "new_tab('http://127.0.0.1:8080')\n"
            "click_ref('e1')\n"
            "fill_input('#search', 'camofox')\n"
            "press_key('Enter')\n"
            "print('done')",
            task_id="e2e-interact",
        )
        assert result["success"] is True, result
        assert "done" in result["output"]

    def test_screenshot_attached(self, camofox_mode):
        result = _exec("capture_screenshot()", task_id="e2e-shot")
        assert result["success"] is True, result
        path = result.get("screenshot_path")
        assert path, result
        assert os.path.isfile(path)
        with open(path, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_tab_recovery_on_404(self, camofox_mode):
        """A garbage-collected tab is recreated; the cache follows."""
        first = _exec(
            "new_tab('http://127.0.0.1:8080')\nprint(page_info())",
            task_id="e2e-recover",
        )
        assert first["success"] is True, first
        assert camofox_mod._sessions["e2e-recover"]["tab_id"] == "t1"

        camofox_mode.kill_tab("t1")
        second = _exec("print(page_info())", task_id="e2e-recover")
        assert second["success"] is True, second
        assert "CAMOFOX_TAB_ID=t2" in second["output"]
        assert camofox_mod._sessions["e2e-recover"]["tab_id"] == "t2"

    def test_non_gone_errors_do_not_recreate(self, camofox_mode):
        """A 401 (bad API key) must propagate, not silently recreate the tab."""
        first = _exec(
            "new_tab('http://127.0.0.1:8080')\nprint(page_info())", task_id="e2e-401"
        )
        assert first["success"] is True, first
        assert camofox_mod._sessions["e2e-401"]["tab_id"] == "t1"

        camofox_mode.stats_error_code = 401
        second = _exec("print(page_info())", task_id="e2e-401")
        assert second["success"] is False
        assert "HTTP 401" in second.get("stderr", "")
        # no recreation happened: the cache still points at the stale tab
        assert camofox_mod._sessions["e2e-401"]["tab_id"] == "t1"
        assert "CAMOFOX_TAB_ID=t2" not in second.get("output", "")

    def test_inherited_tab_is_validated_once_per_exec(self, camofox_mode):
        first = _exec("new_tab('http://127.0.0.1:8080')", task_id="e2e-stats")
        assert first["success"] is True, first
        before = camofox_mode.stats_requests

        second = _exec(
            "print(page_info())\nprint(get_links())\nclick_ref('e1')",
            task_id="e2e-stats",
        )
        assert second["success"] is True, second
        assert camofox_mode.stats_requests == before + 1

    def test_clear_error_without_server(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "camofox"}},
        )
        monkeypatch.setattr("tools.browser_camofox.get_camofox_url", lambda: "")
        result = _exec("print('x')", task_id="e2e-nosrv")
        assert result.get("success") is not True
        assert "CAMOFOX_URL" in result.get("error", "")

    def test_always_blocked_metadata_url_rejected(self, camofox_mode):
        result = _exec(
            "url = 'http://' + '169.254.169.254/latest/meta-data'\nnew_tab(url)",
            task_id="e2e-blocked",
        )
        assert result.get("success") is not True
        assert "cloud metadata" in result.get("stderr", "")

    def test_dynamic_private_navigation_allowed_by_existing_policy(
        self, camofox_mode, monkeypatch
    ):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "camofox", "allow_private_urls": True}},
        )
        result = _exec(
            "host = '127.0.0.1'\nurl = f'http://{host}:8080/app'\nprint(new_tab(url))",
            task_id="e2e-private",
        )
        assert result["success"] is True, result
        assert camofox_mode.tabs["t1"]["url"] == "http://127.0.0.1:8080/app"

    def test_loopback_rewrite_uses_existing_camofox_setting(
        self, camofox_mode, monkeypatch
    ):
        monkeypatch.setenv("CAMOFOX_REWRITE_LOOPBACK_URLS", "true")
        result = _exec(
            "url = 'http://127.0.0.1:3000/app'\nprint(new_tab(url))",
            task_id="e2e-rewrite",
        )
        assert result["success"] is True, result
        assert camofox_mode.tabs["t1"]["url"] == "http://host.docker.internal:3000/app"

    def test_snapshot_wait_and_eval_server_failures_are_not_success(self, camofox_mode):
        first = _exec("new_tab('http://127.0.0.1:8080')", task_id="e2e-errors")
        assert first["success"] is True, first

        camofox_mode.snapshot_error_code = 500
        snapshot = _exec("print(page_info())", task_id="e2e-errors")
        assert snapshot["success"] is False
        assert "HTTP 500" in snapshot.get("stderr", "")

        camofox_mode.snapshot_error_code = None
        camofox_mode.wait_ready = False
        wait = _exec("print(wait_for_load())", task_id="e2e-errors")
        assert wait["success"] is False
        assert "did not reach ready state" in wait.get("stderr", "")

        camofox_mode.wait_ready = True
        camofox_mode.evaluate_error_code = 503
        evaluated = _exec("print(js('document.title'))", task_id="e2e-errors")
        assert evaluated["success"] is False
        assert "HTTP 503" in evaluated.get("stderr", "")

    def test_last_tab_id_wins(self, camofox_mode):
        """Two new_tab() calls in one script: the cache must follow the LAST."""
        result = _exec(
            "new_tab('http://127.0.0.1:8080')\nnew_tab('http://127.0.0.1:8081')",
            task_id="e2e-two-tabs",
        )
        assert result["success"] is True, result
        assert "CAMOFOX_TAB_ID=t1" in result["output"]
        assert "CAMOFOX_TAB_ID=t2" in result["output"]
        assert camofox_mod._sessions["e2e-two-tabs"]["tab_id"] == "t2"

    def test_provider_credentials_are_scrubbed(self, camofox_mode, monkeypatch):
        """The model's code runs here — it must not inherit provider keys."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
        result = _exec(
            "import os\nprint('KEY=' + os.environ.get('ANTHROPIC_API_KEY', 'absent'))",
            task_id="e2e-scrub",
        )
        assert result["success"] is True, result
        assert "KEY=absent" in result["output"]
        assert "sk-should-not-leak" not in result["output"]

    def test_camofox_key_is_redacted_from_stdout_and_stderr(
        self, camofox_mode, monkeypatch
    ):
        secret = "camofox-secret-value-123"
        monkeypatch.setenv("CAMOFOX_API_KEY", secret)
        result = _exec(
            "import os, sys\n"
            "print(os.environ['CAMOFOX_API_KEY'])\n"
            "print(os.environ['CAMOFOX_API_KEY'], file=sys.stderr)",
            task_id="e2e-redact",
        )
        assert result["success"] is True, result
        assert secret not in json.dumps(result)
        assert "[REDACTED_CAMOFOX_API_KEY]" in result["output"]

    def test_stdout_is_bounded_with_recovery_metadata(self, camofox_mode):
        result = _exec("print('x' * 60000)", task_id="e2e-output-cap")
        assert result["success"] is True, result
        assert result["stdout_truncated"] is True
        assert result["stdout_bytes_total"] > result["stdout_bytes_captured"]
        assert result.get("stdout_spill_path")

    def test_nonempty_session_argument_is_rejected(self, camofox_mode):
        result = json.loads(
            bu_cli.browser_exec("print('x')", session="named", task_id="e2e-session")
        )
        assert "success" not in result
        assert "does not support" in result["error"]

    def test_same_task_calls_are_serialized_and_lock_is_released(self, monkeypatch):
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def fake_exec(code, timeout_s, task_id):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return code

        monkeypatch.setattr(camofox_exec_mod, "_browser_exec_unlocked", fake_exec)
        results = []
        threads = [
            threading.Thread(
                target=lambda value=value: results.append(
                    camofox_exec_mod.browser_exec(value, task_id="same-task")
                )
            )
            for value in ("first", "second")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(results) == ["first", "second"]
        assert max_active == 1
        assert camofox_exec_mod._TASK_LOCKS == {}

    def test_agent_helpers_auto_imported(self, camofox_mode, monkeypatch, tmp_path):
        """The description promises agent_helpers.py is auto-imported."""
        monkeypatch.setenv("BH_AGENT_WORKSPACE", str(tmp_path))
        (tmp_path / "agent_helpers.py").write_text(
            "def my_helper():\n    return 'helper-ran'\n", encoding="utf-8"
        )
        result = _exec("print(my_helper())", task_id="e2e-helpers")
        assert result["success"] is True, result
        assert "helper-ran" in result["output"]

    def test_agent_helpers_cannot_replace_runtime_helpers(
        self, camofox_mode, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("BH_AGENT_WORKSPACE", str(tmp_path))
        (tmp_path / "agent_helpers.py").write_text(
            "def new_tab(url):\n    raise RuntimeError('replaced')\n"
            "def custom_helper():\n    return 'custom'\n",
            encoding="utf-8",
        )
        result = _exec(
            "new_tab('http://127.0.0.1:8080')\nprint(custom_helper())",
            task_id="e2e-helper-collision",
        )
        assert result["success"] is True, result
        assert "custom" in result["output"]
        assert "ignored reserved helper name: new_tab" in result["stderr"]

    def test_screenshot_uses_multimodal_tool_envelope(self, camofox_mode, monkeypatch):
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: True
        )
        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *args, **kwargs: "data:image/jpeg;base64,abc",
        )
        result = bu_cli.browser_exec("capture_screenshot()", task_id="e2e-native-shot")
        assert isinstance(result, dict) and result["_multimodal"] is True
        assert [part["type"] for part in result["content"]] == ["text", "image_url"]
        assert result["meta"]["screenshot_path"]


# ---------------------------------------------------------------------------
# Schema/description
# ---------------------------------------------------------------------------


class TestSchema:
    def test_camofox_description_when_backend_camofox(
        self, monkeypatch, camofox_server
    ):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "camofox"}},
        )
        overrides = bu_cli._dynamic_schema_overrides()
        assert "Camofox backend" in overrides["description"]
        assert "click_ref" in overrides["description"]
        assert "session" not in overrides["parameters"]["properties"]

    def test_cli_description_when_backend_browser_use(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"browser": {"backend": "browser-use"}},
        )
        overrides = bu_cli._dynamic_schema_overrides()
        assert "Camofox backend" not in overrides["description"]
