"""Focused tests for the minimal CloakBrowser runtime wrapper."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


async def _async_noop(*args, **kwargs):
    return None


def _awaitable(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


class TestCloakBrowserMode:
    def test_enabled_when_config_selects_cloakbrowser(self):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        cfg = {
            "browser": {
                "cloud_provider": "cloakbrowser",
                "use_gateway": False,
                "cloakbrowser": {"enabled": True},
            }
        }
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is True

    def test_enabled_when_selector_chooses_cloakbrowser_without_enabled_flag(self):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        cfg = {"browser": {"cloud_provider": "cloakbrowser", "use_gateway": False}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is True

    def test_enabled_when_selector_uses_legacy_mixed_case_provider(self):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        cfg = {"browser": {"cloud_provider": " CloakBrowser ", "use_gateway": False}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is True

    def test_disabled_for_other_provider(self):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        cfg = {"browser": {"cloud_provider": "browserbase", "use_gateway": False}}
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is False

    def test_disabled_for_other_provider_even_with_stale_enabled_flag(self):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        cfg = {
            "browser": {
                "cloud_provider": "browser-use",
                "use_gateway": True,
                "cloakbrowser": {"enabled": True},
            }
        }
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is False

    def test_disabled_when_gateway_mode_enabled(self):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        cfg = {
            "browser": {
                "cloud_provider": "cloakbrowser",
                "use_gateway": True,
                "cloakbrowser": {"enabled": True},
            }
        }
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is False

    def test_cdp_override_takes_priority(self, monkeypatch):
        from tools.browser_cloakbrowser import is_cloakbrowser_mode

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        cfg = {
            "browser": {
                "cloud_provider": "cloakbrowser",
                "use_gateway": False,
                "cloakbrowser": {"enabled": True},
            }
        }
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is True

        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={
                "browser": {
                    "cloud_provider": "cloakbrowser",
                    "use_gateway": False,
                    "cdp_url": "http://example-host:9222",
                    "cloakbrowser": {"enabled": True},
                }
            },
        ):
            assert is_cloakbrowser_mode() is False

        monkeypatch.setenv("BROWSER_CDP_URL", "http://example-host:9223")
        with patch("hermes_cli.config.read_raw_config", return_value=cfg):
            assert is_cloakbrowser_mode() is False


class TestCloakBrowserAvailability:
    def test_missing_package_returns_false_without_importing_browser(self):
        from tools.browser_cloakbrowser import check_cloakbrowser_available

        with patch("importlib.util.find_spec", return_value=None) as mock_find_spec:
            assert check_cloakbrowser_available() is False

        mock_find_spec.assert_called_once_with("cloakbrowser")


class TestBuildLaunchOptions:
    def test_missing_config_uses_defaults(self, tmp_path):
        from tools import browser_cloakbrowser as bc

        with patch.object(bc, "get_hermes_home", return_value=tmp_path), patch(
            "tools.browser_cloakbrowser.hermes_config.load_config", return_value={}
        ):
            assert bc.build_cloakbrowser_launch_options() == {
                "headless": False,
                "humanize": True,
                "stealth_args": True,
                "user_data_dir": str(tmp_path / "cloakbrowser_profile"),
            }

    def test_empty_user_data_dir_resolves_to_profile_scoped_default(self, tmp_path):
        from tools import browser_cloakbrowser as bc

        cfg = {"browser": {"cloakbrowser": {"headless": True, "user_data_dir": ""}}}
        with patch.object(bc, "get_hermes_home", return_value=tmp_path), patch(
            "tools.browser_cloakbrowser.hermes_config.load_config", return_value=cfg
        ):
            options = bc.build_cloakbrowser_launch_options()

        assert options["headless"] is True
        assert options["user_data_dir"] == str(tmp_path / "cloakbrowser_profile")

    def test_explicit_user_data_dir_is_respected(self, tmp_path):
        from tools import browser_cloakbrowser as bc

        explicit_dir = tmp_path / "custom-profile"
        cfg = {
            "browser": {
                "cloakbrowser": {
                    "headless": True,
                    "user_data_dir": str(explicit_dir),
                    "timezone": "UTC",
                    "locale": "en-US",
                }
            }
        }
        with patch.object(bc, "get_hermes_home", return_value=tmp_path), patch(
            "tools.browser_cloakbrowser.hermes_config.load_config", return_value=cfg
        ):
            options = bc.build_cloakbrowser_launch_options()

        assert options["user_data_dir"] == str(explicit_dir)
        assert options["headless"] is True
        assert options["timezone"] == "UTC"
        assert options["locale"] == "en-US"
        assert Path(options["user_data_dir"]) == explicit_dir

    def test_supported_schema_keys_are_forwarded_and_unsupported_keys_are_ignored(self, tmp_path):
        from tools import browser_cloakbrowser as bc

        cfg = {
            "browser": {
                "cloakbrowser": {
                    "headless": True,
                    "humanize": False,
                    "proxy": "http://proxy.example:8080",
                    "geoip": True,
                    "locale": "en-US",
                    "timezone": "America/New_York",
                    "stealth_args": False,
                    "user_agent": "Slice4Agent/1.0",
                    "color_scheme": "dark",
                    "extra_args": ["--disable-gpu"],
                    "user_data_dir": str(tmp_path / "profile"),
                    "fingerprint_seed": "stale-unsupported-setting",
                }
            }
        }

        with patch.object(bc, "get_hermes_home", return_value=tmp_path), patch(
            "tools.browser_cloakbrowser.hermes_config.load_config", return_value=cfg
        ):
            options = bc.build_cloakbrowser_launch_options()

        assert options == {
            "headless": True,
            "humanize": False,
            "proxy": "http://proxy.example:8080",
            "geoip": True,
            "locale": "en-US",
            "timezone": "America/New_York",
            "stealth_args": False,
            "user_agent": "Slice4Agent/1.0",
            "color_scheme": "dark",
            "extra_args": ["--disable-gpu"],
            "user_data_dir": str(tmp_path / "profile"),
        }


class TestWrapperEntrypoints:
    def test_screenshot_returns_standard_data_path_shape(self, monkeypatch, tmp_path):
        from tools import browser_cloakbrowser as bc

        screenshot_bytes = b"\x89PNG\r\n\x1a\ncloakbrowser"
        page = SimpleNamespace(screenshot=_awaitable(screenshot_bytes))
        session = {"page": page, "refs": {}}
        shots_dir = tmp_path / "cache" / "screenshots"

        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(bc, "_run_on_session_loop", lambda session, coro, timeout=None: bc._run_async(coro))
        monkeypatch.setattr("hermes_constants.get_hermes_dir", lambda *parts: shots_dir)

        result = json.loads(bc.cloakbrowser_screenshot(task_id="task-shot"))

        assert result["success"] is True
        assert result["data"]["path"].endswith(".png")
        assert Path(result["data"]["path"]).read_bytes() == screenshot_bytes

    def test_screenshot_annotate_true_includes_snapshot_annotations(self, monkeypatch, tmp_path):
        from tools import browser_cloakbrowser as bc

        screenshot_bytes = b"\x89PNG\r\n\x1a\ncloakbrowser"
        page = SimpleNamespace(screenshot=_awaitable(screenshot_bytes))
        session = {"page": page, "refs": {}}
        snapshot = {
            "snapshot": '- button "Submit" [@e1]',
            "refs": {"e1": {"selector": "[data-hermes-ref=\"e1\"]", "tag": "button", "text": "Submit"}},
            "element_count": 1,
        }
        shots_dir = tmp_path / "cache" / "screenshots"

        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(bc, "_ensure_live_page", _awaitable(page))
        monkeypatch.setattr(bc, "_snapshot_page", _awaitable(snapshot))
        monkeypatch.setattr(bc, "_run_on_session_loop", lambda session, coro, timeout=None: bc._run_async(coro))
        monkeypatch.setattr("hermes_constants.get_hermes_dir", lambda *parts: shots_dir)

        result = json.loads(bc.cloakbrowser_screenshot(task_id="task-shot", annotate=True))

        assert result["success"] is True
        assert result["data"]["annotations"] == [{
            "ref": "@e1",
            "label": 'button "Submit"',
            "tag": "button",
            "text": "Submit",
            "selector": '[data-hermes-ref="e1"]',
        }]
        assert session["refs"] == snapshot["refs"]

    def test_launch_session_strips_duplicate_user_data_dir_for_persistent_launch(self, monkeypatch, tmp_path):
        from tools import browser_cloakbrowser as bc

        captured = {}
        page = SimpleNamespace()
        context = SimpleNamespace(pages=[page])

        async def fake_launch_persistent_context_async(*, user_data_dir, **kwargs):
            captured["user_data_dir"] = user_data_dir
            captured["kwargs"] = kwargs
            return context

        monkeypatch.setattr(
            bc,
            "_import_cloakbrowser_api",
            lambda: (None, fake_launch_persistent_context_async),
        )
        monkeypatch.setattr(
            bc,
            "build_cloakbrowser_launch_options",
            lambda: {
                "user_data_dir": str(tmp_path / "profile"),
                "headless": True,
                "humanize": True,
            },
        )

        session = bc._run_async(bc._launch_session())

        assert captured == {
            "user_data_dir": str(tmp_path / "profile"),
            "kwargs": {"headless": True, "humanize": True},
        }
        assert session["context"] is context
        assert session["page"] is page
        assert session["persistent"] is True

    def test_snapshot_click_type_scroll_back_press_use_runtime_state(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        page = SimpleNamespace(
            url="https://example.com/after",
            title=_awaitable("Example Page"),
            go_back=_awaitable(None),
        )
        page.keyboard = SimpleNamespace(press=_async_noop)
        page.mouse = SimpleNamespace(wheel=_async_noop)

        session = {
            "page": page,
            "refs": {
                "e1": {"selector": "#submit"},
                "e2": {"selector": "#name"},
            },
        }

        clicked = []
        filled = []
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(
            bc,
            "_snapshot_page",
            _awaitable({
                "snapshot": "- button \"Submit\" [@e1]",
                "element_count": 1,
                "refs": {"e1": {"selector": "#submit"}, "e2": {"selector": "#name"}},
            }),
        )
        monkeypatch.setattr(bc, "_click_selector", lambda page, selector: clicked.append((page, selector)) or _awaitable(None)())
        monkeypatch.setattr(bc, "_type_into_selector", lambda page, selector, text: filled.append((page, selector, text)) or _awaitable(None)())

        snap = json.loads(bc.cloakbrowser_snapshot(task_id="task-1"))
        assert snap == {"success": True, "snapshot": "- button \"Submit\" [@e1]", "element_count": 1}

        click = json.loads(bc.cloakbrowser_click("@e1", task_id="task-1"))
        assert click == {"success": True, "clicked": "@e1", "url": "https://example.com/after"}
        assert clicked == [(page, "#submit")]

        typed = json.loads(bc.cloakbrowser_type("e2", "hello", task_id="task-1"))
        assert typed == {"success": True, "typed": "hello", "element": "@e2"}
        assert filled == [(page, "#name", "hello")]

        scroll = json.loads(bc.cloakbrowser_scroll("down", task_id="task-1"))
        assert scroll == {"success": True, "scrolled": "down"}

        back = json.loads(bc.cloakbrowser_back(task_id="task-1"))
        assert back == {"success": True, "url": "https://example.com/after"}

        press = json.loads(bc.cloakbrowser_press("Enter", task_id="task-1"))
        assert press == {"success": True, "pressed": "Enter"}

    def test_navigate_uses_launch_session_and_auto_snapshot(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        page = SimpleNamespace(url="https://example.com/final", title=_awaitable("Example Title"))
        session = {"page": page, "refs": {"e1": {"selector": "#link"}}}
        navigated = []

        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)
        monkeypatch.setattr(bc, "_navigate_page", lambda page, url, **kw: navigated.append((page, url)) or _awaitable(None)())
        monkeypatch.setattr(bc, "_snapshot_page", _awaitable({"snapshot": "- link \"Docs\" [@e1]", "element_count": 1}))

        result = json.loads(bc.cloakbrowser_navigate("https://example.com/start", task_id="task-nav"))

        assert result == {
            "success": True,
            "url": "https://example.com/final",
            "title": "Example Title",
            "snapshot": "- link \"Docs\" [@e1]",
            "element_count": 1,
        }
        assert navigated == [(page, "https://example.com/start")]

    def test_navigate_returns_clear_error_when_snapshot_fails_after_successful_nav(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        page = SimpleNamespace(url="https://example.com/final", title=_awaitable("Example Title"))
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: {"page": page, "refs": {}})
        monkeypatch.setattr(bc, "_navigate_page", _awaitable(None))
        monkeypatch.setattr(bc, "_snapshot_page", lambda page, full=False, user_task=None: (_ for _ in ()).throw(RuntimeError("snap broke")))

        result = json.loads(bc.cloakbrowser_navigate("https://example.com/start", task_id="task-nav"))

        assert result["success"] is True
        assert result["snapshot_warning"] == "snap broke"

    def test_get_images_uses_runtime_page_and_matches_browser_tool_shape(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        page = SimpleNamespace()
        session = {"page": page, "refs": {}}
        captured = {}
        images = [
            {"src": "https://example.com/logo.png", "alt": "Logo", "width": 640, "height": 480},
            {"src": "https://example.com/banner.jpg", "alt": "", "width": 1200, "height": 300},
        ]

        async def fake_evaluate(script):
            captured["script"] = script
            return images

        page.evaluate = fake_evaluate
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)

        result = json.loads(bc.cloakbrowser_get_images(task_id="task-images"))

        assert result == {"success": True, "images": images, "count": 2}
        assert "document.images" in captured["script"]
        assert "startsWith('data:')" in captured["script"]

    def test_get_images_falls_back_to_empty_list_when_page_returns_non_list(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        page = SimpleNamespace(evaluate=_awaitable({"unexpected": True}))
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: {"page": page, "refs": {}})

        result = json.loads(bc.cloakbrowser_get_images(task_id="task-images"))

        assert result == {"success": True, "images": [], "count": 0}

    def test_ensure_session_runs_listener_registration_on_session_loop(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        class LoopBoundPage:
            def __init__(self):
                self.listeners = {}
                self.loop_ids = []

            def on(self, event, handler):
                self.loop_ids.append(id(__import__("asyncio").get_running_loop()))
                self.listeners[event] = handler

        loop, thread = bc._start_loop_thread("cloakbrowser-test-loop")
        session = {"page": LoopBoundPage(), "refs": {}, "_loop": loop, "_thread": thread}
        monkeypatch.setitem(bc._sessions, "loop-test", session)
        try:
            ready = bc._ensure_session("loop-test")
        finally:
            bc._sessions.pop("loop-test", None)
            bc._shutdown_loop_thread(loop, thread)

        assert ready is session
        assert len(session["page"].loop_ids) == 2
        assert session["_console_listeners_page"] is session["page"]

    def test_console_returns_buffered_messages_and_errors_and_clear_empties_buffers(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        session = {
            "page": SimpleNamespace(url="https://example.com"),
            "refs": {},
            "console_messages": [
                {"type": "log", "text": "hello", "source": "console"},
                {"type": "error", "text": "oops", "source": "console"},
            ],
            "page_errors": [
                {"message": "Uncaught TypeError", "source": "exception"},
            ],
        }
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: session)

        result = json.loads(bc.cloakbrowser_console(task_id="task-console"))

        assert result == {
            "success": True,
            "console_messages": [
                {"type": "log", "text": "hello", "source": "console"},
                {"type": "error", "text": "oops", "source": "console"},
            ],
            "js_errors": [
                {"message": "Uncaught TypeError", "source": "exception"},
            ],
            "total_messages": 2,
            "total_errors": 1,
        }

        cleared = json.loads(bc.cloakbrowser_console(clear=True, task_id="task-console"))
        assert cleared["success"] is True
        assert cleared["total_messages"] == 2
        assert cleared["total_errors"] == 1
        assert session["console_messages"] == []
        assert session["page_errors"] == []

    def test_eval_returns_structured_serializable_result(self, monkeypatch):
        from tools import browser_cloakbrowser as bc

        page = SimpleNamespace(url="https://example.com", evaluate=_awaitable('{"title":"Example"}'))
        monkeypatch.setattr(bc, "_ensure_session", lambda task_id=None: {"page": page, "refs": {}})

        result = json.loads(bc.cloakbrowser_eval("document.title", task_id="task-eval"))

        assert result == {
            "success": True,
            "result": {"title": "Example"},
            "result_type": "dict",
            "method": "cloakbrowser_native",
        }

    def test_session_listener_registration_buffers_console_and_pageerror_events(self):
        from tools import browser_cloakbrowser as bc

        class FakePage:
            def __init__(self):
                self.listeners = {}
                self.loop_ids = []

            def on(self, event, handler):
                self.loop_ids.append(id(__import__("asyncio").get_running_loop()))
                self.listeners[event] = handler

        page = FakePage()
        session = {"page": page, "refs": {}}

        bc._run_async(bc._ensure_session_ready(session))

        page.listeners["console"](SimpleNamespace(type="warning", text="watch out"))
        page.listeners["pageerror"](RuntimeError("boom"))

        assert len(page.loop_ids) == 2
        assert len(set(page.loop_ids)) == 1
        assert session["console_messages"] == [
            {"type": "warning", "text": "watch out", "source": "console"},
        ]
        assert session["page_errors"] == [
            {"message": "boom", "source": "exception"},
        ]

    def test_session_listener_registration_reregisters_after_page_recreation(self):
        from tools import browser_cloakbrowser as bc

        class FakePage:
            def __init__(self, closed=False):
                self.closed = closed
                self.listeners = {}

            def is_closed(self):
                return self.closed

            def on(self, event, handler):
                __import__("asyncio").get_running_loop()
                self.listeners[event] = handler

        first_page = FakePage(closed=True)
        replacement_page = FakePage()
        context = SimpleNamespace(new_page=_awaitable(replacement_page))
        session = {"page": first_page, "context": context, "refs": {"e1": {"selector": "#x"}}}

        page = bc._run_async(bc._ensure_session_ready(session))
        assert page is replacement_page
        assert session["page"] is replacement_page
        assert session["refs"] == {}
        assert session["_console_listeners_page"] is replacement_page
        replacement_page.listeners["console"](SimpleNamespace(type="info", text="fresh page"))

        assert session["console_messages"] == [
            {"type": "info", "text": "fresh page", "source": "console"},
        ]


class TestBrowserToolRouting:
    def test_check_requirements_requires_cloakbrowser_package(self, monkeypatch):
        from tools.browser_tool import check_browser_requirements

        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("tools.browser_tool.check_cloakbrowser_available", lambda: False)
        assert check_browser_requirements() is False

        monkeypatch.setattr("tools.browser_tool.check_cloakbrowser_available", lambda: True)
        assert check_browser_requirements() is True

    def test_is_local_backend_treats_cloakbrowser_as_local(self, monkeypatch):
        from tools.browser_tool import _is_local_backend

        monkeypatch.setattr("tools.browser_tool._get_cdp_override", lambda: "")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("tools.browser_tool._get_cloud_provider", lambda: "cloakbrowser")
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _is_local_backend() is True

    def test_is_local_backend_keeps_containerized_cloakbrowser_non_local(self, monkeypatch):
        from tools.browser_tool import _is_local_backend

        monkeypatch.setattr("tools.browser_tool._get_cdp_override", lambda: "")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("tools.browser_tool._get_cloud_provider", lambda: "cloakbrowser")
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        assert _is_local_backend() is False

    def test_browser_navigate_routes_to_cloakbrowser(self, monkeypatch):
        from tools.browser_tool import browser_navigate

        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("tools.browser_tool._sensitive_query_param_name", lambda url: None)
        monkeypatch.setattr("tools.browser_tool._is_always_blocked_url", lambda url: False)
        monkeypatch.setattr("tools.browser_tool._is_safe_url", lambda url: True)
        monkeypatch.setattr("tools.browser_tool.check_website_access", lambda url: None)
        monkeypatch.setattr(
            "tools.browser_tool.cloakbrowser_navigate",
            lambda url, task_id=None: json.dumps(
                {
                    "success": True,
                    "url": url,
                    "task_id": task_id,
                    "backend": "cloakbrowser",
                }
            ),
        )

        result = json.loads(browser_navigate("https://example.com", task_id="t-route"))
        assert result == {
            "success": True,
            "url": "https://example.com",
            "task_id": "t-route",
            "backend": "cloakbrowser",
        }

    def test_cloud_blocks_redirect_to_private_in_cloakbrowser_mode(self, monkeypatch):
        from tools.browser_tool import browser_navigate

        calls = []

        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._is_local_backend", lambda: False)
        monkeypatch.setattr("tools.browser_tool._allow_private_urls", lambda: False)
        monkeypatch.setattr("tools.browser_tool._sensitive_query_param_name", lambda url: None)
        monkeypatch.setattr("tools.browser_tool._is_always_blocked_url", lambda url: False)
        monkeypatch.setattr("tools.browser_tool._is_safe_url", lambda url: "192.168" not in url)
        monkeypatch.setattr("tools.browser_tool.check_website_access", lambda url: None)
        monkeypatch.setattr(
            "tools.browser_tool.cloakbrowser_navigate",
            lambda url, task_id=None: calls.append((url, task_id)) or json.dumps(
                {
                    "success": True,
                    "url": "http://192.168.1.1/internal",
                    "title": "Internal",
                    "snapshot": "secret",
                }
            ),
        )

        result = json.loads(browser_navigate("https://example.com/start", task_id="t-ssrf"))

        assert result == {
            "success": False,
            "error": "Blocked: redirect landed on a private/internal address",
        }
        assert calls == [
            ("https://example.com/start", "t-ssrf"),
            ("about:blank", "t-ssrf"),
        ]

    def test_containerized_cloakbrowser_blocks_redirect_to_imds(self, monkeypatch):
        from tools.browser_tool import browser_navigate

        calls = []

        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._is_local_backend", lambda: False)
        monkeypatch.setattr("tools.browser_tool._allow_private_urls", lambda: False)
        monkeypatch.setattr("tools.browser_tool._sensitive_query_param_name", lambda url: None)
        monkeypatch.setattr(
            "tools.browser_tool._is_always_blocked_url",
            lambda url: "169.254.169.254" in url,
        )
        monkeypatch.setattr("tools.browser_tool._is_safe_url", lambda url: True)
        monkeypatch.setattr("tools.browser_tool.check_website_access", lambda url: None)
        monkeypatch.setattr(
            "tools.browser_tool.cloakbrowser_navigate",
            lambda url, task_id=None: calls.append((url, task_id)) or json.dumps(
                {
                    "success": True,
                    "url": "http://169.254.169.254/latest/meta-data/",
                    "title": "IMDS",
                }
            ),
        )

        result = json.loads(browser_navigate("https://example.com/start", task_id="t-imds"))

        assert result == {
            "success": False,
            "error": "Blocked: redirect landed on a cloud metadata endpoint",
        }
        assert calls == [
            ("https://example.com/start", "t-imds"),
            ("about:blank", "t-imds"),
        ]

    def test_browser_snapshot_routes_to_cloakbrowser(self, monkeypatch):
        from tools.browser_tool import browser_snapshot

        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr(
            "tools.browser_tool.cloakbrowser_snapshot",
            lambda full=False, task_id=None, user_task=None: json.dumps(
                {
                    "success": True,
                    "full": full,
                    "task_id": task_id,
                    "user_task": user_task,
                    "backend": "cloakbrowser",
                }
            ),
        )

        result = json.loads(browser_snapshot(full=True, task_id="t-snap", user_task="inspect"))
        assert result == {
            "success": True,
            "full": True,
            "task_id": "t-snap",
            "user_task": "inspect",
            "backend": "cloakbrowser",
        }

    def test_browser_actions_route_to_cloakbrowser(self, monkeypatch):
        from tools import browser_tool as bt

        monkeypatch.setattr(bt, "_is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr(bt, "cloakbrowser_click", lambda ref, task_id=None: json.dumps({"tool": "click", "ref": ref, "task_id": task_id}))
        monkeypatch.setattr(bt, "cloakbrowser_type", lambda ref, text, task_id=None: json.dumps({"tool": "type", "ref": ref, "text": text, "task_id": task_id}))
        monkeypatch.setattr(bt, "cloakbrowser_scroll", lambda direction, task_id=None: json.dumps({"tool": "scroll", "direction": direction, "task_id": task_id}))
        monkeypatch.setattr(bt, "cloakbrowser_back", lambda task_id=None: json.dumps({"tool": "back", "task_id": task_id}))
        monkeypatch.setattr(bt, "cloakbrowser_press", lambda key, task_id=None: json.dumps({"tool": "press", "key": key, "task_id": task_id}))
        monkeypatch.setattr(bt, "cloakbrowser_get_images", lambda task_id=None: json.dumps({"tool": "get_images", "task_id": task_id}))
        monkeypatch.setattr(bt, "cloakbrowser_console", lambda clear=False, task_id=None: json.dumps({"tool": "console", "clear": clear, "task_id": task_id}))

        assert json.loads(bt.browser_click("@e1", task_id="t1")) == {"tool": "click", "ref": "@e1", "task_id": "t1"}
        assert json.loads(bt.browser_type("@e2", "hello", task_id="t2")) == {"tool": "type", "ref": "@e2", "text": "hello", "task_id": "t2"}
        assert json.loads(bt.browser_scroll("down", task_id="t3")) == {"tool": "scroll", "direction": "down", "task_id": "t3"}
        assert json.loads(bt.browser_back(task_id="t4")) == {"tool": "back", "task_id": "t4"}
        assert json.loads(bt.browser_press("Enter", task_id="t5")) == {"tool": "press", "key": "Enter", "task_id": "t5"}
        assert json.loads(bt.browser_get_images(task_id="t6")) == {"tool": "get_images", "task_id": "t6"}
        assert json.loads(bt.browser_console(clear=True, task_id="t7")) == {"tool": "console", "clear": True, "task_id": "t7"}

    def test_cloakbrowser_activity_updates_for_headed_and_headless_sessions(self, monkeypatch):
        from tools import browser_tool as bt

        monkeypatch.setattr(bt, "_is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr(bt, "_last_session_key", lambda task_id: task_id)
        monkeypatch.setattr(bt, "_blocked_private_page_snapshot", lambda task_id: None)
        monkeypatch.setattr(bt, "_blocked_private_page_action", lambda task_id, action: None)

        started = []
        touched = []
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: started.append(True))
        monkeypatch.setattr(bt, "_update_session_activity", lambda task_id: touched.append(task_id))
        monkeypatch.setattr(bt, "cloakbrowser_snapshot", lambda full=False, task_id=None, user_task=None: json.dumps({"success": True, "task_id": task_id, "full": full}))
        monkeypatch.setattr(bt, "cloakbrowser_click", lambda ref, task_id=None: json.dumps({"success": True, "task_id": task_id, "ref": ref}))

        headed = json.loads(bt.browser_snapshot(task_id="headed-task"))
        headless = json.loads(bt.browser_click("@e1", task_id="headless-task"))

        assert headed == {"success": True, "task_id": "headed-task", "full": False}
        assert headless == {"success": True, "task_id": "headless-task", "ref": "@e1"}
        assert started == [True, True]
        assert touched == ["headed-task", "headless-task"]


class TestPerTurnCleanup:
    def test_native_cloakbrowser_survives_per_turn_cleanup(self, monkeypatch):
        from agent.chat_completion_helpers import cleanup_task_resources

        calls = []
        monkeypatch.setattr("tools.browser_tool._is_cloakbrowser_mode", lambda: True)
        monkeypatch.setattr("run_agent.cleanup_vm", lambda task_id: calls.append(("vm", task_id)))
        monkeypatch.setattr("run_agent.cleanup_browser", lambda task_id: calls.append(("browser", task_id)))

        agent = SimpleNamespace(verbose_logging=True)
        cleanup_task_resources(agent, "cloak-task")

        assert calls == [("vm", "cloak-task")]
