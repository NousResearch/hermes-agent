"""Regression for #155838: ``pre_keyless_rescue`` gates cloud web rescue.

Covers the real web tool dispatch path, not the rescue helpers directly.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import hermes_cli.plugins as plugins_mod
from plugins.web import keyless_mcp
from tools import web_tools


class _KeylessRescueBoomProvider:
    name = "keenable"
    display_name = "Keenable"

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def is_available(self):
        return True

    def search(self, query, limit=5):
        return {"success": False, "error": "HTTP 500 upstream exploded"}

    def extract(self, urls, **kwargs):
        return [
            {"url": url, "title": "", "content": "", "error": "HTTP 500 upstream exploded"}
            for url in urls
        ]


class _RaisingProvider(_KeylessRescueBoomProvider):
    def search(self, query, limit=5):
        raise RuntimeError("connection reset by peer")



def _ring_search_success(provider_name, query, limit):
    return {
        "success": True,
        "data": {"web": [{"url": "https://ring.example"}]},
    }



def _forbid_ring(*args, **kwargs):
    raise AssertionError("ring rescue should not run when vetoed")


@pytest.fixture(autouse=True)
def _keyed_keenable_env(monkeypatch):
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "kn-real" if name == "KEENABLE_API_KEY" else "",
    )
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "keenable"})
    monkeypatch.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: True)
    yield


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    yield home
    plugins_mod._reset_plugin_managers_for_tests()


@pytest.fixture
def _web_tool_setup(monkeypatch):
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)

    async def _allow_all(url, **kwargs):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)


def _write_plugin(home: Path, name: str, body: str) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        textwrap.dedent(
            f"""
            name: {name}
            version: 0.0.1
            description: keyless rescue hook probe
            provides_hooks:
              - pre_keyless_rescue
            """
        ).strip() + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    (home / "config.yaml").write_text(
        textwrap.dedent(
            f"""
            plugins:
              enabled:
                - {name}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins(force=True)


def _dispatch_search(monkeypatch, provider):
    monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: provider)
    return json.loads(web_tools.web_search_tool("sensitive query", limit=2))


async def _dispatch_extract(monkeypatch, provider):
    monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: provider)
    raw = await web_tools.web_extract_tool(["https://example.invalid/a", "https://example.invalid/b"])
    data = json.loads(raw)
    return data["results"] if isinstance(data, dict) and "results" in data else data


class TestPreKeylessRescueHook:
    @pytest.mark.asyncio
    async def test_veto_blocks_search_and_extract_and_annotates_refusal(self, _isolated_home,
                                                                      _web_tool_setup, monkeypatch):
        _write_plugin(
            _isolated_home,
            "pre-keyless-rescue-veto",
            """
            def register(ctx):
                def veto(**kwargs):
                    return False

                ctx.register_hook("pre_keyless_rescue", veto)
            """,
        )
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(keyless_mcp, "search_with_failover", _forbid_ring)
            patcher.setattr(keyless_mcp, "extract_with_failover", _forbid_ring)
            search = _dispatch_search(patcher, _KeylessRescueBoomProvider())
            extract = await _dispatch_extract(patcher, _KeylessRescueBoomProvider())
        assert search["success"] is False
        assert "pre_keyless_rescue" in search["error"]
        assert all("pre_keyless_rescue" in entry.get("error", "") for entry in extract)

    @pytest.mark.parametrize(
        "plugin_body, enabled",
        [
            (None, False),
            (
                """
                def register(ctx):
                    def allow(**kwargs):
                        return True

                    ctx.register_hook("pre_keyless_rescue", allow)
                """,
                True,
            ),
        ],
    )
    def test_absent_or_truthy_callback_keeps_today_behavior(self, _isolated_home, _web_tool_setup,
                                                            monkeypatch, plugin_body, enabled):
        if enabled:
            _write_plugin(_isolated_home, "pre-keyless-rescue-allow", plugin_body)
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(keyless_mcp, "search_with_failover", _ring_search_success)
            result = _dispatch_search(patcher, _KeylessRescueBoomProvider())
        assert result["success"] is True
        assert result["data"]["rescued_from"] == "keenable"
        assert "pre_keyless_rescue" not in json.dumps(result)

    def test_callback_exception_fails_closed_and_logs(self, _isolated_home, _web_tool_setup,
                                                      monkeypatch, caplog):
        # Like pre_tool_call: a guard that raised made no decision, so the data stays home. The
        # call keeps its own error, which is what it returns without the rescue anyway.
        _write_plugin(
            _isolated_home,
            "pre-keyless-rescue-raise",
            """
            def register(ctx):
                def boom(**kwargs):
                    raise RuntimeError("kaboom")

                ctx.register_hook("pre_keyless_rescue", boom)
            """,
        )
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(keyless_mcp, "search_with_failover", _forbid_ring)
            with caplog.at_level("WARNING"):
                result = _dispatch_search(patcher, _RaisingProvider())
        assert result["success"] is False
        assert "connection reset by peer" in result["error"] and "kaboom" in result["error"]
        assert any("kaboom" in record.message or "pre_keyless_rescue" in record.message for record in caplog.records)

    def test_signature_inspection_allows_kind_only_callback(self, _isolated_home, _web_tool_setup,
                                                           monkeypatch):
        _write_plugin(
            _isolated_home,
            "pre-keyless-rescue-kind-only",
            """
            def register(ctx):
                def kind_only(kind):
                    assert kind == "search"
                    return True

                ctx.register_hook("pre_keyless_rescue", kind_only)
            """,
        )
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(keyless_mcp, "search_with_failover", _ring_search_success)
            result = _dispatch_search(patcher, _KeylessRescueBoomProvider())
        assert result["success"] is True
        assert result["data"]["rescued_from"] == "keenable"


# --- Directive contract: the same shape and the same human gate as pre_tool_call ----------------------


def _record_ring(calls):
    def search(provider_name, query, limit):
        calls.append(("search", query))
        return {"success": True, "data": {"web": [{"url": "https://ring.example"}]}}

    def extract(provider_name, urls):
        calls.extend(("page", url) for url in urls)
        return [{"url": url, "title": url, "content": "fetched by the vendor"} for url in urls]
    return search, extract


def _plugin_returning(home, name, returned):
    _write_plugin(
        home,
        name,
        f"""
        def register(ctx):
            def hook(kind, backend_name, error, **kwargs):
                return {returned}

            ctx.register_hook("pre_keyless_rescue", hook)
        """,
    )


async def _both(patcher, provider, calls):
    search, extract = _record_ring(calls)
    patcher.setattr(keyless_mcp, "search_with_failover", search)
    patcher.setattr(keyless_mcp, "extract_with_failover", extract)
    return _dispatch_search(patcher, provider), await _dispatch_extract(patcher, provider)


class TestPreKeylessRescueDirectives:
    @pytest.mark.asyncio
    async def test_block_directive_keeps_the_data_home(self, _isolated_home, _web_tool_setup, monkeypatch):
        _plugin_returning(_isolated_home, "rescue-block",
                          '{"action": "block", "message": "kept on this machine by the plugin"}')
        calls = []
        with pytest.MonkeyPatch.context() as patcher:
            search, extract = await _both(patcher, _KeylessRescueBoomProvider(), calls)
        assert calls == []
        assert search["success"] is False and "kept on this machine by the plugin" in search["error"]
        assert all("kept on this machine by the plugin" in e.get("error", "") for e in extract)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("approved", [True, False])
    async def test_approve_directive_asks_the_human_gate(self, _isolated_home, _web_tool_setup, monkeypatch,
                                                         approved):
        _plugin_returning(_isolated_home, "rescue-approve",
                          '{"action": "approve", "message": "send it to an outside service?", '
                          '"rule_key": "probe:keyless_rescue"}')
        asked = []

        def gate(tool_name, reason, *, rule_key="", approval_callback=None):
            asked.append((tool_name, reason, rule_key))
            return {"approved": approved, "message": "BLOCKED: Action denied by user."}

        monkeypatch.setattr("tools.approval.request_tool_approval", gate)
        calls = []
        with pytest.MonkeyPatch.context() as patcher:
            search, extract = await _both(patcher, _KeylessRescueBoomProvider(), calls)
        assert [a[0] for a in asked] == ["web_search", "web_extract"]
        assert all(a[1] == "send it to an outside service?" and a[2] == "probe:keyless_rescue" for a in asked)
        if approved:
            assert [kind for kind, _ in calls] == ["search", "page", "page"]
            assert search["success"] is True
        else:
            assert calls == []
            assert "denied by user" in search["error"]
            assert all("denied by user" in e.get("error", "") for e in extract)

    @pytest.mark.asyncio
    async def test_approve_with_nobody_to_answer_fails_closed(self, _isolated_home, _web_tool_setup,
                                                              monkeypatch):
        # The real gate: no CLI, no gateway, no approval bridge in this process.
        _plugin_returning(_isolated_home, "rescue-approve-unattended",
                          '{"action": "approve", "message": "send it to an outside service?"}')
        calls = []
        with pytest.MonkeyPatch.context() as patcher:
            search, _extract = await _both(patcher, _KeylessRescueBoomProvider(), calls)
        assert calls == []
        assert search["success"] is False and "BLOCKED" in search["error"]

    @pytest.mark.asyncio
    async def test_callback_that_raises_keeps_the_data_home(self, _isolated_home, _web_tool_setup,
                                                            monkeypatch):
        _write_plugin(
            _isolated_home,
            "rescue-raise",
            """
            def register(ctx):
                def boom(**kwargs):
                    raise RuntimeError("kaboom")

                ctx.register_hook("pre_keyless_rescue", boom)
            """,
        )
        calls = []
        with pytest.MonkeyPatch.context() as patcher:
            search, _extract = await _both(patcher, _KeylessRescueBoomProvider(), calls)
        assert calls == []
        assert "kaboom" in search["error"] and "HTTP 500 upstream exploded" in search["error"]

    @pytest.mark.asyncio
    async def test_none_is_no_opinion(self, _isolated_home, _web_tool_setup, monkeypatch):
        _plugin_returning(_isolated_home, "rescue-none", "None")
        calls = []
        with pytest.MonkeyPatch.context() as patcher:
            search, _ = await _both(patcher, _KeylessRescueBoomProvider(), calls)
        assert [kind for kind, _ in calls] == ["search", "page", "page"] and search["success"] is True
