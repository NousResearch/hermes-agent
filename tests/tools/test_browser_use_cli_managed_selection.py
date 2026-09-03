"""Regression coverage for canonical managed Browser Use selection (#93865)."""

import tools.browser_use_cli as bu_cli


def test_canonical_nous_selection_routes_browser_use_through_gateway(monkeypatch):
    import tools.browser_tool as bt

    class _BrowserUseProvider:
        name = "browser-use"

    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {
            "browser": {
                "backend": "browser-use",
                "cloud_provider": "nous",
            }
        },
    )
    monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
    monkeypatch.setattr(bt, "_get_cloud_provider", lambda: _BrowserUseProvider())

    seen = []

    def session_info(cache_key):
        seen.append(cache_key)
        return {"cdp_url": "wss://managed.example/cdp/session"}

    monkeypatch.setattr(bt, "_get_session_info", session_info)

    env = {}
    assert bu_cli._resolve_backend_cdp(env, "task-1") is None
    assert seen == ["task-1"]
    assert env["BU_CDP_WS"] == "wss://managed.example/cdp/session"


def test_direct_browser_use_selection_stays_on_native_cli_path(monkeypatch):
    import tools.browser_tool as bt

    class _BrowserUseProvider:
        name = "browser-use"

    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {
            "browser": {
                "backend": "browser-use",
                "cloud_provider": "browser-use",
            }
        },
    )
    monkeypatch.setattr(bt, "_get_cdp_override", lambda: "")
    monkeypatch.setattr(bt, "_get_cloud_provider", lambda: _BrowserUseProvider())
    monkeypatch.setattr(
        bt,
        "_get_session_info",
        lambda cache_key: (_ for _ in ()).throw(
            AssertionError("direct Browser Use must not provision a gateway session")
        ),
    )

    env = {}
    assert bu_cli._resolve_backend_cdp(env, "task-1") is None
    assert "BU_CDP_WS" not in env and "BU_CDP_URL" not in env


def test_empty_browser_provider_clears_stale_managed_selection():
    from hermes_cli.tools_config import _write_provider_config

    config = {
        "browser": {
            "cloud_provider": "nous",
            "use_gateway": True,
            "backend": "browser-use",
            "keep_me": "value",
        }
    }

    _write_provider_config(
        {"browser_provider": ""}, config, managed_feature=None
    )

    browser = config["browser"]
    assert "cloud_provider" not in browser
    assert "use_gateway" not in browser
    assert browser["backend"] == "browser-use"
    assert browser["keep_me"] == "value"

