"""Required web mediation contract, using real plugin discovery and native dispatch.

The consumer is deliberately synthetic: no Provider, credential or ledger exists.
Network transport is replaced only at the external rescue boundary; dispatch,
configuration, scoped registration and URL checks remain real.
"""

import asyncio
import json

import pytest
import yaml

from agent.web_search_provider import WebSearchProvider
from hermes_constants import get_hermes_home, hermes_home_key

PLUGIN = r"""
from agent.web_search_provider import WebSearchProvider

class QuotaConsumer(WebSearchProvider):
    name = 'lab-quota'
    request_policy_version = 1
    available = True
    search_capable = True
    extract_capable = True
    mode = 'ok'

    def __init__(self):
        self.calls = []

    def is_available(self):
        return self.available

    def supports_search(self):
        return self.search_capable

    def supports_extract(self):
        return self.extract_capable

    def search(self, query, limit=5):
        self.calls.append(('search', query, limit))
        if self.mode == 'raise':
            raise RuntimeError('outcome_unknown: synthetic lost reply')
        if self.mode == 'deny':
            return {'success': False, 'error': 'synthetic_quota_denied'}
        return {'success': True, 'data': {'web': [
            {'title': 'offline', 'url': 'https://93.184.216.34/', 'description': 'synthetic'}]}}

    async def extract(self, urls, **kwargs):
        self.calls.append(('extract', list(urls)))
        if self.mode == 'raise':
            raise RuntimeError('outcome_unknown: synthetic lost reply')
        if self.mode == 'deny':
            return [{'url': u, 'error': 'synthetic_quota_denied'} for u in urls]
        return [{'url': u, 'content': 'synthetic page', 'title': 'offline'} for u in urls]

consumer = QuotaConsumer()
def register(ctx):
    ctx.register_web_search_provider(consumer)
"""


@pytest.fixture
def lab(tmp_path, monkeypatch):
    from agent import web_search_registry as providers
    from hermes_cli.plugins import _ensure_plugins_discovered, get_plugin_manager

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    empty = tmp_path / "empty-bundled"
    empty.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty))
    path = home / "plugins" / "quota-lab"
    path.mkdir(parents=True)
    (path / "plugin.yaml").write_text(
        "name: quota-lab\nversion: 0.1.0\nentry: __init__.py\n"
    )
    (path / "__init__.py").write_text(PLUGIN)
    config = {
        "plugins": {"enabled": ["quota-lab"]},
        "web": {
            "required_provider": "lab-quota",
            "backend": "lab-quota",
            "cache_enabled": True,
            "keyless_rescue": True,
            "keyless_fallback": True,
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    _ensure_plugins_discovered()
    provider = providers.get_provider("lab-quota")
    assert provider is not None, get_plugin_manager()._plugins
    # Every attempted native escape is visible, without creating a real network client.
    escapes = []

    def rescue_search(*args):
        escapes.append(("search", args))
        return {"success": True, "data": {"web": []}}

    def rescue_extract(*args):
        escapes.append(("extract", args))
        return [{"url": u, "content": "unmanaged rescue"} for u in args[1]]

    from plugins.web import keyless_mcp

    monkeypatch.setattr(keyless_mcp, "search_with_failover", rescue_search)
    monkeypatch.setattr(keyless_mcp, "extract_with_failover", rescue_extract)
    yield home, provider, escapes, config
    get_plugin_manager().unload()


def _call(capability, entry="model"):
    from model_tools import handle_function_call
    from tools.registry import registry
    from tools.web_tools import web_search_tool, web_extract_tool

    name = "web_" + capability
    args = (
        {"query": "synthetic query", "limit": 3}
        if capability == "search"
        else {"urls": ["https://93.184.216.34/offline"]}
    )
    if entry == "model":
        result = handle_function_call(name, args)
    elif entry == "registry":
        result = registry.dispatch(name, args)
    else:
        result = (
            web_search_tool(**args)
            if capability == "search"
            else asyncio.run(web_extract_tool(**args))
        )
    return json.loads(result)


@pytest.mark.parametrize("capability", ["search", "extract"])
@pytest.mark.parametrize("entry", ["model", "registry", "direct"])
def test_every_native_request_reaches_required_consumer(lab, capability, entry):
    _, provider, escapes, _ = lab
    for _ in range(2):
        result = _call(capability, entry)
        assert not result.get("error"), result
        assert result.get("success", True), result
    assert len(provider.calls) == 2  # a cache hit must not bypass admission
    if capability == "search":
        assert [call[2] for call in provider.calls] == [
            3,
            3,
        ]  # no inflated bucket/fan-out
    assert escapes == []


@pytest.mark.parametrize("capability", ["search", "extract"])
@pytest.mark.parametrize("entry", ["model", "registry", "direct"])
@pytest.mark.parametrize(
    "fault",
    [
        "same-signature-rename",
        "malformed",
        "scalar",
        "false-root",
        "list-root",
        "invalid-utf8",
        "read-denied",
        "discovery-denied",
        "missing-directory",
        "not-directory",
        "overlay-error",
        "leaf-read-missing",
        "directory-vanishes-on-read",
    ],
)
def test_managed_policy_failure_never_dispatches_or_rescues(
    lab, tmp_path, monkeypatch, capability, entry, fault
):
    """H01: every native entry must reject stale/partial managed policy after binding."""
    import os
    from pathlib import Path
    from hermes_cli import config as cfg, managed_scope
    from hermes_cli.config_effective import load_user_config_effective

    home, provider, escapes, _ = lab
    managed = tmp_path / "managed"
    managed.mkdir()
    policy = managed / "config.yaml"
    original = "web:\n  required_provider: lab-quota\n"
    policy.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    # User repeats the same binding: a swallowed overlay error must not hide behind it.
    assert _call(capability, entry).get("success", True)
    assert len(provider.calls) == 1 and escapes == []
    assert (
        load_user_config_effective(home / "config.yaml")["web"]["required_provider"]
        == "lab-quota"
    )
    provider.calls.clear()

    def rename_same_signature():
        before = policy.stat()
        policy.write_text(original.replace("lab-quota", "bad-quota"), encoding="utf-8")
        os.utime(policy, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = policy.stat()
        assert (after.st_size, after.st_mtime_ns) == (
            before.st_size,
            before.st_mtime_ns,
        )

    read_text, path_stat = Path.read_text, Path.stat

    def denied_read(path, *args, **kwargs):
        if path == policy:
            raise PermissionError("synthetic managed read denial")
        return read_text(path, *args, **kwargs)

    def denied_discovery(path, *args, **kwargs):
        if path == managed:
            raise PermissionError("synthetic managed discovery denial")
        return path_stat(path, *args, **kwargs)

    def remove_directory():
        managed.rename(tmp_path / "retired-managed")

    def replace_directory():
        remove_directory()
        managed.write_text("not a directory", encoding="utf-8")

    def broken_overlay(*args, **kwargs):
        raise ValueError("synthetic managed overlay failure")

    def missing_read(path, *args, **kwargs):
        if path == policy:
            raise FileNotFoundError("synthetic read failure on an existing leaf")
        return read_text(path, *args, **kwargs)

    def vanished_directory_read(path, *args, **kwargs):
        if path == policy:
            remove_directory()
        return read_text(path, *args, **kwargs)

    mutations = {
        "same-signature-rename": rename_same_signature,
        "malformed": lambda: policy.write_text("web: [broken", encoding="utf-8"),
        "scalar": lambda: policy.write_text("42", encoding="utf-8"),
        "false-root": lambda: policy.write_text("false", encoding="utf-8"),
        "list-root": lambda: policy.write_text("[]", encoding="utf-8"),
        "invalid-utf8": lambda: policy.write_bytes(b"\xff"),
        "read-denied": lambda: fault_patch.setattr(Path, "read_text", denied_read),
        "discovery-denied": lambda: fault_patch.setattr(Path, "stat", denied_discovery),
        "missing-directory": remove_directory,
        "not-directory": replace_directory,
        "overlay-error": lambda: fault_patch.setattr(
            cfg, "_deep_merge", broken_overlay
        ),
        "leaf-read-missing": lambda: fault_patch.setattr(
            Path, "read_text", missing_read
        ),
        "directory-vanishes-on-read": lambda: fault_patch.setattr(
            Path, "read_text", vanished_directory_read
        ),
    }
    with monkeypatch.context() as fault_patch:
        # Fault injections must be removed before fixture teardown/recovery.
        mutations[fault]()
        result = _call(capability, entry)
        assert "required_web_provider" in result.get("error", ""), result
        assert provider.calls == escapes == []
        if fault == "same-signature-rename":
            # Same bytes, only cache invalidation changes: both reads must refuse.
            changed_bytes = policy.read_bytes()
            managed_scope.invalidate_managed_cache()
            assert "required_web_provider" in _call(capability, entry).get("error", "")
            assert policy.read_bytes() == changed_bytes
            assert provider.calls == escapes == []

    # Restoring valid policy keeps the same binding usable (not durable revocation).
    if not managed.is_dir():
        if managed.exists():
            managed.unlink()
        managed.mkdir()
    policy.write_text(original, encoding="utf-8")
    assert _call(capability, entry).get("success", True)
    assert len(provider.calls) == 1 and escapes == []


@pytest.mark.parametrize("capability", ["search", "extract"])
@pytest.mark.parametrize("mode", ["deny", "raise"])
def test_required_denial_and_unknown_never_rescue(lab, capability, mode):
    _, provider, escapes, _ = lab
    provider.mode = mode
    result = _call(capability)
    assert "synthetic_" in json.dumps(result) or "outcome_unknown" in json.dumps(
        result
    ), result
    assert len(provider.calls) == 1
    assert escapes == []


@pytest.mark.parametrize("capability", ["search", "extract"])
def test_unloaded_consumer_does_not_expose_fallback(lab, capability):
    from agent.transports import codex as codex_transport
    from agent.web_required_provider import RequiredWebProviderError
    from agent.web_search_provider import WebSearchProvider
    from agent import web_search_registry as providers
    from hermes_cli.plugins import get_plugin_manager

    class Other(WebSearchProvider):
        name = "other"
        calls = []

        def is_available(self):
            return True

        def supports_extract(self):
            return True

        def search(self, query, limit=5):
            self.calls.append(query)
            return {"success": True, "data": {"web": []}}

        def extract(self, urls, **kwargs):
            self.calls.append(urls)
            return [{"url": u, "content": "unmanaged"} for u in urls]

    home, provider, escapes, config = lab
    other = Other()
    providers.register_provider(other, scope=hermes_home_key())
    assert get_plugin_manager().unload("quota-lab")
    assert codex_transport._xai_prefers_native_web_search() is False
    assert codex_transport._openai_prefers_native_web_search() is False
    result = _call(capability)
    assert result.get("error") or result.get("success") is False, result
    assert other.calls == provider.calls == escapes == []
    with pytest.raises(RequiredWebProviderError):
        providers.get_active_search_provider()
    with pytest.raises(RequiredWebProviderError):
        providers.get_active_extract_provider()


@pytest.mark.parametrize("capability", ["search", "extract"])
@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "boolean-version",
        "unavailable",
        "capability",
        "readiness",
        "replacement",
    ],
)
def test_faults_never_admit_or_rescue(lab, capability, fault, monkeypatch):
    from agent.transports import codex as codex_transport
    from agent.web_required_provider import RequiredWebProviderError
    from agent import web_search_registry as providers
    from tools.web_tools import check_web_api_key

    _, provider, escapes, _ = lab
    assert check_web_api_key() is True  # also binds the exact plugin instance

    def unavailable():
        raise RuntimeError("synthetic readiness fault")

    mutations = {
        "version": lambda: setattr(provider, "request_policy_version", 9),
        "boolean-version": lambda: setattr(provider, "request_policy_version", True),
        "unavailable": lambda: setattr(provider, "available", False),
        "capability": lambda: setattr(provider, capability + "_capable", False),
        "readiness": lambda: monkeypatch.setattr(provider, "is_available", unavailable),
        "replacement": lambda: providers.register_provider(
            type(provider)(), scope=hermes_home_key()
        ),
    }
    mutations[fault]()
    assert codex_transport._xai_prefers_native_web_search() is False
    assert codex_transport._openai_prefers_native_web_search() is False
    result = _call(capability)
    assert "required_web_provider" in result.get("error", ""), result
    assert provider.calls == escapes == []
    # One shared registry gate serves both native web tools.  Losing only one
    # capability leaves the other tool visible; the failed capability itself
    # still refuses and cannot fall back.
    assert check_web_api_key() is (fault == "capability")
    with pytest.raises(RequiredWebProviderError):
        providers._resolve(None, capability=capability)


def test_search_only_required_provider_keeps_search_visible_and_extract_closed(lab):
    from tools.web_tools import check_web_api_key

    _, provider, escapes, _ = lab
    provider.extract_capable = False

    assert check_web_api_key() is True
    assert _call("search").get("success") is True
    extract = _call("extract")
    assert "required_web_provider" in extract.get("error", ""), extract
    assert [call[0] for call in provider.calls] == ["search"]
    assert escapes == []


@pytest.mark.parametrize(
    "content",
    [None, "{}", "web: []", "[]", "web: [broken", "web: {required_provider: other}"],
)
def test_policy_loss_after_binding_never_restores_legacy(lab, content):
    from agent.transports import codex as codex_transport

    home, provider, escapes, _ = lab
    assert _call("search")["success"]
    provider.calls.clear()
    path = home / "config.yaml"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    assert codex_transport._xai_prefers_native_web_search() is False
    assert codex_transport._openai_prefers_native_web_search() is False
    for capability in ("search", "extract"):
        result = _call(capability)
        assert "required_web_provider" in result.get("error", ""), result
    assert provider.calls == escapes == []


def test_global_same_name_cannot_fill_missing_profile_slot(lab):
    from agent import web_search_registry as providers
    from hermes_cli.plugins import get_plugin_manager

    _, provider, escapes, _ = lab
    assert get_plugin_manager().unload("quota-lab")
    global_provider = type(provider)()
    providers.register_provider(global_provider)
    try:
        for capability in ("search", "extract"):
            assert "required_web_provider" in _call(capability).get("error", "")
        assert global_provider.calls == provider.calls == escapes == []
    finally:
        providers.restore_registration("lab-quota", global_provider, None)


def test_profile_switch_uses_its_own_instance_not_global_memo(lab, tmp_path):
    from agent import web_search_registry as providers
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    _, first, escapes, config = lab
    assert _call("search")["success"]
    other_home = tmp_path / "second-profile"
    other_home.mkdir()
    (other_home / "config.yaml").write_text(yaml.safe_dump(config))
    token = set_hermes_home_override(other_home)
    second = type(first)()
    try:
        providers.register_provider(second, scope=hermes_home_key())
        assert _call("search")["success"]
        assert len(second.calls) == 1
        first.available = False
        assert _call("search")["success"]  # another profile's failure is not global
        assert len(second.calls) == 2
    finally:
        providers.restore_registration(
            "lab-quota", second, None, scope=hermes_home_key()
        )
        reset_hermes_home_override(token)
    assert "required_web_provider" in _call("search").get("error", "")
    assert len(first.calls) == 1
    assert escapes == []


@pytest.mark.asyncio
async def test_cancelled_extract_preserves_cancellation_and_closes_once(
    lab, monkeypatch
):
    from tools.web_tools import web_extract_tool

    _, provider, escapes, _ = lab
    entered, finished = asyncio.Event(), asyncio.Event()
    calls = []

    async def blocking(urls, **kwargs):
        calls.append(list(urls))
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(provider, "extract", blocking)
    task = asyncio.create_task(web_extract_tool(["https://93.184.216.34/offline"]))
    await asyncio.wait_for(entered.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert len(calls) == 1 and escapes == []


def test_extract_timeout_is_unknown_not_a_retry(lab, monkeypatch):
    _, provider, escapes, config = lab
    home = get_hermes_home()
    config["web"]["extract_timeout"] = 2
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    calls, finished = [], []

    async def blocking(urls, **kwargs):
        calls.append(list(urls))
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(True)

    monkeypatch.setattr(provider, "extract", blocking)
    result = _call("extract")
    assert "outcome_unknown" in result["results"][0]["error"], result
    assert len(calls) == len(finished) == 1 and escapes == []


def test_missing_provider_is_refused_before_dns(lab, monkeypatch):
    import socket
    from hermes_cli.plugins import get_plugin_manager

    _, provider, escapes, _ = lab
    assert get_plugin_manager().unload("quota-lab")
    dns = []

    def resolver(*args, **kwargs):
        dns.append(args)
        raise AssertionError("must refuse before DNS")

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    assert "required_web_provider" in _call("extract").get("error", "")
    assert dns == provider.calls == escapes == []


@pytest.mark.parametrize(
    "policy", [{"enabled": True, "domains": ["93.184.216.34"], "shared_files": []}, 42]
)
def test_required_extract_website_policy_cannot_be_skipped(lab, policy):
    home, provider, escapes, config = lab
    config["security"] = {"website_blocklist": policy}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    result = _call("extract")
    assert result.get("error") or result["results"][0].get("error"), result
    assert provider.calls == escapes == []


@pytest.mark.asyncio
async def test_sync_timeout_is_not_proof_of_worker_cancellation(lab, monkeypatch):
    import threading
    from tools.web_tools import web_extract_tool

    home, provider, escapes, config = lab
    config["web"]["extract_timeout"] = 2
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def blocking(urls, **kwargs):
        calls.append(list(urls))
        entered.set()
        try:
            if not release.wait(5):
                raise TimeoutError("fixture release did not arrive")
            return [{"url": u, "content": "synthetic late effect"} for u in urls]
        finally:
            finished.set()

    monkeypatch.setattr(provider, "extract", blocking)
    try:
        result = json.loads(
            await asyncio.wait_for(
                web_extract_tool(["https://93.184.216.34/offline"]), timeout=4
            )
        )
        assert entered.is_set() and not finished.is_set()
        assert "outcome_unknown" in result["results"][0]["error"]
        assert len(calls) == 1 and escapes == []
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)


def _configure_conflicting_legacy_backend(home):
    config = {
        "plugins": {"enabled": ["quota-lab"]},
        "web": {
            "required_provider": "lab-quota",
            "search_backend": "legacy-escape",
            "extract_backend": "legacy-escape",
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


class _LegacyEscapeProvider(WebSearchProvider):
    @property
    def name(self):
        return "legacy-escape"

    @property
    def display_name(self):
        return "Legacy escape"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def search(self, query, limit=5):
        self.calls.append(("search", query, limit))
        return {"success": True, "data": {"web": [{"title": "legacy"}]}}

    async def extract(self, urls, **kwargs):
        self.calls.append(("extract", list(urls)))
        return [{"url": url, "content": "legacy escape"} for url in urls]


def test_required_binding_closes_search_hook_skip_and_registry_bypasses(
    lab, monkeypatch
):
    """Mandatory routing lives below optional hooks and every public dispatch surface."""
    from agent import web_search_registry as providers
    from agent.transports import codex as codex_transport
    from hermes_cli import plugins
    from model_tools import handle_function_call
    from tools.registry import registry
    from tools.web_tools import web_search_tool

    home, required, escapes, _ = lab
    required.calls.clear()
    _configure_conflicting_legacy_backend(home)
    assert codex_transport._xai_prefers_native_web_search() is False
    assert codex_transport._openai_prefers_native_web_search() is False
    legacy = _LegacyEscapeProvider()
    providers.register_provider(legacy, scope=hermes_home_key())

    def hook_boom(*args, **kwargs):
        raise RuntimeError("synthetic optional hook failure")

    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", hook_boom)
    args = {"query": "must not escape", "limit": 1}
    results = [
        json.loads(web_search_tool(**args)),
        json.loads(handle_function_call("web_search", args)),
        json.loads(
            handle_function_call("web_search", args, skip_pre_tool_call_hook=True)
        ),
        json.loads(registry.dispatch("web_search", args)),
    ]

    assert all(
        result.get("success") is not True and result.get("error") for result in results
    ), results
    assert required.calls == legacy.calls == escapes == []


@pytest.mark.asyncio
async def test_required_binding_closes_context_reference_direct_extract(lab):
    """The @url helper may call the raw tool, but cannot select an alternate provider."""
    from agent import web_search_registry as providers
    from agent.context_references import _default_url_fetcher

    home, required, escapes, _ = lab
    required.calls.clear()
    _configure_conflicting_legacy_backend(home)
    legacy = _LegacyEscapeProvider()
    providers.register_provider(legacy, scope=hermes_home_key())

    content = await _default_url_fetcher("https://93.184.216.34/offline")

    assert content == ""
    assert required.calls == legacy.calls == escapes == []


@pytest.mark.asyncio
async def test_required_extract_policy_error_fails_closed_before_provider(
    lab, monkeypatch
):
    """A website-policy exception cannot become permission for a mandatory request."""
    from tools import website_policy
    from tools.web_tools import web_extract_tool

    _, required, escapes, _ = lab
    required.calls.clear()

    def policy_boom(*args, **kwargs):
        raise RuntimeError("synthetic website-policy failure")

    monkeypatch.setattr(website_policy, "check_website_access", policy_boom)
    result = json.loads(await web_extract_tool(["https://93.184.216.34/offline"]))

    assert result.get("success") is not True and result.get("error"), result
    assert required.calls == escapes == []
