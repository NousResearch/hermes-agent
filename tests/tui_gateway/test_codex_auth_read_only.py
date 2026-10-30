"""Process-local Codex auth policy, with real runtime/auth/recovery imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


PROBE = r'''
import base64
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import httpx

home = Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True, exist_ok=True)
kind, explicit, readonly = sys.argv[1:]
readonly = readonly == "yes"
token = "eyJhbGciOiJub25lIn0." + base64.urlsafe_b64encode(json.dumps({"exp": 1 if kind == "expired" else 4102444800}).encode()).decode().rstrip("=") + ".synthetic"
row = {"id": "synthetic", "source": "manual:device_code", "auth_type": "oauth", "access_token": token, "refresh_token": "synthetic-refresh", "base_url": "https://pool.invalid/codex"}
store = ({"providers": {"openai-codex": {"tokens": {"access_token": token, "refresh_token": "synthetic-refresh"}}}} if kind in {"singleton", "expired"} else {} if kind == "missing" else {"credential_pool": {"openai-codex": [row]}})
auth = home / "auth.json"
auth.write_text(json.dumps(store))
before = auth.read_bytes()
(home / "config.yaml").write_text("model:\n  provider: openai-codex\n  default: synthetic-model\nfallback_providers:\n  - provider: openrouter\n    model: synthetic-fallback\n")

requests_seen = []
def forbidden(*args, **kwargs):
    request = args[1]
    url = str(getattr(request, "url", "unknown"))
    if "/models?" in url or url.endswith("/api/show"):
        if isinstance(request, httpx.Request):
            return httpx.Response(404, request=request)
        response = requests.Response()
        response.status_code = 404
        response._content = b"{}"
        return response
    requests_seen.append(url)
    raise AssertionError("auth must not issue HTTP requests")
httpx.Client.send = forbidden
import requests
requests.Session.send = forbidden

from tui_gateway import entry
from gateway.runtime_ownership import process_ownership
entry._claim_launch_home = lambda: home
process_ownership.release = lambda home: None

def exercise():
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import resolve_codex_runtime_credentials
    from agent.client_lifecycle import ClientLifecycleMixin
    kwargs = {"explicit_base_url": "https://explicit.invalid/codex"} if explicit == "yes" else {}
    if kind == "missing":
        from hermes_cli.auth import AuthError
        from tui_gateway import server
        try:
            server._resolve_runtime_with_fallback({"requested": "openai-codex"})
        except AuthError as exc:
            assert exc.provider == "openai-codex"
        else:
            raise AssertionError("missing Codex auth did not fail closed")
        assert auth.read_bytes() == before
        assert not (home / "auth.lock").exists()
        assert not requests_seen
        return
    if not readonly:
        safe = resolve_runtime_provider(requested="openai-codex", codex_auth_read_only=True, **kwargs)
        assert safe["api_key"] == token
        assert not (home / "auth.lock").exists()
    runtime = resolve_runtime_provider(requested="openai-codex", codex_auth_read_only=False, **kwargs)
    assert runtime["api_key"] == token
    if readonly:
        assert runtime.get("credential_pool") is None
        assert runtime["base_url"] == ("https://explicit.invalid/codex" if explicit == "yes" else "https://pool.invalid/codex" if kind == "pool" else "https://chatgpt.com/backend-api/codex")
        assert resolve_codex_runtime_credentials(force_refresh=True, read_only=False)["api_key"] == token
        from agent.auxiliary_client import _resolve_codex_credential_and_base
        assert _resolve_codex_credential_and_base() == (token, "https://pool.invalid/codex" if kind == "pool" else "https://chatgpt.com/backend-api/codex")
        agent = SimpleNamespace(api_mode="codex_responses", provider="openai-codex", api_key=token)
        assert ClientLifecycleMixin._try_refresh_codex_client_credentials(agent, force=True) is False
        # A rebuild or model change resolves again on another thread, without ContextVar transport.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            again = executor.submit(resolve_runtime_provider, requested="openai-codex").result()
        assert again["api_key"] == token
        assert again.get("credential_pool") is None
        from agent.credential_pool import load_pool
        try:
            load_pool("openai-codex")
        except RuntimeError:
            pass
        else:
            raise AssertionError("mutable Codex pool admitted")
        from run_agent import AIAgent
        real_agent = AIAgent(model="synthetic-model", provider="openai-codex", api_mode="codex_responses",
                             api_key=token, base_url=runtime["base_url"], enabled_toolsets=[],
                             skip_memory=True, skip_context_files=True, quiet_mode=True)
        assert real_agent.codex_auth_read_only
        assert real_agent._credential_pool is None
        for status in (401, 429):
            assert real_agent._recover_with_credential_pool(status_code=status, has_retried_429=False) == (False, False)
        assert real_agent._try_activate_fallback() is False
        real_agent.switch_model(new_model="synthetic-model-2", new_provider="openai-codex",
                                api_key=token, base_url=runtime["base_url"], api_mode="codex_responses")
        assert real_agent._credential_pool is None
        try:
            real_agent.switch_model(new_model="synthetic-model", new_provider="openai-codex", api_mode="codex_app_server")
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsupported app-server runtime admitted")
        real_agent.close()
        from agent.codex_auth_policy import bind_codex_auth_policy
        try:
            bind_codex_auth_policy(read_only=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("readonly process policy downgraded")
        assert auth.read_bytes() == before
        assert not (home / "auth.lock").exists()
        assert not requests_seen, requests_seen
    else:
        assert (home / "auth.lock").exists()

entry._serve = exercise
sys.argv = ["tui_gateway.entry"] + (["--codex-auth-read-only"] if readonly else [])
entry.main(sys.argv[1:])
'''


@pytest.mark.parametrize("kind", ["singleton", "pool", "expired", "missing"])
@pytest.mark.parametrize("explicit", ["yes", "no"])
def test_stdio_readonly_auth_preserves_store_across_runtime_and_recovery(tmp_path, kind, explicit):
    _probe(tmp_path, kind, explicit, "yes")


def test_separate_default_process_keeps_mutable_auth(tmp_path):
    _probe(tmp_path, "singleton", "yes", "no")


def _probe(tmp_path: Path, kind: str, explicit: str, readonly: str) -> None:
    env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes")}
    result = subprocess.run(
        [sys.executable, "-c", PROBE, kind, explicit, readonly],
        env=env, capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
