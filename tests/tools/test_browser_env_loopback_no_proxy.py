"""Browser child envs must bypass proxies for loopback hosts (#110565).

websockets>=14 defaults to ``proxy=True`` and resolves proxies via
``urllib.request.getproxies()``, which reads the macOS/Windows *system* proxy
config even with no ``*_proxy`` env vars set — so a local CDP WebSocket
(``ws://127.0.0.1:<port>/devtools/...``) gets routed into the system proxy and
the handshake fails with "did not receive a valid HTTP response".
``_build_browser_env`` therefore appends the loopback hosts to NO_PROXY/no_proxy
for every browser subprocess (append, never overwrite operator entries).
"""

import pytest

import tools.browser_tool as bt


class TestEnsureLoopbackNoProxy:
    def test_empty_env_sets_both_casings(self):
        env = {}
        bt._ensure_loopback_no_proxy(env)
        assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
        assert env["no_proxy"] == "127.0.0.1,localhost,::1"

    def test_appends_without_dropping_operator_entries(self):
        env = {"NO_PROXY": "corp.example.com,.internal"}
        bt._ensure_loopback_no_proxy(env)
        assert env["NO_PROXY"] == "corp.example.com,.internal,127.0.0.1,localhost,::1"

    def test_lowercase_value_appended_too(self):
        env = {"no_proxy": "10.0.0.0/8"}
        bt._ensure_loopback_no_proxy(env)
        assert env["no_proxy"] == "10.0.0.0/8,127.0.0.1,localhost,::1"

    def test_idempotent_when_loopback_already_present(self):
        env = {"NO_PROXY": "127.0.0.1,localhost,::1"}
        bt._ensure_loopback_no_proxy(env)
        assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"

    def test_partial_overlap_only_adds_missing(self):
        env = {"NO_PROXY": "localhost"}
        bt._ensure_loopback_no_proxy(env)
        assert env["NO_PROXY"] == "localhost,127.0.0.1,::1"

    def test_unrelated_keys_untouched(self):
        env = {"PATH": "/usr/bin", "NO_PROXY": "x.example"}
        bt._ensure_loopback_no_proxy(env)
        assert env["PATH"] == "/usr/bin"
        assert set(env) == {"PATH", "NO_PROXY", "no_proxy"}


class TestBuildBrowserEnvLoopback:
    @pytest.fixture
    def stub_sanitized_env(self, monkeypatch):
        """Replace the credential-scrub layer with a fixed dict so the test sees
        exactly what _build_browser_env adds on top."""
        import tools.environments.local as local
        holder = {}

        def _fake(inherit_credentials=False):
            return dict(holder)

        monkeypatch.setattr(local, "hermes_subprocess_env", _fake)
        return holder

    def test_sets_loopback_no_proxy_when_scrubbed_env_has_none(self, stub_sanitized_env, monkeypatch):
        for key in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)

        env = bt._build_browser_env()

        assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
        assert env["no_proxy"] == "127.0.0.1,localhost,::1"

    def test_keeps_operator_loopback_entries(self, stub_sanitized_env, monkeypatch):
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        stub_sanitized_env["NO_PROXY"] = "git.internal"
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)

        env = bt._build_browser_env()

        assert env["NO_PROXY"] == "git.internal,127.0.0.1,localhost,::1"

    def test_passthrough_keys_still_readded(self, stub_sanitized_env, monkeypatch):
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.setenv("BROWSER_USE_API_KEY", "test-key")

        env = bt._build_browser_env()

        assert env["BROWSER_USE_API_KEY"] == "test-key"
        assert env["NO_PROXY"] == "127.0.0.1,localhost,::1"
