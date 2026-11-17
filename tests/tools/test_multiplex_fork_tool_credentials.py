"""Fork tool credentials resolve from the owning profile under multiplex (fix 4).

Each tool is exercised with:
  * ``os.environ`` holding a "default"-profile value (the multiplexer's env),
  * two secret scopes (profiles A and B) holding different values,
and asserts the scope wins, a scoped miss is a miss (never ``os.environ``),
an unscoped read is "not configured" with a WARNING naming the tool and key,
and the ``.env`` self-heal never runs. A multiplex-off block pins today's
behaviour (``os.environ`` + self-heal).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _multiplex_reset():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@contextmanager
def scope(mapping):
    tok = ss.set_secret_scope(mapping)
    try:
        yield
    finally:
        ss.reset_secret_scope(tok)


@pytest.fixture()
def no_dotenv(monkeypatch):
    """Fail loudly if any self-heal reaches the dotenv loader."""
    import hermes_cli.env_loader as el

    calls = []
    monkeypatch.setattr(el, "load_hermes_dotenv", lambda *a, **k: calls.append(1))
    return calls


# ── agent.fork_secrets ────────────────────────────────────────────────────


class TestReadSecret:
    def test_off_reads_os_environ(self, monkeypatch):
        from agent.fork_secrets import read_secret

        monkeypatch.setenv("JIRA_API_TOKEN", " env-tok ")
        assert read_secret("JIRA_API_TOKEN", tool="jira_api") == "env-tok"

    def test_on_reads_scope_not_env(self, monkeypatch):
        from agent.fork_secrets import read_secret

        monkeypatch.setenv("JIRA_API_TOKEN", "default-tok")
        ss.set_multiplex_active(True)
        with scope({"JIRA_API_TOKEN": "a-tok"}):
            assert read_secret("JIRA_API_TOKEN", tool="jira_api") == "a-tok"
        with scope({}):
            assert read_secret("JIRA_API_TOKEN", tool="jira_api") == ""

    def test_on_unscoped_is_not_configured_with_warning(self, monkeypatch, caplog):
        from agent.fork_secrets import read_secret

        monkeypatch.setenv("JIRA_API_TOKEN", "default-tok")
        ss.set_multiplex_active(True)
        with caplog.at_level(logging.WARNING, logger="agent.fork_secrets"):
            assert read_secret("JIRA_API_TOKEN", tool="jira_api") == ""
        assert any(
            "jira_api" in r.getMessage() and "JIRA_API_TOKEN" in r.getMessage()
            for r in caplog.records
        )

    def test_heal_dotenv_skipped_under_multiplex(self, no_dotenv):
        from agent.fork_secrets import heal_dotenv

        ss.set_multiplex_active(True)
        assert heal_dotenv() is False
        assert no_dotenv == []
        ss.set_multiplex_active(False)
        assert heal_dotenv() is True
        assert no_dotenv == [1]


class TestScopedSubprocessEnv:
    def test_off_is_os_environ_copy(self, monkeypatch):
        from agent.fork_secrets import scoped_subprocess_env

        monkeypatch.setenv("NOTION_API_KEY", "default-key")
        env = scoped_subprocess_env()
        assert env["NOTION_API_KEY"] == "default-key"

    def test_on_strips_default_secrets_and_overlays_scope(self, monkeypatch, tmp_path):
        from agent.fork_secrets import scoped_subprocess_env

        home = tmp_path / "default"
        home.mkdir()
        (home / ".env").write_text(
            "NOTION_API_KEY=default-key\nJIRA_API_TOKEN=default-jira\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("NOTION_API_KEY", "default-key")
        monkeypatch.setenv("JIRA_API_TOKEN", "default-jira")
        ss.set_multiplex_active(True)
        with scope({"NOTION_API_KEY": "b-key"}):
            env = scoped_subprocess_env()
        assert env["NOTION_API_KEY"] == "b-key"
        assert "JIRA_API_TOKEN" not in env  # default's secret, B has none
        assert env.get("PATH")  # globals survive
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("X")  # sanity: still unscoped here
        env = scoped_subprocess_env()  # unscoped → no overlay (fail closed)
        assert "NOTION_API_KEY" not in env


    def test_on_strips_every_non_global_key_when_default_secrets_unknown(self, monkeypatch):
        import agent.fork_secrets as fs

        monkeypatch.setattr(fs, "_process_profile_secret_keys", lambda: None)
        monkeypatch.setenv("SOME_DEFAULT_SECRET", "x")
        ss.set_multiplex_active(True)
        with scope({"NOTION_API_KEY": "b-key"}):
            env = fs.scoped_subprocess_env()
        assert "SOME_DEFAULT_SECRET" not in env
        assert env["NOTION_API_KEY"] == "b-key"
        assert env.get("PATH")


# ── Jira ──────────────────────────────────────────────────────────────────


class TestJira:
    CREDS = ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN")

    @pytest.fixture(autouse=True)
    def _default_env(self, monkeypatch):
        monkeypatch.setenv("JIRA_BASE_URL", "https://default.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "default@x")
        monkeypatch.setenv("JIRA_API_TOKEN", "default-tok")

    def test_each_profile_gets_its_own(self, no_dotenv):
        from tools.jira_api_tool import _jira_creds

        ss.set_multiplex_active(True)
        with scope({"JIRA_BASE_URL": "https://a.atlassian.net/", "JIRA_EMAIL": "a@x", "JIRA_API_TOKEN": "a-tok"}):
            assert _jira_creds() == ("https://a.atlassian.net", "a@x", "a-tok")
        with scope({"JIRA_BASE_URL": "https://b.atlassian.net", "JIRA_EMAIL": "b@x", "JIRA_API_TOKEN": "b-tok"}):
            assert _jira_creds() == ("https://b.atlassian.net", "b@x", "b-tok")
        assert no_dotenv == []

    def test_scoped_miss_is_not_configured(self, no_dotenv):
        from tools.jira_api_tool import _check_jira_api, _jira_creds

        ss.set_multiplex_active(True)
        with scope({}):
            assert _jira_creds() == ("", "", "")
            assert _check_jira_api() is False
        assert no_dotenv == []

    def test_unscoped_is_not_configured(self, no_dotenv, caplog):
        from tools.jira_api_tool import _check_jira_api

        ss.set_multiplex_active(True)
        with caplog.at_level(logging.WARNING):
            assert _check_jira_api() is False
        assert any("JIRA_API_TOKEN" in r.getMessage() for r in caplog.records)
        assert no_dotenv == []

    def test_off_reads_env_and_heals(self, monkeypatch, no_dotenv):
        from tools.jira_api_tool import _jira_creds

        assert _jira_creds() == ("https://default.atlassian.net", "default@x", "default-tok")
        monkeypatch.delenv("JIRA_API_TOKEN")
        _jira_creds()
        assert no_dotenv == [1]


# ── Notion ────────────────────────────────────────────────────────────────


class TestNotion:
    def test_scope_key_reaches_subprocess_env(self, monkeypatch, tmp_path, no_dotenv):
        from tools.notion_api_tool import _ntn_env

        home = tmp_path / "default"
        home.mkdir()
        (home / ".env").write_text("NOTION_API_KEY=default-key\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("NOTION_API_KEY", "default-key")
        monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
        ss.set_multiplex_active(True)
        with scope({"NOTION_API_KEY": "a-key"}):
            env = _ntn_env()
        assert env["NOTION_API_TOKEN"] == "a-key"
        assert env["NOTION_KEYRING"] == "0"
        with scope({}):
            env = _ntn_env()
        assert not env.get("NOTION_API_TOKEN")
        assert not env.get("NOTION_API_KEY")
        assert no_dotenv == []

    def test_off_is_unchanged(self, monkeypatch, no_dotenv):
        from tools.notion_api_tool import _ntn_env

        monkeypatch.setenv("NOTION_API_KEY", "default-key")
        monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
        env = _ntn_env()
        assert env["NOTION_API_TOKEN"] == "default-key"
        assert no_dotenv == []
        monkeypatch.delenv("NOTION_API_KEY")
        _ntn_env()
        assert no_dotenv == [1]


# ── Webflow ───────────────────────────────────────────────────────────────


class TestWebflow:
    @pytest.fixture(autouse=True)
    def _default_env(self, monkeypatch):
        monkeypatch.setenv("WEBFLOW_API_TOKEN", "default-tok")
        monkeypatch.setenv("WEBFLOW_API_TOKEN_SITE2", "default-site2")
        monkeypatch.setenv("WEBFLOW_SITE_TOKENS", "site2:WEBFLOW_API_TOKEN_SITE2")

    def test_site_map_and_named_var_resolve_through_scope(self, no_dotenv):
        from tools.webflow_asset_tool import _check_webflow_asset_upload, _webflow_token

        ss.set_multiplex_active(True)
        with scope({
            "WEBFLOW_API_TOKEN": "a-tok",
            "WEBFLOW_SITE_TOKENS": "siteX:A_SITE_X",
            "A_SITE_X": "a-x",
        }):
            assert _webflow_token("siteX") == "a-x"
            assert _webflow_token("site2") == "a-tok"  # unmapped in A
            assert _webflow_token() == "a-tok"
        with scope({}):
            assert _webflow_token("site2") == ""
            assert _check_webflow_asset_upload() is False
        assert no_dotenv == []

    def test_off_is_unchanged(self):
        from tools.webflow_asset_tool import _webflow_token

        assert _webflow_token("site2") == "default-site2"
        assert _webflow_token() == "default-tok"


# ── Slack react / post_thread ─────────────────────────────────────────────


class TestSlackToken:
    @pytest.fixture(autouse=True)
    def _no_config_token(self, monkeypatch):
        import gateway.config as gc

        class _Cfg:
            platforms = {}

        monkeypatch.setattr(gc, "load_gateway_config", lambda: _Cfg())
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-default")

    def test_scope_token_and_miss(self, no_dotenv):
        from tools.slack_react_tool import _resolve_slack_token

        ss.set_multiplex_active(True)
        with scope({"SLACK_BOT_TOKEN": "xoxb-educator"}):
            assert _resolve_slack_token() == "xoxb-educator"
        with scope({}):
            assert _resolve_slack_token() == ""
        assert _resolve_slack_token() == ""  # unscoped
        assert no_dotenv == []

    def test_unscoped_never_reaches_config_env_token(self, monkeypatch, no_dotenv):
        """Unscoped, load_gateway_config() would read the default bot's token
        from os.environ — the resolver must not consult it."""
        import gateway.config as gc

        monkeypatch.setattr(
            gc, "load_gateway_config",
            lambda: (_ for _ in ()).throw(AssertionError("config must not be read unscoped")),
        )
        from tools.slack_react_tool import _resolve_slack_token

        ss.set_multiplex_active(True)
        assert _resolve_slack_token() == ""

    def test_post_thread_check_uses_same_resolution(self):
        from tools.slack_post_thread_tool import _check_slack_post_thread

        ss.set_multiplex_active(True)
        with scope({}):
            assert _check_slack_post_thread() is False

    def test_off_is_unchanged(self, monkeypatch, no_dotenv):
        from tools.slack_react_tool import _resolve_slack_token

        assert _resolve_slack_token() == "xoxb-default"
        monkeypatch.delenv("SLACK_BOT_TOKEN")
        _resolve_slack_token()
        assert no_dotenv == [1]


# ── Google Drive service account ──────────────────────────────────────────


class TestDriveCredentials:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        from plugins.google_drive_sa import client as gd

        monkeypatch.setattr(gd, "_credentials", None)
        monkeypatch.setattr(gd, "_profile_credentials", {})
        monkeypatch.setattr(gd, "_deps_ready", True)
        monkeypatch.setenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", '{"who": "default"}')
        built = []

        def fake_from_info(info, scopes=None):
            built.append(info["who"])
            return ("creds", info["who"])

        import types
        import sys

        sa_mod = types.SimpleNamespace(
            Credentials=types.SimpleNamespace(from_service_account_info=fake_from_info)
        )
        oauth2 = types.ModuleType("google.oauth2")
        oauth2.service_account = sa_mod
        monkeypatch.setitem(sys.modules, "google.oauth2", oauth2)
        monkeypatch.setitem(sys.modules, "google.oauth2.service_account", sa_mod)

        def adc_forbidden(*a, **k):
            raise AssertionError("ADC must not be used under multiplex")

        auth = types.ModuleType("google.auth")
        auth.default = adc_forbidden
        monkeypatch.setitem(sys.modules, "google.auth", auth)
        return built

    def test_per_profile_credentials(self, tmp_path):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.google_drive_sa import client as gd

        ss.set_multiplex_active(True)
        out = {}
        for name in ("a", "b"):
            home = tmp_path / name
            home.mkdir()
            htok = set_hermes_home_override(str(home))
            try:
                with scope({"GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": f'{{"who": "{name}"}}'}):
                    out[name] = gd._get_credentials()
                    assert gd._get_credentials() is out[name]  # cached per profile
            finally:
                reset_hermes_home_override(htok)
        assert out["a"] == ("creds", "a")
        assert out["b"] == ("creds", "b")

    def test_scoped_miss_is_not_configured_no_adc(self, tmp_path):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from plugins.google_drive_sa import client as gd

        ss.set_multiplex_active(True)
        htok = set_hermes_home_override(str(tmp_path))
        try:
            with scope({}):
                with pytest.raises(RuntimeError, match="not configured"):
                    gd._get_credentials()
        finally:
            reset_hermes_home_override(htok)

    def test_check_available_is_per_profile(self, monkeypatch, tmp_path):
        from plugins.google_drive_sa import client as gd

        monkeypatch.setattr(gd, "_avail", None)
        ss.set_multiplex_active(True)
        with scope({"GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"who": "a"}'}):
            assert gd.check_available() is True
        with scope({}):
            assert gd.check_available() is False
        assert gd._avail is None

    def test_off_uses_process_env(self):
        from plugins.google_drive_sa import client as gd

        assert gd._get_credentials() == ("creds", "default")
