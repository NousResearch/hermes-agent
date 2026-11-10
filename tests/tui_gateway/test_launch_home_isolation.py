"""The launch home the TUI server binds launch-profile turns to must follow the process home."""
import os
from pathlib import Path

# Imported at collection on purpose: that is when real test modules import the server,
# before any per-test fixture redirects HERMES_HOME, so ``_hermes_home`` freezes to the
# pre-fixture home exactly as it does for the rest of the suite.
from tui_gateway import server
from tui_gateway.launch_profile_policy import launch_secret_scope


def test_launch_defaults_remain_identity_bound_across_source_revocation(tmp_path, monkeypatch):
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from hermes_constants import pin_process_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import launch_profile_policy as lpp

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(lpp, "_snapshot", None)
    monkeypatch.setenv("ENV_ONLY_LOGIN", "fake-injected")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".env").write_text("REVOKED_LOGIN=fake-file\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
    lpp.capture_launch_env()
    pin_process_hermes_home(tmp_path)
    ss.set_multiplex_active(True)
    ht = set_hermes_home_override(tmp_path)
    scope = launch_secret_scope(tmp_path)
    token = ss.set_secret_scope(scope, profile_home=str(tmp_path))
    try:
        assert isinstance(scope, ss.ProfileSecretScope)
        boundary = ss.build_profile_env_boundary(tmp_path, tmp_path)
        assert boundary.target_values['ENV_ONLY_LOGIN'] == 'fake-injected'
        (tmp_path / ".env").write_text("", encoding="utf-8")
        assert ss.refresh_installed_secret_scope(tmp_path)
        boundary = ss.build_profile_env_boundary(tmp_path, tmp_path)
        assert boundary.target_values['ENV_ONLY_LOGIN'] == 'fake-injected'
        assert 'REVOKED_LOGIN' not in boundary.target_values
        assert scope['REVOKED_LOGIN'] == 'fake-file'
        assert 'REVOKED_LOGIN' in ss.get_profile_owned_secret_names(tmp_path)
    finally:
        ss.reset_secret_scope(token)
        reset_hermes_home_override(ht)
        ss.set_multiplex_active(False)
        pin_process_hermes_home(None)
        env_loader.reset_secret_source_cache()


def test_launch_home_follows_the_process_home_redirected_after_import():
    """``server._hermes_home`` is get_hermes_home() at import — under a developer shell the
    honored custom HERMES_HOME, a guarded root. Launch-profile turns read ``<launch home>/.env``
    (``launch_secret_scope``), so the home they bind must be resolved at call time from the
    process env, like the launch ``state.db`` handle (#112692), never the import-time value."""
    sandbox = Path(os.environ["HERMES_HOME"])
    assert server._launch_home() == sandbox
    (sandbox / ".env").write_text("HERMES_LAUNCH_HOME_PROBE=from-sandbox\n", encoding="utf-8")
    assert launch_secret_scope(server._launch_home()).get("HERMES_LAUNCH_HOME_PROBE") == "from-sandbox"
