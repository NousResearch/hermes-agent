"""Terminal run envs resolve the routed profile's scope under multiplex (#142798).

A host gateway keeps the environment of whatever session started it. Before the fix,
``_make_run_env`` built a served profile's terminal commands from that env alone: the
routed profile's own ``.env`` names only reached the child when the launch process also
defined the name or it was a declared passthrough, and a launch-residue value for the
same name won. The builder now overlays the bound secret scope first — the same
semantics as ``build_subprocess_env(strip_launch_profile=True)``."""

from tools.environments.local import _make_run_env

# A Hermes-managed provider credential name, built at runtime so no credential-shaped
# literal sits in the source: the test only needs the NAME to be blocklisted.
_PROVIDER_KEY = "OPENAI_" + "API_KEY"


def _multiplex(monkeypatch, scope):
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr("agent.secret_scope.current_secret_scope", lambda: scope)


def test_served_profile_commands_see_their_own_scope(monkeypatch):
    monkeypatch.setenv("MYAPP_AGENT_ID", "from-p1")  # what started the host gateway
    _multiplex(monkeypatch, {"MYAPP_AGENT_ID": "from-p2"})
    env = _make_run_env({})
    assert env["MYAPP_AGENT_ID"] == "from-p2"


def test_single_profile_keeps_the_process_env(monkeypatch):
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    monkeypatch.setenv("MYAPP_AGENT_ID", "from-p1")
    env = _make_run_env({})
    assert env["MYAPP_AGENT_ID"] == "from-p1"


def test_backend_env_still_beats_the_routed_scope(monkeypatch):
    monkeypatch.setenv("MYAPP_AGENT_ID", "from-p1")
    _multiplex(monkeypatch, {"MYAPP_AGENT_ID": "from-p2"})
    env = _make_run_env({"MYAPP_AGENT_ID": "backend-declared"})
    assert env["MYAPP_AGENT_ID"] == "backend-declared"


def test_scope_values_pass_the_same_credential_scrub(monkeypatch):
    _multiplex(monkeypatch, {_PROVIDER_KEY: "routed-profile-scrub-probe"})
    monkeypatch.setenv(_PROVIDER_KEY, "launch-profile-scrub-probe")
    env = _make_run_env({})
    # The overlay rides along only names the scrub already allows: a provider credential
    # stays out whether it came from the launch env or the routed profile's scope.
    assert _PROVIDER_KEY not in env


def test_managed_policy_keys_beat_the_routed_scope(monkeypatch):
    monkeypatch.setenv("ORG_POLICY_FLAG", "policy-value")
    monkeypatch.setattr(
        "hermes_cli.env_loader.managed_dotenv_keys", lambda: ["ORG_POLICY_FLAG"])
    _multiplex(monkeypatch, {"ORG_POLICY_FLAG": "routed-user-value"})
    env = _make_run_env({})
    assert env["ORG_POLICY_FLAG"] == "policy-value"
