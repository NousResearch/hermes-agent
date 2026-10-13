"""Terminal/background children inherit no undeclared secrets.

Forwarding every ambient variable that is not a *declared* Hermes credential would let a
user's own ``.env`` keys (e.g. ``EXAMPLE_SERVICE_TOKEN``) and runtime-injected secrets reach
``env`` in the agent's shell, one prompt injection away from exfiltration. The policy is
deny-by-default for the ambient environment: any key a dotenv file or external secret source
supplied, and any secret-shaped name, is dropped unless explicitly registered for passthrough.
Values the operator passes to the backend explicitly (``terminal.env`` / caller ``env``) still win.
"""

import os

import pytest

from tests.tools import _child_env_fixtures
from tests.tools._child_env_fixtures import observe_child, observe_terminal
from tools.environments import local

child_env = _child_env_fixtures.child_env  # fixture, requested by name below

DOTENV_SECRET = "EXAMPLE_SERVICE_TOKEN"
DOTENV_PLAIN = "HERMES_FORK_SETTING"  # dotenv-supplied even though the name is not secret-shaped
SOURCE_SUPPLIED = "VAULT_SUPPLIED_VALUE"  # supplied by an external secret source (bitwarden, ...)
AMBIENT_SECRET_SHAPED = [
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "MY_APP_KEY", "DEPLOY_WEBHOOK_SECRET",
    "DB_PASSWORD", "CLAUDE_CODE_OAUTH_TOKEN", "STRIPE_API_KEY", "SENTRY_DSN", "MY_PRIVATE_KEY",
]
AMBIENT_PLAIN = ["MY_CUSTOM_VAR", "EDITOR", "AWS_REGION", "KEYBOARD_LAYOUT", "TOKENIZERS_PARALLELISM"]


@pytest.fixture
def secret_env(child_env, monkeypatch):
    from hermes_cli import env_loader
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", {DOTENV_SECRET, DOTENV_PLAIN})
    monkeypatch.setattr(env_loader, "_SOURCE_SUPPLIED_NAMES", {SOURCE_SUPPLIED})
    for name in [DOTENV_SECRET, DOTENV_PLAIN, SOURCE_SUPPLIED, *AMBIENT_SECRET_SHAPED, *AMBIENT_PLAIN]:
        monkeypatch.setenv(name, "fake-" + name)
    return child_env


ALL = [DOTENV_SECRET, DOTENV_PLAIN, SOURCE_SUPPLIED, *AMBIENT_SECRET_SHAPED, *AMBIENT_PLAIN]
EXPECTED = {**dict.fromkeys([DOTENV_SECRET, DOTENV_PLAIN, SOURCE_SUPPLIED, *AMBIENT_SECRET_SHAPED]),
            **{k: "fake-" + k for k in AMBIENT_PLAIN}}


def test_foreground_terminal_drops_dotenv_and_secret_shaped_ambient_vars(secret_env):
    env = local.LocalEnvironment(cwd=str(secret_env), timeout=30)
    try:
        observed = observe_terminal(env, ALL)
    finally:
        env.cleanup()
    assert observed == EXPECTED


def test_background_spawn_env_drops_dotenv_and_secret_shaped_ambient_vars(secret_env):
    observed = observe_child(local._sanitize_subprocess_env(os.environ, {}), ALL)
    assert observed == EXPECTED


def test_passthrough_registration_is_the_only_way_back_in(secret_env):
    from tools.env_passthrough import register_env_passthrough
    register_env_passthrough([DOTENV_SECRET, "STRIPE_API_KEY"])
    env = local.LocalEnvironment(cwd=str(secret_env), timeout=30)
    try:
        observed = observe_terminal(env, [DOTENV_SECRET, "STRIPE_API_KEY", "DB_PASSWORD"])
    finally:
        env.cleanup()
    assert observed == {DOTENV_SECRET: "fake-" + DOTENV_SECRET,
                        "STRIPE_API_KEY": "fake-STRIPE_API_KEY", "DB_PASSWORD": None}


def test_explicit_backend_env_from_operator_config_still_reaches_the_child(secret_env):
    env = local.LocalEnvironment(cwd=str(secret_env), timeout=30, env={"DEPLOY_TOKEN": "operator-set"})
    try:
        observed = observe_terminal(env, ["DEPLOY_TOKEN"])
    finally:
        env.cleanup()
    assert observed == {"DEPLOY_TOKEN": "operator-set"}
    bg = observe_child(local._sanitize_subprocess_env(os.environ, {"DEPLOY_TOKEN": "operator-set"}),
                       ["DEPLOY_TOKEN"])
    assert bg == {"DEPLOY_TOKEN": "operator-set"}


@pytest.mark.parametrize("name", ["PWD", "OLDPWD", "PATH", "HOME", "LANG", "TERM", "SSH_TTY"])
def test_secret_shape_heuristic_spares_core_shell_vars(name):
    from tools.environments.local_env_policy import _is_secret_shaped_env_name
    assert not _is_secret_shaped_env_name(name)
