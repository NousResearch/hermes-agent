"""A secret source's bootstrap credential (the vault machine token Hermes itself reads
to fetch every other secret) must never reach a spawned child. Provider keys are Tier 2
and already stripped; the vault token outranks them and was not on any list, so a
``terminal("env")`` or any LSP / browser / TTS child inherited it. Each source hands the
token to its own CLI explicitly (``_run_bws_list``, ``_op_child_env``), so no child of
the generic spawn paths needs it."""

import pytest

from tests.tools._child_env_fixtures import child_env, observe_child  # child_env: fixture
from tools.environments import local

# Default bootstrap env names declared by the bundled sources
# (agent/secret_sources/bitwarden.py, agent/secret_sources/onepassword.py).
VAULT_TOKENS = ["BWS_ACCESS_TOKEN", "OP_SERVICE_ACCOUNT_TOKEN"]
# Operator-shell credentials that stay inheritable on purpose (SECURITY.md §3.2 posture,
# local_env_policy.py AWS note); the fix must not widen the scrub to them.
OPERATOR_KEPT = ["AWS_SECRET_ACCESS_KEY", "MY_CUSTOM_VAR"]


def _seed(monkeypatch):
    for name in [*VAULT_TOKENS, *OPERATOR_KEPT, "OPENAI_API_KEY"]:
        monkeypatch.setenv(name, "fake-" + name)


@pytest.mark.parametrize("builder", [
    pytest.param(lambda: local._make_run_env({}), id="terminal"),
    pytest.param(lambda: local.build_subprocess_env(scrub_secrets=True), id="background"),
    pytest.param(lambda: local.hermes_subprocess_env(), id="helper"),
    pytest.param(lambda: local.hermes_subprocess_env(inherit_credentials=True), id="inherit_credentials"),
])
def test_vault_bootstrap_tokens_never_reach_spawned_children(child_env, monkeypatch, builder):
    _seed(monkeypatch)
    observed = observe_child(builder(), [*VAULT_TOKENS, *OPERATOR_KEPT])
    assert {k: observed[k] for k in VAULT_TOKENS} == dict.fromkeys(VAULT_TOKENS)
    assert {k: observed[k] for k in OPERATOR_KEPT} == {k: "fake-" + k for k in OPERATOR_KEPT}


def test_sources_still_hand_the_token_to_their_own_cli(child_env, monkeypatch):
    # The scrub is about *other* children. The vault CLIs get the token explicitly.
    from agent.secret_sources import onepassword
    from agent.secret_sources.base import source_child_env

    _seed(monkeypatch)
    assert onepassword._op_child_env("op-token")["OP_SERVICE_ACCOUNT_TOKEN"] == "op-token"
    bws_env = source_child_env()
    bws_env["BWS_ACCESS_TOKEN"] = "bws-token"  # what _run_bws_list does
    assert observe_child(bws_env, ["BWS_ACCESS_TOKEN"]) == {"BWS_ACCESS_TOKEN": "bws-token"}
