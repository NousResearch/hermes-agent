"""Plugin-declared ``strip_env_keys`` must be applied on the foreground command path.

``_make_run_env`` builds the child environment for the ordinary ``terminal`` tool
(``LocalEnvironment._run_bash``), long-lived spawns and the file-search subprocess. It
passed a hardcoded ``frozenset()`` as the plugin-strip set to ``_scrubbed_env``, so a
plugin's declared ``strip_env_keys`` were silently ignored there while the background
``_sanitize_subprocess_env`` path honoured the same union. A registered strip-only
provider therefore had no effect on the surface a model-authored command runs on.
"""

import pytest

from agent import terminal_env_registry as reg
from agent.terminal_env_provider import TerminalEnvironmentProvider
from tests.tools._child_env_fixtures import child_env, observe_child
from tools.environments import local

STRIP_KEYS = ("PLUGIN_TERMINAL_TOKEN", "PLUGIN_TERMINAL_TRAIL")
KEPT_KEY = "OPERATOR_APP_KEY"


class _StripOnlyProvider(TerminalEnvironmentProvider):
    """Declares credential names to strip but provides no backend of its own."""

    name = "striponly"
    display_name = "StripOnly"
    is_remote = False
    is_container = False

    @property
    def strip_env_keys(self):
        return frozenset(STRIP_KEYS)

    def is_available(self):
        return False

    def create_environment(self, **kwargs):
        raise RuntimeError("strip-only provider: no environment of its own")


@pytest.fixture
def registered():
    reg._reset_for_tests()
    reg.register_provider(_StripOnlyProvider())
    try:
        yield
    finally:
        reg._reset_for_tests()


def test_foreground_child_env_strips_plugin_declared_keys(child_env, registered):  # health: allow F811 -- pytest resolves the fixture by its parameter name
    """The real child process observes the declared keys gone and other vars intact."""
    source = {key: "value-" + key for key in (*STRIP_KEYS, KEPT_KEY)}
    observed = observe_child(local._make_run_env(source), [*STRIP_KEYS, KEPT_KEY, "HERMES_HOME"])
    # HERMES_HOME also proves the child ran under the fixture's isolated home, not the operator's.
    assert observed == {**dict.fromkeys(STRIP_KEYS), KEPT_KEY: "value-" + KEPT_KEY,
                        "HERMES_HOME": str(child_env / "hermes")}


def test_foreground_child_env_keeps_operator_vars_without_a_provider(child_env):  # health: allow F811 -- pytest resolves the fixture by its parameter name
    """No registered provider: nothing plugin-declared is stripped, so the keys survive."""
    source = {key: "value-" + key for key in (*STRIP_KEYS, KEPT_KEY)}
    observed = observe_child(local._make_run_env(source), [*STRIP_KEYS, KEPT_KEY, "HERMES_HOME"])
    assert observed == {**{key: "value-" + key for key in (*STRIP_KEYS, KEPT_KEY)},
                        "HERMES_HOME": str(child_env / "hermes")}
