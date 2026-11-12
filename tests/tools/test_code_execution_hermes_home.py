"""execute_code child env honors the multiplexed per-turn HERMES_HOME override (#110303).

Under a multiplexed Desktop/Dashboard connection one server process serves several
profiles, binding a context-local HERMES_HOME override per turn. ``_build_child_env``
scrubs the server process's ``os.environ`` — which carries the machine-default
HERMES_HOME — so without the rewrite below, skill scripts run via ``execute_code``
silently read/write the wrong profile's directory.
"""

import sys

import pytest

from hermes_constants import (
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.code_execution_env import _build_child_env


@pytest.fixture
def home_override():
    tokens = []

    def _set(path):
        tokens.append(set_hermes_home_override(path))
        return str(path)

    yield _set
    for token in reversed(tokens):
        reset_hermes_home_override(token)


def _child_env():
    return _build_child_env(
        rpc_endpoint="sock",
        rpc_token="tok",
        tmpdir="/tmp/hermes-test",
        child_python=sys.executable,
    )


class TestMultiplexedHermesHome:
    def test_override_rewrites_stale_server_default_per_turn(self, monkeypatch, home_override, tmp_path):
        """The reported bug: a child must see the ACTIVE profile's home, not the server default,
        and sequential turns for different profiles each see their own."""
        monkeypatch.setenv("HERMES_HOME", "/machine/default/.hermes")
        alpha = home_override(tmp_path / "profiles" / "alpha")
        assert _child_env()["HERMES_HOME"] == alpha
        beta = home_override(tmp_path / "profiles" / "beta")
        assert _child_env()["HERMES_HOME"] == beta

    def test_no_override_leaves_inherited_value_untouched(self, monkeypatch):
        """Dedicated per-profile processes (no override): zero behavior change."""
        assert get_hermes_home_override() is None
        monkeypatch.setenv("HERMES_HOME", "/machine/default/.hermes")

        assert _child_env()["HERMES_HOME"] == "/machine/default/.hermes"


class TestMultiplexedScratchTmp:
    """TMPDIR must follow the routed profile like HERMES_HOME does (#138501).

    The scrub keeps TMPDIR/TMP/TEMP but used to drop the HERMES_SCRATCH_DIR ownership
    marker, so the per-turn ``apply_scratch_tmp_env`` rebase could not tell the backend's
    own exported temp from a user override and left the startup profile's scratch in the
    child env.
    """

    def test_hermes_owned_tmpdir_rebases_to_routed_profile_scratch(self, monkeypatch, home_override, tmp_path):
        """Backend booted under beta (scratch exported + marked), turn routed to alpha: the
        child's TMP triple and the marker all point at alpha's scratch."""
        home_b = tmp_path / "profiles" / "beta"
        scratch_b = str(home_b / "cache" / "scratch")
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        monkeypatch.setenv("HERMES_SCRATCH_DIR", scratch_b)
        for key in ("TMPDIR", "TMP", "TEMP"):
            monkeypatch.setenv(key, scratch_b)
        alpha = home_override(tmp_path / "profiles" / "alpha")
        scratch_a = str(tmp_path / "profiles" / "alpha" / "cache" / "scratch")

        child_env = _child_env()
        assert child_env["HERMES_HOME"] == alpha
        assert child_env["TMPDIR"] == child_env["TMP"] == child_env["TEMP"] == scratch_a
        assert child_env["HERMES_SCRATCH_DIR"] == scratch_a

    def test_user_tmpdir_survives_the_routed_rewrite(self, monkeypatch, home_override, tmp_path):
        """A user-set TMPDIR (unequal to the marker) is a deliberate override: the child
        keeps it even while a profile is routed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "beta"))
        monkeypatch.setenv("TMPDIR", "/var/folders/user-set-tmp")
        home_override(tmp_path / "profiles" / "alpha")

        assert _child_env()["TMPDIR"] == "/var/folders/user-set-tmp"
