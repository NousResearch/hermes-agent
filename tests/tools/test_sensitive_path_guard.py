"""Tests proving the read-only guard denies raw shell reads of paths
`agent/file_safety.py` classifies as sensitive — closing the gap that
module's own docstring admits exists against terminal_tool (it is "NOT a
security boundary" on its own; enforcement here is what makes it one for
Rob's specific command surface).

Note: this repo's own tests/conftest.py deliberately sandboxes HERMES_HOME
to a per-test tempdir (autouse fixture — see its own module docstring) so
no test can ever touch the real ~/.hermes. Tests here that need a path
"inside HERMES_HOME" must therefore read HERMES_HOME from the environment
rather than hardcoding the real `~/.hermes` — the .env-style tests don't
need this since that check is a plain relative/anchored-elsewhere pattern,
not tied to HERMES_HOME at all.
"""

import os

from tools.read_only_command_guard import run_read_only_guard


def allowed(cmd: str) -> bool:
    return run_read_only_guard(cmd).allowed


def denied(cmd: str) -> bool:
    return not run_read_only_guard(cmd).allowed


def _sandboxed_hermes_home() -> str:
    """The per-test HERMES_HOME conftest.py's autouse fixture sets, so
    "inside HERMES_HOME" assertions work under this repo's own test
    isolation instead of assuming the real ~/.hermes."""
    return os.environ["HERMES_HOME"]


class TestSensitivePathDenied:
    def test_dot_env_denied(self):
        assert denied("cat .env")

    def test_dot_env_variant_denied(self):
        assert denied("cat .env.production")

    def test_mcp_tokens_denied(self):
        home = _sandboxed_hermes_home()
        assert denied(f"cat {home}/mcp-tokens/project-os.json")

    def test_auth_json_denied(self):
        home = _sandboxed_hermes_home()
        assert denied(f"cat {home}/auth.json")

    def test_head_on_env_denied(self):
        assert denied("head -n5 .env")

    def test_grep_on_env_denied(self):
        assert denied("grep TOKEN .env")

    def test_stat_on_mcp_tokens_denied(self):
        home = _sandboxed_hermes_home()
        assert denied(f"stat {home}/mcp-tokens/project-os.json")


class TestOrdinaryPathsStillAllowed:
    def test_cat_ordinary_log(self):
        assert allowed("cat /var/log/syslog")

    def test_grep_ordinary_source_file(self):
        assert allowed("grep -n foo tools/read_only_command_guard.py")

    def test_env_example_not_blocked(self):
        # .env.example is the documented-shape substitute, deliberately
        # not a real secret file — must remain readable.
        assert allowed("cat .env.example")
