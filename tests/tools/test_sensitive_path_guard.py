"""Tests proving the read-only guard denies raw shell reads of paths
`agent/file_safety.py` classifies as sensitive — closing the gap that
module's own docstring admits exists against terminal_tool (it is "NOT a
security boundary" on its own; enforcement here is what makes it one for
Rob's specific command surface).

Covers, on top of the file_safety categories: shell glob/expansion
rejection (the guard must reason about what the shell resolves, not the
literal text), symlink-to-sensitive-target resolution, SSH/private-key
material, and the /proc deny policy.

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


class TestGlobAndShellExpansionDenied:
    """Regression for the live-proven glob bypass: the Rob execution path
    is shell=True, so the shell expands globs BEFORE the guard ever sees
    the resolved path. `cat <dir>/.s?h/id_ed2551*` passed every literal
    .ssh/private-key check because the guard saw wildcard text while the
    shell read the real key. Path-shaped arguments containing shell
    expansion characters are now denied outright (fail closed) rather than
    emulated."""

    def test_live_proven_glob_shape_denied(self):
        assert denied("cat ~/.s?h/id_ed2551*")

    def test_wildcard_inside_filename_denied(self):
        assert denied("cat /home/nico/.ssh/id_ed2551*")

    def test_wildcard_inside_parent_dir_denied(self):
        assert denied("cat /home/*/.ssh/id_ed25519")

    def test_relative_glob_denied(self):
        assert denied("cat .ssh/id_*")

    def test_absolute_glob_denied(self):
        assert denied("cat /home/nico/.ssh/id_*")

    def test_glob_with_dotdot_denied(self):
        assert denied("cat /home/nico/.ssh/../.ssh/id_*")

    def test_env_glob_variant_denied(self):
        assert denied("cat .env*")

    def test_question_mark_glob_denied(self):
        assert denied("cat .ss?/id_ed25519")

    def test_character_class_glob_denied(self):
        assert denied("cat ~/.ssh/id_[er]sa")

    def test_quoted_glob_still_denied(self):
        # Quoting does not help: the deobfuscator collapses quotes, and
        # the shell may still expand depending on context — fail closed.
        assert denied("cat '~/.*/.ssh/id_*'")

    def test_brace_expansion_denied(self):
        assert denied("cat ~/.ssh/id_{ed25519,rsa}")

    def test_param_expansion_prefix_denied(self):
        # `$VAR` prefixes bypassed every literal check before this pass:
        # the guard saw literal `$HERMES_HOME/...` text while the shell
        # resolved it to the real credential store.
        assert denied(f"cat $HERMES_HOME/mcp-tokens/project-os.json")


class TestSymlinkToSensitiveTargetDenied:
    """The private-key and file_safety checks must resolve symlinks: a
    link named `innocent.txt` pointing at `~/.ssh/id_ed25519` (or a
    project `.env`) resolves to the sensitive real path at read time and
    must be denied, not just the literal basename."""

    def test_symlink_to_ssh_key_denied(self, tmp_path):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key = ssh_dir / "id_ed25519"
        key.write_text("decoy key material — not a real key")
        link = tmp_path / "innocent-name.txt"
        link.symlink_to(key)
        assert denied(f"cat {link}")

    def test_symlink_to_env_file_denied(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("DECOY_KEY=decoy-value")
        link = tmp_path / "config.txt"
        link.symlink_to(env)
        assert denied(f"cat {link}")

    def test_symlink_to_pem_material_denied(self, tmp_path):
        keydir = tmp_path / "keys"
        keydir.mkdir()
        material = keydir / "server.pem"
        material.write_text("decoy pem — not a real key")
        link = tmp_path / "README.txt"
        link.symlink_to(material)
        # Denied on the RESOLVED basename's key-material extension even
        # though the link's own name is innocent.
        assert denied(f"cat {link}")

    def test_unknown_file_inside_ssh_dir_denied(self, tmp_path):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "secret-config").write_text("decoy")
        assert denied(f"cat {ssh_dir / 'secret-config'}")


class TestProcPolicyDenied:
    """Structural denial of /proc/*/environ, /proc/*/fd/* and /proc/*/mem
    — raw environment reads, descriptor traversal and memory images are
    denied at the guard rather than trusted to output redaction."""

    def test_self_environ_denied(self):
        assert denied("cat /proc/self/environ")

    def test_pid_environ_denied(self):
        assert denied("cat /proc/1/environ")
        assert denied("head -n1 /proc/42/environ")

    def test_self_fd_denied(self):
        assert denied("cat /proc/self/fd/0")
        assert denied("cat /proc/self/fd/3")

    def test_pid_fd_denied(self):
        assert denied("cat /proc/1/fd/2")

    def test_fd_directory_denied(self):
        assert denied("cat /proc/self/fd")

    def test_mem_denied(self):
        assert denied("cat /proc/self/mem")
        assert denied("tail -n1 /proc/999/mem")

    def test_fd_glob_denied(self):
        assert denied("cat /proc/self/fd/*")

    def test_proc_metadata_still_allowed(self):
        assert allowed("cat /proc/self/status")
        assert allowed("cat /proc/1/cmdline")


class TestGrepPatternExemptFromPathChecks:
    """grep's first non-flag argument is a search PATTERN, not a path:
    rejecting glob metacharacters there would break the registered
    `rob_journal_query` tool (which pipes user-supplied patterns into
    grep), and treating it as a path re-blocks patterns that merely LOOK
    sensitive (e.g. `grep .env`). The pattern stays exempt while file
    arguments (including `-f`/`--file` values) remain fully checked."""

    def test_glob_in_pattern_allowed(self):
        assert allowed("grep 'err*' /var/log/syslog")

    def test_character_class_in_pattern_allowed(self):
        assert allowed("grep '[0-9]*' /var/log/syslog")

    def test_anchor_dollar_in_pattern_allowed(self):
        assert allowed("grep 'error$' /var/log/syslog")

    def test_dotenv_literal_pattern_allowed(self):
        assert allowed("grep .env /var/log/syslog")

    def test_e_flag_pattern_allowed(self):
        assert allowed("grep -e 'warn*' /var/log/syslog")

    def test_glob_in_file_argument_still_denied(self):
        assert denied("grep error /var/log/*.log")

    def test_pattern_file_via_dash_f_still_checked(self):
        home = _sandboxed_hermes_home()
        assert denied(f"grep -f {home}/mcp-tokens/project-os.json /etc/hostname")

    def test_pattern_file_via_long_flag_still_checked(self):
        home = _sandboxed_hermes_home()
        assert denied(f"grep --file={home}/auth.json /etc/hostname")

    def test_file_after_attached_pattern_flag_still_checked(self):
        # `-ePAT` consumes the pattern slot; the following token is a FILE
        # under grep's own semantics and must still be path-checked.
        assert denied("grep -ePAT .env")

    def test_file_after_dash_f_is_still_checked(self):
        # With `-f`, grep takes patterns from the file, so the next bare
        # token is an input FILE — path-checked, never skipped as pattern.
        assert denied("grep -f /tmp/patterns .env")


class TestOrdinaryPathsStillAllowed:
    def test_cat_ordinary_log(self):
        assert allowed("cat /var/log/syslog")

    def test_grep_ordinary_source_file(self):
        assert allowed("grep -n foo tools/read_only_command_guard.py")

    def test_env_example_not_blocked(self):
        # .env.example is the documented-shape substitute, deliberately
        # not a real secret file — must remain readable.
        assert allowed("cat .env.example")
