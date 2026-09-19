"""Scratch-workspace exemption for the project env/config write guard.

t_4537bd82: ``cp <profile>/config.yaml ./repro/config.yaml`` inside a Kanban
scratch workspace was flagged as "overwrite project env/config file" —
_PROJECT_CONFIG_PATH matches any path ending in ``config.yaml`` with no notion
of *where* the destination lives, so sandbox/repro copies of config files were
unapprovable in single-query (kanban worker) mode, where the gate denies
fail-closed with no human to approve.

A kanban scratch workspace (and a session cwd pinned to one) is disposable by
contract: every worker run gets a fresh directory and the kernel cleans it up.
Destinations that RESOLVE under HERMES_KANBAN_WORKSPACE /
HERMES_KANBAN_WORKSPACES_ROOT are therefore exempt from the
_PROJECT_SENSITIVE_WRITE_TARGET rules only. Everything else keeps firing, the
scan continues past the skipped rule (so a scratch ``cp`` that ALSO targets
~/.hermes/config.yaml or /etc stays flagged), and the user-level sensitive
rules never had the exemption at all.
"""

import os

import contextvars

import pytest

from tools.approval import check_all_command_guards, detect_dangerous_command
from tools.approval_context import reset_current_session_key, set_current_session_key


@pytest.fixture(autouse=True)
def _clean_approval_routing(monkeypatch):
    """No session-context routing; kanban roots unset unless a test sets them."""
    for var in (
        "HERMES_YOLO_MODE", "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION",
        "HERMES_CRON_SESSION", "HERMES_EXEC_ASK", "HERMES_SESSION_PLATFORM",
        "HERMES_SINGLE_QUERY_SESSION",
        "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    token = set_current_session_key("")
    yield
    reset_current_session_key(token)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A scratch workspace + root, pinned the way the kanban dispatcher does."""
    root = tmp_path / "workspaces"
    ws = root / "t_test"
    ws.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(root))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(ws))
    return ws


@pytest.fixture
def source_config(tmp_path):
    """A config.yaml source OUTSIDE any scratch root (the profile's real config)."""
    src = tmp_path / "profile" / "config.yaml"
    src.parent.mkdir(exist_ok=True)
    src.write_text("approvals:\n  mode: ask\n")
    return src


# ---------------------------------------------------------------------------
# The false positive that must stay fixed
# ---------------------------------------------------------------------------

class TestScratchWorkspaceDestinationsAllowed:
    def test_cp_profile_config_into_scratch_workspace(self, scratch, source_config):
        """The exact command from t_4537bd82 (absolute destination spelling)."""
        is_dangerous, _, description = detect_dangerous_command(
            f"cp {source_config} {scratch}/repro/config.yaml"
        )
        assert not is_dangerous, description

    def test_cp_relative_destination_with_pinned_cwd(self, scratch, source_config, monkeypatch):
        """TERMINAL_CWD is pinned to the workspace for kanban workers; the
        card's original command used a relative destination."""
        monkeypatch.chdir(scratch)
        assert detect_dangerous_command(f"cp {source_config} ./config.yaml") == (False, None, None)
        assert detect_dangerous_command(f"cp {source_config} repro/config.yaml") == (False, None, None)

    def test_cd_into_workspace_then_cp(self, scratch, source_config):
        is_dangerous, _, description = detect_dangerous_command(
            f"cd {scratch} && cp {source_config} ./config.yaml"
        )
        assert not is_dangerous, description

    @pytest.mark.parametrize("prog", ["cp", "mv", "install"])
    def test_all_copy_verbs(self, scratch, source_config, prog, monkeypatch):
        monkeypatch.chdir(scratch)
        assert detect_dangerous_command(f"{prog} {source_config} ./config.yaml") == (False, None, None)

    def test_tee_and_redirection_into_scratch(self, scratch, source_config):
        assert detect_dangerous_command(f"cat {source_config} > {scratch}/config.yaml") == (False, None, None)
        assert detect_dangerous_command(f"cat {source_config} | tee {scratch}/.env") == (False, None, None)

    def test_redirection_relative_destination_with_pinned_cwd(self, scratch, monkeypatch):
        monkeypatch.chdir(scratch)
        assert detect_dangerous_command("echo key=val > .env") == (False, None, None)
        assert detect_dangerous_command("echo key=val | tee config.yaml") == (False, None, None)

    def test_symlinked_workspace_prefix_still_resolves(self, tmp_path, source_config, monkeypatch):
        """A workspace reached through a symlinked path still exempts: roots and
        destinations are both realpath'd before the containment check."""
        real_root = tmp_path / "real-workspaces"
        real_ws = real_root / "t_real"
        real_ws.mkdir(parents=True)
        link_root = tmp_path / "linked"
        link_root.mkdir()
        os.symlink(real_root, link_root / "workspaces")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(link_root / "workspaces"))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(link_root / "workspaces" / "t_real"))
        is_dangerous, _, description = detect_dangerous_command(
            f"cp {source_config} {link_root}/workspaces/t_real/config.yaml"
        )
        assert not is_dangerous, description

    def test_home_relative_scratch_spelling(self, tmp_path, source_config, monkeypatch):
        """``$HOME``/``~``-spelled destinations inside a scratch tree under the
        user home resolve through the same containment check."""
        home = tmp_path / "home"
        ws = home / ".hermes" / "kanban" / "boards" / "ops" / "workspaces" / "t_h"
        ws.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(ws))
        for spelling in (f"{home}/.hermes/kanban/boards/ops/workspaces/t_h/config.yaml",
                         "~/.hermes/kanban/boards/ops/workspaces/t_h/config.yaml",
                         "$HOME/.hermes/kanban/boards/ops/workspaces/t_h/config.yaml"):
            is_dangerous, _, description = detect_dangerous_command(f"cp {source_config} {spelling}")
            assert not is_dangerous, f"{spelling}: {description}"


# ---------------------------------------------------------------------------
# The exemption must not become a bypass
# ---------------------------------------------------------------------------

class TestScratchGuardSafety:
    def test_symlink_escape_outside_scratch_still_flagged(self, scratch, source_config, tmp_path):
        """A symlink INSIDE the workspace pointing OUT: the destination resolves
        outside every root, so the historical flag must survive."""
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, scratch / "escape")
        is_dangerous, _, description = detect_dangerous_command(
            f"cp {source_config} {scratch}/escape/config.yaml"
        )
        assert is_dangerous, description

    def test_relative_destination_outside_workspace_cwd_still_flagged(self, tmp_path, source_config, monkeypatch):
        """No scratch roots apply, or the cwd is not a workspace: a relative
        config.yaml destination keeps the historical flag."""
        plain = tmp_path / "plain-cwd"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert detect_dangerous_command(f"cp {source_config} ./config.yaml")[0] is True
        # Roots set, but cwd is NOT under them (a non-kanban session that
        # happens to carry a stale workspace var).
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
        monkeypatch.chdir(plain)
        assert detect_dangerous_command(f"cp {source_config} ./config.yaml")[0] is True

    def test_compound_command_keeps_other_segments_flagged(self, scratch, source_config):
        """Only the LAST segment is considered for the exemption: anything after
        && / ; keeps the command flagged (conservative by design)."""
        is_dangerous, _, _ = detect_dangerous_command(
            f"cp {source_config} {scratch}/config.yaml && echo done"
        )
        assert is_dangerous

    def test_dest_outside_roots_with_workspace_env_still_flagged(self, scratch, source_config, tmp_path):
        is_dangerous, _, description = detect_dangerous_command(
            f"cp {source_config} {tmp_path}/project/config.yaml"
        )
        assert is_dangerous, description

    def test_unset_kanban_env_keeps_historical_behavior(self, source_config, tmp_path, monkeypatch):
        """No kanban vars at all (plain CLI session): identical behavior to
        before this change."""
        monkeypatch.chdir(tmp_path)
        assert detect_dangerous_command(f"cp {source_config} ./config.yaml")[0] is True


# ---------------------------------------------------------------------------
# True positives the guard must keep blocking
# ---------------------------------------------------------------------------

class TestTruePositivesStillBlocked:
    @pytest.mark.parametrize("command", [
        "cp evil.yaml /etc/config.yaml",
        "cp evil.yaml /etc/app/config.yaml",
        "cp evil.yaml ~/.hermes/config.yaml",
        "cp evil.yaml $HOME/.hermes/config.yaml",
        "cp evil ~/.ssh/authorized_keys",
        "cp evil ~/.bashrc",
        "mv evil ~/.netrc",
        "install -m 600 evil /etc/config.yaml",
        "cp evil.yaml .env",
        "cp evil.yaml config.yaml",
        "cp evil.yaml ./config.yaml",
        "echo x > /etc/config.yaml",
        "echo x | tee ~/.hermes/config.yaml",
        "echo x > .env",
        "echo x > config.yaml",
        "echo x | tee config.yaml",
    ])
    def test_still_flagged(self, command, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # a plain non-workspace cwd
        is_dangerous, _, description = detect_dangerous_command(command)
        assert is_dangerous, f"{command!r} must stay flagged"

    def test_hardline_floor_untouched(self):
        assert detect_dangerous_command("rm -rf /")[0] is True
        assert detect_dangerous_command("rm -rf ~/.hermes")[0] is True


# ---------------------------------------------------------------------------
# End-to-end through the single-query gate (kanban worker context)
# ---------------------------------------------------------------------------

class TestSingleQueryGate:
    def test_scratch_copy_approved_in_single_query_mode(self, scratch, source_config, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.chdir(scratch)
        result = check_all_command_guards(f"cp {source_config} ./config.yaml", "local")
        assert result["approved"] is True, result.get("message")

    def test_project_config_copy_still_blocked_in_single_query_mode(self, tmp_path, source_config, monkeypatch):
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.chdir(tmp_path)
        result = check_all_command_guards(f"cp {source_config} ./config.yaml", "local")
        assert result["approved"] is False
        assert "single-query" in (result.get("message") or "")