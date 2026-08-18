"""Regression tests for the terminal-scanner false-positive class.

Four shape fixes in the command-approval pipeline, each pinned from a real
incident class where a legitimate operation was blocked (or hardline-blocked)
by a scanner heuristic misreading shell syntax:

- Fix A (malformed-exec / oversize): normalization's global backslash-strip
  flips quote parity on legal grep/sed patterns, and quoted payload bulk was
  measured against the separator-free size ceiling.
- Fix B (heredoc data): the stream-write DOTALL rules walked into heredoc
  bodies and matched config-file mentions that were report DATA, never
  redirection targets.
- Fix C (scratch cleanup): `rm` of literal scratch paths demanded interactive
  approval (a permanent block in headless workers).
- Fix D (tirith benign shapes): append-only writes to *.log under a logs/
  directory were flagged as dotfile_overwrite; scratch-dir deletion bursts
  as mass_file_deletion.
"""

import subprocess
from unittest.mock import MagicMock, patch

import tools.approval as approval_module
from tools.approval import (
    detect_dangerous_command,
    detect_hardline_command,
)
from tools.tirith_security import check_command_security


def _mock_run(returncode=0, stdout="", stderr=""):
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _tirith_cfg():
    return {"tirith_enabled": True, "tirith_path": "tirith",
            "tirith_timeout": 5, "tirith_fail_open": True}


def _tirith_stdout(findings, summary=""):
    import json
    return json.dumps({"findings": findings, "summary": summary})


# ---------------------------------------------------------------------------
# Fix A — malformed-exec verdicts must require the RAW command to be unlexable
# ---------------------------------------------------------------------------


class TestMalformedExecFalsePositives:
    def test_grep_with_escaped_quote_class_is_not_malformed(self):
        # Legal grep patterns whose author escapes normalization strips —
        # previously classified malformed and HARDLINE-blocked.
        for cmd in (
            "grep x\\\"y file",
            "grep '\\\"[^\\\"]*' logfile.txt",
        ):
            is_hl, _ = detect_hardline_command(cmd)
            assert not is_hl, f"{cmd!r} must not hardline-block"
            is_dangerous, _, _ = detect_dangerous_command(cmd)
            assert not is_dangerous

    def test_sed_alternation_with_pipe_escape_is_not_malformed(self):
        cmd = "sed 's/a\\|b/x/g' notes.md"
        is_hl, _ = detect_hardline_command(cmd)
        assert not is_hl

    def test_genuinely_malformed_quoting_still_blocks(self):
        # An actually unterminated quote the author wrote — the raw lexer
        # fails too, so the malformed verdict is trusted and the hardline
        # floor holds.
        is_hl, desc = detect_hardline_command("grep 'unterminated")
        assert is_hl
        assert desc == "command parser limit or malformed executable payload"

    def test_large_quoted_python_program_not_parser_limit_blocked(self):
        # A multi-KB python -c one-liner: quoted bulk is data; the unquoted
        # skeleton is small. Previously tripped the separator-free 4096-char
        # ceiling and hardline-blocked. (The -c execution flag itself keeps
        # its approval-tier finding — that is policy, not this fix.)
        body = "print('x' * %d)" % 3000
        cmd = f"python3 -c '{body}'"
        is_hl, _ = detect_hardline_command(cmd)
        assert not is_hl, "quoted payload bulk must not trip the parser-limit hardline"


# ---------------------------------------------------------------------------
# Fix B — heredoc bodies are data, not redirection targets
# ---------------------------------------------------------------------------


class TestHeredocDataMasking:
    def test_config_mention_inside_heredoc_body_is_not_a_stream_write(self):
        # A tee-written report whose heredoc PROSE mentions config.yaml —
        # the mention is data, not a write target. The stream-write rules
        # use DOTALL and previously walked into the body.
        cmd = (
            "tee /tmp/deploy-report.md > /dev/null <<'EOF'\n"
            "deploy steps:\n"
            "1. update ~/.hermes/config.yaml with new tokens\n"
            "EOF"
        )
        is_dangerous, key, _ = detect_dangerous_command(cmd)
        assert not is_dangerous, f"heredoc prose matched dangerous pattern: {key}"

    def test_plain_heredoc_report_is_not_a_stream_write(self):
        cmd = (
            "cat > /tmp/report.md <<'EOF'\n"
            "Section: deployment\n"
            "We considered storing this in config.yaml but decided against it.\n"
            "EOF"
        )
        is_dangerous, key, _ = detect_dangerous_command(cmd)
        assert not is_dangerous, f"heredoc prose matched dangerous pattern: {key}"

    def test_real_config_redirect_outside_heredoc_still_flags(self):
        cmd = (
            "echo 'key: value' >> ~/.hermes/config.yaml <<'EOF'\n"
            "report body\n"
            "EOF"
        )
        is_dangerous, _, _ = detect_dangerous_command(cmd)
        assert is_dangerous, "an actual config.yaml append must stay flagged"


# ---------------------------------------------------------------------------
# Fix C — scratch-space cleanup
# ---------------------------------------------------------------------------


class TestScratchCleanup:
    def test_rm_of_literal_tmp_file_is_not_dangerous(self):
        for cmd in (
            "rm -f /tmp/scratch-notes.md",
            "rm -rf /tmp/build-probe",
            "cd /tmp && rm -f probe1.txt probe2.txt",
        ):
            is_dangerous, key, _ = detect_dangerous_command(cmd)
            assert not is_dangerous, f"{cmd!r} flagged as: {key}"

    def test_rm_with_glob_still_demands_approval(self):
        is_dangerous, _, _ = detect_dangerous_command("rm -f /tmp/*.log")
        assert is_dangerous, "globbed rm must keep the approval flow"

    def test_rm_with_command_substitution_still_demands_approval(self):
        is_dangerous, _, _ = detect_dangerous_command("rm -f /tmp/$(whoami)-data")
        assert is_dangerous

    def test_rm_of_home_documents_still_demands_approval(self):
        is_dangerous, _, _ = detect_dangerous_command("rm -rf ~/Documents/project")
        assert is_dangerous

    def test_rm_traversal_escape_still_demands_approval(self):
        is_dangerous, _, _ = detect_dangerous_command("rm -f /tmp/../home/user/x")
        assert is_dangerous, "path traversal out of scratch must keep the approval flow"

    def test_hardline_floor_untouched_by_scratch_exemption(self):
        # Scratch cleanup only skips the approval-tier rm patterns; the
        # hardline floor still blocks recursive deletes of system dirs.
        is_hl, _ = detect_hardline_command("rm -rf /etc")
        assert is_hl
        # Mixed operands (scratch + system path) decline the exemption and
        # keep the dangerous-tier approval flow.
        is_dangerous, key, _ = detect_dangerous_command("rm -rf /tmp/scratch /etc/app.conf")
        assert is_dangerous, f"mixed scratch+system rm must keep approval: {key}"


# ---------------------------------------------------------------------------
# Fix D — tirith benign shapes
# ---------------------------------------------------------------------------


class TestTirithBenignShapes:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_append_to_log_under_logs_dir_is_suppressed(self, mock_cfg, mock_run):
        findings = [{"rule_id": "dotfile_overwrite", "severity": "high"}]
        mock_cfg.return_value = _tirith_cfg()
        mock_run.return_value = _mock_run(
            1, _tirith_stdout(findings, "dotfile overwrite"))
        result = check_command_security(
            "printf 'entry\\n' >> ~/.hermes/logs/audit.log")
        assert result["action"] == "allow", result
        assert result["findings"] == []

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_single_redirect_overwrite_keeps_finding(self, mock_cfg, mock_run):
        findings = [{"rule_id": "dotfile_overwrite", "severity": "high"}]
        mock_cfg.return_value = _tirith_cfg()
        mock_run.return_value = _mock_run(
            1, _tirith_stdout(findings, "dotfile overwrite"))
        result = check_command_security(
            "printf 'entry\\n' > ~/.hermes/logs/audit.log")
        assert result["action"] != "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_append_to_real_rc_dotfile_keeps_finding(self, mock_cfg, mock_run):
        findings = [{"rule_id": "dotfile_overwrite", "severity": "high"}]
        mock_cfg.return_value = _tirith_cfg()
        mock_run.return_value = _mock_run(
            1, _tirith_stdout(findings, "dotfile overwrite"))
        result = check_command_security(
            "echo 'alias x=y' >> ~/.bashrc")
        assert result["action"] != "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_scratch_deletion_burst_is_suppressed(self, mock_cfg, mock_run):
        findings = [{"rule_id": "mass_file_deletion", "severity": "medium"}]
        mock_cfg.return_value = _tirith_cfg()
        mock_run.return_value = _mock_run(
            1, _tirith_stdout(findings, "mass deletion"))
        result = check_command_security(
            "rm -f /tmp/probe-a.json /tmp/probe-b.json /tmp/probe-c.json")
        assert result["action"] == "allow", result

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_non_scratch_deletion_burst_keeps_finding(self, mock_cfg, mock_run):
        findings = [{"rule_id": "mass_file_deletion", "severity": "medium"}]
        mock_cfg.return_value = _tirith_cfg()
        mock_run.return_value = _mock_run(
            1, _tirith_stdout(findings, "mass deletion"))
        result = check_command_security(
            "rm -f ~/docs/a.md ~/docs/b.md ~/docs/c.md")
        assert result["action"] != "allow"
