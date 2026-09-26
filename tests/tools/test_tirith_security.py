"""Tests for the tirith security scanning subprocess wrapper."""

import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import tools.tirith_security as _tirith_mod
from tools.tirith_security import check_command_security, ensure_installed


def _reset_state():
    _tirith_mod._install_attempted.clear()
    _tirith_mod._install_threads.clear()
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False
    _tirith_mod._circuit_open_at = 0.0


@pytest.fixture(autouse=True)
def _reset_tirith_state():
    _reset_state()
    yield
    _reset_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run(returncode=0, stdout="", stderr=""):
    """Build a mock subprocess.CompletedProcess."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _json_stdout(findings=None, summary=""):
    return json.dumps({"findings": findings or [], "summary": summary})


# ---------------------------------------------------------------------------
# Exit code → action mapping
# ---------------------------------------------------------------------------

class TestExitCodeMapping:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_0_allow(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(0, _json_stdout())
        result = check_command_security("echo hello")
        assert result["action"] == "allow"
        assert result["findings"] == []

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_1_block_with_findings(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": "homograph_url", "severity": "high"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "homograph detected"))
        result = check_command_security("curl http://gооgle.com")
        assert result["action"] == "block"
        assert len(result["findings"]) == 1
        assert result["summary"] == "homograph detected"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_2_warn_with_findings(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": "shortened_url", "severity": "medium"}]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "shortened URL"))
        result = check_command_security("curl https://bit.ly/abc")
        assert result["action"] == "warn"
        assert len(result["findings"]) == 1
        assert result["summary"] == "shortened URL"


# ---------------------------------------------------------------------------
# JSON parse failure (exit code still wins)
# ---------------------------------------------------------------------------

class TestJsonParseFailure:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_1_invalid_json_still_blocks(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(1, "NOT JSON")
        result = check_command_security("bad command")
        assert result["action"] == "block"
        assert "details unavailable" in result["summary"]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_0_invalid_json_allows(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(0, "NOT JSON")
        result = check_command_security("safe command")
        assert result["action"] == "allow"


# ---------------------------------------------------------------------------
# Operational failures + fail_open
# ---------------------------------------------------------------------------

class TestOSErrorFailOpen:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_file_not_found_fail_open(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        result = check_command_security("echo hi")
        assert result["action"] == "allow"
        assert "unavailable" in result["summary"]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_os_error_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        result = check_command_security("echo hi")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]


class TestTimeoutFailOpen:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_timeout_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tirith", timeout=5)
        result = check_command_security("slow command")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]


class TestUnknownExitCode:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_unknown_exit_code_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.return_value = _mock_run(99, "")
        result = check_command_security("cmd")
        assert result["action"] == "block"
        assert "exit code 99" in result["summary"]


# ---------------------------------------------------------------------------
# Circuit breaker: half-open recovery
# ---------------------------------------------------------------------------

def _open_breaker(age_s):
    """Put the breaker in the open state as if it tripped ``age_s`` seconds ago."""
    _tirith_mod._crash_count = _tirith_mod._CRASH_LIMIT
    _tirith_mod._circuit_open = True
    _tirith_mod._circuit_open_at = time.monotonic() - age_s


class TestCircuitBreakerHalfOpen:
    @pytest.mark.parametrize("returncode, action", [(0, "allow"), (1, "block"), (2, "warn")])
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_completed_probe_after_retry_window_closes_breaker(self, mock_cfg, mock_run, returncode, action):
        """Once the retry window has elapsed, one real scan runs; any verdict (allow/block/warn)
        proves the binary healthy and closes the breaker, so the next command is scanned again."""
        mock_cfg.return_value = _CFG
        _open_breaker(age_s=_tirith_mod._CIRCUIT_RETRY_S + 1)
        mock_run.return_value = _mock_run(returncode, _json_stdout())

        result = check_command_security("echo hi")

        assert result["action"] == action
        assert mock_run.call_count == 1
        assert (_tirith_mod._circuit_open, _tirith_mod._crash_count) == (False, 0)
        # Breaker closed: the following command is scanned rather than short-circuited.
        check_command_security("echo again")
        assert mock_run.call_count == 2

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_open_breaker_probes_once_per_window_and_failed_probe_rearms(self, mock_cfg, mock_run):
        """Inside the window nothing spawns; after it exactly one probe runs, and a probe that
        fails re-arms the window so the next caller is fail-open without spawning again."""
        mock_cfg.return_value = _CFG
        _open_breaker(age_s=1)
        mock_run.side_effect = OSError("binary gone")

        assert check_command_security("echo hi")["summary"] == "tirith disabled (circuit breaker)"
        assert mock_run.call_count == 0

        _open_breaker(age_s=_tirith_mod._CIRCUIT_RETRY_S + 1)
        assert check_command_security("echo hi")["action"] == "allow"  # probe spawned and failed
        assert mock_run.call_count == 1
        assert _tirith_mod._circuit_open is True
        assert check_command_security("echo hi")["summary"] == "tirith disabled (circuit breaker)"
        assert mock_run.call_count == 1  # re-armed: no second probe inside the fresh window


# ---------------------------------------------------------------------------
# Disabled
# ---------------------------------------------------------------------------

class TestDisabled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_allow(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        result = check_command_security("rm -rf /")
        assert result["action"] == "allow"


# ---------------------------------------------------------------------------
# Findings cap + summary cap
# ---------------------------------------------------------------------------

class TestCaps:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_findings_and_summary_capped(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": f"rule_{i}"} for i in range(100)]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "x" * 1000))
        result = check_command_security("cmd")
        assert len(result["findings"]) == 50
        assert len(result["summary"]) == 500


# ---------------------------------------------------------------------------
# Programming errors propagate
# ---------------------------------------------------------------------------

class TestProgrammingErrors:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_attribute_error_propagates(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = AttributeError("unexpected bug")
        with pytest.raises(AttributeError):
            check_command_security("cmd")


# ---------------------------------------------------------------------------
# ensure_installed
# ---------------------------------------------------------------------------

class TestEnsureInstalled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_none(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        assert ensure_installed() is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/tirith")
    @patch("tools.tirith_security._load_security_config")
    def test_found_on_path_returns_immediately(self, mock_cfg, mock_which):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        assert ensure_installed() == "/usr/local/bin/tirith"


# ---------------------------------------------------------------------------
# Unsupported platform (Windows etc.) — silent fast-path everywhere
# ---------------------------------------------------------------------------

class TestUnsupportedPlatform:
    """When PM has no tirith build for this OS+arch, the entire subsystem
    must stay silent: no install thread, no spawn attempts, no CLI banner. Pattern-matching
    guards still cover the gap; tirith content scanning is just absent."""

    @pytest.mark.parametrize("target, expected", [
        ("linux-x64", True),
        ("win32-x64", False),
        (RuntimeError("unsupported architecture: riscv64"), False),
    ])
    def test_is_platform_supported(self, target, expected):
        # Table inputs, not a host fake: support is PM's per-target mapping.
        current_target = MagicMock(side_effect=[target])
        with patch("pm.current_target", current_target):
            assert _tirith_mod.is_platform_supported() is expected

    @patch("tools.tirith_security._load_security_config")
    def test_check_command_security_unsupported_allows_silently(self, mock_cfg):
        """Windows: skip the resolver and spawn entirely — return allow with
        an empty summary so callers can't accidentally surface 'tirith
        unavailable' messaging to the user."""
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.subprocess.run") as mock_run, \
             patch("tools.tirith_security._resolve_tirith_path") as mock_resolve:
            result = check_command_security("rm -rf /")
            assert result == {"action": "allow", "findings": [], "summary": ""}
            mock_run.assert_not_called()
            mock_resolve.assert_not_called()

    def test_explicit_path_still_honored_on_unsupported_platform(self, tmp_path):
        """If a user explicitly configured a tirith_path (e.g. they built it
        themselves under WSL), the unsupported-platform short-circuit must
        NOT override that — explicit config wins."""
        custom = tmp_path / "tirith"
        custom.write_text("#!/bin/sh\nexit 0\n")
        custom.chmod(0o755)
        with patch("tools.tirith_security.is_platform_supported", return_value=False):
            assert _tirith_mod._resolve_tirith_path(str(custom)) == str(custom)


# ---------------------------------------------------------------------------
# PM-provisioned binary: one install attempt per home, explicit paths never download
# ---------------------------------------------------------------------------

_BARE_CFG = {"tirith_enabled": True, "tirith_path": "tirith",
             "tirith_timeout": 5, "tirith_fail_open": True}


@pytest.fixture
def pm_tirith(monkeypatch):
    """Nothing on PATH, nothing installed yet, lazy installs allowed."""
    import pm

    installed = MagicMock(return_value=None)
    ensure = MagicMock()
    monkeypatch.setattr("tools.tirith_security.shutil.which", lambda _name: None)
    monkeypatch.setattr(pm, "installed_package", installed)
    monkeypatch.setattr(pm, "ensure", ensure)
    monkeypatch.setattr(pm, "lazy_installs_allowed", lambda: True)
    return ensure, installed


class TestPmInstall:
    def test_default_path_installs_through_pm(self, pm_tirith):
        """The default bare 'tirith' is provisioned by PM on a cold scan."""
        ensure, installed = pm_tirith
        ensure.side_effect = lambda *_a, **_k: setattr(
            installed, "return_value", MagicMock(binary="/pm/tirith"))

        assert _tirith_mod._resolve_tirith_path("tirith") == "/pm/tirith"
        ensure.assert_called_once_with("tirith")

    def test_failed_install_is_not_retried(self, pm_tirith):
        """After a failed install, subsequent resolves fall back without retrying."""
        ensure, _ = pm_tirith
        ensure.side_effect = RuntimeError("download failed")

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        assert ensure.call_count == 1

    def test_tilde_explicit_path_missing_no_download(self, pm_tirith):
        """An explicit ~/path that doesn't exist must NOT trigger an install."""
        ensure, _ = pm_tirith

        result = _tirith_mod._resolve_tirith_path("~/bin/tirith")

        ensure.assert_not_called()
        assert "~" not in result  # tilde still expanded

    def test_install_proceeds_without_cosign(self, tmp_path):
        """Provenance is optional without cosign: SHA-256 verification alone proceeds."""
        with patch("tools.tirith_security.shutil.which", return_value=None):
            verified, reason = _tirith_mod.verify_release_provenance(tmp_path, MagicMock())
        assert (verified, reason) == (False, "")


# ---------------------------------------------------------------------------
# Background install / non-blocking startup (P2)
# ---------------------------------------------------------------------------

class TestBackgroundInstall:
    def test_ensure_installed_non_blocking(self, pm_tirith):
        """ensure_installed must return immediately when an install is needed."""
        with patch("tools.tirith_security._load_security_config", return_value=_BARE_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            assert ensure_installed() is None  # not available yet
            MockThread.assert_called_once()
            MockThread.return_value.start.assert_called_once()

    def test_scan_does_not_wait_on_startup_install(self, pm_tirith):
        """A scan during the startup install returns the default instead of installing again."""
        ensure, _ = pm_tirith
        with patch("tools.tirith_security._load_security_config", return_value=_BARE_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.threading.Thread"):
            ensure_installed()

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        ensure.assert_not_called()


# ---------------------------------------------------------------------------
# Warn-once dedupe (issue: tirith spawn failed spamming on Windows)
# ---------------------------------------------------------------------------

class TestSpawnWarningDedup:
    """When tirith isn't installed yet (background install in flight, or
    install marked failed), every terminal command spammed an identical
    ``tirith spawn failed: [WinError 2]`` warning to ``errors.log``. The
    dedupe set in ``_warn_once`` collapses repeats by ``(exc class, errno)``
    while still surfacing the first occurrence so users see the failure.
    """

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_repeated_spawn_failure_logs_once(self, mock_cfg, mock_run, caplog):
        mock_cfg.return_value = {
            "tirith_enabled": True, "tirith_path": "tirith",
            "tirith_timeout": 5, "tirith_fail_open": True,
        }
        mock_run.side_effect = FileNotFoundError("[WinError 2]")
        # Fresh dedupe state — clear any keys left by other tests.
        _tirith_mod._warned_messages.clear()

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            for i in range(15):
                result = check_command_security("echo hi")
                # Behavior must remain the same on every call —
                # fail-open allow, with the exception captured in summary.
                assert result["action"] == "allow"
                if i < _tirith_mod._CRASH_LIMIT:
                    # Before circuit breaker opens, summary has the exception
                    assert "unavailable" in result["summary"]
                else:
                    # After circuit breaker opens, summary is generic
                    assert "circuit breaker" in result["summary"]

        spawn_warnings = [
            rec for rec in caplog.records
            if "tirith spawn failed" in rec.message
        ]
        assert len(spawn_warnings) == 1, (
            f"expected exactly 1 spawn-failed warning across 15 commands, "
            f"got {len(spawn_warnings)}: {[r.message for r in spawn_warnings]}"
        )


# ---------------------------------------------------------------------------
# .app TLD suppression (issue #24461)
# ---------------------------------------------------------------------------

_CFG = {"tirith_enabled": True, "tirith_path": "tirith",
        "tirith_timeout": 5, "tirith_fail_open": True}


class TestAppTldSuppression:
    """warn verdicts whose only finding is lookalike_tld/.app are downgraded to allow."""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_app_only_warn_downgraded_to_allow(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "value": ".app",
                     "message": "Domain uses '.app' TLD which can be confused with file extensions"}]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, ".app TLD warning"))
        result = check_command_security("curl https://example.app")
        assert result["action"] == "allow"
        assert result["findings"] == []
        assert result["summary"] == ""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_mixed_findings_preserve_warn(self, mock_cfg, mock_run):
        """If .app finding is accompanied by another finding, warn is preserved."""
        mock_cfg.return_value = _CFG
        findings = [
            {"rule_id": "lookalike_tld", "value": ".app"},
            {"rule_id": "shortened_url", "severity": "medium"},
        ]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "mixed"))
        result = check_command_security("curl https://bit.ly/test.app")
        assert result["action"] == "warn"
        assert len(result["findings"]) == 2

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_block_verdict_never_suppressed(self, mock_cfg, mock_run):
        """block exit code is never downgraded, even if finding looks like .app."""
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "value": ".app"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "block"))
        result = check_command_security("curl https://example.app")
        assert result["action"] == "block"


class TestEmojiVariationSelectorSuppression:
    """VS16 after an emoji-capable base is presentation, not obfuscation: no approval prompt."""

    _VS = [{"rule_id": "variation_selector", "severity": "medium"}]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_emoji_only_variation_selector_warn_is_downgraded(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(2, _json_stdout(self._VS, "variation selector"))

        # SMP emoji, Dingbats/Misc Symbols, and BMP singletons outside those blocks (ℹ ▶).
        result = check_command_security('ls "🗞️ Journal/" "✅️ Projects/" "ℹ️ Info/" "▶️ Media/"')

        assert result == {"action": "allow", "findings": [], "summary": ""}

    @pytest.mark.parametrize("command, findings", [
        ("printf 'a️'", _VS),            # VS16 after a letter
        ("printf '0️'", _VS),            # VS16 after a digit (keycap base)
        ("printf 'x󠄀'", _VS),        # a non-VS16 selector
        ('curl https://bit.ly/x --output "🗞️ Journal/file"',  # emoji path + another finding
         _VS + [{"rule_id": "shortened_url", "severity": "medium"}]),
    ])
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_other_selectors_or_mixed_findings_keep_warn(self, mock_cfg, mock_run, command, findings):
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "variation selector"))

        result = check_command_security(command)

        assert result["action"] == "warn"
        assert result["findings"] == findings

# Pipe-to-interpreter false positive on local producers (issue #32737)
# ---------------------------------------------------------------------------

def _pipe_finding(lhs="my-wrapper", interp="python3"):
    """A tirith pipe_to_interpreter finding shaped like the real JSON payload."""
    return {"rule_id": "pipe_to_interpreter", "severity": "HIGH",
            "title": f"Pipe to interpreter: {lhs} | {interp}",
            "description": f"Command pipes local output into interpreter '{interp}'.",
            "evidence": [{"type": "command_pattern", "pattern": "pipe to interpreter",
                          "matched": f"{lhs} | {interp}"}],
            "mitre_id": "T1059.004",
            "remediation": "Write the output to a file first, review it, then execute."}


def _write_exe(path, body="#!/bin/sh\necho ok\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _cfg(**extra):
    cfg = dict(_CFG)
    cfg.update(extra)
    return cfg


class TestPipeToInterpreterLocalExemption:
    """#32737: a pipeline whose producer is a local user-owned executable is reported HIGH by
    tirith's pipe_to_interpreter rule exactly like a remote download piped to a shell. When every
    producer stage of the flagged pipe resolves to a user-owned executable inside a trusted
    scripts directory the finding is a false positive and is downgraded to allow; anything else
    keeps the block."""

    @pytest.fixture(autouse=True)
    def _trusted_dirs_stay_in_the_sandbox(self, monkeypatch, tmp_path):
        """Every default trusted directory is resolved and stat'ed, so pointing them at the real
        Hermes home would make these tests I/O against live state (the home guard flags it) on a
        shell that exports HERMES_HOME or HOME."""
        home = tmp_path / "hermes-home"
        monkeypatch.setattr(_tirith_mod, "get_hermes_home", lambda: home)
        monkeypatch.setattr(_tirith_mod, "get_default_hermes_root", lambda: home)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_local_producer_downgraded_to_allow(self, mock_cfg, mock_run, tmp_path):
        trusted = _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding(str(trusted))], "pipe to interpreter"))

        result = check_command_security(f"{trusted} | python3")

        assert result == {"action": "allow", "findings": [], "summary": ""}

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_bare_name_resolved_from_trusted_dir(self, mock_cfg, mock_run, tmp_path):
        """The issue's spelling: a bare wrapper name that exists only in a trusted bin dir."""
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        result = check_command_security("my-wrapper clients --limit 100 | python3")

        assert result["action"] == "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_default_hermes_bin_dir_is_trusted(self, mock_cfg, mock_run, tmp_path, monkeypatch):
        """$HERMES_HOME/bin and $HERMES_HOME/profiles/*/bin need no config opt-in."""
        home = tmp_path / "hermes"
        _write_exe(home / "bin" / "my-wrapper")
        monkeypatch.setattr(_tirith_mod, "get_hermes_home", lambda: home)
        monkeypatch.setattr(_tirith_mod, "get_default_hermes_root", lambda: home)
        mock_cfg.return_value = _cfg()
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        result = check_command_security("my-wrapper | python3")

        assert result["action"] == "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_system_producer_keeps_block(self, mock_cfg, mock_run, tmp_path):
        """`ls | python3`: /usr/bin/ls sits outside every trusted scripts directory."""
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding("ls")], "pipe to interpreter"))

        result = check_command_security("ls | python3")

        assert result["action"] == "block"
        assert result["findings"][0]["rule_id"] == "pipe_to_interpreter"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_unresolvable_producer_keeps_block(self, mock_cfg, mock_run, tmp_path):
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding("no-such-tool")], "pipe to interpreter"))

        assert check_command_security("no-such-tool | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_second_untrusted_stage_keeps_block(self, mock_cfg, mock_run, tmp_path):
        """`my-wrapper | tee f | python3`: tee lives in /usr/bin, so the pipe is not all-local."""
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        assert check_command_security(
            "my-wrapper | tee /tmp/out.json | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_network_fetcher_producer_never_exempted(self, mock_cfg, mock_run, tmp_path):
        """A fetcher named `curl` inside a trusted dir still cannot feed an interpreter, even
        when a user lists /usr/bin-shaped directories as trusted."""
        _write_exe(tmp_path / "scripts" / "curl")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding("curl")], "pipe to interpreter"))

        assert check_command_security("curl https://example.com | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_extra_finding_never_exempted(self, mock_cfg, mock_run, tmp_path):
        """Any second finding keeps the verdict, whatever the producer is."""
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        findings = [_pipe_finding(), {"rule_id": "shortened_url", "severity": "medium"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "pipe to interpreter"))

        result = check_command_security("my-wrapper | python3")

        assert result["action"] == "block"
        assert len(result["findings"]) == 2

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_command_substitution_producer_keeps_block(self, mock_cfg, mock_run, tmp_path):
        """A substitution hides the producer: content origin unknowable, so fail closed."""
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding("subst")], "pipe to interpreter"))

        assert check_command_security("$(curl http://x) | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_backtick_substitution_keeps_block(self, mock_cfg, mock_run, tmp_path):
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        assert check_command_security("`curl http://x` | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_subshell_producer_keeps_block(self, mock_cfg, mock_run, tmp_path):
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        assert check_command_security("(my-wrapper) | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_symlink_escaping_trusted_dir_keeps_block(self, mock_cfg, mock_run, tmp_path):
        """A symlink inside the trusted dir pointing at /bin/sh must not count as trusted."""
        scripts = tmp_path / "scripts"
        _write_exe(tmp_path / "scripts" / "my-wrapper")
        (scripts / "link").symlink_to("/bin/sh")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(scripts)])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        assert check_command_security("link | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_world_writable_producer_keeps_block(self, mock_cfg, mock_run, tmp_path):
        """World-writable means any local account can rewrite the producer after the scan."""
        scripts = tmp_path / "scripts"
        wrapper = _write_exe(scripts / "my-wrapper")
        wrapper.chmod(0o777)
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(scripts)])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding()], "pipe to interpreter"))

        assert check_command_security("my-wrapper | python3")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_warn_verdict_also_downgraded(self, mock_cfg, mock_run, tmp_path):
        trusted = _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(2, _json_stdout(
            [_pipe_finding(str(trusted))], "pipe to interpreter"))

        assert check_command_security(f"{trusted} | python3")["action"] == "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_unrelated_trailing_statement_still_exempt(self, mock_cfg, mock_run, tmp_path):
        """A trailing statement with no pipe is irrelevant to the rule."""
        trusted = _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding(str(trusted))], "pipe to interpreter"))

        assert check_command_security(f"{trusted} | python3 && echo done")["action"] == "allow"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_second_pipeline_must_also_be_local(self, mock_cfg, mock_run, tmp_path):
        """A second pipe into an interpreter whose producer is /usr/bin keeps the verdict."""
        trusted = _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding(str(trusted))], "pipe to interpreter"))

        assert check_command_security(
            f"{trusted} | python3 && ls | sh")["action"] == "block"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_no_pipe_into_interpreter_means_no_downgrade(self, mock_cfg, mock_run, tmp_path):
        """Nothing looks like `<stage> | <interpreter>`, so the helper must not vouch."""
        trusted = _write_exe(tmp_path / "scripts" / "my-wrapper")
        mock_cfg.return_value = _cfg(trusted_executable_dirs=[str(tmp_path / "scripts")])
        mock_run.return_value = _mock_run(1, _json_stdout(
            [_pipe_finding(str(trusted))], "pipe to interpreter"))

        assert check_command_security(f"{trusted} > /tmp/out.json")["action"] == "block"


class TestSplitPipelineGroups:
    """Unit tests for the conservative top-level pipeline splitter."""

    def test_simple_pipeline(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups("a | b") == [["a", "b"]]

    def test_quoted_pipe_is_not_a_split(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups('a "x|y" | b') == [['a "x|y"', "b"]]

    def test_or_and_and_split_statements(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups("a | b || c | d") == [["a", "b"], ["c", "d"]]
        assert _split_pipeline_groups("a | b && c") == [["a", "b"], ["c"]]

    def test_stderr_redirect_is_not_a_statement_break(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups("a 2>&1 | b") == [["a 2>&1", "b"]]

    def test_command_substitution_is_rejected(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups("$(a) | b") is None
        assert _split_pipeline_groups("a `b` | c") is None

    def test_subshell_and_brace_group_are_rejected(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups("(a) | b") is None
        assert _split_pipeline_groups("{ a; } | b") is None
        assert _split_pipeline_groups("a | b {x}") is None

    def test_quoted_braces_are_kept(self):
        from tools.tirith_security import _split_pipeline_groups
        assert _split_pipeline_groups("a '{x}' | b") == [["a '{x}'", "b"]]
