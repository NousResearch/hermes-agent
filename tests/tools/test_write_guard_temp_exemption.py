"""Tests for the temp-dir exemption of the stale write_file overwrite blocker.

Cron agents reuse canonical temp-file names across runs (e.g.
``/tmp/clawdtalk-report.txt``), so the stale-overwrite blocker refused the
fresh run's write: this task never read the previous run's artifact. Temp
dirs are world-writable scratch space, so ``_temp_path_exempt`` skips the
staleness refusal for them. Two load-bearing properties:

- The decision runs on the REALPATH, so a symlink planted in a temp dir
  pointing at a real user file resolves to the real path and is NOT exempt.
- Both ``/var/folders`` spellings must be checked: macOS realpath
  canonicalises ``/var/...`` to ``/private/var/...``, so a lone
  ``"/var/folders/"`` prefix is dead code there.
"""

import os

import pytest

from tools.file_tools_write_guards import _stale_overwrite_blocker, _temp_path_exempt


class TestTempPathExempt:
    def test_tmp_prefix_exempt(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        assert _temp_path_exempt("/tmp/clawdtalk-report.txt") is True

    def test_private_tmp_prefix_exempt(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        assert _temp_path_exempt("/private/tmp/clawdtalk-report.txt") is True

    def test_var_folders_dual_form_exempt(self, monkeypatch):
        # macOS realpath canonicalises /var -> /private/var, so an input
        # spelled "/var/folders/..." only matches via the /private form; on
        # Linux it matches via the /var form. TMPDIR is unset so no other
        # rule can answer this.
        monkeypatch.delenv("TMPDIR", raising=False)
        assert _temp_path_exempt("/var/folders/h6/x.txt") is True

    def test_private_var_folders_prefix_exempt(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        assert _temp_path_exempt("/private/var/folders/h6/x.txt") is True

    def test_tmpdir_env_rule(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        assert _temp_path_exempt(str(tmp_path / "send-clawdtalk-report.py")) is True
        assert _temp_path_exempt("/Users/tester/documents/outside.md") is False

    def test_symlink_to_real_user_file_not_exempt(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        target = "/Users/tester/notes/real.md"  # outside every temp prefix
        link = tmp_path / "link.md"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
        # realpath resolves through the link: the TARGET decides, so a real
        # user file stays protected even when reached from a temp dir.
        assert _temp_path_exempt(str(link)) is False

    def test_real_user_file_not_exempt(self, monkeypatch):
        monkeypatch.delenv("TMPDIR", raising=False)
        assert _temp_path_exempt("/Users/tester/notes.md") is False

    def test_sensitive_socket_path_not_exempt(self, monkeypatch):
        # The sensitive-path hard-deny remains the enforcement for
        # /var/run/docker.sock; the exemption must not claim it either.
        monkeypatch.delenv("TMPDIR", raising=False)
        assert _temp_path_exempt("/var/run/docker.sock") is False

    def test_none_and_empty_not_exempt(self):
        assert _temp_path_exempt(None) is False
        assert _temp_path_exempt("") is False


class TestStaleOverwriteBlockerExemption:
    """The early return in ``_stale_overwrite_blocker``: a temp path skips the
    staleness refusal even when the existing file was never read by this task."""

    def test_temp_artifact_overwrite_allowed_without_read(self, tmp_path, monkeypatch):
        # The production failure: a fresh task reusing last run's canonical
        # temp name, previous content never read. Pre-fix this was refused.
        monkeypatch.delenv("HERMES_DISABLE_FILE_STATE_GUARD", raising=False)
        stale = tmp_path / "clawdtalk-report.txt"
        stale.write_text("previous run's report", encoding="utf-8")
        result = _stale_overwrite_blocker(str(stale), str(stale), "temp-exempt-allow")
        assert result is None

    def test_same_state_refused_without_the_exemption(self, tmp_path, monkeypatch):
        # Negative control: identical never-read state, exemption switched
        # off (the pre-fix behavior) -> the staleness refusal fires. Proves
        # the setup reaches the refusal branch, so the exempt assertion above
        # tests the exemption, not an accidental pass-through.
        monkeypatch.delenv("HERMES_DISABLE_FILE_STATE_GUARD", raising=False)
        stale = tmp_path / "clawdtalk-report.txt"
        stale.write_text("previous run's report", encoding="utf-8")
        monkeypatch.setattr(
            "tools.file_tools_write_guards._temp_path_exempt", lambda resolved: False)
        result = _stale_overwrite_blocker(str(stale), str(stale), "temp-exempt-refuse")
        assert result is not None
        assert "has not seen its full current content" in result
