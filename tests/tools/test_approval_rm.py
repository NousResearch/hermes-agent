"""Regression tests for rm approval detection."""

from unittest.mock import patch as mock_patch

import pytest

from tools import approval, approval_detection
from tools.approval import detect_dangerous_command


def test_previous_drive_regex_approval_key_still_matches(monkeypatch):
    monkeypatch.setattr(approval, "_permanent_approved", set())
    approval.load_permanent({r'''rm\s+(-[^\s]*\s+)*["\']?(?:[a-z]:)?/'''})
    dangerous, key, _ = detect_dangerous_command("rm X:/file")
    assert dangerous and approval.is_approved("drive-regex-legacy", key)


class TestDetectDangerousRm:
    def test_rm_flags_after_operands_detected(self):
        # GNU rm permutes options: `rm build/ -rf` == `rm -rf build/`.
        # Port of openai/codex#33464.
        for cmd in (
            "rm build/ -rf",
            "rm build/ --recursive --force",
            "rm ~/projects -rf",
            "sudo rm build/ -rf",
            "rm one two three -rf",
        ):
            is_dangerous, _key, desc = detect_dangerous_command(cmd)
            assert is_dangerous is True, f"{cmd!r} should require approval"
            assert "delete" in desc.lower()


    @pytest.mark.platforms("linux")
    @pytest.mark.parametrize("command", [
        "rm -f /tmp/arbitrary-probe.py",
        "git diff --check && git status --short && rm -f /tmp/arbitrary-probe.py",
        "rm -f /var/tmp/arbitrary-probe.py",
    ])
    def test_nonrecursive_absolute_temp_cleanup_is_not_dangerous(self, command):
        """A single-file temp cleanup is not a root-directory deletion."""
        assert detect_dangerous_command(command) == (False, None, None)

    @pytest.mark.platforms("linux")
    def test_nonrecursive_verification_artifact_cleanup_is_not_dangerous(self):
        with mock_patch("tempfile.gettempdir", return_value="/tmp"):
            for prefix in ("hermes-verify-", "hermes-ad-hoc-"):
                assert detect_dangerous_command(f"rm -f /tmp/{prefix}example.py") == (
                    False,
                    None,
                    None,
                )

    @pytest.mark.platforms("linux")
    @pytest.mark.require_symlinks
    def test_symlinked_temp_dir_only_exempts_canonical_target(self, tmp_path):
        real_temp = tmp_path / "real-temp"
        real_temp.mkdir()
        linked_temp = tmp_path / "linked-temp"
        linked_temp.symlink_to(real_temp, target_is_directory=True)
        basename = "hermes-verify-example.py"

        with mock_patch("tempfile.gettempdir", return_value=str(linked_temp)):
            assert approval_detection._is_verification_artifact_cleanup(
                f"rm -f {linked_temp / basename}"
            ) is False
            assert detect_dangerous_command(f"rm -f {linked_temp / basename}") == (
                False,
                None,
                None,
            )
            assert detect_dangerous_command(f"rm -f {real_temp / basename}") == (
                False,
                None,
                None,
            )

    @pytest.mark.parametrize("command", [
        "rm -f foo.txt",
        "rm -f build/foo.o",
        "unlink /tmp/arbitrary-probe.py",
        "rm foo.txt -f",
    ])
    def test_existing_nonrecursive_cleanup_forms_remain_unprompted(self, command):
        assert detect_dangerous_command(command) == (False, None, None)

    @pytest.mark.parametrize("command", [
        "rm -f /",
        "(rm -f /)",
        "`rm -f /`",
        "rm -f />/dev/null",
        'rm -f "/"',
        "rm -f '/'",
        "rm -f X:/notes.txt",
        'rm -f "C:/notes.txt"',
        "rm -rf /",
        "rm -rf /etc/nginx",
        "rm -rf /home/alice/project",
        "rm build/ -rf",
    ])
    def test_root_and_recursive_deletions_remain_dangerous(self, command):
        is_dangerous, key, description = detect_dangerous_command(command)
        assert is_dangerous is True, command
        assert key is not None, command
        assert "delete" in description.lower(), command

    def test_verification_cleanup_exemption_still_rejects_noncanonical_shapes(self):
        commands = (
            "rm -rf /tmp/hermes-verify-example.py",
            "rm -f /tmp/hermes-verify-example.py /tmp/other.py",
            "rm -f /tmp/nested/../hermes-verify-example.py",
            "rm -f /tmp/a/../../tmp/hermes-verify-example.py",
            "rm -f /var/tmp/hermes-verify-example.py",
            "rm -f /tmp/hermes-verify-*",
            "rm -f /tmp/hermes-verify-$(touch>/tmp/pwned).py",
            "rm -f /tmp/hermes-ad-hoc-`touch>/tmp/pwned`.py",
            "rm -f /tmp/hermes-verify-example.py; touch /tmp/pwned",
        )
        assert detect_dangerous_command("rm -rf /tmp/hermes-verify-example.py")[0] is True
        with mock_patch("tempfile.gettempdir", return_value="/tmp"):
            assert all(
                approval_detection._is_verification_artifact_cleanup(command) is False
                for command in commands
            )
