"""Public POSIX rm gating complements the host-native canonical-exemption test."""

from unittest.mock import patch

import pytest

from tools.approval import detect_dangerous_command


@pytest.mark.platforms("posix")
@pytest.mark.require_symlinks
def test_symlinked_temp_dir_keeps_public_rm_gate(tmp_path):
    real_temp = tmp_path / "real-temp"
    real_temp.mkdir()
    linked_temp = tmp_path / "linked-temp"
    linked_temp.symlink_to(real_temp, target_is_directory=True)
    basename = "hermes-verify-example.py"

    with patch("tempfile.gettempdir", return_value=str(linked_temp)):
        assert detect_dangerous_command(f'rm -f "{linked_temp / basename}"')[0] is True
        assert detect_dangerous_command(f'rm -f "{real_temp / basename}"') == (
            False, None, None,
        )
