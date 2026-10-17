"""An addition-only patch must retain all existing lines when appending."""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.mark.parametrize("original", [b"", b"\n", b"\n\n", b"existing", b"existing\n", b"existing\n\n"])
def test_addition_only_append_preserves_existing_prefix(tmp_path, original):
    target = tmp_path / "notes.txt"
    target.write_bytes(original)
    environment = LocalEnvironment(str(tmp_path))
    try:
        operations = ShellFileOperations(environment)
        result = operations.patch_v4a(
            "*** Begin Patch\n*** Update File: notes.txt\n+added\n*** End Patch"
        )
        assert result.success, result.error
        separator = b"\n" if original and not original.endswith(b"\n") else b""
        assert target.read_bytes() == original + separator + b"added\n"
    finally:
        environment.cleanup()
