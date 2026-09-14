"""UTF-16 pagination must preserve the same logical lines as UTF-8 reads."""

import json

import pytest

from tools import file_tools
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.fixture
def read_tool(tmp_path, monkeypatch):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)), cwd=str(tmp_path))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": ops)

    def read(path, offset=1, limit=2000):
        result = json.loads(registry.dispatch(
            "read_file", {"path": str(path), "offset": offset, "limit": limit},
            task_id=f"utf16-pagination-{tmp_path.name}",
        ))
        assert not result.get("error"), result
        return result

    return read


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("native", ["0", "1"])
@pytest.mark.parametrize("text", ["first\nlast\n", "first\r\nlast\r\n", "first\nlast"])
def test_utf16_exact_page_has_no_phantom_line(
    tmp_path, monkeypatch, read_tool, encoding, native, text,
):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    encoded = tmp_path / "encoded.txt"
    plain = tmp_path / "plain.txt"
    encoded.write_bytes(("\ufeff" + text).encode(encoding))
    plain.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))

    expected = read_tool(plain, limit=2)
    actual = read_tool(encoded, limit=2)

    assert actual["total_lines"] == expected["total_lines"]
    assert actual["content"] == expected["content"]
    assert actual.get("truncated", False) == expected.get("truncated", False)
    assert "Transcoded from" in actual.get("hint", "")


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("native", ["0", "1"])
@pytest.mark.parametrize("text", ["first\n\nlast\n", "first\n\n"])
def test_utf16_page_preserves_a_selected_blank_line(
    tmp_path, monkeypatch, read_tool, encoding, native, text,
):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", native)
    encoded = tmp_path / "encoded.txt"
    plain = tmp_path / "plain.txt"
    encoded.write_bytes(("\ufeff" + text).encode(encoding))
    plain.write_bytes(text.encode("utf-8"))

    expected = read_tool(plain, limit=2)
    actual = read_tool(encoded, limit=2)

    assert actual["content"] == expected["content"] == "1|first\n2|"
    assert actual["total_lines"] == expected["total_lines"]
    assert actual.get("truncated", False) == expected.get("truncated", False)
