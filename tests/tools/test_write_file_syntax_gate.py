"""Regression coverage for #60525: write_file() must reject invalid
JSON/YAML/TOML content BEFORE it touches disk, not write first and only
report the parse failure afterward (which left files_modified gating in
tools/file_tools.py reporting a corrupt write as a success).
"""

import json
from pathlib import Path

import pytest

import tools.file_operations as file_operations
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def ops(tmp_path: Path):
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


class TestWriteFileSyntaxGate:
    @pytest.mark.parametrize(
        "filename,bad_content",
        [
            ("config.json", '{"a": 1,'),
            ("config.yaml", 'key: "unclosed\n'),
            ("config.yml", 'key: "unclosed\n'),
            ("config.toml", "[section\nk = 'v'"),
        ],
    )
    def test_invalid_content_refused_nothing_written(self, ops, tmp_path, filename, bad_content):
        target = tmp_path / filename
        res = ops.write_file(str(target), bad_content)
        assert res.error is not None
        assert not target.exists()

    def test_invalid_content_leaves_parent_dir_uncreated(self, ops, tmp_path):
        """The gate must run before mkdir -- a rejected write has to have
        zero disk side effects, not just an unwritten target file."""
        target = tmp_path / "new_subdir" / "config.json"
        res = ops.write_file(str(target), '{"a": 1,')
        assert res.error is not None
        assert not target.parent.exists()

    def test_invalid_json_existing_file_left_untouched(self, ops, tmp_path):
        target = tmp_path / "config.json"
        target.write_text('{"a": 1}')
        res = ops.write_file(str(target), '{"a": 1,')
        assert res.error is not None
        assert target.read_text() == '{"a": 1}'

    def test_valid_json_and_yaml_still_written(self, ops, tmp_path):
        json_target = tmp_path / "config.json"
        content = json.dumps({"a": 1, "b": [1, 2, 3]})
        res = ops.write_file(str(json_target), content)
        assert res.error is None, res.error
        assert json_target.read_text() == content

        yaml_target = tmp_path / "config.yaml"
        content = "a: 1\nb:\n  - 1\n  - 2\n"
        res = ops.write_file(str(yaml_target), content)
        assert res.error is None, res.error
        assert yaml_target.read_text() == content

    def test_bom_marked_file_still_writable(self, ops, tmp_path):
        """Regression guard: the gate must validate the RAW content the
        caller passed in, before the BOM-preservation shim re-adds a BOM
        to an already-BOM-stripped rewrite. Checking post-shim content
        false-positives every rewrite of a BOM-marked file as invalid
        JSON (a leading U+FEFF is not valid JSON on its own)."""
        target = tmp_path / "config.json"
        target.write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}')

        res = ops.write_file(str(target), json.dumps({"a": 2}))
        assert res.error is None, res.error
        assert target.read_bytes() == b"\xef\xbb\xbf" + json.dumps({"a": 2}).encode()

    def test_non_linted_extension_with_garbage_still_written(self, ops, tmp_path):
        target = tmp_path / "notes.txt"
        garbage = "{{{ not json, not yaml, not anything ]]] <<<"
        res = ops.write_file(str(target), garbage)
        assert res.error is None, res.error
        assert target.read_text() == garbage

    def test_json_content_parsed_only_once(self, ops, tmp_path, monkeypatch):
        """The pre-write gate and the post-write lint-delta report must
        share a single parse of the same content, not each call
        json.loads independently on identical bytes. Two disconnected
        call sites re-deriving the same fact is the shape of bug that
        let the original gate go missing (a check existed, nothing
        wired its result to the write decision)."""
        calls = []
        real = file_operations._lint_json_inproc

        def counting(content):
            calls.append(content)
            return real(content)

        monkeypatch.setitem(file_operations.LINTERS_INPROC, ".json", counting)

        target = tmp_path / "config.json"
        content = json.dumps({"a": 1})
        res = ops.write_file(str(target), content)
        assert res.error is None, res.error
        assert len(calls) == 1, f"expected exactly one parse, got {len(calls)}"

    def test_invalid_python_is_not_hard_refused(self, ops, tmp_path):
        """Deliberate scope decision: .py keeps the pre-existing
        non-blocking lint-delta report rather than a hard refusal --
        this codebase's own test suite writes arbitrary non-Python
        content through *.py paths as generic write-mechanics
        fixtures."""
        target = tmp_path / "broken.py"
        bad_python = "def foo(:\n    pass\n"
        res = ops.write_file(str(target), bad_python)
        assert res.error is None, res.error
        assert target.read_text() == bad_python
