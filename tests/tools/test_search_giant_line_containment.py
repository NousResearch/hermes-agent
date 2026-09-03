"""Giant single-line file containment in content search (cline/cline#13525 port).

A match inside a serialized dump (multi-MB single-line JSON, minified
bundle) used to make rg/grep emit the ENTIRE matched line into stdout:
``head -n`` counts lines, so a 40MB match line crossed the transport
untruncated and was buffered whole into Python before the per-match
[:500] clamp ran (measured 42MB transport / ~180MB peak alloc for one
match). The fix bounds lines at the search-engine layer: rg gets
``--max-columns 2000 --max-columns-preview``; the grep fallbacks pipe
through ``cut -c1-2000``.

These tests run the REAL pipelines via bash (no mocked stdout) so the
flag/pipe behavior of the installed rg/grep is what's exercised.
"""

import os
import shutil
import subprocess

import pytest

from tools.file_operations import ShellFileOperations

# Big enough to prove containment, small enough to keep the test fast.
GIANT = 5 * 1024 * 1024  # 5MB single line
# Generous ceiling: pre-fix stdout for one giant match is >= GIANT bytes.
STDOUT_CEILING = 1 * 1024 * 1024


class RecordingEnv:
    """Local bash executor that records the largest stdout it returned."""

    def __init__(self, cwd):
        self.cwd = cwd
        self.max_stdout = 0

    def execute(self, command, timeout=60, **kwargs):
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, errors="replace",
            timeout=timeout + 30,
        )
        out = proc.stdout + (proc.stderr or "")
        self.max_stdout = max(self.max_stdout, len(out))
        return {"output": out, "returncode": proc.returncode}


@pytest.fixture()
def giant_dir(tmp_path):
    (tmp_path / "trace.json").write_text(
        '{"needle": "' + "x" * GIANT + '"}', encoding="utf-8"
    )
    (tmp_path / "small.py").write_text("needle = 1\n", encoding="utf-8")
    return tmp_path


def _ops(giant_dir, engine):
    env = RecordingEnv(str(giant_dir))
    ops = ShellFileOperations(env)
    ops._has_command = lambda cmd: cmd == engine
    return ops, env


@pytest.mark.parametrize("engine", ["rg", "grep"])
def test_giant_single_line_match_is_bounded(giant_dir, engine):
    if shutil.which(engine) is None:
        pytest.skip(f"{engine} not installed")
    ops, env = _ops(giant_dir, engine)

    result = ops.search("needle", path=str(giant_dir), target="content")

    assert result.error is None
    paths = {os.path.basename(m.path) for m in result.matches}
    # The giant-file match must still be REPORTED (preview, not omission)...
    assert paths == {"trace.json", "small.py"}
    assert all(len(m.content) <= 500 for m in result.matches)
    # ...but its full line must never have crossed the transport.
    assert env.max_stdout < STDOUT_CEILING, (
        f"{engine} pipeline returned {env.max_stdout} bytes of stdout — "
        "giant matched line was not truncated at the engine layer"
    )


@pytest.mark.parametrize("engine", ["rg", "grep"])
@pytest.mark.parametrize("output_mode", ["files_only", "count"])
def test_line_cap_skipped_for_path_and_count_modes(giant_dir, engine, output_mode):
    """files_only/count lines are paths/counts — never giant, never cut."""
    if shutil.which(engine) is None:
        pytest.skip(f"{engine} not installed")
    ops, env = _ops(giant_dir, engine)

    result = ops.search("needle", path=str(giant_dir), target="content",
                        output_mode=output_mode)

    assert result.error is None
    if output_mode == "files_only":
        assert {os.path.basename(f) for f in result.files} == {"trace.json", "small.py"}
    else:
        assert {os.path.basename(k) for k in result.counts} == {"trace.json", "small.py"}
    assert env.max_stdout < STDOUT_CEILING
