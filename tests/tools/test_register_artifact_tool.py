"""Tests for the register_artifact file-staging tool.

The tool copies a workspace file into ``<workspace>/outputs/<product_run_id>/``
(or bare ``outputs/`` when no managed run id is bound) so the backend's post-run
sweep can deliver it. It must be path-safe (no traversal / absolute escape) and
degrade gracefully without a run id. No SSE/threads are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from karinai.runtime.session_bridge import bind_karinai_run_context
from tools.register_artifact_tool import _handle_register_artifact


def _bind_run(product_run_id: str) -> list:
    """Bind an api_server session with a KarinAI product run id, as the
    api_server's _bind_api_server_session does: base session vars via
    set_session_vars, the run-scoped KarinAI vars via bind_karinai_run_context,
    one concatenated token list."""
    tokens = set_session_vars(platform="api_server", async_delivery=False)
    tokens += bind_karinai_run_context(product_run_id=product_run_id)
    return tokens


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean workspace pinned as both the safe-write root and the cwd anchor."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(ws))
    monkeypatch.setenv("TERMINAL_CWD", str(ws))
    monkeypatch.delenv("HERMES_PRODUCT_RUN_ID", raising=False)
    yield ws
    # Restore the _UNSET sentinel so state never leaks into later tests.
    # NOT clear_session_vars([]): that pins every var to "" ("explicitly
    # cleared"), which suppresses the os.environ fallback in get_session_env
    # for the REST OF THE PYTEST PROCESS — breaking any later test in this
    # process that binds session env via monkeypatch.setenv (e.g.
    # tests/karinai/test_image_gateway_provider.py).
    reset_session_vars()


def _call(path=None, name=None, description=None):
    args = {}
    if path is not None:
        args["path"] = path
    if name is not None:
        args["name"] = name
    if description is not None:
        args["description"] = description
    return json.loads(_handle_register_artifact(args, task_id="default"))


def test_managed_run_stages_into_outputs_run_id(workspace: Path) -> None:
    _bind_run("run_abc123")
    (workspace / "report.txt").write_text("hello", encoding="utf-8")

    res = _call(path="report.txt", name="Q3 Report.txt")

    # Managed branch reports staged (delivery is the backend sweep's job; the
    # tool does not over-promise "delivered").
    assert res["success"] is True and res["staged"] is True
    assert "delivered" not in res
    dest = workspace / "outputs" / "run_abc123" / "Q3 Report.txt"
    assert dest.is_file() and dest.read_text(encoding="utf-8") == "hello"
    assert res["path"] == "outputs/run_abc123/Q3 Report.txt"


def test_no_run_id_degrades_to_outputs_undelivered(workspace: Path) -> None:
    # Explicitly bind an empty product run id (the dev/non-managed case).
    _bind_run("")
    (workspace / "data.csv").write_text("a,b", encoding="utf-8")

    res = _call(path="data.csv")

    assert res["success"] is True and res["delivered"] is False
    assert (workspace / "outputs" / "data.csv").is_file()
    assert "not be auto-delivered" in res["message"]


def test_default_name_is_source_filename(workspace: Path) -> None:
    _bind_run("run_x")
    (workspace / "chart.png").write_bytes(b"\x89PNG")

    res = _call(path="chart.png")

    assert (workspace / "outputs" / "run_x" / "chart.png").is_file()
    assert res["name"] == "chart.png"


def test_name_reduced_to_safe_basename(workspace: Path) -> None:
    _bind_run("run_x")
    (workspace / "f.txt").write_text("x", encoding="utf-8")

    res = _call(path="f.txt", name="../../etc/evil.txt")

    # The traversal in `name` is stripped to a plain basename inside the run dir.
    assert res["name"] == "evil.txt"
    dest = workspace / "outputs" / "run_x" / "evil.txt"
    assert dest.is_file()
    assert not (workspace.parent / "etc" / "evil.txt").exists()


def test_missing_path_errors(workspace: Path) -> None:
    res = _call()
    assert "error" in res and "path" in res["error"]


def test_missing_file_errors(workspace: Path) -> None:
    _bind_run("run_x")
    res = _call(path="nope.txt")
    assert "error" in res and "no file found" in res["error"]


def test_relative_traversal_rejected(workspace: Path) -> None:
    _bind_run("run_x")
    # A file outside the workspace must not be reachable via '..'.
    (workspace.parent / "secret.txt").write_text("top secret", encoding="utf-8")
    res = _call(path="../secret.txt")
    assert "error" in res and "inside your workspace" in res["error"]


def test_absolute_path_outside_workspace_rejected(workspace: Path, tmp_path: Path) -> None:
    _bind_run("run_x")
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    res = _call(path=str(outside))
    assert "error" in res and "inside your workspace" in res["error"]


def test_symlinked_output_dir_rejected(workspace: Path, tmp_path: Path) -> None:
    # The agent (which also has shell) plants outputs/<run_id> as a symlink to an
    # external dir, hoping copy2 follows it out of the workspace.
    import os as _os

    _bind_run("run_x")
    (workspace / "f.txt").write_text("payload", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    (workspace / "outputs").mkdir()
    _os.symlink(str(external), str(workspace / "outputs" / "run_x"))

    res = _call(path="f.txt", name="evil.txt")

    assert "error" in res and "symlinked output directory" in res["error"]
    assert not (external / "evil.txt").exists()  # nothing written outside the workspace


def test_leaf_symlink_not_followed(workspace: Path, tmp_path: Path) -> None:
    # A pre-planted leaf symlink at the destination name must not be written
    # through; the tool sidesteps it to a fresh, contained file.
    import os as _os

    _bind_run("run_x")
    (workspace / "f.txt").write_text("payload", encoding="utf-8")
    out_dir = workspace / "outputs" / "run_x"
    out_dir.mkdir(parents=True)
    external_target = tmp_path / "target.txt"
    external_target.write_text("original", encoding="utf-8")
    _os.symlink(str(external_target), str(out_dir / "report.txt"))

    res = _call(path="f.txt", name="report.txt")

    assert res["success"] is True
    assert external_target.read_text(encoding="utf-8") == "original"  # not overwritten
    assert res["name"] != "report.txt"  # sidestepped to a fresh name
    assert (out_dir / res["name"]).is_file() and not (out_dir / res["name"]).is_symlink()


def test_filename_collision_preserves_both(workspace: Path) -> None:
    _bind_run("run_x")
    (workspace / "a.txt").write_text("first", encoding="utf-8")
    (workspace / "b.txt").write_text("second", encoding="utf-8")

    r1 = _call(path="a.txt", name="report.txt")
    r2 = _call(path="b.txt", name="report.txt")

    out_dir = workspace / "outputs" / "run_x"
    assert r1["name"] == "report.txt"
    assert r2["name"] == "report (2).txt"  # second is uniquified, not overwritten
    assert (out_dir / "report.txt").read_text(encoding="utf-8") == "first"
    assert (out_dir / "report (2).txt").read_text(encoding="utf-8") == "second"


def test_oversize_file_warns(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.register_artifact_tool as mod

    monkeypatch.setattr(mod, "_BACKEND_MAX_FILE_BYTES", 4)
    _bind_run("run_x")
    (workspace / "big.bin").write_bytes(b"0123456789")  # 10 bytes > 4

    res = _call(path="big.bin", name="big.bin")

    assert res["success"] is True
    assert "256 MB" in res["message"] and "may not be delivered" in res["message"]


def test_run_id_visible_in_worker_thread(workspace: Path) -> None:
    # Production runs tool handlers on a ThreadPoolExecutor worker via
    # propagate_context_to_thread (copy_context). Prove HERMES_PRODUCT_RUN_ID
    # set on the parent thread reaches the handler on the worker thread.
    import concurrent.futures

    from tools.thread_context import propagate_context_to_thread

    _bind_run("run_thread")
    (workspace / "f.txt").write_text("x", encoding="utf-8")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        raw = ex.submit(propagate_context_to_thread(_handle_register_artifact), {"path": "f.txt"}, task_id="default").result()
    res = json.loads(raw)

    assert res["success"] is True
    # Lands in the parent run's dir, proving the contextvar crossed the thread.
    assert (workspace / "outputs" / "run_thread" / "f.txt").is_file()
    assert res["path"] == "outputs/run_thread/f.txt"


def test_session_context_product_run_id_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.session_context import get_session_env

    monkeypatch.delenv("HERMES_PRODUCT_RUN_ID", raising=False)
    tokens = _bind_run("run_zzz")
    try:
        assert get_session_env("HERMES_PRODUCT_RUN_ID") == "run_zzz"
    finally:
        clear_session_vars(tokens)
    try:
        # Cleared context returns "" (no os.environ fallback), per the documented contract.
        assert get_session_env("HERMES_PRODUCT_RUN_ID") == ""
    finally:
        # This test doesn't use the workspace fixture, so restore _UNSET itself
        # rather than leaving the vars pinned to "" for later tests.
        reset_session_vars()
