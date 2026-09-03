"""F-17 / F-18 — attachment integrity for the ``kanban_attach`` tool surface.

F-17: base64 is prefix-decodable, so a payload truncated by the model's
output cap still decodes cleanly and used to be stored silently. Attachments
are the only durable output path a completed worker has (scratch workspaces
are deleted on completion), so a silent short write is data loss.

F-18: ``base64.b64decode(..., validate=True)`` rejects whitespace, so ordinary
line-wrapped (MIME) base64 failed outright.

The fix has two halves:

* ``content_base64`` now requires ``expected_bytes`` **and**
  ``expected_sha256``, both verified *before* anything reaches
  ``store_attachment_bytes``;
* a ``path`` branch reads the bytes off disk inside the task's own resolved
  workspace, so they never travel through the model at all.

Case ids (U1-U13, P1-P14) match the test plan in
``F17-ATTACHMENT-INTEGRITY-DESIGN.md`` plus the fail-closed
``workspace_path IS NULL`` case decided in HERMES-NL-05B.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _isolate_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Worker whose task has NO resolved workspace (``workspace_path`` NULL).

    This is the state ``create_task`` leaves a scratch task in until the
    dispatcher calls ``resolve_workspace``. Inline attaches must still work
    here; ``path=`` must fail closed (P14).
    """
    kb = _isolate_home(monkeypatch, tmp_path)
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        assert kb.get_task(conn, tid).workspace_path is None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


@pytest.fixture
def worker_ws(monkeypatch, tmp_path):
    """Worker whose task HAS a resolved workspace, as after a real dispatch.

    Returns ``(task_id, workspace_dir)``.
    """
    kb = _isolate_home(monkeypatch, tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-ws", assignee="test-worker")
        kb.claim_task(conn, tid)
        # Mirror what resolve_workspace() persists at dispatch time.
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?", (str(ws), tid)
        )
        conn.commit()
        assert kb.get_task(conn, tid).workspace_path == str(ws)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid, ws


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(args):
    from tools import kanban_tools as kt

    return json.loads(kt._handle_attach(args))


def _attachments(tid):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        return kb.list_attachments(conn, tid)
    finally:
        conn.close()


def _assert_refused(out, tid, *, contains=None):
    assert "error" in out, f"expected refusal, got {out}"
    if contains:
        assert contains.lower() in out["error"].lower(), out["error"]
    assert _attachments(tid) == [], "nothing may be stored on a refusal"


def _inline(payload: bytes, **over):
    args = {
        "filename": "d.txt",
        "content_base64": base64.b64encode(payload).decode(),
        "expected_bytes": len(payload),
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.update(over)
    return args


PAYLOAD = b"hermes attachment integrity payload " * 8


# ---------------------------------------------------------------------------
# U1-U13 — inline content_base64
# ---------------------------------------------------------------------------


def test_U1_inline_without_expected_is_refused(worker_env):
    """Clean base64 with no declared expectation must be refused outright."""
    args = _inline(PAYLOAD)
    args.pop("expected_bytes")
    args.pop("expected_sha256")
    _assert_refused(_call(args), worker_env, contains="expected_bytes")


def test_U2_line_wrapped_base64_is_accepted(worker_env):
    """F-18: 76-char MIME wrapping with a trailing newline must decode."""
    wrapped = textwrap.fill(base64.b64encode(PAYLOAD).decode(), 76) + "\n"
    out = _call(_inline(PAYLOAD, content_base64=wrapped))
    assert out.get("ok") is True, out
    assert out["size"] == len(PAYLOAD)
    assert len(_attachments(worker_env)) == 1


def test_U3_whitespace_variants_are_accepted(worker_env):
    """Spaces, tabs and CRLF are transport artefacts, not content."""
    b64 = base64.b64encode(PAYLOAD).decode()
    noisy = " " + b64[:10] + "\t" + b64[10:30] + "\r\n" + b64[30:] + "  \n"
    out = _call(_inline(PAYLOAD, content_base64=noisy))
    assert out.get("ok") is True, out
    assert out["size"] == len(PAYLOAD)


def test_U4_truncated_base64_is_refused(worker_env):
    """F-17: a prefix-truncated payload still decodes — the length must catch it."""
    b64 = base64.b64encode(PAYLOAD).decode()
    truncated = b64[: (len(b64) // 2) // 4 * 4]  # keep a valid 4-char boundary
    out = _call(_inline(PAYLOAD, content_base64=truncated))
    _assert_refused(out, worker_env, contains="truncated")
    assert str(len(PAYLOAD)) in out["error"], "the error must name the declared size"


def test_U5_expected_sha256_alone_missing_is_refused(worker_env):
    args = _inline(PAYLOAD)
    args.pop("expected_sha256")
    _assert_refused(_call(args), worker_env, contains="expected_sha256")


def test_U6_wrong_expected_sha256_is_refused(worker_env):
    args = _inline(PAYLOAD, expected_sha256="0" * 64)
    _assert_refused(_call(args), worker_env, contains="hash")


def test_U7_correct_expectations_store_and_echo(worker_env):
    out = _call(_inline(PAYLOAD))
    assert out.get("ok") is True, out
    assert out["size"] == len(PAYLOAD)
    assert out["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    stored = _attachments(worker_env)
    assert len(stored) == 1
    assert Path(stored[0].stored_path).read_bytes() == PAYLOAD


def test_U8_non_base64_characters_are_refused(worker_env):
    _assert_refused(
        _call(_inline(PAYLOAD, content_base64="!!!not base64!!!")),
        worker_env,
        contains="base64",
    )


def test_U9_both_path_and_inline_is_refused(worker_ws):
    tid, ws = worker_ws
    src = ws / "f.txt"
    src.write_bytes(PAYLOAD)
    args = _inline(PAYLOAD, path=str(src))
    _assert_refused(_call(args), tid, contains="exactly one")


def test_U10_neither_path_nor_inline_is_refused(worker_env):
    _assert_refused(_call({"filename": "d.txt"}), worker_env, contains="one of")


def test_U11_data_uri_prefix_is_refused_by_name(worker_env):
    """A data: URI must be named, not silently rejected as 'bad base64'."""
    data_uri = "data:text/plain;base64," + base64.b64encode(PAYLOAD).decode()
    out = _call(_inline(PAYLOAD, content_base64=data_uri))
    _assert_refused(out, worker_env, contains="data:")


def test_U12_oversize_inline_payload_is_refused(worker_env, monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "KANBAN_ATTACHMENT_MAX_BYTES", 32)
    _assert_refused(_call(_inline(PAYLOAD)), worker_env, contains="limit")


def test_U13_right_length_wrong_bytes_is_refused_by_hash(worker_env):
    """Proves the two guards are not redundant: length passes, hash must fail."""
    corrupted = bytearray(PAYLOAD)
    corrupted[0] ^= 0xFF
    args = _inline(PAYLOAD, content_base64=base64.b64encode(bytes(corrupted)).decode())
    out = _call(args)
    assert "error" in out and "hash" in out["error"].lower(), out
    assert _attachments(worker_env) == []


# ---------------------------------------------------------------------------
# P1-P14 — path branch
# ---------------------------------------------------------------------------


def test_P1_file_inside_workspace_is_stored(worker_ws):
    tid, ws = worker_ws
    src = ws / "report.md"
    src.write_bytes(PAYLOAD)
    out = _call({"path": str(src)})
    assert out.get("ok") is True, out
    assert out["size"] == len(PAYLOAD)
    assert out["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    stored = _attachments(tid)
    assert len(stored) == 1
    assert stored[0].filename == "report.md", "filename defaults to the basename"
    assert Path(stored[0].stored_path).read_bytes() == PAYLOAD


def test_P2_traversal_out_of_workspace_is_refused(worker_ws, tmp_path):
    tid, ws = worker_ws
    secret = tmp_path / ".hermes" / ".env"
    secret.write_text("TOKEN=redacted\n")
    _assert_refused(
        _call({"path": str(ws / ".." / ".hermes" / ".env")}), tid, contains="workspace"
    )


def test_P3_absolute_path_outside_workspace_is_refused(worker_ws, tmp_path):
    tid, _ws = worker_ws
    outside = tmp_path / "outside.txt"
    outside.write_bytes(PAYLOAD)
    _assert_refused(_call({"path": str(outside)}), tid, contains="workspace")


def test_P4_symlink_escaping_workspace_is_refused(worker_ws, tmp_path):
    tid, ws = worker_ws
    secret = tmp_path / "auth.json"
    secret.write_text('{"token": "redacted"}')
    link = ws / "looks-harmless.json"
    link.symlink_to(secret)
    _assert_refused(_call({"path": str(link)}), tid, contains="workspace")


def test_P5_symlink_inside_workspace_is_allowed(worker_ws):
    tid, ws = worker_ws
    real = ws / "real.txt"
    real.write_bytes(PAYLOAD)
    link = ws / "link.txt"
    link.symlink_to(real)
    out = _call({"path": str(link)})
    assert out.get("ok") is True, out
    assert out["size"] == len(PAYLOAD)


def test_P6_directory_is_refused(worker_ws):
    tid, ws = worker_ws
    d = ws / "subdir"
    d.mkdir()
    _assert_refused(_call({"path": str(d)}), tid, contains="regular file")


def test_P7_fifo_is_refused(worker_ws):
    tid, ws = worker_ws
    fifo = ws / "pipe"
    os.mkfifo(fifo)
    _assert_refused(_call({"path": str(fifo)}), tid, contains="regular file")


def test_P8_oversize_file_is_refused_without_reading(worker_ws, monkeypatch):
    """The cap is enforced from fstat, before a single byte is read."""
    from hermes_cli import kanban_db as kb

    tid, ws = worker_ws
    big = ws / "big.bin"
    big.write_bytes(b"x" * 4096)
    monkeypatch.setattr(kb, "KANBAN_ATTACHMENT_MAX_BYTES", 1024)

    reads = []
    real_read = os.read

    def counting_read(fd, n):
        reads.append(fd)
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", counting_read)
    _assert_refused(_call({"path": str(big)}), tid, contains="limit")
    assert reads == [], "an oversize file must not be read at all"


def test_P9_missing_file_is_a_clean_error(worker_ws):
    tid, ws = worker_ws
    _assert_refused(_call({"path": str(ws / "nope.txt")}), tid, contains="no such file")


def test_P10_traversal_in_explicit_filename_is_sanitised(worker_ws):
    tid, ws = worker_ws
    src = ws / "ok.txt"
    src.write_bytes(PAYLOAD)
    out = _call({"path": str(src), "filename": "../../escape.txt"})
    assert out.get("ok") is True, out
    stored = _attachments(tid)
    assert stored[0].filename == "escape.txt"
    assert "/" not in stored[0].filename


def test_P11_same_name_twice_gets_collision_suffix(worker_ws):
    tid, ws = worker_ws
    src = ws / "dup.txt"
    src.write_bytes(PAYLOAD)
    assert _call({"path": str(src)}).get("ok") is True
    assert _call({"path": str(src)}).get("ok") is True
    names = sorted(a.filename for a in _attachments(tid))
    assert names == ["dup (1).txt", "dup.txt"], names


def test_P12_file_grown_between_fstat_and_read_is_refused(worker_ws, monkeypatch):
    """TOCTOU: the bytes read must match the size the fd was measured at."""
    tid, ws = worker_ws
    src = ws / "racy.txt"
    src.write_bytes(PAYLOAD)

    real_fstat = os.fstat

    class _Shrunk:
        def __init__(self, st):
            self.st_mode = st.st_mode
            self.st_size = st.st_size - 5  # pretend the file was smaller

    monkeypatch.setattr(os, "fstat", lambda fd: _Shrunk(real_fstat(fd)))
    _assert_refused(_call({"path": str(src)}), tid, contains="changed")


def test_P13_path_outside_workspace_without_extra_roots_is_refused(
    worker_ws, tmp_path
):
    """No attach_extra_roots exists in NL-05B: the workspace is the only root."""
    tid, _ws = worker_ws
    projects = tmp_path / "Projects" / "thing"
    projects.mkdir(parents=True)
    src = projects / "note.md"
    src.write_bytes(PAYLOAD)
    _assert_refused(_call({"path": str(src)}), tid, contains="workspace")


def test_P14_null_workspace_path_fails_closed(worker_env, tmp_path):
    """A-2 (Thomas): no resolved workspace → path= is refused, never guessed."""
    src = tmp_path / "anywhere.txt"
    src.write_bytes(PAYLOAD)
    out = _call({"path": str(src)})
    _assert_refused(out, worker_env, contains="workspace")
    # The refusal must point at the usable alternative.
    assert "content_base64" in out["error"], out["error"]


# ---------------------------------------------------------------------------
# Regression guard — the property that makes "nothing was stored" true
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["no_expectation", "truncated", "wrong_hash", "data_uri", "bad_base64"],
)
def test_verification_failure_never_reaches_storage(worker_env, monkeypatch, case):
    """No refused inline payload may call store_attachment_bytes."""
    from hermes_cli import kanban_db as kb

    calls = []
    real = kb.store_attachment_bytes
    monkeypatch.setattr(
        kb,
        "store_attachment_bytes",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )

    args = _inline(PAYLOAD)
    if case == "no_expectation":
        args.pop("expected_bytes")
        args.pop("expected_sha256")
    elif case == "truncated":
        b64 = args["content_base64"]
        args["content_base64"] = b64[: (len(b64) // 2) // 4 * 4]
    elif case == "wrong_hash":
        args["expected_sha256"] = "1" * 64
    elif case == "data_uri":
        args["content_base64"] = "data:text/plain;base64," + args["content_base64"]
    elif case == "bad_base64":
        args["content_base64"] = "###"

    assert "error" in _call(args)
    assert calls == [], f"{case}: storage was reached despite a failed verification"


def test_path_verification_failure_never_reaches_storage(worker_ws, monkeypatch):
    from hermes_cli import kanban_db as kb

    tid, ws = worker_ws
    calls = []
    real = kb.store_attachment_bytes
    monkeypatch.setattr(
        kb,
        "store_attachment_bytes",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    assert "error" in _call({"path": str(ws / "missing.txt")})
    assert calls == []


# ---------------------------------------------------------------------------
# Schema — the description is what steers the model away from the bad path
# ---------------------------------------------------------------------------


def test_schema_exposes_path_and_expectations():
    from tools.kanban_tools import KANBAN_ATTACH_SCHEMA

    props = KANBAN_ATTACH_SCHEMA["parameters"]["properties"]
    for key in ("path", "content_base64", "expected_bytes", "expected_sha256"):
        assert key in props, f"{key} missing from kanban_attach schema"
    assert props["expected_bytes"]["type"] == "integer"
    # Nothing is schema-required: filename defaults to the basename on the path
    # branch, and the either-or contract cannot be expressed in JSON Schema —
    # it is enforced in the handler instead (see the two tests below).
    assert KANBAN_ATTACH_SCHEMA["parameters"]["required"] == []
    desc = KANBAN_ATTACH_SCHEMA["description"].lower()
    assert "path" in desc and "expected_bytes" in desc


def test_contract_is_enforced_in_code_not_by_schema_required(worker_ws):
    """`required: []` is safe only because the handler enforces the contract."""
    tid, ws = worker_ws
    src = ws / "only-path.txt"
    src.write_bytes(PAYLOAD)
    # path alone is a complete, valid call
    assert _call({"path": str(src)}).get("ok") is True
    # inline alone is complete only with filename + both expectations
    assert _call(_inline(PAYLOAD, filename="only-inline.txt")).get("ok") is True
    # ...and inline without a filename is refused by the handler, not the schema
    args = _inline(PAYLOAD)
    args.pop("filename")
    out = _call(args)
    assert "error" in out and "filename" in out["error"].lower(), out


# ---------------------------------------------------------------------------
# Hardening added after adversarial review
# ---------------------------------------------------------------------------


def test_malformed_expected_sha256_is_a_contract_error(worker_env):
    """A typo'd digest must say so, not masquerade as a content mismatch."""
    out = _call(_inline(PAYLOAD, expected_sha256="not-a-digest"))
    _assert_refused(out, worker_env, contains="64 hex")
    assert "mismatch" not in out["error"].lower()


def test_non_string_content_base64_is_refused(worker_env):
    _assert_refused(
        _call(_inline(PAYLOAD, content_base64=12345)), worker_env, contains="string"
    )


def test_parent_directory_swapped_after_check_is_refused(worker_ws, monkeypatch):
    """The resolve()/open() window: a parent that becomes a symlink is refused.

    O_NOFOLLOW alone only guards the leaf, so the descent walks from an fd on
    the workspace root with O_NOFOLLOW on every hop. Simulated by swapping the
    parent directory for a symlink right after the containment check.
    """
    import tools.kanban_tools as kt

    tid, ws = worker_ws
    sub = ws / "sub"
    sub.mkdir()
    src = sub / "f.txt"
    src.write_bytes(PAYLOAD)

    outside = ws.parent / "elsewhere"
    outside.mkdir()
    (outside / "f.txt").write_bytes(b"attacker-controlled")

    real_relative_to = Path.relative_to
    swapped = {"done": False}

    def racing_relative_to(self, other):
        # Fires once, between the containment check and the descent.
        if not swapped["done"]:
            swapped["done"] = True
            import shutil

            shutil.rmtree(sub)
            (ws / "sub").symlink_to(outside)
        return real_relative_to(self, other)

    monkeypatch.setattr(Path, "relative_to", racing_relative_to)
    out = _call({"path": str(src)})
    assert "error" in out, out
    assert _attachments(tid) == []


def test_path_refused_without_dir_fd_support(worker_ws, monkeypatch):
    """No dir_fd → the workspace boundary is unenforceable → path= fails closed."""
    tid, ws = worker_ws
    src = ws / "f.txt"
    src.write_bytes(PAYLOAD)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    out = _call({"path": str(src)})
    _assert_refused(out, tid, contains="unavailable on this platform")
    assert "content_base64" in out["error"], "must point at the safe alternative"


def test_descent_does_not_leak_file_descriptors(worker_ws):
    """Repeated refusals must not orphan directory fds."""
    import resource

    tid, ws = worker_ws
    deep = ws / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "f.txt").write_bytes(PAYLOAD)
    missing = deep / "gone.txt"

    def open_fd_count():
        n = 0
        soft = min(resource.getrlimit(resource.RLIMIT_NOFILE)[0], 4096)
        for fd in range(3, soft):
            try:
                os.fstat(fd)
                n += 1
            except OSError:
                pass
        return n

    _call({"path": str(missing)})  # warm any lazy imports/connections
    before = open_fd_count()
    for _ in range(25):
        assert "error" in _call({"path": str(missing)})
        assert _call({"path": str(deep / "f.txt")}).get("ok") is True
    after = open_fd_count()
    assert after <= before + 2, f"fd leak: {before} → {after}"


def test_fifo_open_does_not_block(worker_ws):
    """O_NONBLOCK on the open: a writer-less FIFO must not park the worker."""
    import threading

    tid, ws = worker_ws
    fifo = ws / "blocking-pipe"
    os.mkfifo(fifo)

    result = {}

    def run():
        result["out"] = _call({"path": str(fifo)})

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "opening a FIFO blocked the worker"
    assert "error" in result["out"]
