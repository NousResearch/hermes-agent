"""Tests for the binary-document write guard (port of nearai/ironclaw#7109).

A plain-text write can never produce a valid OOXML/OLE/ODF container, so
write_file/patch must refuse to write text into .docx/.xlsx/.pptx (and
friends), and must refuse to OVERWRITE an existing .pdf — while still
allowing new-.pdf creation (raw PDF syntax is text-authorable).
"""

import json
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from tools.binary_extensions import (
    has_opaque_document_extension,
    is_pdf_path,
)
from tools.file_tools import (
    _check_binary_document_write,
    patch_tool,
    write_file_tool,
)
from tools.file_operations import (
    FilePrefixResult,
    PatchResult,
    ShellFileOperations,
    WriteResult,
)
from tools.environments.local import LocalEnvironment


class _RemoteEnvironment:
    cwd = "/workspace"


class _FakeRemoteFileOps:
    def __init__(self, prefix: FilePrefixResult):
        self.env = _RemoteEnvironment()
        self.prefix = prefix
        self.prefix_reads = []
        self.write_calls = 0
        self.patch_calls = 0

    def read_file_prefix(self, path: str, length: int) -> FilePrefixResult:
        self.prefix_reads.append((path, length))
        return self.prefix

    def write_file(self, path: str, content: str) -> WriteResult:
        self.write_calls += 1
        return WriteResult(bytes_written=len(content), verified=True)

    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        self.patch_calls += 1
        return PatchResult(success=True, files_modified=[path])


def _install_remote_file_ops(monkeypatch, prefix: FilePrefixResult) -> _FakeRemoteFileOps:
    import tools.file_tools as file_tools

    file_ops = _FakeRemoteFileOps(prefix)
    monkeypatch.setattr(
        file_tools,
        "_resolve_path_for_task",
        lambda path, task_id="default": PurePosixPath("/workspace") / path,
    )
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": file_ops)
    return file_ops


def _make_minimal_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="xml" '
            'ContentType="application/xml"/></Types>',
        )
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.'
            'openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r>'
            "<w:t>Quarterly numbers look good.</w:t></w:r></w:p></w:body>"
            "</w:document>",
        )


class TestFilePrefixRead:
    def test_reads_only_requested_binary_prefix(self, tmp_path: Path):
        target = tmp_path / "large.pot"
        target.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"x" * 100_000)
        file_ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

        result = file_ops.read_file_prefix(str(target), 8)

        assert result.error is None
        assert result.missing is False
        assert result.content == bytes.fromhex("D0CF11E0A1B11AE1")

    def test_distinguishes_missing_file(self, tmp_path: Path):
        file_ops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

        result = file_ops.read_file_prefix(str(tmp_path / "missing.pot"), 8)

        assert result.error is None
        assert result.missing is True
        assert result.content == b""


class TestExtensionHelpers:
    def test_opaque_document_extensions(self):
        for p in ("a.docx", "b.XLSX", "c.pptx", "d.doc", "e.odt", "f.ods", "g.odp",
                  "h.docm", "i.xlsm", "j.xlsb", "k.pptm", "l.ppsx", "m.ppsm",
                  "n.pps", "p.rtf", "q.epub"):
            assert has_opaque_document_extension(p) is True, f"{p} should be opaque"

    def test_non_opaque_paths(self):
        for p in ("a.txt", "b.py", "c.pdf", "d.md", "noext", "e.csv", "messages.pot"):
            assert has_opaque_document_extension(p) is False

    def test_is_pdf_path(self):
        assert is_pdf_path("report.pdf") is True
        assert is_pdf_path("report.PDF") is True
        assert is_pdf_path("report.txt") is False


class TestCheckBinaryDocumentWrite:
    def test_docx_always_rejected(self, tmp_path: Path):
        # Even a NON-existing docx is rejected — text can't be a valid container.
        err = _check_binary_document_write(str(tmp_path / "new.docx"))
        assert err is not None
        assert ".docx" in err

    def test_existing_pdf_rejected(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
        err = _check_binary_document_write(str(pdf))
        assert err is not None
        assert "overwrite" in err.lower()

    def test_new_pdf_allowed(self, tmp_path: Path):
        assert _check_binary_document_write(str(tmp_path / "fresh.pdf")) is None

    def test_plain_text_allowed(self, tmp_path: Path):
        assert _check_binary_document_write(str(tmp_path / "notes.txt")) is None

    def test_new_gettext_pot_allowed(self, tmp_path: Path):
        assert _check_binary_document_write(str(tmp_path / "messages.pot")) is None

    def test_existing_gettext_pot_allowed(self, tmp_path: Path):
        pot = tmp_path / "messages.pot"
        pot.write_text('msgid "hello"\nmsgstr ""\n', encoding="utf-8")
        assert _check_binary_document_write(str(pot)) is None

    def test_existing_ole_powerpoint_pot_rejected(self, tmp_path: Path):
        pot = tmp_path / "slides.pot"
        pot.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"legacy powerpoint")
        err = _check_binary_document_write(str(pot))
        assert err is not None
        assert "PowerPoint" in err

    def test_existing_non_regular_pot_rejected(self, tmp_path: Path):
        pot = tmp_path / "messages.pot"
        pot.mkdir()

        err = _check_binary_document_write(str(pot))

        assert err is not None
        assert "not a regular file" in err

    def test_remote_inspection_failure_rejected(self, monkeypatch):
        file_ops = _install_remote_file_ops(
            monkeypatch, FilePrefixResult(error="permission denied")
        )

        err = _check_binary_document_write("slides.pot", task_id="remote")

        assert err is not None
        assert "could not be inspected safely" in err
        assert file_ops.prefix_reads == [("/workspace/slides.pot", 8)]

    def test_remote_inspection_exception_rejected(self, monkeypatch):
        file_ops = _install_remote_file_ops(monkeypatch, FilePrefixResult())

        def raise_transport_error(path: str, length: int) -> FilePrefixResult:
            raise RuntimeError("transport unavailable")

        monkeypatch.setattr(file_ops, "read_file_prefix", raise_transport_error)

        err = _check_binary_document_write("slides.pot", task_id="remote")

        assert err is not None
        assert "could not be inspected safely" in err
        assert "transport unavailable" in err


@pytest.mark.parametrize("tool_name", ["write", "patch"])
@pytest.mark.parametrize(
    ("prefix", "should_reject"),
    [
        (FilePrefixResult(content=bytes.fromhex("D0CF11E0A1B11AE1")), True),
        (FilePrefixResult(content=b'msgid "'), False),
        (FilePrefixResult(missing=True), False),
    ],
    ids=["ole-powerpoint", "gettext", "missing"],
)
def test_remote_pot_guard_uses_backend_for_both_write_paths(
    monkeypatch, tool_name, prefix, should_reject
):
    file_ops = _install_remote_file_ops(monkeypatch, prefix)

    if tool_name == "write":
        result = json.loads(
            write_file_tool("slides.pot", 'msgid "hello"\n', task_id="remote")
        )
    else:
        result = json.loads(
            patch_tool(
                mode="replace",
                path="slides.pot",
                old_string="hello",
                new_string="goodbye",
                task_id="remote",
            )
        )

    assert bool(result.get("error")) is should_reject
    expected_reads = 1 if should_reject else 2
    assert file_ops.prefix_reads == [
        ("/workspace/slides.pot", 8)
    ] * expected_reads
    assert file_ops.write_calls == (1 if tool_name == "write" and not should_reject else 0)
    assert file_ops.patch_calls == (1 if tool_name == "patch" and not should_reject else 0)


@pytest.mark.parametrize("tool_name", ["write", "patch"])
def test_pot_guard_reinspects_after_acquiring_mutation_lock(
    monkeypatch, tmp_path: Path, tool_name: str
):
    import tools.file_tools as file_tools

    target = tmp_path / "messages.pot"
    target.write_text('msgid "hello"\nmsgstr ""\n', encoding="utf-8")
    ole_content = bytes.fromhex("D0CF11E0A1B11AE1") + b"legacy powerpoint"
    real_lock_path = file_tools.file_state.lock_path

    class SwapToOleOnEnter:
        def __init__(self, resolved: str):
            self._lock = real_lock_path(resolved)

        def __enter__(self):
            self._lock.__enter__()
            target.write_bytes(ole_content)
            return self

        def __exit__(self, exc_type, exc, traceback):
            return self._lock.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(
        file_tools.file_state,
        "lock_path",
        lambda resolved: SwapToOleOnEnter(resolved),
    )

    if tool_name == "write":
        result = json.loads(write_file_tool(str(target), 'msgid "replacement"\n'))
    else:
        result = json.loads(
            patch_tool(
                mode="replace",
                path=str(target),
                old_string="hello",
                new_string="replacement",
            )
        )

    assert result.get("error")
    assert "PowerPoint" in result["error"]
    assert target.read_bytes() == ole_content


class TestWriteFileToolGuard:
    def test_write_file_rejects_existing_docx(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        _make_minimal_docx(docx)
        original = docx.read_bytes()

        result = json.loads(write_file_tool(str(docx), "edited text"))

        assert result.get("error"), "text write into .docx must be refused"
        assert docx.read_bytes() == original, "document bytes must be untouched"
        assert zipfile.is_zipfile(docx), "document must remain a valid container"

    def test_write_file_rejects_docm(self, tmp_path: Path):
        """Regression: .docm is extractable by read_file (anydoc) but was
        missing from OPAQUE_DOCUMENT_EXTENSIONS in the original PR #82818.
        Flagged by @egilewski — proven live: text write corrupted the zip."""
        docm = tmp_path / "macro.docm"
        _make_minimal_docx(docm)  # same OOXML zip structure
        original = docm.read_bytes()

        result = json.loads(write_file_tool(str(docm), "edited text"))

        assert result.get("error"), "text write into .docm must be refused"
        assert docm.read_bytes() == original, "document bytes must be untouched"
        assert zipfile.is_zipfile(docm), "document must remain a valid container"

    def test_write_file_rejects_new_docx(self, tmp_path: Path):
        result = json.loads(write_file_tool(str(tmp_path / "new.docx"), "hello"))
        assert result.get("error")
        assert not (tmp_path / "new.docx").exists()

    def test_write_file_rejects_existing_pdf_overwrite(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n1 0 obj\nendobj\n%%EOF\n")
        original = pdf.read_bytes()

        result = json.loads(write_file_tool(str(pdf), "replacement text"))

        assert result.get("error")
        assert pdf.read_bytes() == original

    def test_write_file_allows_new_pdf_creation(self, tmp_path: Path):
        pdf = tmp_path / "generated.pdf"
        result = json.loads(write_file_tool(str(pdf), "%PDF-1.4\n%%EOF\n"))
        assert not result.get("error")
        assert pdf.exists()

    def test_write_file_plain_text_unaffected(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        result = json.loads(write_file_tool(str(target), "hello world"))
        assert not result.get("error")
        assert target.read_text() == "hello world"

    def test_write_file_allows_gettext_pot(self, tmp_path: Path):
        target = tmp_path / "messages.pot"
        content = 'msgid "hello"\nmsgstr ""\n'
        result = json.loads(write_file_tool(str(target), content))
        assert not result.get("error")
        assert target.read_text() == content

    def test_write_file_rejects_ole_powerpoint_pot(self, tmp_path: Path):
        target = tmp_path / "slides.pot"
        original = bytes.fromhex("D0CF11E0A1B11AE1") + b"legacy powerpoint"
        target.write_bytes(original)
        result = json.loads(write_file_tool(str(target), "replacement text"))
        assert result.get("error")
        assert target.read_bytes() == original


class TestPatchToolGuard:
    def test_patch_replace_rejects_docx(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        _make_minimal_docx(docx)
        original = docx.read_bytes()

        result = json.loads(
            patch_tool(mode="replace", path=str(docx),
                       old_string="good", new_string="great")
        )

        assert result.get("error")
        assert docx.read_bytes() == original

    def test_patch_v4a_update_rejects_docx(self, tmp_path: Path):
        docx = tmp_path / "report.docx"
        _make_minimal_docx(docx)
        original = docx.read_bytes()

        v4a = (
            "*** Begin Patch\n"
            f"*** Update File: {docx}\n"
            "@@\n"
            "-good\n"
            "+great\n"
            "*** End Patch"
        )
        result = json.loads(patch_tool(mode="patch", patch=v4a))

        assert result.get("error")
        assert docx.read_bytes() == original

    def test_patch_v4a_delete_of_docx_not_blocked_by_guard(self, tmp_path: Path):
        # Delete doesn't write text content — the binary-document guard must
        # not fire for it (delete may still fail/succeed for other reasons).
        docx = tmp_path / "old.docx"
        _make_minimal_docx(docx)

        v4a = (
            "*** Begin Patch\n"
            f"*** Delete File: {docx}\n"
            "*** End Patch"
        )
        result = json.loads(patch_tool(mode="patch", patch=v4a))
        err = result.get("error") or ""
        assert "binary document" not in err.lower()

    def test_patch_replace_plain_text_unaffected(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        target.write_text("hello world")
        result = json.loads(
            patch_tool(mode="replace", path=str(target),
                       old_string="world", new_string="there")
        )
        assert not result.get("error")
        assert target.read_text() == "hello there"

    def test_patch_replace_allows_gettext_pot(self, tmp_path: Path):
        target = tmp_path / "messages.pot"
        target.write_text('msgid "hello"\nmsgstr ""\n', encoding="utf-8")
        result = json.loads(
            patch_tool(mode="replace", path=str(target),
                       old_string='msgid "hello"', new_string='msgid "goodbye"')
        )
        assert not result.get("error")
        assert target.read_text(encoding="utf-8").startswith('msgid "goodbye"')
