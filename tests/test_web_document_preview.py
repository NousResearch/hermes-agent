from pathlib import Path

import pytest

from hermes_cli import web_document_preview as preview


def test_pdf_tokens_stream_bounded_ranges_and_close(tmp_path: Path):
    path = tmp_path / "paper.pdf"
    data = b"%PDF-1.7\n" + bytes(range(256)) * 300
    path.write_bytes(data)

    descriptor = preview.open_pdf(path)

    assert descriptor["byteLength"] == len(data)
    assert descriptor["revision"]
    assert preview.read_pdf_range(descriptor["id"], 20, 100, descriptor["revision"]) == data[20:100]
    assert preview.close_pdf(descriptor["id"]) is True

    with pytest.raises(FileNotFoundError):
        preview.read_pdf_range(descriptor["id"], 20, 100)


def test_pdf_range_rejects_oversized_and_changed_documents(tmp_path: Path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\n" + b"a" * 1024)
    descriptor = preview.open_pdf(path)

    with pytest.raises(ValueError, match="Invalid PDF byte range"):
        preview.read_pdf_range(descriptor["id"], 0, preview.PDF_MAX_RANGE_BYTES + 1)

    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="PDF_CHANGED"):
        preview.read_pdf_range(descriptor["id"], 0, 10, descriptor["revision"])

    preview.close_pdf(descriptor["id"])


def test_tex_directives_allow_known_program_and_confined_root(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    root = tmp_path / "main.tex"
    child = tmp_path / "chapters" / "one.tex"
    child.parent.mkdir()
    root.write_text("\\documentclass{article}", encoding="utf-8")

    resolved, program = preview.tex_directives(
        "% !TeX root = ../main.tex\n% !TeX program = xelatex\n",
        child,
    )

    assert resolved == root
    assert program == "xelatex"


def test_tex_directives_reject_root_outside_source_tree(tmp_path: Path):
    source = tmp_path / "project" / "main.tex"
    source.parent.mkdir()

    with pytest.raises(ValueError, match="inside the source project"):
        preview.tex_directives("% !TeX root = ../outside.tex", source)


def test_tex_directives_allow_root_in_explicit_non_git_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    source = workspace / "chapters" / "chapter.tex"
    root = workspace / "main.tex"
    source.parent.mkdir(parents=True)
    root.write_text("\\documentclass{article}", encoding="utf-8")

    resolved, _ = preview.tex_directives("% !TeX root = ../main.tex", source, workspace)

    assert resolved == root


def test_tex_directives_ignore_workspace_that_does_not_contain_source(tmp_path: Path):
    project = tmp_path / "project"
    workspace = tmp_path / "unrelated"
    source = project / "main.tex"
    project.mkdir()
    workspace.mkdir()
    source.write_text("\\documentclass{article}", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the source project"):
        preview.tex_directives("% !TeX root = ../outside.tex", source, workspace)


def test_compile_tex_uses_credential_scrubbed_environment(tmp_path: Path, monkeypatch):
    source = tmp_path / "main.tex"
    source.write_text("\\documentclass{article}", encoding="utf-8")
    captured: dict = {}

    class FakeProcess:
        returncode = 1

        def __init__(self, *args, **kwargs):
            captured["env"] = kwargs["env"]

        def communicate(self, timeout=None):
            return b"compile failed", None

        def kill(self):
            return None

    def which(name: str):
        return "/usr/bin/xelatex" if name == "xelatex" else None

    monkeypatch.setattr(preview.shutil, "which", which)
    monkeypatch.setattr(preview.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        preview,
        "hermes_subprocess_env",
        lambda *, inherit_credentials: {"PATH": "/usr/bin"} if not inherit_credentials else {},
    )

    result = preview.compile_tex(source, "credential-test")

    assert result["status"] == "error"
    assert captured["env"] == {
        "PATH": "/usr/bin",
        "max_print_line": "1000",
        "openin_any": "p",
        "openout_any": "p",
    }
