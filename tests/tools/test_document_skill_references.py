"""Generated document guidance must name an available bundled skill (#105689).

No model or messaging service is called. Only the PDF page-text reader is
controlled, using the same mixed text/scan fixture shape as test_read_extract.
The prompt builders and skill loader use their production implementations.
"""

import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def bundled_pdf_home(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "skills"
    # Scan the entire unmodified bundled catalog through the supported config
    # surface, so absence is not manufactured by a PDF-only fixture.
    (tmp_path / "skills").mkdir()
    (tmp_path / "config.yaml").write_text(
        json.dumps({"skills": {"external_dirs": [str(source)]}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert load_skill("pdf").get("success") is True
    assert load_skill("pdf", "references/ocr-extraction.md").get("success") is True
    return tmp_path


def load_skill(name, file_path=None):
    from tools import skills_tool  # noqa: F401 -- registers the real tool
    from tools.registry import registry

    args = {"name": name}
    if file_path is not None:
        args["file_path"] = file_path
    return json.loads(registry.dispatch("skill_view", args))


def assert_referenced_skill_resolves(note):
    match = re.search(r"(?:or the |use the )([a-z][a-z0-9-]*) skill", note)
    assert match is not None, f"No named skill in generated note: {note}"
    name = match.group(1)
    loaded = load_skill(name)
    assert loaded.get("success") is True, (
        f"Runtime note points to an unavailable skill: {name!r}; {loaded.get('error')}"
    )


def test_gateway_attachment_note_names_a_loadable_skill(bundled_pdf_home):
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner

    event = MessageEvent(
        text="Summarize this attachment.",
        message_type=MessageType.DOCUMENT,
        source=None,
        media_urls=[str(bundled_pdf_home / "contract.pdf")],
        media_types=["application/pdf"],
    )
    note = GatewayRunner._prepend_inbound_document_notes(event, event.text)
    assert_referenced_skill_resolves(note)


def test_pdf_coverage_note_names_a_loadable_skill(bundled_pdf_home, monkeypatch):
    from tools import read_extract

    pages = ["A readable introductory page with sufficient text."] + [""] * 3
    monkeypatch.setattr(read_extract, "_pdf_page_texts", lambda _path: pages)
    note = read_extract._pdf_coverage_note(str(bundled_pdf_home / "mixed-scan.pdf"))
    assert "3 of 4 pages" in note
    assert_referenced_skill_resolves(note)
