"""Contract tests for the MCP and memory-provider documentation boundary.

MCP memory servers expose a tool surface. Memory providers participate in an
implementation-dependent lifecycle. The developer guide must make that
distinction without presenting dual use of one backend as blanket support.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_MD = REPO_ROOT / "website" / "docs" / "developer-guide" / "memory-provider-plugin.md"

HEADING = "Choosing between an MCP server and a memory provider"


def _choosing_section() -> str:
    text = DOC_MD.read_text(encoding="utf-8")
    assert HEADING in text
    return text.split(HEADING, 1)[1]


def test_choosing_section_distinguishes_the_two_configured_surfaces():
    section = _choosing_section()

    assert "memory.provider" in section
    assert "mcp_servers" in section
    assert "tools only" in section.lower()
    assert "one external memory provider" in section.lower()


def test_choosing_section_describes_provider_lifecycle_as_implementation_dependent():
    section = _choosing_section()

    for hook in ("prefetch()", "sync_turn()", "on_pre_compress()", "on_memory_write()"):
        assert hook in section
    assert "depends on the provider implementation" in section.lower()


def test_choosing_section_warns_about_duplicate_backend_work_without_policy_claim():
    section = _choosing_section().lower()

    assert "duplicate" in section
    assert "recall" in section
    assert "writes" in section
    assert "cost" in section
    assert "not forbidden" not in section


def test_choosing_section_preserves_the_host_owned_mcp_connection_boundary():
    section = _choosing_section().lower()

    assert "own api" in section
    assert "host owns mcp connections" in section
