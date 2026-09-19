"""Contract tests for MCP vs memory-provider developer docs.

Issue #108849: the developer guide must document that MCP memory servers
and memory providers are different surfaces — tools-only vs lifecycle —
without implying Hermes rejects pointing both at the same backend.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_MD = (
    REPO_ROOT
    / "website"
    / "docs"
    / "developer-guide"
    / "memory-provider-plugin.md"
)


HEADING = "Choosing between an MCP server and a memory provider"


def _doc_text() -> str:
    return DOC_MD.read_text(encoding="utf-8")


def _choosing_section() -> str:
    text = _doc_text()
    assert HEADING in text
    return text.split(HEADING, 1)[1]


def test_choosing_section_heading_exists():
    assert HEADING in _doc_text()


def test_comparison_table_covers_required_rows():
    section = _choosing_section()
    for phrase in (
        "recall trigger",
        "turn persistence",
        "pre-compression",
        "built-in memory mirroring",
        "tool loading",
        "consumers",
    ):
        assert phrase in section.lower()


def test_provider_lifecycle_symbols_documented():
    text = _doc_text()
    for symbol in (
        "prefetch()",
        "sync_turn()",
        "on_pre_compress()",
        "on_memory_write()",
    ):
        assert symbol in text


def test_config_keys_contrast_provider_and_mcp():
    section = _choosing_section()
    assert "memory.provider" in section
    assert "mcp_servers" in section


def test_mcp_path_is_tools_only_model_decides():
    section = _choosing_section().lower()
    assert "tools only" in section
    assert "calls a tool" in section or "call a tool" in section


def test_dual_backend_not_forbidden_but_can_duplicate():
    section = _choosing_section().lower()
    assert "not forbidden" in section
    assert "duplicate" in section
    assert "prompt-discipline" in section or "search memory first" in section
    assert "not a substitute" in section


def test_provider_uses_own_api_host_owns_mcp():
    section = _choosing_section()
    assert "own API" in section or "own HTTP" in section
    assert "host owns MCP connections" in section.lower() or "host owns" in section.lower()
