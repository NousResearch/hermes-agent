"""Tests for the path-scoped Scout research publication tool."""

import importlib
import json
from pathlib import Path
from unittest.mock import patch


VALID_PACKET = """\
---
packet_type: scout_evidence_packet
question: What changed?
mode: decision_brief
created: 2026-05-23
as_of: 2026-05-23
collector: scout
publication_status: candidate
---

## Claim Ledger
| ID | Claim | Claim type | Source ID(s) | Source class | Confidence | Freshness | Conflict or limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | A primary source reports the change. | verified_fact | S1 | primary | high | current_as_of:2026-05-23 | None |

## Source Register
| Source ID | Title | Publisher/author | URL or stable ID | Retrieved | Source class | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Primary Record | Publisher | https://example.test/record | 2026-05-23 | primary | Direct record |
"""

VALID_RESOURCE = """\
---
created: 2026-05-23
updated: 2026-05-23
type: resource
tags: [research, test]
source_agent: Scout
evidence_packet_date: 2026-05-23
evidence_status: verified
as_of: 2026-05-23
related: ["[[09 System/MOC - Research & Resources]]"]
---

# Verified Research Note

## Findings

The change is recorded by the primary source
([Primary Record](https://example.test/record)).

## Evidence Quality

Primary evidence, high confidence, current as of 2026-05-23.

## Sources

- [Primary Record](https://example.test/record), retrieved 2026-05-23.

## Open Questions

- None.
"""


def _module():
    return importlib.import_module("tools.vault_publish_tool")


def _configure_vault(monkeypatch, tmp_path: Path):
    module = _module()
    vault_root = tmp_path / "vault"
    monkeypatch.setattr(module, "VAULT_ROOT", vault_root)
    return module, vault_root


def test_publishes_valid_resource_under_resources_allowlist(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    target = vault_root / "05 Resources" / "Verified Note.md"

    with patch.object(
        module,
        "write_file_tool",
        return_value=json.dumps({"status": "ok", "path": str(target)}),
    ) as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=VALID_RESOURCE,
                evidence_packet=VALID_PACKET,
                artifact_type="resource",
            )
        )

    assert result["status"] == "ok"
    write_tool.assert_called_once_with(str(target), VALID_RESOURCE, task_id="default")


def test_rejects_path_outside_vault_allowlist(monkeypatch, tmp_path):
    module, _ = _configure_vault(monkeypatch, tmp_path)
    target = tmp_path / "profiles" / "scout" / "config.yaml"

    with patch.object(module, "write_file_tool") as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=VALID_RESOURCE,
                evidence_packet=VALID_PACKET,
            )
        )

    assert "error" in result
    assert "approved vault path" in result["error"]
    write_tool.assert_not_called()


def test_rejects_parent_traversal_even_when_destination_resolves_inside(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    target = vault_root / "05 Resources" / ".." / "05 Resources" / "Note.md"

    with patch.object(module, "write_file_tool") as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=VALID_RESOURCE,
                evidence_packet=VALID_PACKET,
            )
        )

    assert "error" in result
    assert "traversal" in result["error"].lower()
    write_tool.assert_not_called()


def test_rejects_symlink_escape_from_resources(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    resources = vault_root / "05 Resources"
    outside = tmp_path / "elsewhere"
    resources.mkdir(parents=True)
    outside.mkdir()
    (resources / "escape").symlink_to(outside, target_is_directory=True)
    target = resources / "escape" / "note.md"

    with patch.object(module, "write_file_tool") as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=VALID_RESOURCE,
                evidence_packet=VALID_PACKET,
            )
        )

    assert "error" in result
    assert "symlink" in result["error"].lower()
    write_tool.assert_not_called()


def test_rejects_resource_without_evidence_packet_metadata(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    target = vault_root / "05 Resources" / "Missing Packet.md"

    with patch.object(module, "write_file_tool") as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=VALID_RESOURCE,
                evidence_packet="## Claim Ledger\n",
            )
        )

    assert "error" in result
    assert "evidence packet" in result["error"].lower()
    write_tool.assert_not_called()


def test_rejects_resource_without_inline_source_citation(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    target = vault_root / "05 Resources" / "Uncited.md"
    uncited = VALID_RESOURCE.replace(
        "([Primary Record](https://example.test/record)).",
        "in the record.",
    ).replace(
        "- [Primary Record](https://example.test/record), retrieved 2026-05-23.",
        "- Primary Record, retrieved 2026-05-23.",
    )

    with patch.object(module, "write_file_tool") as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=uncited,
                evidence_packet=VALID_PACKET,
            )
        )

    assert "error" in result
    assert "citation" in result["error"].lower()
    write_tool.assert_not_called()


def test_rejects_resource_when_only_sources_section_is_cited(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    target = vault_root / "05 Resources" / "Uncited Findings.md"
    uncited_findings = VALID_RESOURCE.replace(
        "The change is recorded by the primary source\n"
        "([Primary Record](https://example.test/record)).",
        "The change is recorded by the primary source.",
    )

    with patch.object(module, "write_file_tool") as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=uncited_findings,
                evidence_packet=VALID_PACKET,
            )
        )

    assert "error" in result
    assert "findings" in result["error"].lower()
    assert "citation" in result["error"].lower()
    write_tool.assert_not_called()


def test_appends_operations_log_without_overwriting_existing_entries(monkeypatch, tmp_path):
    module, vault_root = _configure_vault(monkeypatch, tmp_path)
    target = vault_root / "09 System" / "Operations Log.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Operations Log\n\n- Existing entry.\n", encoding="utf-8")
    entry = "- 2026-05-23: Published [[05 Resources/Verified Note]] from verified packet.\n"

    def _replacement_writer(path, content, task_id="default"):
        Path(path).write_text(content, encoding="utf-8")
        return json.dumps({"status": "ok", "path": path})

    with patch.object(module, "write_file_tool", side_effect=_replacement_writer) as write_tool:
        result = json.loads(
            module.publish_research_artifact(
                path=str(target),
                content=entry,
                evidence_packet=VALID_PACKET,
                artifact_type="operations_log",
            )
        )

    assert result["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "# Operations Log\n\n- Existing entry.\n" + entry
    write_tool.assert_not_called()


def test_tool_is_registered_under_vault_publish_toolset():
    module = _module()
    from tools.registry import registry

    entry = registry.get_entry("publish_research_artifact")
    assert entry is not None
    assert entry.toolset == "vault-publish"
    assert entry.handler is module._handle_publish_research_artifact
