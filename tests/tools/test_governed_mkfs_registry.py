"""Registry discovery tests for the governed_mkfs tool.

The registry's AST-based discovery only imports modules with a top-level
``registry.register(...)`` call.  If that invariant is ever broken, the
governed path silently disappears from the model's toolset — which would
be a safety regression (the model would have no governed alternative and
could be tempted to bypass).  These tests pin the discovery contract.

Process-boundary contract (review #100694, 2026-09-02): the governed
handler is the ONLY legitimate issuer — it runs in the long-lived Hermes
process, asks the human for an explicit one-shot approval, and mints the
grant with the SAME process-local authority that claims and verifies it.
The model-facing schema therefore exposes the target tuple only (no
grant_id): the model can request, never self-grant.
"""

from pathlib import Path

from tools.registry import _module_registers_tools, registry


def test_governed_mkfs_module_is_ast_discovered():
    module_path = Path("tools/governed_mkfs_tool.py")
    assert module_path.exists()
    assert _module_registers_tools(module_path) is True


def test_governed_mkfs_registered_after_import():
    import tools.governed_mkfs_tool  # noqa: F401

    entry = registry.get_entry("governed_mkfs")
    assert entry is not None
    assert entry.name == "governed_mkfs"
    assert entry.toolset == "terminal"
    # The schema must expose only structured fields, never a free shell
    # string, and NO grant_id: the model requests the target tuple, the
    # process mints the grant after human approval.
    props = entry.schema["parameters"]["properties"]
    assert set(props) == {"vm_id", "device", "fs_type", "label"}
    assert entry.schema["parameters"]["required"] == [
        "vm_id", "device", "fs_type", "label",
    ]


def test_governed_mkfs_handler_is_the_in_process_issuer():
    """The handler is the legitimate issuer: it requests the human
    approval and mints the grant in the consumer process (same authority
    generation).  The model-facing schema has no grant_id, so a forged or
    externally-minted grant cannot even be referenced."""
    import inspect

    import tools.governed_mkfs_tool as gmt

    src = inspect.getsource(gmt._handle_governed_mkfs)
    # The handler drives the full approval -> issue -> claim -> workflow.
    assert "request_destructive_grant_approval" in src
    assert "issue_grant" in src
    assert "claim_grant" in src
    assert "settle_grant" in src
    # No grant_id in the schema: the model cannot point the tool at a
    # pre-minted grant.
    assert "grant_id" not in gmt.GOVERNED_MKFS_SCHEMA["parameters"]["properties"]
