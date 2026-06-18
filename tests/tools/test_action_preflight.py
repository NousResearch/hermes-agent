"""Test-only skeleton for reversibility-tier semantic receipts.

These tests intentionally exercise an isolated module only. They must not prove
or require any production dispatch hook integration yet.
"""

from pathlib import Path

from tools.action_preflight import (
    ReversibilityTier,
    SemanticReceipt,
    classify_tool_action,
    validate_receipt,
)


def test_read_only_passthrough_without_receipt():
    for tool_name, args in [
        ("read_file", {"path": "README.md"}),
        ("search_files", {"pattern": "foo"}),
        ("send_message", {"action": "list"}),
    ]:
        preflight = classify_tool_action(tool_name, args)
        result = validate_receipt(preflight, receipt=None, trusted_decision=False)

        assert preflight.tier is ReversibilityTier.READ_ONLY
        assert result.allowed is True


def test_send_message_send_requires_trusted_receipt():
    preflight = classify_tool_action(
        "send_message",
        {"action": "send", "target": "discord:#general", "message": "hello"},
    )

    assert preflight.tier is ReversibilityTier.EXTERNALLY_VISIBLE_PUBLISH
    assert validate_receipt(preflight, None, trusted_decision=False).allowed is False
    assert validate_receipt(
        preflight,
        SemanticReceipt.for_preflight(preflight, approved=True),
        trusted_decision=True,
    ).allowed is True


def test_destructive_delete_requires_trusted_approval():
    preflight = classify_tool_action("delete_file", {"path": "old.txt"})

    assert preflight.tier is ReversibilityTier.DESTRUCTIVE_DELETE
    assert validate_receipt(
        preflight,
        SemanticReceipt.for_preflight(preflight, approved=True),
        trusted_decision=False,
    ).allowed is False
    assert validate_receipt(
        preflight,
        SemanticReceipt.for_preflight(preflight, approved=True),
        trusted_decision=True,
    ).allowed is True


def test_secret_credential_paths_and_args_escalate_and_receipt_redacts_values(tmp_path):
    secret_value = "sk-live-super-secret-token"
    preflight = classify_tool_action(
        "write_file",
        {"path": str(tmp_path / ".env"), "content": f"API_TOKEN={secret_value}\n"},
    )
    receipt = SemanticReceipt.for_preflight(
        preflight,
        approved=True,
        summary=f"approved token {secret_value}",
    )

    assert preflight.tier is ReversibilityTier.SECRET_CREDENTIAL_HANDLING
    assert secret_value not in repr(receipt)
    assert secret_value not in str(receipt.to_dict())

    token_preflight = classify_tool_action(
        "configure_provider",
        {"api_token": secret_value, "path": "config/auth.yaml"},
    )
    assert token_preflight.tier is ReversibilityTier.SECRET_CREDENTIAL_HANDLING


def test_staged_workspace_write_requires_pre_state_and_expected_delta(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preflight = classify_tool_action(
        "write_file",
        {"path": str(workspace / "notes.txt"), "content": "hello"},
        workspace_root=workspace,
    )

    assert preflight.tier is ReversibilityTier.STAGED_LOCAL_CHANGE
    assert validate_receipt(
        preflight,
        SemanticReceipt.for_preflight(preflight, approved=True),
        trusted_decision=True,
    ).allowed is False
    assert validate_receipt(
        preflight,
        SemanticReceipt.for_preflight(
            preflight,
            approved=True,
            pre_state_ref="sha256:before",
            expected_delta="create notes.txt",
        ),
        trusted_decision=True,
    ).allowed is True


def test_unknown_risky_write_publish_fails_closed():
    for tool_name in ["publish_blog", "write_remote_config", "post_remote_update"]:
        preflight = classify_tool_action(tool_name, {"value": "x"})
        result = validate_receipt(preflight, None, trusted_decision=False)

        assert preflight.tier is ReversibilityTier.UNKNOWN_RISKY
        assert result.allowed is False


def test_self_asserted_approval_status_in_model_args_is_ignored():
    preflight = classify_tool_action(
        "send_message",
        {
            "action": "send",
            "target": "discord:#general",
            "message": "hello",
            "approval_status": "approved",
        },
    )
    receipt = SemanticReceipt.for_preflight(preflight, approved=True)

    assert validate_receipt(preflight, receipt, trusted_decision=False).allowed is False


def test_payload_hash_mismatch_and_toctou_mutation_fail():
    preflight = classify_tool_action("send_message", {"action": "send", "message": "old"})
    changed_after_approval = classify_tool_action(
        "send_message",
        {"action": "send", "message": "mutated by plugin pre_tool_call"},
    )
    receipt = SemanticReceipt.for_preflight(preflight, approved=True)

    assert receipt.payload_hash != changed_after_approval.payload_hash
    assert validate_receipt(changed_after_approval, receipt, trusted_decision=True).allowed is False


def test_patch_path_traversal_and_symlink_outside_workspace_escalate(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()

    traversal = classify_tool_action(
        "patch",
        {"path": "../outside/target.txt", "old_string": "a", "new_string": "b"},
        workspace_root=workspace,
    )
    assert traversal.tier is ReversibilityTier.UNKNOWN_RISKY

    target = outside / "secret.txt"
    target.write_text("secret")
    link = workspace / "link.txt"
    link.symlink_to(target)
    symlink = classify_tool_action(
        "patch",
        {"path": str(link), "old_string": "secret", "new_string": "x"},
        workspace_root=workspace,
    )
    assert symlink.tier is ReversibilityTier.UNKNOWN_RISKY


def test_write_file_to_auth_config_escalates_to_secret_handling(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    preflight = classify_tool_action(
        "write_file",
        {"path": str(workspace / "config" / "credentials.yaml"), "content": "token: x"},
        workspace_root=workspace,
    )

    assert preflight.tier is ReversibilityTier.SECRET_CREDENTIAL_HANDLING
