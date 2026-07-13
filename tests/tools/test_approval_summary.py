from tools.approval_summary import summarize_command_approval


ODOO_EXPORT_COMMAND = r'''python3 -c '
import zipfile, json, sys, collections
p=sys.argv[1]
with zipfile.ZipFile(p) as z:
    names=z.namelist()
    print(json.dumps(names[:20]))
    print(z.read("manifest.json").decode("utf-8", "replace")[:12000])
' /Users/example/.hermes/cache/documents/odoo-export.zip'''


def test_inline_zip_inspection_explains_data_access_and_model_exposure():
    summary = summarize_command_approval(
        ODOO_EXPORT_COMMAND,
        "script execution via -e/-c flag",
    )

    assert summary.risk_level == "medium"
    assert "ZIP archive" in summary.purpose
    assert any("Reads" in item and "odoo-export.zip" in item for item in summary.effects)
    assert any("not modify or delete" in item for item in summary.effects)
    assert any("model context" in item for item in summary.effects)
    assert "later inline scripts" in summary.session_scope
    assert summary.once_only is True


def test_recursive_delete_is_high_risk_and_does_not_claim_reversibility():
    summary = summarize_command_approval(
        "rm -rf /Users/example/project/build",
        "recursive delete",
    )

    assert summary.risk_level == "high"
    assert "Delete" in summary.purpose
    assert any("permanently" in item for item in summary.effects)
    assert summary.once_only is True


def test_unknown_command_uses_cautious_language():
    summary = summarize_command_approval("custom-tool --flag", "dangerous command")

    assert summary.risk_level == "medium"
    assert "could not be determined" in summary.effects[0]
    assert "matching this safety rule" in summary.session_scope
