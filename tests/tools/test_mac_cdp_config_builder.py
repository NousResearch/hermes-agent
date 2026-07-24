"""Unit tests for guarded Mac CDP sidecar config generation."""
from __future__ import annotations

import json

import pytest

from tools import mac_cdp_config_builder as builder


def test_inventory_config_is_readonly_and_sessionized(tmp_path):
    cfg = builder.build_config(
        {
            "mode": "inventory",
            "url": "https://example.com/form",
            "sessionId": "safe-01",
            "allowedDomains": ["example.com"],
            "outputPrefix": "dryrun",
        },
        shared_root=tmp_path,
    )
    assert cfg["url"] == "https://example.com/form?sessionId=safe-01"
    assert cfg["allowSubmit"] is False
    assert cfg["readOnly"] is True
    assert cfg["minFields"] == 1
    assert cfg["sideEffectPolicy"]["approvalRequiredBeforeRun"] is False
    assert "does not type" in " ".join(cfg["sideEffectPolicy"]["knownSideEffects"])
    assert cfg["outputPath"] == str(tmp_path / "dryrun-readonly-inventory-result.json")
    assert cfg["screenshotPath"] == str(tmp_path / "dryrun-readonly-inventory-screenshot.png")


def test_fill_config_rejects_unlisted_domain(tmp_path):
    with pytest.raises(ValueError, match="not in allowedDomains"):
        builder.build_config(
            {
                "mode": "fill",
                "url": "https://evil.example/form",
                "sessionId": "safe-01",
                "allowedDomains": ["example.com"],
                "fields": [{"selector": "#title", "value": "x"}],
            },
            shared_root=tmp_path,
        )


def test_fill_config_rejects_submit_secret_and_missing_session(tmp_path):
    rejected = [
        (
            {
                "mode": "fill",
                "url": "https://example.com",
                "sessionId": "safe-01",
                "allowedDomains": ["example.com"],
                "allowSubmit": True,
                "fields": [{"selector": "#x", "value": "ok"}],
            },
            "submit",
        ),
        (
            {
                "mode": "fill",
                "url": "https://example.com",
                "sessionId": "safe-01",
                "allowedDomains": ["example.com"],
                "fields": [{"selector": "#x", "value": "token=abc"}],
            },
            "secret",
        ),
        (
            {
                "mode": "fill",
                "url": "https://example.com",
                "allowedDomains": ["example.com"],
                "fields": [{"selector": "#x", "value": "ok"}],
            },
            "sessionId",
        ),
    ]
    for payload, expected in rejected:
        with pytest.raises(ValueError) as exc:
            builder.build_config(payload, shared_root=tmp_path)
        assert expected.lower() in str(exc.value).lower()


def test_fill_config_requires_fields_and_blocks_custom_validation_js(tmp_path):
    cfg = builder.build_config(
        {
            "mode": "fill",
            "url": "https://example.com/form?x=1",
            "sessionId": "abc",
            "allowedDomains": ["example.com"],
            "fields": [{"selector": "#title", "kind": "value", "value": "こんにちは"}],
        },
        shared_root=tmp_path,
    )
    assert cfg["url"] == "https://example.com/form?x=1&sessionId=abc"
    assert cfg["allowSubmit"] is False
    assert cfg["fields"][0]["value"] == "こんにちは"
    assert cfg["allowedDomains"] == ["example.com"]
    assert cfg["sessionId"] == "abc"
    assert "sideEffectApproval" not in cfg
    assert cfg["validationExpression"] == "({ok:true})"
    assert cfg["sideEffectPolicy"]["approvalRequiredBeforeRun"] is True
    assert "autosave" in " ".join(cfg["sideEffectPolicy"]["knownSideEffects"])
    assert "obtain approval" in cfg["sideEffectPolicy"]["approvalInstruction"]

    with pytest.raises(ValueError, match="custom validationExpression"):
        builder.build_config(
            {
                "mode": "fill",
                "url": "https://example.com/form?x=1",
                "sessionId": "abc",
                "allowedDomains": ["example.com"],
                "fields": [{"selector": "#title", "kind": "value", "value": "こんにちは"}],
                "validationExpression": "document.querySelector('form').submit()",
            },
            shared_root=tmp_path,
        )


def test_fill_config_includes_approval_marker_only_with_matching_token(tmp_path):
    cfg = builder.build_config(
        {
            "mode": "fill",
            "url": "https://example.com/form",
            "sessionId": "abc",
            "allowedDomains": ["example.com"],
            "fields": [{"selector": "#title", "value": "ok"}],
            "approvalToken": "APPROVED:abc",
        },
        shared_root=tmp_path,
    )
    assert cfg["sideEffectApproval"] == {
        "approved": True,
        "token": "APPROVED:abc",
        "scope": "url+fields+sessionId",
    }

    with pytest.raises(ValueError, match="approvalToken"):
        builder.build_config(
            {
                "mode": "fill",
                "url": "https://example.com/form",
                "sessionId": "abc",
                "allowedDomains": ["example.com"],
                "fields": [{"selector": "#title", "value": "ok"}],
                "approvalToken": "APPROVED:wrong",
            },
            shared_root=tmp_path,
        )


def test_write_config_uses_mode_specific_default_name(tmp_path):
    out = builder.write_config(
        {"mode": "inventory", "url": "https://example.com", "allowedDomains": ["example.com"]},
        shared_root=tmp_path,
    )
    assert out == tmp_path / "cdp-readonly-inventory-config.json"
    data = json.loads(out.read_text())
    assert data["allowSubmit"] is False
    assert data["readOnly"] is True
