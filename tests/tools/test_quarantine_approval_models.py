from dataclasses import replace
import os

import pytest

from tools.quarantine_approval_models import (
    APPROVAL_BINDING_SCHEMA,
    ApprovalBinding,
    approval_scope_digest,
    canonical_json_bytes,
    evidence_bundle_digest,
)


def sample_binding(**overrides):
    values = dict(
        schema_version=APPROVAL_BINDING_SCHEMA,
        candidate_sha256="a" * 64,
        source_id="owner/repo/demo",
        source_version="v1",
        gate_release_sha256="b" * 64,
        policy_version="skill-admission-gate-v0.1",
        hermes_guard_version="skills-guard-v5",
        cisco_version="skill-scanner 2.1.0",
        nvidia_version="2.11.2",
        scanner_completeness=(
            ("cisco_ai_defense", True),
            ("hermes_skills_guard", True),
            ("nvidia_skillspector", False),
        ),
        scanner_reason_codes=(
            ("cisco_ai_defense", ()),
            ("hermes_skills_guard", ()),
            ("nvidia_skillspector", ("reference_unresolved",)),
        ),
        evidence_digest="c" * 64,
    )
    values.update(overrides)
    return ApprovalBinding(**values)


def test_approval_scope_digest_is_stable_and_field_sensitive():
    binding = sample_binding()
    attempt = "1" * 32
    baseline = approval_scope_digest(binding, attempt)
    assert baseline == approval_scope_digest(binding, attempt)
    assert approval_scope_digest(
        replace(binding, candidate_sha256="d" * 64), attempt
    ) != baseline
    assert approval_scope_digest(binding, "2" * 32) != baseline
    for field, value in (
        ("source_version", "v2"),
        ("policy_version", "policy-v2"),
        ("cisco_version", "skill-scanner 9.9.9"),
        ("nvidia_version", "9.9.9"),
        ("evidence_digest", "e" * 64),
        ("scanner_completeness", (("cisco_ai_defense", False),)),
        ("scanner_reason_codes", (("nvidia_skillspector", ("other",)),)),
    ):
        assert approval_scope_digest(
            replace(binding, **{field: value}), attempt
        ) != baseline


def test_canonical_json_bytes_is_sorted_compact_and_stable():
    payload = {"z": 1, "a": {"y": 2, "x": 3}}
    encoded = canonical_json_bytes(payload)
    assert encoded == b'{"a":{"x":3,"y":2},"z":1}'
    assert canonical_json_bytes(payload) == encoded


def test_evidence_bundle_digest_is_content_and_path_sensitive(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a.json").write_text('{"x":1}', encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "b.txt").write_text("hello", encoding="utf-8")
    first = evidence_bundle_digest(root)
    assert first == evidence_bundle_digest(root)
    (root / "nested" / "b.txt").write_text("changed", encoding="utf-8")
    assert evidence_bundle_digest(root) != first


def test_evidence_bundle_digest_rejects_redirected_entry(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = root / "linked.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this Windows host")
    with pytest.raises(ValueError, match="redirect"):
        evidence_bundle_digest(root)


def test_minimal_model_has_no_production_approval_artifact_runtime():
    import tools.quarantine_approval_models as models

    assert not hasattr(models, "ApprovalArtifact")
    assert not hasattr(models, "APPROVAL_ARTIFACT_SCHEMA")
    assert not hasattr(models, "unsigned_approval_bytes")
