"""Wire-contract tests for the SEP-2640 skills extension.

Every case here is a rule the spec states and a host must not soften: which
capability declarations admit the skills channel, what a manifest must contain
before it may be trusted, and that a body is measured on the bytes the server
sent rather than on any host-side transformation of them.
"""

import pytest

from agent.skills_contracts import SkillError, SkillErrorCode
from tools import mcp_skills_protocol as proto

MAIN = "skill://acme/billing/refunds/SKILL.md"
SUPPORT = "skill://acme/billing/refunds/references/api.md"


def _entry(**overrides):
    entry = {
        "uri": MAIN,
        "frontmatter": {"name": "refunds", "description": "Process billing refunds", "version": "2"},
        "resources": [
            {"uri": MAIN, "digest": "sha256:" + "a" * 64, "size": 151},
            {"uri": SUPPORT, "digest": "sha256:" + "b" * 64, "size": 42},
        ],
    }
    entry.update(overrides)
    return entry


# ── capability gating ───────────────────────────────────────────────────────

def test_extension_absent_means_the_channel_is_unsupported_not_broken():
    verdict = proto.parse_capabilities({"tools": {}, "resources": {}}, "2025-11-25")
    assert verdict.support == "unsupported"
    assert not verdict.supported


def test_extension_without_resources_is_protocol_invalid():
    """The spec requires the resources capability alongside the extension: without it
    the host could list skills but never read one, which is worse than not listing."""
    verdict = proto.parse_capabilities(
        {"extensions": {proto.SKILLS_EXTENSION_ID: {"directoryRead": True}}}, "2026-07-28")
    assert verdict.support == "protocol_invalid"
    assert "resources" in verdict.reason


def test_empty_extension_object_supports_required_methods_and_no_directory_read():
    verdict = proto.parse_capabilities(
        {"resources": {}, "extensions": {proto.SKILLS_EXTENSION_ID: {}}}, "2026-07-28")
    assert verdict.supported
    assert verdict.directory_read is False


def test_directory_read_requires_an_explicit_true():
    for value in (None, False, "true", 1):
        caps = {"resources": {}, "extensions": {proto.SKILLS_EXTENSION_ID: {"directoryRead": value}}}
        assert proto.parse_capabilities(caps, "2026-07-28").directory_read is False


def test_settings_must_be_an_object():
    caps = {"resources": {}, "extensions": {proto.SKILLS_EXTENSION_ID: True}}
    assert proto.parse_capabilities(caps, "2026-07-28").support == "protocol_invalid"


def test_legacy_capability_objects_are_accepted_through_the_mapping_normalizer():
    """The declaration point moved between protocol eras; only its shape is normalized."""
    class _Caps:
        def model_dump(self, **kw):
            return {"resources": {}, "extensions": {proto.SKILLS_EXTENSION_ID: {"directoryRead": True}}}

    verdict = proto.parse_capabilities(proto.capabilities_mapping(_Caps()), "2025-11-25")
    assert verdict.supported and verdict.directory_read


# ── URI handling ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("uri", [
    "skill://acme/billing/../secrets/SKILL.md",
    "skill://acme/billing/%2e%2e/secrets/SKILL.md",
    "skill://acme//SKILL.md",
    "skill://acme/billing/refunds/SKILL.md/",
    "skill:///SKILL.md",
    "SKILL.md",
    "",
])
def test_traversal_and_malformed_uris_are_refused(uri):
    with pytest.raises(SkillError) as exc:
        proto.skill_root_uri(uri)
    assert exc.value.code is SkillErrorCode.PROTOCOL_INVALID


def test_entry_uri_must_address_the_main_file():
    with pytest.raises(SkillError):
        proto.skill_root_uri("skill://acme/billing/refunds/README.md")


def test_skill_root_and_relative_path():
    assert proto.skill_root_uri(MAIN) == "skill://acme/billing/refunds"
    assert proto.relative_path_within("skill://acme/billing/refunds", SUPPORT) == "references/api.md"


def test_skill_root_at_the_authority_is_supported():
    """``skill://team/SKILL.md`` has no path segments to strip."""
    assert proto.skill_root_uri("skill://team/SKILL.md") == "skill://team"
    assert proto.relative_path_within("skill://team", "skill://team/nested/file.md") == "nested/file.md"


def test_containment_compares_segments_not_string_prefixes():
    with pytest.raises(SkillError):
        proto.relative_path_within("skill://acme/refunds", "skill://acme/refunds-old/SKILL.md")


def test_cross_origin_resource_is_refused():
    with pytest.raises(SkillError):
        proto.relative_path_within("skill://acme/refunds", "skill://other/refunds/references/x.md")


def test_the_resource_may_not_be_the_root_itself():
    with pytest.raises(SkillError):
        proto.relative_path_within("skill://acme/refunds", "skill://acme/refunds")


# ── entry and manifest validation ───────────────────────────────────────────

def test_valid_entry_becomes_a_routable_contract_object():
    entry = proto.parse_skill_entry(_entry(), origin_id="acme-mcp", label="acme-mcp")
    assert entry.identity.origin_id == "acme-mcp"
    assert entry.identity.locator == MAIN
    assert entry.name == "refunds"
    assert entry.manifest.file_count() == 2
    assert entry.manifest.total_size() == 193
    assert entry.content_revision.startswith("sha256:")
    assert entry.frontmatter["version"] == "2", "unknown frontmatter fields must survive"


def test_manifest_must_list_its_own_main_file():
    entry = _entry(resources=[{"uri": SUPPORT, "digest": "sha256:" + "b" * 64, "size": 42}])
    with pytest.raises(SkillError) as exc:
        proto.parse_skill_entry(entry, origin_id="s", label="s")
    assert "SKILL.md" in exc.value.message


@pytest.mark.parametrize("bad", [
    {"uri": MAIN, "digest": "sha256:XYZ", "size": 1},
    {"uri": MAIN, "digest": "md5:" + "a" * 32, "size": 1},
    {"uri": MAIN, "digest": "sha256:" + "A" * 64, "size": 1},
    {"uri": MAIN, "digest": "sha256:" + "a" * 64, "size": -1},
    {"uri": MAIN, "digest": "sha256:" + "a" * 64},
    {"uri": MAIN, "digest": "sha256:" + "a" * 64, "size": True},
])
def test_malformed_manifest_rows_are_refused(bad):
    with pytest.raises(SkillError) as exc:
        proto.parse_skill_entry(_entry(resources=[bad]), origin_id="s", label="s")
    assert exc.value.code is SkillErrorCode.PROTOCOL_INVALID


def test_duplicate_resource_uris_are_refused():
    row = {"uri": MAIN, "digest": "sha256:" + "a" * 64, "size": 1}
    with pytest.raises(SkillError) as exc:
        proto.parse_skill_entry(_entry(resources=[row, dict(row)]), origin_id="s", label="s")
    assert "twice" in exc.value.message


def test_resources_pointing_outside_the_skill_root_are_refused():
    outside = {"uri": "skill://acme/other/SKILL.md", "digest": "sha256:" + "c" * 64, "size": 9}
    with pytest.raises(SkillError):
        proto.parse_skill_entry(_entry(resources=[outside]), origin_id="s", label="s")


def test_frontmatter_name_is_required():
    with pytest.raises(SkillError):
        proto.parse_skill_entry(_entry(frontmatter={"description": "x"}), origin_id="s", label="s")


def test_too_many_resources_is_a_limit_error_not_a_protocol_error():
    files = [{"uri": MAIN, "digest": "sha256:" + "a" * 64, "size": 1}]
    files += [{"uri": f"skill://acme/billing/refunds/f{i}.md", "digest": "sha256:" + "b" * 64, "size": 1}
              for i in range(proto.MAX_RESOURCES)]
    with pytest.raises(SkillError) as exc:
        proto.parse_skill_entry(_entry(resources=files), origin_id="s", label="s")
    assert exc.value.code is SkillErrorCode.LIMIT_EXCEEDED


def test_oversized_skill_is_a_limit_error():
    huge = [{"uri": MAIN, "digest": "sha256:" + "a" * 64, "size": proto.MAX_TOTAL_BYTES + 1}]
    with pytest.raises(SkillError) as exc:
        proto.parse_skill_entry(_entry(resources=huge), origin_id="s", label="s")
    assert exc.value.code is SkillErrorCode.LIMIT_EXCEEDED


def test_dynamic_manifests_parse_but_carry_no_content_binding():
    entry = proto.parse_skill_entry(_entry(resources="dynamic"), origin_id="s", label="s")
    assert entry.manifest.dynamic
    assert entry.content_revision is None, "a dynamic listing cannot bind a version"


def test_linked_files_groups_everything_but_the_main_file():
    entry = proto.parse_skill_entry(_entry(), origin_id="s", label="s")
    assert proto.linked_files_of(entry) == {"references": ["references/api.md"]}


# ── byte verification ───────────────────────────────────────────────────────

def test_bytes_are_verified_against_the_manifest():
    raw = b"---\nname: refunds\n---\nbody\n"
    good = proto.ManifestFile(uri=MAIN, digest=proto._SHA256_PREFIX + __import__("hashlib").sha256(raw).hexdigest(),
                              size=len(raw), relative_path="SKILL.md")
    assert proto.verify_bytes(raw, good, expect_text=True).ok

    wrong_size = proto.ManifestFile(uri=MAIN, digest=good.digest, size=len(raw) + 1, relative_path="SKILL.md")
    verdict = proto.verify_bytes(raw, wrong_size, expect_text=True)
    assert not verdict.ok and not verdict.size_ok and verdict.digest_ok

    wrong_digest = proto.ManifestFile(uri=MAIN, digest="sha256:" + "0" * 64, size=len(raw),
                                      relative_path="SKILL.md")
    verdict = proto.verify_bytes(raw, wrong_digest, expect_text=True)
    assert not verdict.ok and verdict.size_ok and not verdict.digest_ok


def test_a_main_file_without_frontmatter_fails_verification():
    raw = b"just prose\n"
    import hashlib
    row = proto.ManifestFile(uri=MAIN, digest="sha256:" + hashlib.sha256(raw).hexdigest(),
                             size=len(raw), relative_path="SKILL.md")
    verdict = proto.verify_bytes(raw, row, expect_text=True)
    assert verdict.size_ok and verdict.digest_ok and not verdict.frontmatter_ok
    assert not verdict.ok


def test_binary_resources_skip_the_frontmatter_rule():
    raw = b"\x00\x01\x02"
    import hashlib
    row = proto.ManifestFile(uri=SUPPORT, digest="sha256:" + hashlib.sha256(raw).hexdigest(),
                             size=len(raw), relative_path="references/api.md")
    assert proto.verify_bytes(raw, row, expect_text=False).ok


def test_text_decoding_reports_binary_rather_than_replacing_bytes():
    assert proto.decode_text("héllo".encode()) == "héllo"
    assert proto.decode_text(b"\xff\xfe\x00") is None
