import base64
import hashlib
import json
from pathlib import Path

import pytest

from hermes_wisdom.contract import (
    CONTRACT_PIN,
    ContentFile,
    PackageManifest,
    SystemSpecification,
    author_description_hash,
    canonical_json_bytes,
    derive_content_hash,
    parse_manifest_bytes,
    sanitize_author_description,
    sha256_address,
)


VECTORS = Path("hermes_wisdom/contracts/canonical-hash-vectors.v1.json")
OPENAPI = Path("hermes_wisdom/contracts/gateway-openapi.json")


def test_checked_in_gateway_openapi_matches_the_pinned_digest():
    assert hashlib.sha256(OPENAPI.read_bytes()).hexdigest() == (
        CONTRACT_PIN.openapi_sha256
    )
    contract = json.loads(OPENAPI.read_text(encoding="utf-8"))
    draft = contract["components"]["schemas"]["WisdomDraftRecord"]
    assert "changes_requested" in draft["properties"]["state"]["enum"]
    assert "/v1/sync/org/proposals/{n}/return" in contract["paths"]


def test_gateway_canonical_vectors_match_exactly():
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    files = []
    for item in vectors["files"]:
        body = base64.b64decode(item["content_base64"], validate=True)
        assert sha256_address(body) == item["hash"]
        files.append(
            ContentFile(path=item["path"], mode=item["mode"], hash=item["hash"])
        )
    assert derive_content_hash(files) == vectors["content_hash"]
    canonical = sanitize_author_description(vectors["author_description_input"])
    assert canonical == vectors["canonical_author_description"]
    assert author_description_hash(canonical) == vectors["author_description_hash"]


def test_content_hash_commits_to_mode_and_path():
    blob = sha256_address(b"same")
    plain = derive_content_hash([ContentFile(path="SKILL.md", mode="file", hash=blob)])
    executable = derive_content_hash([
        ContentFile(path="SKILL.md", mode="exec", hash=blob)
    ])
    moved = derive_content_hash([
        ContentFile(path="refs/SKILL.md", mode="file", hash=blob)
    ])
    assert len({plain, executable, moved}) == 3


def _valid_manifest_bytes() -> bytes:
    manifest = PackageManifest(
        name="strict-manifest",
        requirements=SystemSpecification.model_validate({
            "hermes": {"minimum_version": "0.1.0"}
        }),
    )
    return canonical_json_bytes(manifest.model_dump(mode="json"))


def test_manifest_parser_rejects_duplicate_keys_and_nonfinite_numbers():
    valid = _valid_manifest_bytes()
    duplicate = valid.replace(b'{"name":', b'{"schema_version":1,"name":', 1)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_manifest_bytes(duplicate)

    nonfinite = valid.replace(
        b'"minimum_context_window":null', b'"minimum_context_window":NaN'
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        parse_manifest_bytes(nonfinite)


def test_manifest_parser_bounds_integer_values_to_json_safe_range():
    value = json.loads(_valid_manifest_bytes())
    value["requirements"]["model"]["minimum_context_window"] = 9_007_199_254_740_992
    with pytest.raises(ValueError):
        parse_manifest_bytes(canonical_json_bytes(value))
