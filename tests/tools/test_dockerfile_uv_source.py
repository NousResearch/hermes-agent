"""Security contract for the uv binaries copied into the container image."""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"

# Each entry must be backed by an audit of the immutable registry image and its
# dependency graph. uv 0.11.33 embeds the rustls-webpki 0.103.13 fix required by
# #51292; its GHCR manifest resolves to the digest below.
_APPROVED_UV_SOURCE_DIGESTS = {
    "0.11.33-python3.13-trixie": (
        "83447533ae5dab3fbe1b09dff4d34bce0b796d65392179007c04ce1d72d3d752"
    ),
}
_VULNERABLE_UV_SOURCE_DIGEST = (
    "b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b"
)
_UV_SOURCE_RE = re.compile(
    r"^FROM ghcr\.io/astral-sh/uv:"
    r"(?P<source>\d+\.\d+\.\d+-[^@\s]+)"
    r"@sha256:(?P<digest>[0-9a-f]{64}) AS uv_source$",
    re.MULTILINE,
)


def _assert_approved_uv_source(dockerfile: str) -> None:
    match = _UV_SOURCE_RE.search(dockerfile)

    assert match is not None, (
        "Dockerfile must pin the uv source image by version and sha256 digest; "
        "keep the source stage name stable so this container security contract applies"
    )
    source = match.group("source")
    assert source in _APPROVED_UV_SOURCE_DIGESTS, (
        f"uv source {source} has not been audited for the rustls-webpki security "
        "floor required by #51292"
    )
    assert match.group("digest") == _APPROVED_UV_SOURCE_DIGESTS[source], (
        f"uv source {source} must use its independently audited, approved digest"
    )


def test_container_uv_source_excludes_vulnerable_rustls_webpki():
    """The copied uv/uvx binaries must use the patched Rust dependency graph."""
    _assert_approved_uv_source(DOCKERFILE.read_text(encoding="utf-8"))


def test_container_uv_source_rejects_vulnerable_digest_with_approved_tag():
    """A safe-looking tag cannot mask selection of the retired vulnerable image."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    approved_digest = _APPROVED_UV_SOURCE_DIGESTS["0.11.33-python3.13-trixie"]
    vulnerable_source = dockerfile.replace(
        approved_digest, _VULNERABLE_UV_SOURCE_DIGEST
    )

    assert vulnerable_source != dockerfile
    with pytest.raises(AssertionError, match="independently audited, approved digest"):
        _assert_approved_uv_source(vulnerable_source)
