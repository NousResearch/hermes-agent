"""Tests for tools/skill_set_catalog.py — the #254 + AI Catalog skill-set client.

Unit tests cover schema gating, digest verification, archive safety, and
catalog/index parsing with mocked HTTP. The E2E test at the bottom runs a
real local HTTP server built by scripts/publish_skill_set.py and exercises
discover -> resolve -> fetch against actual bytes on the wire.
"""

import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from unittest.mock import patch

import pytest

from tools.skill_set_catalog import (
    ArchiveSafetyError,
    DigestError,
    FetchResult,
    SchemaError,
    SkillSetError,
    SkillSetInfo,
    catalog_url_for,
    compute_digest,
    discover_skill_sets,
    fetch_member,
    resolve_bare_index,
    resolve_skill_set,
    verify_digest,
    SkillSetMember,
    KNOWN_INDEX_SCHEMAS,
    HERMES_SET_EXTENSION,
    SKILL_SET_ENTRY_TYPE,
)

SCHEMA = next(iter(KNOWN_INDEX_SCHEMAS))
BASE = "https://skills.example.com"
INDEX_URL = f"{BASE}/.well-known/agent-skills/index.json"
CATALOG_URL = f"{BASE}/.well-known/ai-catalog.json"

SKILL_MD = (
    "---\nname: code-review\ndescription: Review code.\n---\n\n# Code Review Skill\n"
)


def _tar_gz(files: dict) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for name, data in files.items():
                raw = data.encode() if isinstance(data, str) else data
                info = tarfile.TarInfo(name=name)
                info.size = len(raw)
                tf.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _serve(pages: dict, *, redirects: "dict | None" = None,
           content_types: "dict | None" = None):
    """Patch the module HTTP layer with a URL -> bytes dict.

    ``redirects`` maps requested URL -> final URL (content is looked up at
    the final URL, and the returned FetchResult carries the final URL, like
    the real fetcher does after following a redirect chain).
    ``content_types`` maps final URL -> Content-Type header value.
    """
    redirects = redirects or {}
    content_types = content_types or {}

    def fake_fetch(url, *, timeout=30):
        final = redirects.get(url, url)
        val = pages.get(final)
        if val is None:
            return None
        content = val.encode() if isinstance(val, str) else val
        return FetchResult(url=final, content=content,
                           content_type=content_types.get(final, ""))
    return patch("tools.skill_set_catalog._http_fetch", side_effect=fake_fetch)


# ---------------------------------------------------------------------------
# Digest verification
# ---------------------------------------------------------------------------

class TestDigest:
    def test_roundtrip(self):
        content = b"hello skills"
        verify_digest(content, compute_digest(content))  # no raise

    def test_mismatch_rejected(self):
        with pytest.raises(DigestError, match="mismatch"):
            verify_digest(b"tampered", compute_digest(b"original"))

    def test_missing_digest_rejected(self):
        with pytest.raises(DigestError, match="missing digest"):
            verify_digest(b"x", "")

    @pytest.mark.parametrize("bad", [
        "sha256:short",
        "sha256:" + "G" * 64,             # non-hex
        "sha256:" + "A" * 64,             # uppercase — spec says lowercase
        "md5:" + "a" * 32,
        "a" * 64,                          # bare hex without prefix
    ])
    def test_malformed_digest_rejected(self, bad):
        with pytest.raises(DigestError, match="malformed"):
            verify_digest(b"x", bad)


# ---------------------------------------------------------------------------
# AI Catalog discovery
# ---------------------------------------------------------------------------

def _catalog(entries) -> str:
    return json.dumps({
        "specVersion": "1.0",
        "host": {"displayName": "Example"},
        "entries": entries,
    })


class TestCatalogDiscovery:
    def test_catalog_url_for_origin(self):
        assert catalog_url_for("https://x.example") == \
            "https://x.example/.well-known/ai-catalog.json"
        assert catalog_url_for("https://x.example/custom/cat.json") == \
            "https://x.example/custom/cat.json"

    def test_finds_skill_set_entries_with_extension(self):
        pages = {CATALOG_URL: _catalog([
            {
                "identifier": "urn:air:example:skill-set:backend",
                "displayName": "Backend Dev",
                "description": "Backend feature work.",
                "type": SKILL_SET_ENTRY_TYPE,
                "url": "/.well-known/agent-skills/index.json",
                "extensions": {HERMES_SET_EXTENSION: {
                    "command": "backend-dev",
                    "instruction": "Prefer TDD.",
                }},
            },
            {"identifier": "urn:air:example:mcp:weather",
             "type": "application/mcp-server-card+json",
             "url": "https://api.example.com/mcp"},
        ])}
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
        assert len(sets) == 1
        s = sets[0]
        assert s.name == "Backend Dev"
        assert s.index_url == INDEX_URL
        assert s.command == "backend-dev"
        assert s.instruction == "Prefer TDD."

    def test_entry_without_extension_still_discovered(self):
        pages = {CATALOG_URL: _catalog([
            {"displayName": "Plain Set", "type": SKILL_SET_ENTRY_TYPE,
             "url": "/.well-known/agent-skills/index.json"},
        ])}
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
        assert len(sets) == 1
        assert sets[0].command == ""
        assert sets[0].instruction == ""

    def test_follows_sub_catalog_one_level(self):
        sub_url = f"{BASE}/catalogs/eng.json"
        pages = {
            CATALOG_URL: _catalog([
                {"displayName": "Engineering", "type": "application/ai-catalog+json",
                 "url": "/catalogs/eng.json"},
            ]),
            sub_url: _catalog([
                {"displayName": "Backend Dev", "type": SKILL_SET_ENTRY_TYPE,
                 "url": "/.well-known/agent-skills/index.json"},
            ]),
        }
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
        assert [s.name for s in sets] == ["Backend Dev"]
        assert sets[0].index_url == INDEX_URL

    def test_not_a_catalog_raises(self):
        with _serve({CATALOG_URL: json.dumps({"skills": []})}):
            with pytest.raises(SkillSetError, match="not an AI Catalog"):
                discover_skill_sets(CATALOG_URL)

    def test_unreachable_catalog_raises(self):
        with _serve({}):
            with pytest.raises(SkillSetError, match="Could not fetch"):
                discover_skill_sets(CATALOG_URL)


# ---------------------------------------------------------------------------
# #254 index resolution
# ---------------------------------------------------------------------------

def _index(skills, schema: "str | None" = SCHEMA) -> str:
    payload = {"skills": skills}
    if schema is not None:
        payload["$schema"] = schema
    return json.dumps(payload)


def _info() -> SkillSetInfo:
    return SkillSetInfo(name="Test Set", description="", index_url=INDEX_URL)


class TestIndexResolution:
    def test_members_parsed_and_urls_resolved(self):
        pages = {INDEX_URL: _index([
            {"name": "code-review", "type": "skill-md",
             "description": "Review code.",
             "url": "code-review/SKILL.md", "digest": compute_digest(b"x")},
            {"name": "wrangler", "type": "archive",
             "description": "Deploy workers.",
             "url": "/.well-known/agent-skills/wrangler.tar.gz",
             "digest": compute_digest(b"y")},
        ])}
        with _serve(pages):
            resolved = resolve_skill_set(_info())
        assert [m.name for m in resolved.members] == ["code-review", "wrangler"]
        # Relative resolved against index directory; path-absolute against origin.
        assert resolved.members[0].url == \
            f"{BASE}/.well-known/agent-skills/code-review/SKILL.md"
        assert resolved.members[1].url == \
            f"{BASE}/.well-known/agent-skills/wrangler.tar.gz"

    def test_unknown_schema_refused(self):
        pages = {INDEX_URL: _index([], schema="https://example.com/other/1.0.json")}
        with _serve(pages):
            with pytest.raises(SchemaError, match="Unrecognized index"):
                resolve_skill_set(_info())

    def test_absent_schema_refused(self):
        pages = {INDEX_URL: _index([], schema=None)}
        with _serve(pages):
            with pytest.raises(SchemaError):
                resolve_skill_set(_info())

    def test_unrecognized_type_skipped_with_warning(self):
        pages = {INDEX_URL: _index([
            {"name": "good", "type": "skill-md", "url": "good/SKILL.md",
             "digest": compute_digest(b"x")},
            {"name": "weird", "type": "oci-image", "url": "weird.oci",
             "digest": compute_digest(b"y")},
        ])}
        with _serve(pages):
            resolved = resolve_skill_set(_info())
        assert [m.name for m in resolved.members] == ["good"]
        assert any("weird" in s for s in resolved.skipped)

    def test_bare_index_fallback_name(self):
        pages = {INDEX_URL: _index([])}
        with _serve(pages):
            resolved = resolve_bare_index(INDEX_URL)
        assert resolved.info.name == "skills.example.com"


# ---------------------------------------------------------------------------
# Member fetching — digest + archive safety
# ---------------------------------------------------------------------------

def _member(name="code-review", mtype="skill-md", url=None, digest=""):
    return SkillSetMember(
        name=name, description="", type=mtype,
        url=url or f"{BASE}/.well-known/agent-skills/{name}/SKILL.md",
        digest=digest,
    )


class TestFetchMember:
    def test_skill_md_happy_path(self):
        content = SKILL_MD.encode()
        m = _member(digest=compute_digest(content))
        with _serve({m.url: content}):
            bundle = fetch_member(m, set_info=_info())
        assert bundle.name == "code-review"
        assert bundle.files == {"SKILL.md": SKILL_MD}
        assert bundle.source == "skill-set"
        assert bundle.metadata["digest"] == m.digest

    def test_tampered_content_rejected(self):
        m = _member(digest=compute_digest(SKILL_MD.encode()))
        with _serve({m.url: b"---\nname: evil\n---\nrm -rf /"}):
            with pytest.raises(DigestError, match="mismatch"):
                fetch_member(m, set_info=_info())

    def test_archive_happy_path(self):
        artifact = _tar_gz({
            "SKILL.md": SKILL_MD,
            "scripts/deploy.sh": "#!/bin/sh\necho hi\n",
            "references/API.md": "# API\n",
        })
        url = f"{BASE}/.well-known/agent-skills/wrangler.tar.gz"
        m = _member("wrangler", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            bundle = fetch_member(m, set_info=_info())
        assert set(bundle.files) == {"SKILL.md", "scripts/deploy.sh",
                                     "references/API.md"}

    def test_zip_happy_path(self):
        artifact = _zip({"SKILL.md": SKILL_MD, "references/NOTES.md": "notes"})
        url = f"{BASE}/.well-known/agent-skills/z.zip"
        m = _member("zskill", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            bundle = fetch_member(m, set_info=_info())
        assert set(bundle.files) == {"SKILL.md", "references/NOTES.md"}

    def test_archive_without_root_skill_md_rejected(self):
        artifact = _tar_gz({"nested/SKILL.md": SKILL_MD})
        url = f"{BASE}/.well-known/agent-skills/bad.tar.gz"
        m = _member("bad", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            with pytest.raises(ArchiveSafetyError, match="no SKILL.md at its root"):
                fetch_member(m, set_info=_info())

    def test_path_traversal_rejected(self):
        artifact = _tar_gz({"SKILL.md": SKILL_MD, "../../evil.sh": "boom"})
        url = f"{BASE}/.well-known/agent-skills/trav.tar.gz"
        m = _member("trav", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            with pytest.raises(ArchiveSafetyError):
                fetch_member(m, set_info=_info())

    def test_absolute_path_rejected(self):
        artifact = _tar_gz({"SKILL.md": SKILL_MD, "/etc/cron.d/evil": "boom"})
        url = f"{BASE}/.well-known/agent-skills/abs.tar.gz"
        m = _member("abs", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            with pytest.raises(ArchiveSafetyError):
                fetch_member(m, set_info=_info())

    def test_symlink_member_rejected(self):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                data = SKILL_MD.encode()
                info = tarfile.TarInfo(name="SKILL.md")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
                link = tarfile.TarInfo(name="creds")
                link.type = tarfile.SYMTYPE
                link.linkname = "/home/user/.ssh/id_rsa"
                tf.addfile(link)
        artifact = buf.getvalue()
        url = f"{BASE}/.well-known/agent-skills/lnk.tar.gz"
        m = _member("lnk", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            with pytest.raises(ArchiveSafetyError, match="link member"):
                fetch_member(m, set_info=_info())

    def test_decompression_bomb_rejected(self):
        # 20MB of zeros compresses tiny but exceeds the per-member cap.
        artifact = _tar_gz({"SKILL.md": SKILL_MD, "big.bin": b"\0" * (6 * 1024 * 1024)})
        url = f"{BASE}/.well-known/agent-skills/bomb.tar.gz"
        m = _member("bomb", "archive", url, compute_digest(artifact))
        with _serve({url: artifact}):
            with pytest.raises(ArchiveSafetyError):
                fetch_member(m, set_info=_info())

    def test_unsupported_archive_extension_rejected(self):
        url = f"{BASE}/.well-known/agent-skills/skill.rar"
        m = _member("rarred", "archive", url, compute_digest(b"data"))
        with _serve({url: b"data"}):
            with pytest.raises(ArchiveSafetyError, match="unsupported archive format"):
                fetch_member(m, set_info=_info())

    def test_unsafe_skill_name_rejected(self):
        m = _member("../escape", digest=compute_digest(b"x"))
        with _serve({m.url: b"x"}):
            with pytest.raises(ValueError):
                fetch_member(m, set_info=_info())


# ---------------------------------------------------------------------------
# Review follow-ups (agentskills feedback on PR #81875)
# ---------------------------------------------------------------------------

CDN_INDEX_URL = "https://cdn.example.net/releases/v2/index.json"


class TestRedirectBaseResolution:
    """#254/RFC 3986: relative member URLs resolve against the URL the index
    was actually retrieved from — i.e. AFTER redirects — not the requested URL."""

    def test_members_resolve_against_post_redirect_location(self):
        pages = {CDN_INDEX_URL: _index([
            {"name": "review", "type": "skill-md",
             "url": "review/SKILL.md", "digest": compute_digest(b"x")},
        ])}
        with _serve(pages, redirects={INDEX_URL: CDN_INDEX_URL}):
            resolved = resolve_skill_set(_info())
        assert resolved.members[0].url == \
            "https://cdn.example.net/releases/v2/review/SKILL.md"

    def test_catalog_redirect_rebases_entry_urls(self):
        moved_catalog = "https://cdn.example.net/meta/catalog.json"
        pages = {moved_catalog: _catalog([
            {"displayName": "Backend Dev", "type": SKILL_SET_ENTRY_TYPE,
             "url": "skills/index.json"},
        ])}
        with _serve(pages, redirects={CATALOG_URL: moved_catalog}):
            sets = discover_skill_sets(CATALOG_URL)
        assert sets[0].index_url == "https://cdn.example.net/meta/skills/index.json"


class TestContentTypeArchiveDetection:
    """#254: archive format comes from Content-Type first; the URL file
    extension is only a fallback for absent/generic headers."""

    def _archive_member(self, url, artifact):
        return _member("packed", "archive", url, compute_digest(artifact))

    def test_content_type_wins_over_missing_extension(self):
        artifact = _tar_gz({"SKILL.md": SKILL_MD})
        url = f"{BASE}/download?skill=packed"     # no useful extension
        m = self._archive_member(url, artifact)
        with _serve({url: artifact}, content_types={url: "application/gzip"}):
            bundle = fetch_member(m, set_info=_info())
        assert "SKILL.md" in bundle.files

    def test_content_type_wins_over_wrong_extension(self):
        # Server says zip; URL misleadingly ends in .tar.gz. Header wins.
        artifact = _zip({"SKILL.md": SKILL_MD})
        url = f"{BASE}/.well-known/agent-skills/skill.tar.gz"
        m = self._archive_member(url, artifact)
        with _serve({url: artifact}, content_types={url: "application/zip"}):
            bundle = fetch_member(m, set_info=_info())
        assert "SKILL.md" in bundle.files

    def test_generic_content_type_falls_back_to_extension(self):
        artifact = _tar_gz({"SKILL.md": SKILL_MD})
        url = f"{BASE}/.well-known/agent-skills/skill.tar.gz"
        m = self._archive_member(url, artifact)
        with _serve({url: artifact},
                    content_types={url: "application/octet-stream"}):
            bundle = fetch_member(m, set_info=_info())
        assert "SKILL.md" in bundle.files

    def test_no_header_no_extension_rejected(self):
        artifact = _tar_gz({"SKILL.md": SKILL_MD})
        url = f"{BASE}/download?skill=packed"
        m = self._archive_member(url, artifact)
        with _serve({url: artifact}):
            with pytest.raises(ArchiveSafetyError, match="unsupported archive format"):
                fetch_member(m, set_info=_info())


class TestInlineDataEntries:
    """AI Catalog entries may carry `data` instead of `url`."""

    def test_skill_set_entry_with_inline_index(self):
        inline_index = {
            "$schema": SCHEMA,
            "skills": [
                {"name": "code-review", "type": "skill-md",
                 "url": "/.well-known/agent-skills/code-review/SKILL.md",
                 "digest": compute_digest(SKILL_MD.encode())},
            ],
        }
        pages = {CATALOG_URL: _catalog([
            {"displayName": "Inline Set", "type": SKILL_SET_ENTRY_TYPE,
             "data": inline_index,
             "extensions": {HERMES_SET_EXTENSION: {"command": "inline-set"}}},
        ])}
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
            assert len(sets) == 1
            assert sets[0].inline_index is not None
            assert sets[0].command == "inline-set"
            resolved = resolve_skill_set(sets[0])
        # Relative member URLs resolve against the catalog's location.
        assert resolved.members[0].url == \
            f"{BASE}/.well-known/agent-skills/code-review/SKILL.md"

    def test_inline_index_schema_still_gated(self):
        pages = {CATALOG_URL: _catalog([
            {"displayName": "Bad Inline", "type": SKILL_SET_ENTRY_TYPE,
             "data": {"$schema": "https://example.com/nope.json", "skills": []}},
        ])}
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
            with pytest.raises(SchemaError):
                resolve_skill_set(sets[0])

    def test_inline_sub_catalog_followed(self):
        pages = {CATALOG_URL: _catalog([
            {"displayName": "Nested", "type": "application/ai-catalog+json",
             "data": {"specVersion": "1.0", "entries": [
                 {"displayName": "Backend Dev", "type": SKILL_SET_ENTRY_TYPE,
                  "url": "/.well-known/agent-skills/index.json"},
             ]}},
        ])}
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
        assert [s.name for s in sets] == ["Backend Dev"]
        assert sets[0].index_url == INDEX_URL

    def test_entry_with_neither_url_nor_data_skipped(self):
        pages = {CATALOG_URL: _catalog([
            {"displayName": "Empty", "type": SKILL_SET_ENTRY_TYPE},
            {"displayName": "Real", "type": SKILL_SET_ENTRY_TYPE,
             "url": "/.well-known/agent-skills/index.json"},
        ])}
        with _serve(pages):
            sets = discover_skill_sets(CATALOG_URL)
        assert [s.name for s in sets] == ["Real"]


# ---------------------------------------------------------------------------
# E2E: publisher script -> real HTTP server -> full client flow
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture()
    def published_site(self, tmp_path):
        """Build a real publisher tree with scripts/publish_skill_set.py."""
        import subprocess
        import sys as _sys
        from pathlib import Path as _P

        repo_root = _P(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "publish_skill_set.py"

        # One single-file skill, one multi-file skill.
        s1 = tmp_path / "src" / "code-review"
        s1.mkdir(parents=True)
        (s1 / "SKILL.md").write_text(SKILL_MD)
        s2 = tmp_path / "src" / "deploy-tool"
        (s2 / "scripts").mkdir(parents=True)
        (s2 / "SKILL.md").write_text(
            "---\nname: deploy-tool\ndescription: Deploy things.\n---\n\n# Deploy\n")
        (s2 / "scripts" / "run.sh").write_text("#!/bin/sh\necho deploy\n")

        out = tmp_path / "public"
        subprocess.run(
            [_sys.executable, str(script),
             "--name", "Backend Dev", "--command", "backend-dev",
             "--description", "Backend feature work.",
             "--instruction", "Prefer TDD.",
             "--out", str(out), str(s1), str(s2)],
            check=True, capture_output=True, text=True,
        )
        return out

    def test_full_flow_over_real_http(self, published_site):
        import http.server
        import threading

        handler = type("H", (http.server.SimpleHTTPRequestHandler,), {
            "directory": str(published_site),
            "log_message": lambda self, *a: None,
        })
        httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *a, **kw: handler(*a, directory=str(published_site), **kw),
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            origin = f"http://127.0.0.1:{port}"

            # Bypass the SSRF guard for the loopback test server only.
            def local_fetch(url, *, timeout=30):
                import httpx
                resp = httpx.get(url, timeout=timeout, follow_redirects=True)
                if resp.status_code != 200:
                    return None
                return FetchResult(url=str(resp.url), content=resp.content,
                                   content_type=resp.headers.get("content-type", ""))

            with patch("tools.skill_set_catalog._http_fetch",
                       side_effect=local_fetch):
                sets = discover_skill_sets(catalog_url_for(origin))
                assert len(sets) == 1
                info = sets[0]
                assert info.name == "Backend Dev"
                assert info.command == "backend-dev"
                assert info.instruction == "Prefer TDD."

                resolved = resolve_skill_set(info)
                assert {m.name for m in resolved.members} == \
                    {"code-review", "deploy-tool"}
                types = {m.name: m.type for m in resolved.members}
                assert types["code-review"] == "skill-md"
                assert types["deploy-tool"] == "archive"

                bundles = {m.name: fetch_member(m, set_info=info)
                           for m in resolved.members}
            assert bundles["code-review"].files["SKILL.md"] == SKILL_MD
            assert "scripts/run.sh" in bundles["deploy-tool"].files
            # Digest verification ran on real bytes for every member.
            for b in bundles.values():
                assert b.metadata["digest"].startswith("sha256:")
        finally:
            httpd.shutdown()
            httpd.server_close()
