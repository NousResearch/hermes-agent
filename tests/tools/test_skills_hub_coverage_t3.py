#!/usr/bin/env python3

"""Coverage tests for ClawHubSource and LobeHubSource branches in
tools/skills_hub.py that existing tests do not exercise.

Focus areas (see fix/skills-hub-coverage-36586):
  * ClawHubSource._normalize_tags / _coerce_skill_payload edge cases
  * ClawHubSource version-resolution fallbacks / _get_json error paths
  * ClawHubSource ZIP bundle extraction (unsafe member, large-file, non-text skip)
  * ClawHubSource owner helpers and 429 retry-after cap in _download_zip
  * LobeHubSource search / inspect / fetch / _fetch_index / _fetch_agent branches

All tests are deterministic and offline: httpx calls are mocked, cache reads/
writes are short-circuited, and ZIP bundles are built in-memory.
"""

import io
import json
import unittest
import zipfile
from unittest.mock import patch

from tools.skills_hub import ClawHubSource, LobeHubSource, SkillBundle


class _MockResponse:
    """Minimal httpx.Response stand-in with every attribute the code touches."""

    def __init__(self, status_code=200, json_data=None, text="", content=b"", headers=None):
        self.status_code = status_code
        self.text = text
        self.content = content
        self.headers = headers or {}
        self._json_data = json_data
        self._json_error = None

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


def _make_zip_bytes(members):
    """Build an in-memory ZIP bundle from ``{name: bytes}``.

    ``members`` values may be ``bytes`` (already-encoded) or ``str`` (utf-8).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data if isinstance(data, bytes) else data.encode("utf-8"))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ClawHubSource._coerce_skill_payload
# ---------------------------------------------------------------------------

class TestClawHubCoercePayload(unittest.TestCase):
    def test_non_dict_returns_none(self):
        for bad in ("string", ["x"], 123, True, None):
            self.assertIsNone(ClawHubSource._coerce_skill_payload(bad))

    def test_plain_dict_passthrough(self):
        data = {"slug": "caldav", "tags": ["a"]}
        self.assertIs(ClawHubSource._coerce_skill_payload(data), data)

    def test_nested_skill_merges_version_and_owner(self):
        data = {
            "skill": {"slug": "x", "displayName": "X"},
            "latestVersion": {"version": "1.2.3"},
            "owner": {"handle": "alice"},
        }
        merged = ClawHubSource._coerce_skill_payload(data)
        self.assertEqual(merged["slug"], "x")
        self.assertEqual(merged["latestVersion"], {"version": "1.2.3"})
        self.assertEqual(merged["owner"], {"handle": "alice"})

    def test_nested_skill_does_not_clobber_existing_version(self):
        data = {"skill": {"slug": "x", "latestVersion": {"version": "9.9.9"}}, "latestVersion": {"version": "1.0.0"}}
        merged = ClawHubSource._coerce_skill_payload(data)
        self.assertEqual(merged["latestVersion"], {"version": "9.9.9"})

    def test_nested_skill_without_version_and_owner(self):
        merged = ClawHubSource._coerce_skill_payload({"skill": {"slug": "y"}})
        self.assertEqual(merged, {"slug": "y"})


# ---------------------------------------------------------------------------
# ClawHubSource._normalize_tags
# ---------------------------------------------------------------------------

class TestClawHubNormalizeTags(unittest.TestCase):
    def test_list_coerced_to_str(self):
        self.assertEqual(ClawHubSource._normalize_tags(["a", 5, "b"]), ["a", "5", "b"])

    def test_dict_excludes_latest(self):
        self.assertEqual(ClawHubSource._normalize_tags({"latest": "3.0.2", "auto": "3.0.2"}), ["auto"])

    def test_dict_without_latest_returns_all_keys(self):
        self.assertEqual(ClawHubSource._normalize_tags({"a": "1", "b": "2"}), ["a", "b"])

    def test_other_types_return_empty(self):
        for bad in ("tag", 5, None, 3.14, (1, 2)):
            self.assertEqual(ClawHubSource._normalize_tags(bad), [])


# ---------------------------------------------------------------------------
# ClawHubSource owner helpers
# ---------------------------------------------------------------------------

class TestClawHubOwnerHelpers(unittest.TestCase):
    def test_owner_from_payload_dict_handle(self):
        self.assertEqual(ClawHubSource._owner_from_payload({"owner": {"handle": " alice "}}), "alice")

    def test_owner_from_payload_string(self):
        self.assertEqual(ClawHubSource._owner_from_payload({"owner": "bob"}), "bob")

    def test_owner_from_payload_empty_variants(self):
        self.assertIsNone(ClawHubSource._owner_from_payload(None))
        self.assertIsNone(ClawHubSource._owner_from_payload({}))
        self.assertIsNone(ClawHubSource._owner_from_payload({"owner": {}}))
        self.assertIsNone(ClawHubSource._owner_from_payload({"owner": {"handle": "  "}}))
        self.assertIsNone(ClawHubSource._owner_from_payload({"owner": " "}))

    def test_owner_matches_no_expected_owner(self):
        self.assertTrue(ClawHubSource._owner_matches(None, {"slug": "x"}))
        self.assertTrue(ClawHubSource._owner_matches("", {"slug": "x"}))

    def test_owner_matches_missing_actual(self):
        self.assertTrue(ClawHubSource._owner_matches("alice", {}))
        self.assertTrue(ClawHubSource._owner_matches("alice", {"owner": {}}))

    def test_owner_matches_ci_equal(self):
        self.assertTrue(ClawHubSource._owner_matches("Alice", {"owner": {"handle": "alice"}}))

    def test_owner_matches_mismatch(self):
        self.assertFalse(ClawHubSource._owner_matches("alice", {"owner": {"handle": "bob"}}))


# ---------------------------------------------------------------------------
# ClawHubSource._get_json error paths
# ---------------------------------------------------------------------------

class TestClawHubGetJson(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    @patch("tools.skills_hub.httpx.get")
    def test_returns_none_on_non_200(self, mock_get):
        mock_get.return_value = _MockResponse(status_code=404, json_data={})
        self.assertIsNone(self.src._get_json("https://x.example/skills/a"))

    @patch("tools.skills_hub.httpx.get")
    def test_returns_none_on_http_error(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.ConnectError("refused")
        self.assertIsNone(self.src._get_json("https://x.example/skills/a"))

    @patch("tools.skills_hub.httpx.get")
    def test_returns_none_on_json_decode_error(self, mock_get):
        resp = _MockResponse(status_code=200)
        resp._json_error = json.JSONDecodeError("bad", "doc", 0)
        mock_get.return_value = resp
        self.assertIsNone(self.src._get_json("https://x.example/skills/a"))

    @patch("tools.skills_hub.httpx.get")
    def test_returns_json_on_success(self, mock_get):
        mock_get.return_value = _MockResponse(status_code=200, json_data={"slug": "a"})
        self.assertEqual(self.src._get_json("https://x.example/skills/a"), {"slug": "a"})


# ---------------------------------------------------------------------------
# ClawHubSource._resolve_latest_version fallbacks
# ---------------------------------------------------------------------------

class TestClawHubResolveLatestVersion(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    def test_resolves_from_latest_version_dict(self):
        self.assertEqual(
            self.src._resolve_latest_version("s", {"latestVersion": {"version": "3.0.2"}}),
            "3.0.2",
        )

    def test_latest_version_dict_without_version_falls_through(self):
        # No version field -> should fall through to tags, then versions list.
        self.assertEqual(
            self.src._resolve_latest_version("s", {"latestVersion": {}}),
            None,
        )

    def test_resolves_from_tags_latest(self):
        self.assertEqual(
            self.src._resolve_latest_version("s", {"tags": {"latest": "2.1.0", "x": "1"}}),
            "2.1.0",
        )

    @patch.object(ClawHubSource, "_get_json")
    def test_resolves_from_versions_list(self, mock_get_json):
        mock_get_json.return_value = [{"version": "1.4.0"}, {"version": "1.3.0"}]
        self.assertEqual(self.src._resolve_latest_version("s", {"slug": "s"}), "1.4.0")

    @patch.object(ClawHubSource, "_get_json")
    def test_returns_none_when_no_source(self, mock_get_json):
        mock_get_json.return_value = None
        self.assertIsNone(self.src._resolve_latest_version("s", {"slug": "s"}))


# ---------------------------------------------------------------------------
# ClawHubSource._extract_files branches
# ---------------------------------------------------------------------------

class TestClawHubExtractFiles(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    def test_files_dict_passthrough(self):
        files = self.src._extract_files({"files": {"SKILL.md": "# x", "ref": "y"}})
        self.assertEqual(files, {"SKILL.md": "# x", "ref": "y"})

    def test_files_dict_keeps_only_str_values(self):
        files = self.src._extract_files({"files": {"SKILL.md": "# x", "bin": b"raw"}})
        self.assertEqual(files, {"SKILL.md": "# x"})

    def test_files_list_uses_inline_content_and_raw_url(self):
        files = self.src._extract_files({
            "files": [
                {"path": "SKILL.md", "content": "inline"},
                {"name": "ref.md", "rawUrl": "http://files.example/ref"},
            ]
        })
        self.assertEqual(files, {"SKILL.md": "inline"})

    @patch("tools.skills_hub._guarded_http_get")
    def test_files_list_fetches_raw_text(self, mock_guarded):
        mock_guarded.return_value = _MockResponse(status_code=200, text="fetched")
        files = self.src._extract_files({
            "files": [{"path": "SKILL.md", "rawUrl": "http://files.example/skill"}]
        })
        self.assertEqual(files, {"SKILL.md": "fetched"})
        mock_guarded.assert_called_once_with("http://files.example/skill", timeout=20)

    @patch("tools.skills_hub._guarded_http_get")
    def test_files_list_skips_non_http_raw_url(self, mock_guarded):
        files = self.src._extract_files({
            "files": [{"path": "SKILL.md", "rawUrl": "ftp://files.example/skill"}]
        })
        self.assertEqual(files, {})
        mock_guarded.assert_not_called()

    @patch("tools.skills_hub._guarded_http_get")
    def test_files_list_skips_failed_raw_fetch(self, mock_guarded):
        mock_guarded.return_value = None
        files = self.src._extract_files({
            "files": [{"path": "SKILL.md", "rawUrl": "http://files.example/skill"}]
        })
        self.assertEqual(files, {})

    def test_files_list_skips_missing_path(self):
        files = self.src._extract_files({"files": [{"content": "no path"}]})
        self.assertEqual(files, {})

    def test_files_list_skips_non_dict_meta(self):
        files = self.src._extract_files({"files": ["just a string"]})
        self.assertEqual(files, {})

    def test_files_list_not_list_returns_empty(self):
        self.assertEqual(self.src._extract_files({"files": "nope"}), {})
        self.assertEqual(self.src._extract_files({}), {})


# ---------------------------------------------------------------------------
# ClawHubSource._download_zip — ZIP extraction branches
# ---------------------------------------------------------------------------

class TestClawHubDownloadZip(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    def _zip_with(self, members):
        return _make_zip_bytes(members)

    @patch("tools.skills_hub.httpx.get")
    def test_happy_path_extracts_text_files(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            content=self._zip_with({"SKILL.md": "# Skill\nbody", "references/doc.md": "doc"}),
        )
        files = self.src._download_zip("slug", "1.0.0")
        self.assertEqual(files["SKILL.md"], "# Skill\nbody")
        self.assertEqual(files["references/doc.md"], "doc")
        mock_get.assert_called_once()
        call = mock_get.call_args
        self.assertTrue(call.args[0].endswith("/download"))
        self.assertEqual(call.kwargs["params"], {"slug": "slug", "version": "1.0.0"})

    @patch("tools.skills_hub.httpx.get")
    def test_skips_unsafe_member_path(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            content=self._zip_with({"SKILL.md": "# ok", "../evil.txt": b"evil"}),
        )
        files = self.src._download_zip("slug", "1.0.0")
        self.assertIn("SKILL.md", files)
        self.assertNotIn("../evil.txt", files)

    @patch("tools.skills_hub.httpx.get")
    def test_skips_large_file(self, mock_get):
        mock_get.return_value = _MockResponse(
            status_code=200,
            content=self._zip_with({"SKILL.md": "# ok", "big.bin": b"x" * 600000}),
        )
        files = self.src._download_zip("slug", "1.0.0")
        self.assertIn("SKILL.md", files)
        self.assertNotIn("big.bin", files)

    @patch("tools.skills_hub.httpx.get")
    def test_skips_non_text_file(self, mock_get):
        # Invalid UTF-8 bytes -> decode raises UnicodeDecodeError -> skipped.
        mock_get.return_value = _MockResponse(
            status_code=200,
            content=self._zip_with({"SKILL.md": "# ok", "icon.png": b"\xff\xfe\x00"}),
        )
        files = self.src._download_zip("slug", "1.0.0")
        self.assertIn("SKILL.md", files)
        self.assertNotIn("icon.png", files)

    @patch("tools.skills_hub.time.sleep")
    @patch("tools.skills_hub.httpx.get")
    def test_429_caps_retry_after_and_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _MockResponse(status_code=429, headers={"retry-after": "100"}),
            _MockResponse(status_code=200, content=self._zip_with({"SKILL.md": "# recovered"})),
        ]
        files = self.src._download_zip("slug", "1.0.0")
        self.assertEqual(files["SKILL.md"], "# recovered")
        # Wait is capped at 15s regardless of a 100s Retry-After header.
        mock_sleep.assert_called_once_with(15)
        self.assertEqual(mock_get.call_count, 2)

    @patch("tools.skills_hub.time.sleep")
    @patch("tools.skills_hub.httpx.get")
    def test_429_invalid_retry_after_uses_default_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _MockResponse(status_code=429, headers={"retry-after": "not-a-number"}),
            _MockResponse(status_code=200, content=self._zip_with({"SKILL.md": "# ok"})),
        ]
        files = self.src._download_zip("slug", "1.0.0")
        self.assertEqual(files["SKILL.md"], "# ok")
        mock_sleep.assert_called_once_with(5)

    @patch("tools.skills_hub.httpx.get")
    def test_non_200_returns_empty(self, mock_get):
        mock_get.return_value = _MockResponse(status_code=404)
        self.assertEqual(self.src._download_zip("slug", "1.0.0"), {})

    @patch("tools.skills_hub.httpx.get")
    def test_bad_zip_returns_empty(self, mock_get):
        mock_get.return_value = _MockResponse(status_code=200, content=b"not a zip")
        self.assertEqual(self.src._download_zip("slug", "1.0.0"), {})

    @patch("tools.skills_hub.httpx.get")
    def test_http_error_returns_empty(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.ConnectError("down")
        self.assertEqual(self.src._download_zip("slug", "1.0.0"), {})


# ---------------------------------------------------------------------------
# ClawHubSource.fetch — end-to-end ZIP download path
# ---------------------------------------------------------------------------

class TestClawHubFetchZipPath(unittest.TestCase):
    def setUp(self):
        self.src = ClawHubSource()

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_returns_bundle_from_zip(self, mock_get):
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav"):
                return _MockResponse(
                    status_code=200,
                    json_data={"slug": "caldav", "latestVersion": {"version": "1.0.0"}},
                )
            if url.endswith("/download"):
                return _MockResponse(
                    status_code=200,
                    content=_make_zip_bytes({"SKILL.md": "# CalDAV Skill"}),
                )
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        bundle = self.src.fetch("caldav")

        self.assertIsNotNone(bundle)
        self.assertIsInstance(bundle, SkillBundle)
        self.assertEqual(bundle.name, "caldav")
        self.assertEqual(bundle.files["SKILL.md"], "# CalDAV Skill")
        self.assertEqual(bundle.source, "clawhub")

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_returns_none_when_zip_and_fallback_all_missing(self, mock_get):
        # Detail resolves, ZIP download 404s (empty), version endpoint 404s too.
        def side_effect(url, *args, **kwargs):
            if url.endswith("/skills/caldav"):
                return _MockResponse(status_code=200, json_data={"slug": "caldav", "latestVersion": {"version": "1.0.0"}})
            return _MockResponse(status_code=404, json_data={})

        mock_get.side_effect = side_effect

        self.assertIsNone(self.src.fetch("caldav"))

    @patch("tools.skills_hub.httpx.get")
    def test_fetch_returns_none_when_no_latest_version(self, mock_get):
        mock_get.return_value = _MockResponse(status_code=200, json_data={"slug": "caldav"})
        self.assertIsNone(self.src.fetch("caldav"))


# ---------------------------------------------------------------------------
# LobeHubSource.search
# ---------------------------------------------------------------------------

class TestLobeHubSearch(unittest.TestCase):
    def setUp(self):
        self.src = LobeHubSource()
        self._index = {
            "agents": [
                {
                    "identifier": "web-research",
                    "meta": {"title": "Web Research", "description": "Searches the internet", "tags": ["research", "search"]},
                },
                {
                    "identifier": "code-writer",
                    "meta": {"title": "Code Writer", "description": "Writes Python", "tags": ["code"]},
                },
            ]
        }

    @patch.object(LobeHubSource, "_fetch_index")
    def test_search_matches_and_builds_meta(self, mock_fetch_index):
        mock_fetch_index.return_value = self._index
        results = self.src.search("research")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "lobehub/web-research")
        self.assertEqual(results[0].name, "web-research")
        self.assertEqual(results[0].source, "lobehub")
        self.assertEqual(results[0].tags, ["research", "search"])

    @patch.object(LobeHubSource, "_fetch_index")
    def test_search_returns_empty_when_no_match(self, mock_fetch_index):
        mock_fetch_index.return_value = self._index
        self.assertEqual(self.src.search("nonexistent"), [])

    @patch.object(LobeHubSource, "_fetch_index")
    def test_search_returns_empty_when_index_missing(self, mock_fetch_index):
        mock_fetch_index.return_value = None
        self.assertEqual(self.src.search("research"), [])

    @patch.object(LobeHubSource, "_fetch_index")
    def test_search_returns_empty_when_agents_not_list(self, mock_fetch_index):
        mock_fetch_index.return_value = {"agents": "not-a-list"}
        self.assertEqual(self.src.search("research"), [])

    @patch.object(LobeHubSource, "_fetch_index")
    def test_search_respects_limit(self, mock_fetch_index):
        mock_fetch_index.return_value = {
            "agents": [
                {"identifier": "a", "meta": {"title": "Research A", "description": "research", "tags": []}},
                {"identifier": "b", "meta": {"title": "Research B", "description": "research", "tags": []}},
            ]
        }
        results = self.src.search("research", limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].identifier, "lobehub/a")


# ---------------------------------------------------------------------------
# LobeHubSource.inspect
# ---------------------------------------------------------------------------

class TestLobeHubInspect(unittest.TestCase):
    def setUp(self):
        self.src = LobeHubSource()
        self._index = {
            "agents": [
                {"identifier": "web-research", "meta": {"title": "Web Research", "description": "Searches", "tags": ["research"]}},
            ]
        }

    @patch.object(LobeHubSource, "_fetch_index")
    def test_inspect_finds_agent(self, mock_fetch_index):
        mock_fetch_index.return_value = self._index
        meta = self.src.inspect("lobehub/web-research")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.identifier, "lobehub/web-research")
        self.assertEqual(meta.description, "Searches")
        self.assertEqual(meta.tags, ["research"])

    @patch.object(LobeHubSource, "_fetch_index")
    def test_inspect_not_found_returns_none(self, mock_fetch_index):
        mock_fetch_index.return_value = self._index
        self.assertIsNone(self.src.inspect("lobehub/missing"))

    @patch.object(LobeHubSource, "_fetch_index")
    def test_inspect_returns_none_when_no_index(self, mock_fetch_index):
        mock_fetch_index.return_value = None
        self.assertIsNone(self.src.inspect("lobehub/web-research"))

    @patch.object(LobeHubSource, "_fetch_index")
    def test_inspect_returns_none_when_agents_not_list(self, mock_fetch_index):
        mock_fetch_index.return_value = {"agents": "no"}
        self.assertIsNone(self.src.inspect("lobehub/web-research"))


# ---------------------------------------------------------------------------
# LobeHubSource.fetch
# ---------------------------------------------------------------------------

class TestLobeHubFetch(unittest.TestCase):
    def setUp(self):
        self.src = LobeHubSource()
        self._agent = {
            "identifier": "web-research",
            "meta": {"title": "Web Research", "description": "Searches the web", "tags": ["research"]},
            "config": {"systemRole": "You research the web."},
        }

    @patch.object(LobeHubSource, "_fetch_agent")
    def test_fetch_strips_prefix_and_converts(self, mock_fetch_agent):
        mock_fetch_agent.return_value = self._agent
        bundle = self.src.fetch("lobehub/web-research")
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.name, "web-research")
        self.assertEqual(bundle.identifier, "lobehub/web-research")
        self.assertIn("SKILL.md", bundle.files)
        self.assertIn("You research the web.", bundle.files["SKILL.md"])
        mock_fetch_agent.assert_called_once_with("web-research")

    @patch.object(LobeHubSource, "_fetch_agent")
    def test_fetch_passes_bare_identifier(self, mock_fetch_agent):
        mock_fetch_agent.return_value = self._agent
        bundle = self.src.fetch("web-research")
        self.assertIsNotNone(bundle)
        mock_fetch_agent.assert_called_once_with("web-research")

    @patch.object(LobeHubSource, "_fetch_agent")
    def test_fetch_returns_none_when_no_agent_data(self, mock_fetch_agent):
        mock_fetch_agent.return_value = None
        self.assertIsNone(self.src.fetch("lobehub/web-research"))


# ---------------------------------------------------------------------------
# LobeHubSource._fetch_index
# ---------------------------------------------------------------------------

class TestLobeHubFetchIndex(unittest.TestCase):
    def setUp(self):
        self.src = LobeHubSource()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_success_fetches_and_caches(self, mock_get, _mock_read, mock_write):
        index = {"agents": [{"identifier": "x"}]}
        mock_get.return_value = _MockResponse(status_code=200, json_data=index)
        self.assertEqual(self.src._fetch_index(), index)
        mock_write.assert_called_once_with("lobehub_index", index)

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_non_200_returns_none(self, mock_get, _mock_read, mock_write):
        mock_get.return_value = _MockResponse(status_code=500, json_data={})
        self.assertIsNone(self.src._fetch_index())
        mock_write.assert_not_called()

    @patch("tools.skills_hub._write_index_cache")
    @patch("tools.skills_hub._read_index_cache", return_value=None)
    @patch("tools.skills_hub.httpx.get")
    def test_http_error_returns_none(self, mock_get, _mock_read, mock_write):
        import httpx
        mock_get.side_effect = httpx.ConnectError("down")
        self.assertIsNone(self.src._fetch_index())
        mock_write.assert_not_called()

    @patch("tools.skills_hub.httpx.get")
    @patch("tools.skills_hub._read_index_cache")
    def test_uses_cache_without_http(self, mock_read, mock_get):
        cached = {"agents": [{"identifier": "cached"}]}
        mock_read.return_value = cached
        self.assertEqual(self.src._fetch_index(), cached)
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# LobeHubSource._fetch_agent
# ---------------------------------------------------------------------------

class TestLobeHubFetchAgent(unittest.TestCase):
    def setUp(self):
        self.src = LobeHubSource()

    @patch("tools.skills_hub.httpx.get")
    def test_success_returns_json(self, mock_get):
        agent = {"identifier": "x", "meta": {"title": "X"}}
        mock_get.return_value = _MockResponse(status_code=200, json_data=agent)
        self.assertEqual(self.src._fetch_agent("x"), agent)
        self.assertTrue(mock_get.call_args.args[0].endswith("/x.json"))

    @patch("tools.skills_hub.httpx.get")
    def test_non_200_returns_none(self, mock_get):
        mock_get.return_value = _MockResponse(status_code=404, json_data={})
        self.assertIsNone(self.src._fetch_agent("x"))

    @patch("tools.skills_hub.httpx.get")
    def test_http_error_returns_none(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.ConnectError("down")
        self.assertIsNone(self.src._fetch_agent("x"))

    @patch("tools.skills_hub.httpx.get")
    def test_json_decode_error_returns_none(self, mock_get):
        resp = _MockResponse(status_code=200)
        resp._json_error = json.JSONDecodeError("bad", "doc", 0)
        mock_get.return_value = resp
        self.assertIsNone(self.src._fetch_agent("x"))


# ---------------------------------------------------------------------------
# LobeHubSource._convert_to_skill_md — no system-role branch
# ---------------------------------------------------------------------------

class TestLobeHubConvertSkillMd(unittest.TestCase):
    def test_no_system_role_uses_placeholder(self):
        agent_data = {"identifier": "bare", "meta": {"title": "Bare", "description": "d", "tags": ["t"]}}
        result = LobeHubSource._convert_to_skill_md(agent_data)
        self.assertIn("(No system role defined)", result)
        self.assertIn("name: bare", result)
        self.assertIn("tags: [t]", result)

    def test_non_list_tags_tolerated(self):
        agent_data = {
            "identifier": "x",
            "meta": {"title": "X", "description": "d", "tags": "not-a-list"},
            "config": {"systemRole": "role"},
        }
        result = LobeHubSource._convert_to_skill_md(agent_data)
        self.assertIn("tags: []", result)
        self.assertIn("role", result)


if __name__ == "__main__":
    unittest.main()
