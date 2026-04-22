"""Tests for the NacosSkillSource adapter."""
from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.nacos_cli_client import NacosNotFound, NacosSkillEntry
from tools.skills_nacos import (
    DEFAULT_GROUP,
    DEFAULT_NAMESPACE,
    NacosIdentifier,
    NacosSkillSource,
    parse_nacos_identifier,
)


# ------------------------------------------------------------------ parser

def test_parse_bare_name_uses_defaults():
    ident = parse_nacos_identifier("code-review")
    assert ident == NacosIdentifier(DEFAULT_NAMESPACE, DEFAULT_GROUP, "code-review", None)


def test_parse_full_scheme():
    ident = parse_nacos_identifier("nacos://team-a/my-group/code-review@1.2.0")
    assert ident == NacosIdentifier("team-a", "my-group", "code-review", "1.2.0")


def test_parse_scheme_without_version():
    ident = parse_nacos_identifier("nacos://team-a/hermes-skills/foo")
    assert ident.version is None


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_nacos_identifier("")


def test_parse_malformed_raises():
    with pytest.raises(ValueError):
        parse_nacos_identifier("nacos://onlyname")


def test_canonical_roundtrip():
    ident = parse_nacos_identifier("nacos://x/y/z@1.0.0")
    assert ident.canonical() == "nacos://x/y/z@1.0.0"
    bare = parse_nacos_identifier("nacos://x/y/z")
    assert bare.canonical() == "nacos://x/y/z"


# ------------------------------------------------------------------ source id + trust

def test_source_id():
    assert NacosSkillSource(client=MagicMock()).source_id() == "nacos"


def test_trust_level_for_trusted_ns():
    src = NacosSkillSource(client=MagicMock(), trusted_namespaces=["team-a"])
    assert src.trust_level_for("nacos://team-a/hermes-skills/x") == "trusted"
    assert src.trust_level_for("nacos://other/hermes-skills/x") == "community"
    assert src.trust_level_for("invalid") == "community"


# ------------------------------------------------------------------ search

def _entry(name="foo", version="1.0.0", namespace="public"):
    return NacosSkillEntry(
        name=name, namespace=namespace, group="hermes-skills", version=version,
        description="desc", author="a", updated_at="t", checksum="sha256:xx",
    )


def test_search_delegates_and_translates():
    client = MagicMock()
    client.list_skills.return_value = [_entry("code-review"), _entry("docs-gen")]
    src = NacosSkillSource(client=client)
    results = src.search("code", limit=5)
    assert len(results) == 2
    assert results[0].name == "code-review"
    assert results[0].source == "nacos"
    assert results[0].trust_level == "community"
    client.list_skills.assert_called_once()
    call = client.list_skills.call_args.kwargs
    assert call["query"] == "code"
    assert call["limit"] == 5


def test_search_returns_empty_on_error():
    client = MagicMock()
    client.list_skills.side_effect = Exception("boom")
    # Subclass of NacosCliError is required
    from tools.nacos_cli_client import NacosCliError
    client.list_skills.side_effect = NacosCliError("boom")
    src = NacosSkillSource(client=client)
    assert src.search("x") == []


# ------------------------------------------------------------------ inspect

def test_inspect_returns_none_on_not_found():
    client = MagicMock()
    client.list_skills.side_effect = NacosNotFound("404")
    assert NacosSkillSource(client=client).inspect("nacos://x/y/z") is None


def test_inspect_success():
    client = MagicMock()
    client.list_skills.return_value = [_entry("code-review", "1.2.0")]
    meta = NacosSkillSource(client=client).inspect("nacos://public/hermes-skills/code-review")
    assert meta is not None
    assert meta.identifier == "nacos://public/hermes-skills/code-review"
    assert meta.extra.get("version") == "1.2.0"


def test_inspect_invalid_ident_returns_none():
    assert NacosSkillSource(client=MagicMock()).inspect("nacos://bad") is None


# ------------------------------------------------------------------ fetch

def _make_zip(tmp_path: Path, name: str = "code-review.zip") -> Path:
    zp = tmp_path / name
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("SKILL.md", "---\nname: code-review\nversion: 1.0.0\n---\n# body\n")
        zf.writestr("scripts/run.py", "print(1)")
    return zp


def test_fetch_unzips_into_bundle(tmp_path):
    zp = _make_zip(tmp_path)
    client = MagicMock()
    client.get_skill.return_value = (zp, "sha256:abc")
    client.list_skills.return_value = [_entry("code-review")]
    bundle = NacosSkillSource(client=client).fetch(
        "nacos://public/hermes-skills/code-review"
    )
    assert bundle is not None
    assert bundle.name == "code-review"
    assert bundle.source == "nacos"
    assert bundle.identifier == "nacos://public/hermes-skills/code-review"
    assert "SKILL.md" in bundle.files
    assert bundle.files["SKILL.md"].startswith("---\nname:")
    assert "scripts/run.py" in bundle.files
    assert bundle.metadata["namespace"] == "public"


def test_fetch_returns_none_when_inspect_fails():
    client = MagicMock()
    client.list_skills.return_value = []  # no match
    assert (
        NacosSkillSource(client=client).fetch("nacos://public/hermes-skills/missing")
        is None
    )


def test_fetch_rejects_path_traversal(tmp_path):
    from tools.nacos_cli_client import NacosCliError

    zp = tmp_path / "evil.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("../../etc/passwd", "oops")
    client = MagicMock()
    client.get_skill.return_value = (zp, None)
    client.list_skills.return_value = [_entry("evil")]
    with pytest.raises(NacosCliError, match="unsafe nacos zip"):
        NacosSkillSource(client=client).fetch("nacos://public/hermes-skills/evil")


def test_fetch_rejects_absolute_path(tmp_path):
    from tools.nacos_cli_client import NacosCliError

    zp = tmp_path / "abs.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        # ZipFile normalizes leading slash; construct via ZipInfo to preserve
        info = zipfile.ZipInfo(filename="/etc/shadow")
        zf.writestr(info, "bad")
    client = MagicMock()
    client.get_skill.return_value = (zp, None)
    client.list_skills.return_value = [_entry("bad")]
    with pytest.raises(NacosCliError, match="unsafe nacos zip"):
        NacosSkillSource(client=client).fetch("nacos://public/hermes-skills/bad")


# ------------------------------------------------------------------ default source registration

def test_nacos_source_registered_when_env_set(monkeypatch):
    monkeypatch.setenv("NACOS_SERVER_ADDR", "http://nacos.example:8848")
    # Bypass GitHub auth network calls
    from tools import skills_hub
    sources = skills_hub.create_source_router()
    assert any(s.source_id() == "nacos" for s in sources)


def test_nacos_source_absent_without_env(monkeypatch):
    monkeypatch.delenv("NACOS_SERVER_ADDR", raising=False)
    from tools import skills_hub
    sources = skills_hub.create_source_router()
    assert not any(s.source_id() == "nacos" for s in sources)
