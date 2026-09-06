"""Regression: md5 call sites must declare ``usedforsecurity=False`` so they
run on FIPS-configured hosts.

On FIPS-mode Windows (and any OpenSSL built with the FIPS provider enabled),
``hashlib.md5()`` raises ``ValueError`` unless the call declares
``usedforsecurity=False``. Every md5 call in the runtime is a non-security
checksum or dedup key — QQ/WeChat/WeCom/Yuanbao protocol integrity fields,
Skills Hub cache keys, and tool-output dedup in the context compressor — so the
declaration is semantically honest as well as required for those hosts.

Contract (behavioral, through the real production functions): with a
FIPS-restricted hashlib installed (bare md5 raises, exactly the OpenSSL FIPS
provider rule), every digest path still returns the digest a normal host
computes — the flag unlocks FIPS builds without changing a single byte.
"""

import hashlib
from pathlib import Path

import pytest


@pytest.fixture()
def fips_md5(monkeypatch):
    """hashlib.md5 that mimics the OpenSSL FIPS provider: a call that does not
    declare usedforsecurity=False defaults to 'security use' and raises."""
    real_md5 = hashlib.md5

    def fips(*args, **kwargs):
        if kwargs.get("usedforsecurity", True):
            raise ValueError("[digital envelope routines] unsupported (FIPS mode)")
        kwargs.pop("usedforsecurity")
        return real_md5(*args, **kwargs)

    monkeypatch.setattr(hashlib, "md5", fips)
    return real_md5


class TestFipsModeMd5:
    def test_yuanbao_media_checksum(self, fips_md5, tmp_path):
        """The Yuanbao upload checksum runs under FIPS and matches the normal digest."""
        from gateway.platforms import yuanbao_media

        got = yuanbao_media.md5_hex(b"payload")
        assert got == fips_md5(b"payload").hexdigest()

    def test_context_compressor_tool_dedup(self, fips_md5):
        """Identical oversized tool outputs dedup under FIPS (the md5 dedup key
        path inside ContextCompressor._dedupe_tool_results)."""
        from agent.context_compressor import ContextCompressor, _PRUNE_MIN_CHARS

        big = "x" * (_PRUNE_MIN_CHARS + 10)
        results = [
            {"role": "tool", "content": big},
            {"role": "tool", "content": big},
        ]
        pruned = ContextCompressor._dedupe_tool_results(results)
        assert pruned == 1
        assert "Duplicate tool output" in results[0]["content"]
        assert results[1]["content"] == big

    def test_skills_sync_dir_hash(self, fips_md5, tmp_path):
        """The bundle-change detector hashes under FIPS."""
        from tools import skills_sync

        (tmp_path / "SKILL.md").write_text("hello", encoding="utf-8")
        h = skills_sync._dir_hash(tmp_path)
        expected = fips_md5(usedforsecurity=False)
        expected.update(b"SKILL.md")
        expected.update(b"hello")
        assert h == expected.hexdigest()
