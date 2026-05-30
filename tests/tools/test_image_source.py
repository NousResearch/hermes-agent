"""Unified vision image-source resolver."""
import base64

import pytest

from tools.image_source import (
    NotAnImage,
    ResolveContext,
    ResolvedImage,
    SourceNotFound,
    SourceTooLarge,
    SourceUnsafe,
    UnsupportedScheme,
    resolve_image_source,
)

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.mark.asyncio
async def test_data_url_resolves_to_bytes():
    res = await resolve_image_source(f"data:image/png;base64,{PNG_B64}", ResolveContext())
    assert isinstance(res, ResolvedImage)
    assert res.mime == "image/png"
    assert res.origin == "data"
    assert res.data == base64.b64decode(PNG_B64)


@pytest.mark.asyncio
async def test_data_url_non_image_rejected():
    with pytest.raises(NotAnImage):
        await resolve_image_source("data:text/plain;base64,aGk=", ResolveContext())


@pytest.mark.asyncio
async def test_data_url_oversize_rejected():
    big = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * (40 * 1024 * 1024)).decode()
    with pytest.raises(SourceTooLarge):
        await resolve_image_source(f"data:image/png;base64,{big}", ResolveContext())


@pytest.mark.asyncio
async def test_unknown_scheme_rejected():
    with pytest.raises(UnsupportedScheme):
        await resolve_image_source("s3://bucket/x.png", ResolveContext())


@pytest.mark.asyncio
async def test_blank_source_rejected():
    with pytest.raises(Exception):
        await resolve_image_source("   ", ResolveContext())


@pytest.mark.asyncio
async def test_http_url_downloads_bytes(monkeypatch):
    from tools import image_source

    png = base64.b64decode(PNG_B64)

    async def fake_download(url):
        return png

    monkeypatch.setattr(image_source, "_is_safe_http", lambda u: True)
    monkeypatch.setattr(image_source, "_download_to_bytes", fake_download)
    res = await resolve_image_source("https://ex.com/a.png", ResolveContext())
    assert res.origin == "http"
    assert res.data == png


@pytest.mark.asyncio
async def test_http_url_ssrf_blocked(monkeypatch):
    from tools import image_source

    monkeypatch.setattr(image_source, "_is_safe_http", lambda u: False)
    with pytest.raises(SourceUnsafe):
        await resolve_image_source("http://169.254.169.254/x.png", ResolveContext())


@pytest.mark.asyncio
async def test_file_uri_resolves(tmp_path):
    img = tmp_path / "cache" / "images" / "a.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(base64.b64decode(PNG_B64))
    res = await resolve_image_source(f"file://{img}", ResolveContext())
    assert res.origin == "file"
    assert res.mime == "image/png"


@pytest.mark.asyncio
async def test_local_path_outside_cache_allowed_in_pr1(tmp_path):
    # PR 1 preserves today's permissive behavior; the allowlist (PR 2) is a seam.
    img = tmp_path / "Pictures" / "cat.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(base64.b64decode(PNG_B64))
    res = await resolve_image_source(str(img), ResolveContext())
    assert res.origin == "file"


@pytest.mark.asyncio
async def test_tilde_path_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "shot.png"
    img.write_bytes(base64.b64decode(PNG_B64))
    res = await resolve_image_source("~/shot.png", ResolveContext())
    assert res.origin == "file"


@pytest.mark.asyncio
async def test_missing_local_file_not_found(tmp_path):
    with pytest.raises(SourceNotFound):
        await resolve_image_source(str(tmp_path / "nope.png"), ResolveContext())


class _FakeEnv:
    def __init__(self, files):
        self.files = files

    def execute(self, command, cwd=None, **kw):
        path = command.split()[1]  # base64 <path> | tr -d '\n'
        if path.strip("'\"") in self.files:
            return {"output": base64.b64encode(self.files[path.strip("'\"")]).decode(), "returncode": 0}
        return {"output": "", "returncode": 1}


@pytest.mark.asyncio
async def test_container_path_exec_read_fallback(monkeypatch):
    from tools import image_source

    png = base64.b64decode(PNG_B64)
    monkeypatch.setattr(image_source, "_get_active_env",
                        lambda tid: _FakeEnv({"/workspace/shot.png": png}))
    monkeypatch.setattr(image_source, "_maybe_translate_container_path", lambda p, ctx: p)
    res = await resolve_image_source("/workspace/shot.png", ResolveContext(task_id="t1"))
    assert res.origin == "container"
    assert res.data == png


@pytest.mark.asyncio
async def test_container_fallback_fails_closed_without_env(monkeypatch):
    from tools import image_source

    monkeypatch.setattr(image_source, "_get_active_env", lambda tid: None)
    monkeypatch.setattr(image_source, "_maybe_translate_container_path", lambda p, ctx: p)
    with pytest.raises(SourceNotFound):
        await resolve_image_source("/workspace/x.png", ResolveContext(task_id="t1"))


@pytest.mark.asyncio
async def test_container_exec_read_nonimage_rejected(monkeypatch):
    from tools import image_source

    monkeypatch.setattr(image_source, "_get_active_env",
                        lambda tid: _FakeEnv({"/workspace/x.png": b"not an image"}))
    monkeypatch.setattr(image_source, "_maybe_translate_container_path", lambda p, ctx: p)
    with pytest.raises(NotAnImage):
        await resolve_image_source("/workspace/x.png", ResolveContext(task_id="t1"))
