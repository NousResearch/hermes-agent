"""Unified vision image-source resolver."""
import base64

import pytest

from tools.image_source import (
    NotAnImage,
    ResolveContext,
    ResolvedImage,
    SourceTooLarge,
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
