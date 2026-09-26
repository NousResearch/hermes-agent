"""Dimension caps should not discard resolution that fits the byte budget."""
import asyncio
import base64
from io import BytesIO

import pytest
from PIL import Image

from tools import vision_tools as vt
from tools.registry import registry


def _size(url):
    with Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
        image.load()
        return image.size


@pytest.mark.parametrize('size', [(1569, 1100), (3539, 2499), (1100, 1569)])
def test_native_embed_fits_dimension_without_extra_halving(tmp_path, monkeypatch, size):
    path = tmp_path / 'document.jpg'
    Image.new('RGB', size, 'white').save(path)
    monkeypatch.setattr(vt, '_should_use_native_vision_fast_path', lambda: True)
    result = asyncio.run(registry.get_entry('vision_analyze').handler(
        {'image_url': str(path), 'question': 'Read this document'}))
    assert isinstance(result, dict), result
    url = next(p['image_url']['url'] for p in result['content'] if p['type'] == 'image_url')
    emitted = _size(url)
    cap = vt._EMBED_MAX_DIMENSION
    assert max(emitted) == cap
    assert emitted == tuple(max(1, int(n * cap / max(size))) for n in size)
    assert len(url) <= vt._resolve_embed_target_bytes()


@pytest.mark.parametrize('cap', [800, 1600])
def test_caller_cap_and_scale_disclosure(tmp_path, cap):
    size = (1900, 1400)
    path = tmp_path / 'image.png'
    Image.new('RGB', size, 'white').save(path)
    scale = {}
    url = vt._resize_image_for_vision(path, max_dimension=cap, scale_out=scale)
    emitted = _size(url)
    assert max(emitted) == cap
    assert (scale['orig_width'], scale['orig_height']) == size
    assert (scale['new_width'], scale['new_height']) == emitted


def test_small_image_is_byte_identical(tmp_path):
    path = tmp_path / 'small.png'
    Image.new('RGB', (80, 60), 'white').save(path)
    scale = {}
    url = vt._resize_image_for_vision(path, max_dimension=100, scale_out=scale)
    assert base64.b64decode(url.split(',', 1)[1]) == path.read_bytes()
    assert not scale


def test_missing_pillow_preserves_raw_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'image.jpg'
    Image.new('RGB', (1900, 1400), 'white').save(path)
    monkeypatch.setattr(vt, '_import_pillow_for_resize', lambda: None)
    url = vt._resize_image_for_vision(path, max_dimension=800)
    assert base64.b64decode(url.split(',', 1)[1]) == path.read_bytes()


def test_byte_budget_still_downscales_after_dimension_fit(tmp_path):
    import random

    size = (1900, 1400)
    path = tmp_path / 'noise.png'
    Image.frombytes('RGB', size, random.Random(42).randbytes(size[0] * size[1] * 3)).save(path)
    budget = 64 * 1024
    scale = {}
    url = vt._resize_image_for_vision(
        path, max_dimension=1600, max_base64_bytes=budget,
        force_jpeg=True, scale_out=scale)
    assert len(url) <= budget
    assert max(_size(url)) < 1600
    assert (scale['new_width'], scale['new_height']) == _size(url)


def test_unreadable_image_preserves_raw_fallback(tmp_path):
    path = tmp_path / 'broken.jpg'
    path.write_bytes(b'not an image')
    url = vt._resize_image_for_vision(path, max_base64_bytes=1, max_dimension=800)
    assert base64.b64decode(url.split(',', 1)[1]) == path.read_bytes()


def test_byte_floor_does_not_enlarge_dimension_fitted_strip(tmp_path):
    path = tmp_path / 'strip.png'
    Image.new('RGB', (2000, 20), 'white').save(path)
    url = vt._resize_image_for_vision(path, max_dimension=1000, max_base64_bytes=1)
    assert _size(url) == (1000, 10)
