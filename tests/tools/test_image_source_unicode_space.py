"""A media path the model retyped with a plain space still opens the macOS screenshot it names.

macOS names screenshots ``Screenshot 2026-07-10 at 9.23.58<U+202F>AM.png`` (a NARROW NO-BREAK
SPACE before AM/PM). A model echoing that name back emits U+0020, so ``vision_analyze``,
``video_analyze`` and image/video generation reported "not found" for a file it had just been
shown (#76334). The retry takes only the one same-directory file that differs invisibly, and
never guesses between twins.
"""
import asyncio

import pytest

from tools.image_source import (
    ResolveContext, SourceNotFound, local_source_spelling, resolve_image_source,
)

def _png() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png()


def _resolve(path):
    return asyncio.run(resolve_image_source(str(path), ResolveContext()))


def test_flattened_screenshot_name_resolves_and_generation_gets_the_real_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    real = tmp_path / "Screenshot 2026-07-10 at 9.23.58\u202fAM.png"
    real.write_bytes(PNG)
    typed = tmp_path / "Screenshot 2026-07-10 at 9.23.58 AM.png"

    assert _resolve(typed).data == PNG
    assert local_source_spelling(str(typed)) == str(real)


def test_twins_and_missing_files_are_never_guessed(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    (tmp_path / "a\u202fb.png").write_bytes(PNG)
    (tmp_path / "a\u00a0b.png").write_bytes(PNG)

    with pytest.raises(SourceNotFound):
        _resolve(tmp_path / "a b.png")
    with pytest.raises(SourceNotFound):
        _resolve(tmp_path / "other.png")
    assert local_source_spelling(str(tmp_path / "a b.png")) == str(tmp_path / "a b.png")
