"""A media path the model retyped with a plain space still opens the macOS screenshot it names.

macOS names screenshots ``Screenshot 2026-07-10 at 9.23.58<U+202F>AM.png`` (a NARROW NO-BREAK
SPACE before AM/PM). A model echoing that name back emits U+0020, so ``vision_analyze``,
``video_analyze`` and image/video generation reported "not found" for a file it had just been
shown (#76334). The retry takes only the one same-directory file that differs invisibly, and
never guesses between twins.
"""
import asyncio

import pytest

from hermes_agent.tools.image_source import (
    ResolveContext, SourceNotFound, resolve_image_source,
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


def test_flattened_screenshot_name_opens_in_vision_video_and_generation(tmp_path, monkeypatch):
    from hermes_agent.tools.image_generation_tool import _confine_source_images
    from hermes_agent.tools.vision_tools import _materialize_video

    monkeypatch.setenv("TERMINAL_ENV", "local")
    real = tmp_path / "Screenshot 2026-07-10 at 9.23.58\u202fAM.png"
    real.write_bytes(PNG)
    typed = str(tmp_path / "Screenshot 2026-07-10 at 9.23.58 AM.png")
    clip = tmp_path / "Recording at 9.23\u202fAM.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    assert _resolve(typed).data == PNG
    assert _confine_source_images(typed, [typed, "https://x/y.png"], None) == (
        str(real), [str(real), "https://x/y.png"], None)
    assert asyncio.run(_materialize_video(
        str(tmp_path / "Recording at 9.23 AM.mp4"), None, [])) == clip


def test_twins_and_missing_files_are_never_guessed(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    (tmp_path / "a\u202fb.png").write_bytes(PNG)
    (tmp_path / "a\u00a0b.png").write_bytes(PNG)

    with pytest.raises(SourceNotFound):
        _resolve(tmp_path / "a b.png")
    with pytest.raises(SourceNotFound):
        _resolve(tmp_path / "other.png")
    from hermes_agent.tools.image_source import local_source_spelling
    assert local_source_spelling(str(tmp_path / "a b.png")) == str(tmp_path / "a b.png")
