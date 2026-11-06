"""file:// URI handling in tools/image_source.py, across platform path shapes.

The resolver accepts "a file:// URI" as a documented source form, and on Windows
the RFC-correct spelling of an absolute path puts three slashes before the drive
letter: file:///C:/Users/... Stripping the literal "file://" leaves /C:/Users/...,
which Path() on Windows resolves to \\C:\\Users\\... — a path that does not exist.

The user-visible effect is that a model which writes the CORRECT URI for a Windows
path gets "media file not found", while the identical bare path works, and nothing
in the error suggests the URI form was the problem.

The two-slash spelling (file://C:/Users/...) is also covered, because the obvious
fix — routing the whole source through urllib's url2pathname — repairs the
three-slash form and BREAKS this one, by parsing the drive letter as a URI netloc
and dropping it. POSIX absolute paths are covered for the same reason: they must
survive untouched.
"""

import base64
import importlib
import sys
from pathlib import Path, PurePath

import pytest


# Minimal valid 1x1 PNG. The resolver validates that the payload decodes.
PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    return importlib.reload(isrc)


@pytest.fixture(autouse=True)
def _no_task_env(monkeypatch):
    """Keep the resolver off any real ssh/docker bring-up, per this suite's convention."""
    import tools.terminal_tool as tt
    monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)


class TestFileUriPathShapes:
    @pytest.mark.asyncio
    async def test_bare_absolute_path_resolves(self, tmp_path, monkeypatch):
        """The control. Without this passing, the URI cases below prove nothing:
        a failure could just mean the fixture or the backend is wrong."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_rfc_file_uri_resolves(self, tmp_path, monkeypatch):
        """file:///<abs> — three slashes, the RFC-correct absolute form.

        On Windows this is the case that regressed: the drive letter ends up behind
        a leading slash and the file is never found.
        """
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        uri = "file:///" + PurePath(img).as_posix().lstrip("/")
        res = await isrc.resolve_image_source(uri, isrc.ResolveContext())
        assert res.data == PNG

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows-only")
    async def test_two_slash_windows_uri_still_resolves(self, tmp_path, monkeypatch):
        """file://C:/... — not RFC-correct, but emitted in the wild and previously
        working. It is asserted so that a fix for the three-slash form cannot quietly
        break it; url2pathname would, by reading the drive letter as a netloc.
        """
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        uri = "file://" + PurePath(img).as_posix()
        res = await isrc.resolve_image_source(uri, isrc.ResolveContext())
        assert res.data == PNG

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX absolute-path form")
    async def test_posix_file_uri_unaffected(self, tmp_path, monkeypatch):
        """A POSIX absolute path inside a file:// URI keeps its leading slash: the
        drive-letter repair must be narrow enough not to touch it."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(
            f"file://{img.as_posix()}", isrc.ResolveContext())
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_missing_file_still_reports_not_found(self, tmp_path, monkeypatch):
        """The repair must not turn a genuine miss into something else: a URI that
        points nowhere still raises SourceNotFound."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        missing = tmp_path / "nope.png"
        uri = "file:///" + PurePath(missing).as_posix().lstrip("/")
        with pytest.raises(isrc.SourceNotFound):
            await isrc.resolve_image_source(uri, isrc.ResolveContext())
