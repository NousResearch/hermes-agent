"""Tests for tools/image_source.py — the unified vision image-source resolver.

Covers the delivery contract (data:/http/file/local/container source handling,
size cap, magic-byte sniff) AND the terminal-backend confinement security model
(GHSA-gpxw-6wxv-w3qq): under a non-local backend, host reads are confined to the
media caches and every other path is read inside the sandbox via exec-read.
"""

import base64
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff" + b"\x00" * 64


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    importlib.reload(isrc)
    return isrc


class TestDataUrl:
    @pytest.mark.asyncio
    async def test_valid_data_url_resolves_to_bytes(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(PNG).decode()
        res = await isrc.resolve_image_source(
            f"data:image/png;base64,{b64}", isrc.ResolveContext())
        assert res.data == PNG
        assert res.mime == "image/png"
        assert res.origin == "data"

class TestWindowsFileURIParsing:
    """Helper-level and end-to-end coverage for the file:// URI handler.

    The helper is the chokepoint for Windows drive letters, localhost
    authority, and remote UNC / device paths. Every assertion below is
    a direct invariant on what _file_uri_to_path produces — no filesystem
    is touched (no Path.resolve(), is_file(), stat(), read).
    """

    # --- happy paths ----------------------------------------------------

    def test_file_uri_to_path_windows(self):
        from tools.image_source import _file_uri_to_path

        # Standard file URI
        assert _file_uri_to_path("file:///C:/Users/test/image.png", windows=True) == "C:\\Users\\test\\image.png"

        # Localhost authority (case insensitive)
        assert _file_uri_to_path("file://localhost/C:/Users/test/image.png", windows=True) == "C:\\Users\\test\\image.png"
        assert _file_uri_to_path("file://LOCALHOST/C:/Users/test/image.png", windows=True) == "C:\\Users\\test\\image.png"

        # Drive-letter authority (Windows-only)
        assert _file_uri_to_path("file://C:/image.png", windows=True) == "C:\\image.png"

        # Root-relative path (single leading backslash on the current drive)
        # — legal, NOT a UNC.
        assert _file_uri_to_path("file:///images/photo.png", windows=True) == "\\images\\photo.png"
        assert _file_uri_to_path("file://localhost/images/photo.png", windows=True) == "\\images\\photo.png"

        # Encoded space -> literal space
        assert _file_uri_to_path("file:///C:/images/a%20b.png", windows=True) == "C:\\images\\a b.png"

        # Double encoded space -> literal %20 (single decode only)
        assert _file_uri_to_path("file:///C:/images/a%2520b.png", windows=True) == "C:\\images\\a%20b.png"

    def test_file_uri_to_path_posix(self):
        from tools.image_source import _file_uri_to_path

        # Standard file URI
        assert _file_uri_to_path("file:///tmp/image.png", windows=False) == "/tmp/image.png"

        # Localhost authority
        assert _file_uri_to_path("file://localhost/tmp/image.png", windows=False) == "/tmp/image.png"

        # Drive-letter authority is NOT a hostname on POSIX -> rejected
        with pytest.raises(ValueError, match="Resolving non-local authority 'C:'"):
            _file_uri_to_path("file://C:/image.png", windows=False)

        # Encoded space
        assert _file_uri_to_path("file:///tmp/a%20b.png", windows=False) == "/tmp/a b.png"

        # Double encoded space -> literal %20 (single decode only)
        assert _file_uri_to_path("file:///tmp/a%2520b.png", windows=False) == "/tmp/a%20b.png"

    # --- remote authority rejection (the simple case) -------------------

    def test_remote_authority_rejected_windows(self):
        from tools.image_source import _file_uri_to_path
        with pytest.raises(ValueError, match="Resolving non-local authority 'server'"):
            _file_uri_to_path("file://server/share/image.png", windows=True)

    def test_remote_authority_rejected_posix(self):
        from tools.image_source import _file_uri_to_path
        with pytest.raises(ValueError, match="Resolving non-local authority 'server'"):
            _file_uri_to_path("file://server/share/image.png", windows=False)

    # --- UNC / device-path bypasses that netloc-only checking misses ----

    @pytest.mark.parametrize("uri", [
        "file:////server/share/image.png",                    # empty netloc + //server
        "file://localhost//server/share/image.png",           # localhost + //server
        "file://LOCALHOST//server/share/image.png",           # localhost (case) + //server
        "file://localhost/%5C%5Cserver/share/image.png",      # localhost + %5C%5C -> \\\\
        "file:///%2Fserver/share/image.png",                  # empty netloc + %2F -> //
    ])
    def test_unc_bypass_blocked_windows(self, uri):
        """These shapes bypass the netloc check (netloc is empty or
        localhost), but produce a Windows UNC after url2pathname. They
        must all be rejected on Windows before any filesystem access."""
        from tools.image_source import _file_uri_to_path
        with pytest.raises(ValueError, match="Remote UNC and Windows device paths are not allowed"):
            _file_uri_to_path(uri, windows=True)

    @pytest.mark.parametrize("uri", [
        "file:////server/share/image.png",                    # //server decoded as //
        "file:///%2Fserver/share/image.png",                  # %2F -> //
        "file://localhost//server/share/image.png",           # //server
        "file://LOCALHOST//server/share/image.png",
    ])
    def test_double_slash_bypass_blocked_posix(self, uri):
        """POSIX shares the no-leading-// invariant: any URI that resolves
        to a path starting with ``//`` is rejected as a remote UNC."""
        from tools.image_source import _file_uri_to_path
        with pytest.raises(ValueError, match="Remote UNC and Windows device paths are not allowed"):
            _file_uri_to_path(uri, windows=False)

    # --- drive-relative paths are rejected BEFORE nturl2path ------------
    #
    # The check runs on a single-pass decoded copy of the encoded path, so
    # it does not depend on ``nturl2path.url2pathname`` behavior — which
    # changed between Python 3.11 and 3.13 (3.11 silently rewrites
    # ``C:image.png`` to ``C:\\image.png``; 3.13 returns the raw
    # drive-relative ``C:image.png``). The contract here is stable across
    # versions: any URI that decodes to ``/X:foo`` (drive + colon + no
    # separator, or end of path) is rejected.
    #
    # Percent-encoded colon forms (``%3A``) are rejected by a separate
    # check that fires on the raw encoded path *before* percent-decoding
    # (see ``test_percent_encoded_drive_delimiter_blocked``).

    @pytest.mark.parametrize("uri", [
        "file:///C:image.png",                # empty netloc, raw colon
        "file://localhost/C:image.png",       # localhost netloc, raw colon
        "file:///C:",                         # trailing colon, no body
    ])
    def test_drive_relative_blocked_windows(self, uri):
        """A drive-relative URI resolves against the current directory on
        the named drive — reject it from a vision input so we never
        silently depend on the host's cwd."""
        from tools.image_source import _file_uri_to_path
        with pytest.raises(ValueError, match="Drive-relative file URI paths are not allowed"):
            _file_uri_to_path(uri, windows=True)

    # --- percent-encoded drive delimiter rejection --------------------

    @pytest.mark.parametrize("uri", [
        "file:///C%3Aimage.png",               # encoded colon, no separator
        "file://localhost/C%3Aimage.png",      # localhost + encoded colon
        "file:///C%3A/image.png",               # encoded colon + slash
        "file:///C%3A%2Fimage.png",             # encoded colon + encoded slash
        "file://localhost/C%3A/image.png",      # localhost + encoded colon + slash
        "file://localhost/C%3A%2Fimage.png",    # localhost + encoded colon + encoded slash
    ])
    def test_percent_encoded_drive_delimiter_blocked(self, uri):
        """Percent-encoded colon in a drive-letter position produces
        version-dependent results from nturl2path — reject outright
        before reaching the stdlib."""
        from tools.image_source import _file_uri_to_path
        with pytest.raises(ValueError, match="Percent-encoded Windows drive delimiters"):
            _file_uri_to_path(uri, windows=True)

    # --- drive-absolute literals are accepted with precise output ----

    @pytest.mark.parametrize("uri, expected", [
        ("file:///C:/image.png",            "C:\\image.png"),
        ("file://localhost/C:/image.png",   "C:\\image.png"),
        ("file://C:/image.png",             "C:\\image.png"),
        ("file:///C:/image%20one.png",      "C:\\image one.png"),
    ])
    def test_drive_absolute_or_root_relative_allowed_windows(self, uri, expected):
        """Only literal ``:`` + separator forms are stable across Python
        versions — assert the exact expected path."""
        from tools.image_source import _file_uri_to_path
        assert _file_uri_to_path(uri, windows=True) == expected

    # --- "file:logo.png" must NOT enter the helper ----------------------

    @pytest.mark.asyncio
    async def test_bare_file_colon_name_is_treated_as_path(self, tmp_path, monkeypatch):
        """A POSIX filename containing a literal colon (e.g. ``file:logo.png``)
        is a legitimate local path; the URI helper must not be invoked.

        This test passes the literal string ``"file:logo.png"`` so that
        widening the entry condition back to ``startswith("file:")`` would
        route it through the URI helper and fail (the helper raises
        ValueError because ``file:logo.png`` parses with netloc=``file``
        and path=``logo.png`` — not ``file://`` — and netloc ``file`` is
        not localhost). The chdir + tmp_path layout mirrors how a real
        cwd-relative bare filename would resolve."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.chdir(tmp_path)

        img = tmp_path / "file:logo.png"
        img.write_bytes(PNG)

        res = await isrc.resolve_image_source(
            "file:logo.png",
            isrc.ResolveContext(),
        )
        assert res.data == PNG
        assert res.origin == "file"

    # --- end-to-end: UNC bypasses become SourceNotFound -----------------

    @pytest.mark.asyncio
    async def test_full_resolution_flow_remote_authority_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")

        with pytest.raises(isrc.SourceNotFound) as exc_info:
            await isrc.resolve_image_source("file://server/share/image.png", isrc.ResolveContext())

        # Verify the underlying ValueError is retained as __cause__
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Resolving non-local authority 'server'" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("uri", [
        "file:////server/share/image.png",
        "file://localhost//server/share/image.png",
        "file://LOCALHOST//server/share/image.png",
        "file://localhost/%5C%5Cserver/share/image.png",
        "file:///%2Fserver/share/image.png",
    ])
    async def test_full_resolution_flow_unc_bypass_rejected(self, tmp_path, monkeypatch, uri):
        """UNC bypasses wrapped by resolve_image_source must surface as
        SourceNotFound with a ValueError __cause__, and the rejection
        must occur BEFORE any filesystem read."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")

        with pytest.raises(isrc.SourceNotFound) as exc_info:
            await isrc.resolve_image_source(uri, isrc.ResolveContext())

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Remote UNC" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_full_resolution_flow_drive_relative_rejected(self, tmp_path, monkeypatch):
        """A drive-relative URI wrapped by resolve_image_source must
        surface as SourceNotFound with a ValueError __cause__ carrying
        the drive-relative message — never as an open file read against
        the host's cwd.

        The helper defaults to ``windows=platform.system()``, so a
        pure-POSIX runner would treat ``file:///C:image.png`` as a
        POSIX path (``/C:image.png``) and never enter the drive-relative
        rejection branch. Monkey-patch ``_file_uri_to_path`` to force
        Windows semantics so the rejection is exercised on every OS."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")

        real_helper = isrc._file_uri_to_path

        monkeypatch.setattr(
            isrc,
            "_file_uri_to_path",
            lambda uri: real_helper(uri, windows=True),
        )

        with pytest.raises(isrc.SourceNotFound) as exc_info:
            await isrc.resolve_image_source(
                "file:///C:image.png",
                isrc.ResolveContext(),
            )

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Drive-relative" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_full_resolution_flow(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")

        test_img = tmp_path / "test image.png"
        test_img.write_bytes(PNG)

        # Windows path representation of tmp_path
        file_uri = test_img.as_uri()

        # Full end-to-end integration ensures signature is compatible
        # Returned object is ResolvedImage
        res = await isrc.resolve_image_source(file_uri, isrc.ResolveContext())
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_non_image_data_url_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(isrc.NotAnImage):
            await isrc.resolve_image_source(
                f"data:text/plain;base64,{b64}", isrc.ResolveContext())


class TestLocalBackend:
    @pytest.mark.asyncio
    async def test_local_backend_reads_any_host_path(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "outside" / "pic.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_file_uri_scheme_stripped(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.jpg"
        img.write_bytes(JPEG)
        res = await isrc.resolve_image_source(f"file://{img}", isrc.ResolveContext())
        assert res.mime == "image/jpeg"

    @pytest.mark.asyncio
    async def test_bare_relative_path_resolves(self, tmp_path, monkeypatch):
        """A cwd-relative bare filename ('pic.png') is a valid local source —
        main accepted it; the resolver must not regress it (PR review)."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        monkeypatch.chdir(tmp_path)
        res = await isrc.resolve_image_source("pic.png", isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_unknown_url_scheme_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        with pytest.raises(isrc.UnsupportedScheme):
            await isrc.resolve_image_source(
                "ftp://example.com/pic.png", isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_svg_passes_through_for_rasterization(self, tmp_path, monkeypatch):
        """SVG has no raster magic bytes but is passed through with mime
        image/svg+xml so the vision call sites can rasterize it to PNG."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        svg = tmp_path / "art.svg"
        svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        svg.write_bytes(svg_bytes)
        res = await isrc.resolve_image_source(str(svg), isrc.ResolveContext())
        assert res.mime == "image/svg+xml"
        assert res.data == svg_bytes


class TestNonLocalBackendConfinement:
    """The security model: under a sandbox backend, host reads are confined to
    the media caches; every other path is read inside the sandbox."""

    @pytest.mark.asyncio
    async def test_media_cache_path_host_read(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        cached = home / "cache" / "images" / "inbound.png"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(PNG)
        # No sandbox env needed — a cache path is host-read directly.
        res = await isrc.resolve_image_source(str(cached), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_host_secret_outside_cache_routes_to_sandbox_not_host(self, tmp_path, monkeypatch):
        """A non-cache host path (e.g. /etc/passwd) must NOT be host-read — it
        routes to the in-sandbox exec-read, which reads the CONTAINER's file."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        # A real host file outside the caches, holding a "secret".
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY-DO-NOT-LEAK")

        # Fake sandbox env: its exec-read returns a *different* (container) image,
        # proving we read the container filesystem, not the host secret.
        container_png_b64 = base64.b64encode(PNG).decode()
        calls = {}

        def fake_execute(cmd, **kw):
            calls["cmd"] = cmd
            return {"returncode": 0, "output": container_png_b64}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            res = await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))

        # Read came from the sandbox exec-read, returning the container image —
        # the host secret bytes never appear.
        assert res.origin == "container"
        assert res.data == PNG
        assert b"HOST-PRIVATE-KEY" not in res.data
        assert "head -c" in calls["cmd"] and "< " in calls["cmd"]  # bounded, redirect-safe form

    @pytest.mark.asyncio
    async def test_non_cache_path_fails_closed_without_sandbox(self, tmp_path, monkeypatch):
        """No active sandbox env -> refuse rather than fall back to a host read."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY")

        with patch("tools.image_source._get_active_env", return_value=None):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))

    @pytest.mark.asyncio
    async def test_symlink_in_cache_pointing_outside_is_not_host_read(self, tmp_path, monkeypatch):
        """A symlink planted inside a cache dir that points at a host secret must
        not be host-read (resolve() escapes the cache) — it routes to sandbox."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        secret = tmp_path / "outside" / "id_rsa"
        secret.parent.mkdir(parents=True)
        secret.write_bytes(b"HOST-PRIVATE-KEY")
        cache_dir = home / "cache" / "images"
        cache_dir.mkdir(parents=True)
        link = cache_dir / "sneaky.png"
        try:
            link.symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported")

        # Fails closed (no sandbox) rather than host-reading the symlink target.
        with patch("tools.image_source._get_active_env", return_value=None):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(str(link), isrc.ResolveContext(task_id="t1"))


class TestExecReadSafety:
    @pytest.mark.asyncio
    async def test_exec_read_is_bounded_and_redirect_safe(self, tmp_path, monkeypatch):
        """Leading-dash paths go through an input redirect (no argv exposure)
        and the read is size-bounded via head -c."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        captured = {}

        def fake_execute(cmd, **kw):
            captured["cmd"] = cmd
            return {"returncode": 0, "output": base64.b64encode(PNG).decode()}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            await isrc.resolve_image_source(
                "/workspace/-i-etc-shadow.png", isrc.ResolveContext(task_id="t1"))
        assert f"head -c {isrc._MAX_INGEST_BYTES + 1} < " in captured["cmd"]
        assert "'-i-etc-shadow.png'" in captured["cmd"] or "-i-etc-shadow.png" in captured["cmd"]

    @pytest.mark.asyncio
    async def test_exec_read_over_cap_rejected(self, tmp_path, monkeypatch):
        """A sandbox file larger than the ingest cap is rejected, not embedded."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        # head -c returns cap+1 bytes for an oversized file.
        over = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * (isrc._MAX_INGEST_BYTES - 7)).decode()

        def fake_execute(cmd, **kw):
            return {"returncode": 0, "output": over}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            with pytest.raises(isrc.SourceTooLarge):
                await isrc.resolve_image_source(
                    "/workspace/huge.png", isrc.ResolveContext(task_id="t1"))

    @pytest.mark.asyncio
    async def test_exec_read_nonzero_returncode_raises(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        def fake_execute(cmd, **kw):
            return {"returncode": 1, "output": ""}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(
                    "/workspace/nope.png", isrc.ResolveContext(task_id="t1"))


class TestSvgNormalization:
    """SVG resolves end-to-end: the resolver passes it through as
    image/svg+xml and the vision call sites rasterize it to PNG via
    _normalize_to_supported_image (PR #52688, folded in)."""

    @pytest.mark.asyncio
    async def test_svg_rasterized_when_converter_available(self, tmp_path, monkeypatch):
        from tools import vision_tools as vt
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        svg = tmp_path / "art.svg"
        svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>')

        def fake_rasterize(svg_path, out_path):
            out_path.write_bytes(PNG)
            return True

        with patch.object(vt, "_rasterize_svg_to_png", side_effect=fake_rasterize):
            res = await isrc.resolve_image_source(str(svg), isrc.ResolveContext())
            assert res.mime == "image/svg+xml"
            path, mime, err = vt._normalize_to_supported_image(svg, "image/svg+xml")
        assert err is None
        assert mime == "image/png"
        assert path.read_bytes() == PNG
        path.unlink()

    def test_svg_actionable_error_when_no_converter(self, tmp_path, monkeypatch):
        from tools import vision_tools as vt
        _reload(monkeypatch, tmp_path / "hermes")
        svg = tmp_path / "art.svg"
        svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
        with patch.object(vt, "_rasterize_svg_to_png", return_value=False):
            path, mime, err = vt._normalize_to_supported_image(svg, "image/svg+xml")
        assert path is None
        assert "rasterizer" in err
