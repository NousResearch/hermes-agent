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


# Minimal valid 1x1 PNG bytes. Resolver validation requires a decodable fixture.
PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
JPEG = b"\xff\xd8\xff" + b"\x00" * 64
CORRUPT_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFElEQVR4nGP8z8Dwn4EIwESJ5gAAVQ4CH1evYJQAAAAASUVORK5CYII="
)


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    importlib.reload(isrc)
    return isrc


@pytest.fixture(autouse=True)
def _no_real_sandbox_bringup(monkeypatch):
    """Neutralize the resolver's lazy sandbox bring-up (issue #62825) so unit
    tests never spawn a real ssh/docker env. Patched on terminal_tool (which
    _reload does not touch) and resolved at call time, so it survives the
    per-test image_source reload. The bring-up tests override it."""
    import tools.terminal_tool as tt
    monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)


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

    @pytest.mark.asyncio
    async def test_non_image_data_url_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(isrc.NotAnImage):
            await isrc.resolve_image_source(
                f"data:text/plain;base64,{b64}", isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_corrupt_png_rejected_at_resolver_boundary(self, tmp_path, monkeypatch):
        """A PNG-shaped but undecodable payload never becomes a resolved image."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "corrupt.png"
        img.write_bytes(CORRUPT_PNG)
        with pytest.raises(isrc.NotAnImage):
            await isrc.resolve_image_source(str(img), isrc.ResolveContext())


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
    async def test_desktop_upload_images_dir_host_read(self, tmp_path, monkeypatch):
        """Desktop/clipboard uploads under ``HERMES_HOME/images`` are host-read.

        Regression for #69575: uploads land in the flat top-level ``images/``
        dir (not ``cache/images``). Under a sandbox backend the vision resolver
        must permit reading them host-side — otherwise it falls through to the
        task-id-less sandbox reader and fails with "not reachable inside the
        sandbox".
        """
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        upload = home / "images" / "upload_20260722_181019_1.png"
        upload.parent.mkdir(parents=True)
        upload.write_bytes(PNG)
        # No sandbox env: an uploads path must be host-read directly, not routed
        # to the in-sandbox exec-read.
        res = await isrc.resolve_image_source(str(upload), isrc.ResolveContext())
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

    @pytest.mark.asyncio
    async def test_exec_read_retries_cold_start_then_succeeds(self, tmp_path, monkeypatch):
        """#76566: under Docker, vision's first exec-read can fail (cold
        container / pipe setup) and an identical retry succeeds. The
        resolver must transparently retry before raising, so users don't
        see 'could not read inside the sandbox' on a file that is fully
        readable on the second attempt."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        calls = {"n": 0}
        b64 = base64.b64encode(PNG).decode()

        def fake_execute(cmd, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                # First call: cold start — empty pipe, exit non-zero.
                return {"returncode": 1, "output": ""}
            return {"returncode": 0, "output": b64}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            res = await isrc.resolve_image_source(
                "/workspace/cold.png", isrc.ResolveContext(task_id="t1"))
        assert res.origin == "container"
        assert res.data == PNG
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_exec_read_retries_exhausted_includes_diagnostic(
        self, tmp_path, monkeypatch
    ):
        """#76566: when every retry still fails, the error must carry the
        container's stderr/stdout so the user can tell 'no such file'
        from 'permission denied' from 'cold start never came up'."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        def fake_execute(cmd, **kw):
            return {"returncode": 1, "output": "head: can't open '/x': No such file or directory"}

        with patch("tools.image_source._get_active_env",
                   return_value=SimpleNamespace(execute=fake_execute)):
            with pytest.raises(isrc.SourceNotFound) as excinfo:
                await isrc.resolve_image_source(
                    "/workspace/missing.png", isrc.ResolveContext(task_id="t1"))
        # Diagnostic surfaced — the user can act on it.
        assert "No such file or directory" in str(excinfo.value)


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


class TestLazySandboxBringUp:
    """Issue #62825: under a non-local backend, the FIRST vision_analyze of a
    session (before any terminal command) must bring the sandbox up itself
    instead of failing with 'no active sandbox session'."""

    @pytest.mark.asyncio
    async def test_first_read_brings_up_sandbox_then_reads(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "ssh")

        brought_up = []
        fake_env = SimpleNamespace(
            execute=lambda cmd, **kw: {"returncode": 0, "output": base64.b64encode(PNG).decode()}
        )

        def fake_ensure(task_id):
            brought_up.append(task_id)

        # Env is absent until the lazy bring-up runs, then available — exactly
        # the SSH-handshake ordering the bug was about.
        def fake_get_active(task_id):
            return fake_env if brought_up else None

        import tools.terminal_tool as tt
        monkeypatch.setattr(tt, "ensure_task_env", fake_ensure)
        monkeypatch.setattr(isrc, "_get_active_env", fake_get_active)

        res = await isrc.resolve_image_source("/tmp/test.png", isrc.ResolveContext(task_id="t1"))

        assert brought_up == ["t1"]  # bring-up was triggered before the read
        assert res.origin == "container"
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_bringup_that_yields_no_env_still_fails_closed(self, tmp_path, monkeypatch):
        """If the bring-up can't produce an env, the resolver still refuses
        rather than falling back to a host read."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY")

        import tools.terminal_tool as tt
        monkeypatch.setattr(tt, "ensure_task_env", lambda *_a, **_k: None)
        monkeypatch.setattr(isrc, "_get_active_env", lambda *_a, **_k: None)

        with pytest.raises(isrc.SourceNotFound):
            await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))
