"""Confinement interplay between the generation chokepoint and the backends.

Every image-gen backend resolves its source images through the sanctioned
resolver (tools.image_source). Under a non-local terminal backend the handler
chokepoint (_confine_source_images) has already converted path-like sources
to data: URLs, so per-backend resolution must degrade to pure
decode-sniff-cap validation: no sandbox exec, no download, one decode. And a
path that does slip to a backend under a sandbox with no env must fail
closed rather than fall back to a host read.
"""

import base64
import importlib

import pytest

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _fal(src):
    import tools.image_generation_tool as igt

    return igt._resolve_fal_source_image(src, managed=False)


def _xai(src):
    from plugins.image_gen.xai import _xai_image_field

    return _xai_image_field(src)["url"]


def _krea(src):
    from plugins.image_gen.krea import _resolve_style_refs

    return _resolve_style_refs([src])[0]["url"]


def _openai(src):
    from plugins.image_gen.openai import _load_image_bytes

    return _load_image_bytes(src)[0]


def _codex(src):
    codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")

    return codex_plugin._to_input_image_part(src)["image_url"]


_WRAPPERS = {
    "fal": _fal,
    "xai": _xai,
    "krea": _krea,
    "openai": _openai,
    "codex": _codex,
}


@pytest.fixture(autouse=True)
def _no_real_sandbox(monkeypatch):
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_real_fal(monkeypatch):
    import tools.image_generation_tool as igt

    saved = igt.fal_client
    monkeypatch.setattr(igt, "fal_client", object())
    yield
    igt.fal_client = saved


@pytest.mark.parametrize("backend", sorted(_WRAPPERS))
class TestDataUrlUnderSandbox:
    def test_single_decode_no_exec_no_download(self, backend, monkeypatch, tmp_path):
        """A data: source under docker is validated in-process: the sandbox
        exec-read and the downloader must never run, and the payload is
        decoded exactly once."""
        import tools.image_source as isrc

        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))

        async def _no_exec(*a, **k):
            raise AssertionError("sandbox exec-read must not run for a data: URL")

        async def _no_download(*a, **k):
            raise AssertionError("download must not run for a data: URL")

        monkeypatch.setattr(isrc, "_resolve_container_fallback", _no_exec)
        monkeypatch.setattr(isrc, "_download_to_bytes", _no_download)

        decodes = {"n": 0}
        real_resolve_data_url = isrc._resolve_data_url

        def _counting(s, max_bytes=None):
            decodes["n"] += 1
            return real_resolve_data_url(s, max_bytes)

        monkeypatch.setattr(isrc, "_resolve_data_url", _counting)

        out = _WRAPPERS[backend](_data_url(PNG))

        assert decodes["n"] == 1
        if isinstance(out, bytes):
            assert out == PNG
        else:
            assert base64.b64decode(out.split(",", 1)[1]) == PNG


@pytest.mark.parametrize("backend", sorted(_WRAPPERS))
class TestFailClosedBackstop:
    def test_bare_path_without_sandbox_env_errors(self, backend, monkeypatch, tmp_path):
        """A readable host path outside the media caches, under a sandbox with
        no active env, must error instead of leaking the host bytes."""
        import tools.image_source as isrc

        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        monkeypatch.setattr(isrc, "_get_active_env", lambda tid: None)

        # A real image on the host, so a naive host read WOULD succeed.
        marker = b"HOST-ONLY-PIXELS"
        host_img = tmp_path / "host.png"
        host_img.write_bytes(PNG + marker)

        with pytest.raises(ValueError) as excinfo:
            _WRAPPERS[backend](str(host_img))

        assert marker.decode("latin-1") not in str(excinfo.value)


class TestCapComposition:
    def test_backend_cap_fires_inside_the_ingest_budget(self, monkeypatch, tmp_path):
        """A cache-resident file under docker passes confinement (host read is
        permitted, within the 50MB ingest budget) and is then rejected by the
        Codex per-backend cap with the backend's own message."""
        import tools.image_source as isrc

        codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")

        monkeypatch.setenv("TERMINAL_ENV", "docker")
        cache_dir = tmp_path / "h" / "cache" / "images"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(isrc, "_media_cache_roots", lambda: [cache_dir.parent])

        big = cache_dir / "big.png"
        big.write_bytes(PNG + b"\x00" * 4096)
        monkeypatch.setattr(codex_plugin, "_MAX_INPUT_IMAGE_BYTES", 1024)

        with pytest.raises(isrc.SourceTooLarge, match="exceeds the .*MB limit"):
            codex_plugin._to_input_image_part(str(big))
