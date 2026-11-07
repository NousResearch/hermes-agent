"""Tests for the generation-tool source-image confinement chokepoint.

Under a non-local terminal backend, model-supplied local paths passed to
image_generate / video_generate must resolve through the sandbox-aware media
resolver (tools.image_source) and reach providers as data: URLs — the same
boundary vision/video analysis enforce. URLs pass through untouched; on the
local backend image_generate is a no-op while video_generate still inlines.
"""

import base64
import json

import pytest

import tools.image_generation_tool as igt

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def _no_real_sandbox(monkeypatch):
    import tools.terminal_tool as tt

    monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)


class TestConfineSourceImages:
    def test_local_backend_is_passthrough(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        url, refs, err = igt._confine_source_images(
            "/some/host/pic.png", ["/other/ref.png"], "t1")
        assert url == "/some/host/pic.png"
        assert refs == ["/other/ref.png"]
        assert err is None

    def test_urls_pass_through_under_sandbox(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
        url, refs, err = igt._confine_source_images(
            "https://x/y.png", ["data:image/png;base64,AAAA"], "t1")
        assert url == "https://x/y.png"
        assert refs == ["data:image/png;base64,AAAA"]
        assert err is None

    def test_path_resolves_to_data_url_under_sandbox(self, monkeypatch, tmp_path):
        """A path under docker resolves through the sandbox exec-read and
        arrives as a data: URL carrying the CONTAINER's bytes."""
        from types import SimpleNamespace

        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))

        import tools.image_source as isrc

        b64 = base64.b64encode(PNG).decode()
        monkeypatch.setattr(
            isrc, "_get_active_env",
            lambda tid: SimpleNamespace(
                execute=lambda cmd, **kw: {"returncode": 0, "output": b64}),
        )

        url, refs, err = igt._confine_source_images(
            "/workspace/pic.png", None, "t1")
        assert err is None
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG
        assert refs is None

    def test_unreadable_path_returns_error_payload(self, monkeypatch, tmp_path):
        """No sandbox env + non-cache path -> structured error, not a host read."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))

        import tools.image_source as isrc

        monkeypatch.setattr(isrc, "_get_active_env", lambda tid: None)

        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY")
        _url, _refs, err = igt._confine_source_images(str(secret), None, "t1")
        assert err is not None
        payload = json.loads(err)
        assert payload["success"] is False
        assert "Could not read source image" in payload["error"]
        # The host secret's bytes never left the chokepoint.
        assert "HOST-PRIVATE-KEY" not in err

    def test_handler_rejects_before_provider_dispatch(self, monkeypatch, tmp_path):
        """_handle_image_generate returns the confinement error without ever
        reaching plugin/FAL dispatch."""
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))

        import tools.image_source as isrc

        monkeypatch.setattr(isrc, "_get_active_env", lambda tid: None)

        dispatched = []
        monkeypatch.setattr(
            igt, "_dispatch_to_plugin_provider",
            lambda *a, **k: dispatched.append(1) or None)

        out = igt._handle_image_generate(
            {"prompt": "edit it", "image_url": str(tmp_path / "nope.png")},
            task_id="t1",
        )
        payload = json.loads(out)
        assert payload["success"] is False
        assert dispatched == []

    def test_video_generate_uses_same_chokepoint(self, monkeypatch, tmp_path):
        """video_generate's handler routes its image sources through the
        shared confinement helper too."""
        import tools.video_generation_tool as vgt

        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))

        import tools.image_source as isrc

        monkeypatch.setattr(isrc, "_get_active_env", lambda tid: None)

        out = vgt._handle_video_generate(
            {"prompt": "animate", "image_url": str(tmp_path / "nope.png")},
            task_id="t1",
        )
        payload = json.loads(out)
        assert payload["success"] is False
        assert "Could not read source image" in payload["error"]

    def test_video_generate_inlines_local_paths_on_local_backend(self, monkeypatch, tmp_path):
        """URL-only video backends (FAL, DeepInfra) never see a raw host path on the local
        backend: the provider gets a data: URL, and secret files are still refused."""
        import tools.video_generation_tool as vgt

        monkeypatch.setenv("TERMINAL_ENV", "local")
        home = tmp_path / "h"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        seen = []

        class _Provider:
            name = "fake"

            def default_model(self):
                return "m"

            def generate(self, prompt, **kw):
                seen.append(kw.get("image_url"))
                return {"success": True, "video": "https://x/v.mp4"}

        monkeypatch.setattr(vgt, "_resolve_active_provider", lambda: _Provider())
        pic = tmp_path / "pic.png"
        pic.write_bytes(PNG)
        payload = json.loads(vgt._handle_video_generate({"prompt": "animate", "image_url": str(pic)}, task_id="t1"))
        assert payload["success"] is True
        assert base64.b64decode(seen[0].split(",", 1)[1]) == PNG and seen[0].startswith("data:image/png;base64,")

        (home / ".env").write_text("OPENAI_API_KEY=HOST-PRIVATE-KEY\n")
        out = vgt._handle_video_generate({"prompt": "animate", "image_url": str(home / ".env")}, task_id="t1")
        assert json.loads(out)["success"] is False and len(seen) == 1
        assert "HOST-PRIVATE-KEY" not in out
