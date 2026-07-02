"""Tests for xAI Imagine storage helper behavior."""

from __future__ import annotations

import yaml


def _invalidate_config_cache():
    try:
        import hermes_cli.config as cfg_mod

        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass


def test_storage_defaults_to_permanent_public_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "image_gen",
        filename_prefix="hermes-xai-image",
        extension="png",
    )

    assert storage is not None
    assert storage["public_url"] is True
    assert "expires_after" not in storage
    assert storage["filename"].startswith("hermes-xai-image-")
    assert storage["filename"].endswith(".png")


def test_storage_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "video_gen": {
            "xai": {
                "storage": {
                    "enabled": False,
                },
            },
        },
    }))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options, xai_storage_notice_text

    assert build_xai_storage_options(
        "video_gen",
        filename_prefix="hermes-xai-video",
        extension="mp4",
    ) is None
    assert xai_storage_notice_text("video_gen") == ""


def test_storage_can_be_permanent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "image_gen": {
            "xai": {
                "storage": {
                    "expires_after": "permanent",
                },
            },
        },
    }))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "image_gen",
        filename_prefix="hermes-xai-image",
        extension="png",
    )

    assert storage is not None
    assert "expires_after" not in storage


def test_storage_can_use_finite_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "image_gen": {
            "xai": {
                "storage": {
                    "expires_after": 172800,
                },
            },
        },
    }))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "image_gen",
        filename_prefix="hermes-xai-image",
        extension="png",
    )

    assert storage is not None
    assert storage["expires_after"] == 172800


def test_invalid_storage_retention_falls_back_to_bounded_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "video_gen": {
            "xai": {
                "storage": {
                    "expires_after": "definitely-not-a-duration",
                },
            },
        },
    }))
    _invalidate_config_cache()

    from tools.xai_http import build_xai_storage_options

    storage = build_xai_storage_options(
        "video_gen",
        filename_prefix="hermes-xai-video",
        extension="mp4",
    )

    assert storage is not None
    assert storage["expires_after"] == 172800
