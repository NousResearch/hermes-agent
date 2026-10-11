"""Unit tests for tools.neutts_synth error messaging (no neutts dependency)."""

from tools.neutts_synth import _format_neutts_import_error


def test_format_neutts_import_error_includes_exception_text():
    exc = ImportError("dlopen(libtorch.dylib): Library not loaded")
    msg = _format_neutts_import_error(exc)
    assert "dlopen(libtorch.dylib): Library not loaded" in msg
    assert "neutts import failed" in msg
    assert "hermes setup tts" in msg


def test_format_neutts_import_error_for_missing_package():
    exc = ImportError("No module named 'neutts'")
    msg = _format_neutts_import_error(exc)
    assert "No module named 'neutts'" in msg
    assert "If genuinely not installed" in msg
    assert "hermes setup tts" in msg
