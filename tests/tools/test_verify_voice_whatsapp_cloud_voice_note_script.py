import asyncio
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_voice_whatsapp_cloud_voice_note.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_voice_whatsapp_cloud_voice_note", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_cloud_voice_note_paths():
    script = _load_script_module()

    result = asyncio.run(script.verify())

    assert result["success"] is True
    assert result["checks"]["direct_ogg"]["upload_mime"] == "audio/ogg; codecs=opus"
    assert result["checks"]["direct_ogg"]["converted"] is False
    assert result["checks"]["converted_audio"]["upload_mime"] == (
        "audio/ogg; codecs=opus"
    )
    assert result["checks"]["converted_audio"]["converted"] is True
    assert result["checks"]["converted_audio"]["temporary_removed"] is True
    assert result["checks"]["fallback_audio"]["upload_mime"] == "audio/mpeg"
