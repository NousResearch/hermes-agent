import asyncio
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_voice_whatsapp_cloud_webhook.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_voice_whatsapp_cloud_webhook", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_cloud_webhook_http_smoke():
    script = _load_script_module()

    result = asyncio.run(script.verify())

    assert result["success"] is True
    assert result["checks"]["health"]["verify_token_configured"] is True
    assert result["checks"]["health"]["app_secret_configured"] is True
    assert result["checks"]["verify_handshake"] == {
        "status": 200,
        "challenge_echoed": True,
    }
    assert result["checks"]["signed_post"] == {
        "status": 200,
        "payload": "status_delivery_receipt",
        "signature_accepted": True,
        "dispatched_messages": 0,
    }
    assert result["checks"]["voice_note"] == {
        "status": 200,
        "media_type": "audio/ogg; codecs=opus",
        "cached_extension": ".ogg",
        "cached_bytes": len(script.VOICE_BYTES),
        "dispatched_messages": 1,
    }
