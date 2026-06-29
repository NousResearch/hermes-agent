from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "plugins" / "tts" / "cloud-temple" / "__init__.py"
    spec = importlib.util.spec_from_file_location("cloud_temple_tts_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lang_code_aliases_match_kokoro_wire_codes():
    mod = _load_module()

    assert mod._normalize_lang_code("fr") == "f"
    assert mod._normalize_lang_code("en") == "a"
    assert mod._normalize_lang_code("es") == "e"
    assert mod._normalize_lang_code("pt-BR") == "p"


def test_french_text_infers_french_wire_code():
    mod = _load_module()

    assert mod._infer_lang_code("Le service de synthèse audio est disponible.") == "f"


def test_provider_config_lang_code_wins():
    mod = _load_module()

    assert mod._configured_lang_code({"cloud-temple": {"language": "fr"}}) == "f"
    assert mod._configured_lang_code({"lang_code": "en"}) == "a"
