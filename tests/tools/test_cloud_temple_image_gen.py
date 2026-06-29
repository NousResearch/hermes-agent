from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "plugins" / "image_gen" / "cloud-temple" / "__init__.py"
    spec = importlib.util.spec_from_file_location("cloud_temple_image_gen_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_setup_schema_marks_cloud_temple_as_paid():
    mod = _load_module()

    schema = mod.CloudTempleImageGenProvider().get_setup_schema()

    assert schema["badge"] == "paid"
    assert schema["env_vars"][0]["key"] == "CLOUD_TEMPLE_API_KEY"


def test_model_listing_requires_cloud_temple_api_key():
    mod = _load_module()

    models = mod.CloudTempleImageGenProvider().list_models()

    assert models[0]["price"] == "requires Cloud Temple API key"
    assert models[0]["price"] != "free"
