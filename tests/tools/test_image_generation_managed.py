"""The single managed image row: model id → gateway, and a de-duplicated union catalog."""

from unittest.mock import patch

import tools.image_generation_managed as managed
from plugins.image_gen.krea import KREA_MODEL_IDS
from tools.image_generation_catalog import DEFAULT_MODEL, FAL_MODELS


def test_every_fal_and_krea_model_resolves_to_its_own_gateway_and_the_rest_to_portal():
    assert managed.managed_backend_for_model(None) == managed.FAL
    assert {managed.managed_backend_for_model(m) for m in FAL_MODELS} == {managed.FAL}
    assert {managed.managed_backend_for_model(m) for m in KREA_MODEL_IDS} == {managed.KREA}
    assert managed.managed_backend_for_model("vendor/some-portal-only-model") == managed.PORTAL


def test_union_catalog_lists_each_model_once():
    def rows(name):
        return {
            "krea": [{"id": m, "display": m} for m in KREA_MODEL_IDS],
            "nous": [
                {"id": "microsoft/mai-image-2.5-pro"},   # same id as a FAL entry
                {"id": "krea/krea-2-medium"},            # Krea 2 under its Portal id
                {"id": "google/gemini-3-pro-image"},     # Nano Banana Pro under its Portal id
                {"id": "vendor/portal-only"},
            ],
        }[name]

    with patch.object(managed, "_plugin_rows", side_effect=rows):
        catalog, default = managed.managed_image_catalog()

    assert default == DEFAULT_MODEL
    portal = {m for m, meta in catalog.items() if meta["backend"] == managed.PORTAL}
    assert portal == {"vendor/portal-only"}
    assert catalog["microsoft/mai-image-2.5-pro"]["backend"] == managed.FAL
    assert KREA_MODEL_IDS <= set(catalog) and not any(m.startswith("fal-ai/krea/") for m in catalog)
