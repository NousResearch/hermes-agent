"""image_generate reproducibility knobs — handler forwarding + schema advertisement.

``seed`` / ``num_inference_steps`` / ``guidance_scale`` are accepted by the in-tree FAL helper
(``image_generate_tool``) but were neither advertised by the served schema nor read by
``_handle_image_generate``, so nothing reproducible could go through the tool path: every call
fell back to a random seed and the model's default step count.

Contract pinned here: the knobs are advertised and forwarded on the in-tree FAL route only.
The plugin and managed routes take their own ``controls`` family and their ``generate(**kwargs)``
does not accept these three, so neither the schema nor the handler offers them there.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools import image_generation_tool as ig
from tools.image_generation_tool import FAL_MODELS

# name -> advertised JSON-schema type
_KNOBS = {"seed": "integer", "num_inference_steps": "integer", "guidance_scale": "number"}


@pytest.fixture
def fal_route(monkeypatch):
    """Route ``_handle_image_generate`` straight to the in-tree FAL helper, recording its kwargs."""
    calls: list[dict] = []

    def fake_fal(prompt, aspect_ratio, **kwargs):
        calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
        return json.dumps({"success": True, "image": "/tmp/fal.png", "modality": "text", "upscaled": False})

    monkeypatch.setattr(ig, "_dispatch_to_plugin_provider", lambda *a, **k: None)
    monkeypatch.setattr(ig, "_maybe_route_managed_model", lambda *a, **k: None)
    monkeypatch.setattr(ig, "image_generate_tool", fake_fal)
    return calls


class TestHandlerForwardsReproducibilityKnobs:

    def test_handler_forwards_seed_steps_and_guidance_to_fal(self, fal_route):
        ig._handle_image_generate({
            "prompt": "draw cat", "aspect_ratio": "square",
            "seed": 1234, "num_inference_steps": 8, "guidance_scale": 3.5})

        assert len(fal_route) == 1
        assert fal_route[0]["seed"] == 1234
        assert fal_route[0]["num_inference_steps"] == 8
        assert fal_route[0]["guidance_scale"] == 3.5

    def test_handler_omits_knobs_the_caller_did_not_supply(self, fal_route):
        """No key is invented, so the helper's own defaults still apply."""
        ig._handle_image_generate({"prompt": "draw cat"})

        for name in _KNOBS:
            assert name not in fal_route[0]

    def test_handler_coerces_numeric_strings(self, fal_route):
        """A JSON body may send a numeric string; it is coerced rather than forwarded as a string."""
        ig._handle_image_generate({
            "prompt": "draw cat", "seed": "1234",
            "num_inference_steps": "8", "guidance_scale": "3.5"})

        assert fal_route[0]["seed"] == 1234
        assert fal_route[0]["num_inference_steps"] == 8
        assert fal_route[0]["guidance_scale"] == 3.5

    @pytest.mark.parametrize("bad", ["nope", "", {}, [], True])
    def test_handler_drops_unparseable_values(self, fal_route, bad):
        ig._handle_image_generate({"prompt": "draw cat", "seed": bad, "num_inference_steps": bad})

        assert "seed" not in fal_route[0]
        assert "num_inference_steps" not in fal_route[0]


class TestSchemaAdvertisesReproducibilityKnobs:

    def _fal_schema(self):
        model_id = next(iter(FAL_MODELS))
        with patch.object(ig, "_resolve_fal_model", return_value=(model_id, FAL_MODELS[model_id])), \
             patch.object(ig, "_read_configured_image_provider", return_value=None):
            return ig._build_dynamic_image_schema()

    def test_fal_route_advertises_reproducibility_knobs(self):
        props = self._fal_schema()["parameters"]["properties"]

        for name, json_type in _KNOBS.items():
            assert props[name]["type"] == json_type

    def test_plugin_route_does_not_advertise_reproducibility_knobs(self):
        """The plugin route's ``generate()`` takes no such kwargs, so the schema must not offer them."""
        class _Prov:
            display_name = "Codex Images"

            def capabilities(self):
                return {"modalities": ["text"], "max_reference_images": 0}

            def default_model(self):
                return "img-1"

        with patch.object(ig, "_read_configured_image_provider", return_value="codex"), \
             patch("agent.image_gen_registry.get_provider", return_value=_Prov()), \
             patch("hermes_cli.plugins._ensure_plugins_discovered"):
            props = ig._build_dynamic_image_schema()["parameters"]["properties"]

        assert not set(_KNOBS) & set(props)
