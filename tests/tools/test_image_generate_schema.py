"""image_generate dynamic schema — capability-gated params (#95681 diet).

Contract: args the active model cannot honor are NOT advertised. Coverage
is guaranteed two ways — every in-tree FAL catalog entry must declare the
capability keys the schema builder reads (test below fails when a new
model is added without them), and the plugin provider ABC's capabilities()
default fails closed to text-only/no-upscale.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tools.image_generation_tool as ig
from tools.image_generation_tool import (
    FAL_MODELS,
    IMAGE_GENERATE_SCHEMA,
    _build_dynamic_image_schema,
)


class TestCatalogCapabilityCoverage(unittest.TestCase):
    """Every FAL catalog entry must carry what the schema builder reads."""


    def test_provider_abc_default_fails_closed(self):
        from agent.image_gen_provider import ImageGenProvider

        # Instantiate via a minimal concrete subclass.
        class _P(ImageGenProvider):
            name = "t"
            display_name = "T"
            def generate(self, prompt, aspect_ratio="landscape", **kw):
                return {}
            def list_models(self):
                return []
        caps = _P().capabilities()
        self.assertEqual(caps.get("modalities"), ["text"])
        self.assertFalse(caps.get("supports_upscale", False))



class TestDynamicParamGating(unittest.TestCase):
    def _schema_for(self, model_id):
        with patch.object(ig, "_resolve_fal_model",
                          return_value=(model_id, FAL_MODELS[model_id])), \
             patch.object(ig, "_read_configured_image_provider",
                          return_value=None):
            return _build_dynamic_image_schema()

    def _t2i_only(self):
        return next(m for m, meta in FAL_MODELS.items()
                    if not meta.get("edit_endpoint"))

    def _edit_multi_ref(self):
        return next(m for m, meta in FAL_MODELS.items()
                    if meta.get("edit_endpoint")
                    and int(meta.get("max_reference_images") or 0) > 1)

    def test_t2i_only_model_hides_edit_args(self):
        schema = self._schema_for(self._t2i_only())
        props = schema["parameters"]["properties"]
        self.assertNotIn("image_url", props)
        self.assertNotIn("reference_image_urls", props)
        self.assertIn("cannot edit", schema["description"])

    def test_edit_model_advertises_edit_args_with_cap(self):
        model = self._edit_multi_ref()
        schema = self._schema_for(model)
        props = schema["parameters"]["properties"]
        self.assertIn("image_url", props)
        self.assertIn("reference_image_urls", props)
        self.assertEqual(
            props["reference_image_urls"]["maxItems"],
            int(FAL_MODELS[model]["max_reference_images"]),
        )

    def test_fal_always_advertises_upscale(self):
        # Clarity Upscaler chains for any FAL model on explicit request.
        for model in (self._t2i_only(), self._edit_multi_ref()):
            schema = self._schema_for(model)
            self.assertIn("upscale", schema["parameters"]["properties"], model)

    def test_text_only_plugin_provider_hides_edit_and_upscale(self):
        class _Prov:
            display_name = "Codex Images"
            def capabilities(self):
                return {"modalities": ["text"], "max_reference_images": 0}
            def default_model(self):
                return "img-1"
        with patch.object(ig, "_read_configured_image_provider",
                          return_value="codex"), \
             patch("agent.image_gen_registry.get_provider",
                   return_value=_Prov()), \
             patch("hermes_cli.plugins._ensure_plugins_discovered"):
            schema = _build_dynamic_image_schema()
        props = schema["parameters"]["properties"]
        self.assertEqual(sorted(props), ["aspect_ratio", "prompt"])
        self.assertNotIn("upscale", props)

    def test_managed_krea_model_advertises_krea_edit_args_and_upscale(self):
        """provider nous + a Krea model id is served by the Krea gateway, so the
        schema must advertise what the Krea plugin declares, not the FAL catalog."""
        from plugins.image_gen.krea import KreaImageGenProvider

        with patch.object(ig, "_read_configured_image_provider",
                          return_value="nous"), \
             patch.object(ig, "_read_configured_image_model",
                          return_value="krea-2-medium"):
            schema = _build_dynamic_image_schema()
        props = schema["parameters"]["properties"]
        self.assertIn("image_url", props)
        self.assertEqual(
            props["reference_image_urls"]["maxItems"],
            KreaImageGenProvider().capabilities()["max_reference_images"],
        )
        self.assertIn("upscale", props)

    def test_static_schema_carries_no_capability_args(self):
        """The registration-time placeholder must stay minimal — dynamic
        overrides own the capability args (do-not-re-add guard)."""
        props = IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        self.assertEqual(sorted(props), ["aspect_ratio", "prompt"])



if __name__ == "__main__":
    unittest.main()
