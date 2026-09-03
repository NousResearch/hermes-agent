"""image_generate dynamic schema — capability-gated params (#95681 diet).

Contract: args the active model cannot honor are NOT advertised. Coverage
is guaranteed two ways — every in-tree FAL catalog entry must declare the
capability keys the schema builder reads (test below fails when a new
model is added without them), and the plugin provider ABC's capabilities()
default fails closed to text-only/no-upscale.
"""
import json
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

    def test_every_model_declares_capability_keys(self):
        for model_id, meta in FAL_MODELS.items():
            with self.subTest(model=model_id):
                # upscale default flag must be present and boolean.
                self.assertIn("upscale", meta, model_id)
                self.assertIsInstance(meta["upscale"], bool, model_id)
                # Edit-capable models must cap their reference images.
                if meta.get("edit_endpoint"):
                    self.assertIn("max_reference_images", meta, model_id)
                    self.assertGreaterEqual(
                        int(meta["max_reference_images"]), 1, model_id,
                    )

    def test_provider_abc_default_fails_closed(self):
        from agent.image_gen_provider import ImageGenProvider

        caps_src = ImageGenProvider.capabilities
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

    def test_every_intree_plugin_declares_what_it_implements(self):
        """Fleet-wide declaration ⇄ implementation contract (maintainer
        requirement: EVERY provider combo must carry capability info).

        For each in-tree image_gen plugin: if its generate() source
        implements an upscale pass, capabilities() must declare
        supports_upscale — and vice versa, so a stale declaration can't
        advertise an upscale that silently no-ops. modalities and
        max_reference_images must always be declared."""
        import ast
        import pathlib

        plugins_dir = (pathlib.Path(__file__).resolve().parents[2]
                       / "plugins" / "image_gen")
        assert plugins_dir.is_dir(), plugins_dir
        checked = 0
        for plugin in sorted(plugins_dir.iterdir()):
            src_file = plugin / "__init__.py"
            if not src_file.is_file():
                continue
            src = src_file.read_text(encoding="utf-8")
            if "def capabilities" not in src:
                continue
            checked += 1
            with self.subTest(provider=plugin.name):
                # capabilities() must declare the two mandatory axes.
                self.assertIn("modalities", src, plugin.name)
                self.assertIn("max_reference_images", src, plugin.name)
                # upscale: declaration ⇄ implementation, both directions.
                declares = "supports_upscale" in src
                # Implementation = generate() (or its helpers in the same
                # file) reads the upscale kwarg directly, or passes it
                # through in a delegation whitelist (fal plugin → in-tree
                # Clarity chain).
                implements = ('kwargs.get("upscale")' in src
                              or "upscale_requested" in src
                              or 'kwargs["upscale"]' in src
                              or "def _upscale" in src
                              or '"upscale",' in src)
                self.assertEqual(
                    declares, implements,
                    f"{plugin.name}: supports_upscale declaration "
                    f"({declares}) != implementation ({implements}) — "
                    "declare it in capabilities() iff generate() honors it",
                )
        # The audit must actually have covered the fleet.
        self.assertGreaterEqual(checked, 6, "plugin sweep found too few providers")


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

    def test_static_schema_carries_no_capability_args(self):
        """The registration-time placeholder must stay minimal — dynamic
        overrides own the capability args (do-not-re-add guard)."""
        props = IMAGE_GENERATE_SCHEMA["parameters"]["properties"]
        self.assertEqual(sorted(props), ["aspect_ratio", "prompt"])

    def test_handler_still_rejects_unadvertised_edit_with_teaching_error(self):
        """Wire compat: image_url on a t2i-only model is accepted by the
        handler and answered with the capability error (not a schema-level
        unknown-arg failure). Pin the error text's presence in the source."""
        import inspect
        handler_src = inspect.getsource(ig)
        self.assertIn("capable of image-to-image / editing", handler_src)
        self.assertIn("omit image_url", handler_src)


if __name__ == "__main__":
    unittest.main()
