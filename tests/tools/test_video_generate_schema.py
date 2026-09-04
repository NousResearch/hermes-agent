"""video_generate dynamic schema — capability-gated params (#95681 diet).

Mirrors tests/tools/test_image_generate_schema.py (#97057). Coverage is
guaranteed three ways:
1. every in-tree video_gen plugin's capabilities() must declare EVERY axis
   the schema builder reads (a new axis added to the builder without fleet
   declarations fails here);
2. every FAL video family must carry the per-family keys the fal provider's
   active-model capabilities() resolution reads;
3. declaration⇄implementation: a provider that declares seed/upscale must
   implement it, and vice versa (source-level sweep, both directions).
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tools.video_generation_tool as vt
from tools.video_generation_tool import (
    VIDEO_GENERATE_SCHEMA,
    _build_dynamic_video_schema,
)

# Every axis _build_dynamic_video_schema reads from capabilities().
CAPABILITY_AXES = (
    "modalities",
    "aspect_ratios",
    "resolutions",
    "max_duration",
    "min_duration",
    "supports_audio",
    "supports_negative_prompt",
    "supports_seed",
    "supports_upscale",
    "max_reference_images",
)

# Per-family keys the FAL provider's capabilities() resolution reads.
FAL_FAMILY_KEYS = ("durations", "aspect_ratios", "resolutions", "audio",
                   "negative", "seed")


def _plugin_sources():
    import pathlib

    plugins_dir = (pathlib.Path(__file__).resolve().parents[2]
                   / "plugins" / "video_gen")
    assert plugins_dir.is_dir(), plugins_dir
    out = {}
    for plugin in sorted(plugins_dir.iterdir()):
        src_file = plugin / "__init__.py"
        if src_file.is_file():
            out[plugin.name] = src_file.read_text(encoding="utf-8")
    return out


class TestFleetCapabilityCoverage(unittest.TestCase):
    def test_every_provider_declares_every_axis(self):
        """Instantiate each in-tree provider class and check the RETURNED
        capabilities dict — source grep can't see inherited keys."""
        checked = 0
        # fal
        from plugins.video_gen.fal import FALVideoGenProvider

        caps = FALVideoGenProvider().capabilities()
        for axis in CAPABILITY_AXES:
            self.assertIn(axis, caps, f"fal missing {axis}")
        checked += 1
        # xai
        from plugins.video_gen.xai import XAIVideoGenProvider

        caps = XAIVideoGenProvider().capabilities()
        for axis in CAPABILITY_AXES:
            self.assertIn(axis, caps, f"xai missing {axis}")
        checked += 1
        # deepinfra
        from plugins.video_gen.deepinfra import DeepInfraVideoGenProvider

        caps = DeepInfraVideoGenProvider().capabilities()
        for axis in CAPABILITY_AXES:
            self.assertIn(axis, caps, f"deepinfra missing {axis}")
        checked += 1
        # openrouter
        from plugins.video_gen.openrouter import OpenRouterVideoGenProvider

        caps = OpenRouterVideoGenProvider().capabilities()
        for axis in CAPABILITY_AXES:
            self.assertIn(axis, caps, f"openrouter missing {axis}")
        checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_abc_default_fails_closed(self):
        from agent.video_gen_provider import VideoGenProvider

        class _P(VideoGenProvider):
            name = "t"
            display_name = "T"
            def generate(self, prompt, **kw):
                return {}
            def list_models(self):
                return []
        caps = _P().capabilities()
        self.assertEqual(caps.get("modalities"), ["text"])
        for axis in ("supports_audio", "supports_negative_prompt",
                     "supports_seed", "supports_upscale"):
            self.assertFalse(caps.get(axis), axis)
        for axis in CAPABILITY_AXES:
            self.assertIn(axis, caps, f"ABC default missing {axis}")

    def test_every_fal_family_declares_per_family_keys(self):
        from plugins.video_gen.fal import FAL_FAMILIES

        for fam, meta in FAL_FAMILIES.items():
            with self.subTest(family=fam):
                for key in FAL_FAMILY_KEYS:
                    self.assertIn(key, meta, f"{fam} missing {key}")
                self.assertTrue(
                    meta.get("text_endpoint") or meta.get("image_endpoint"),
                    f"{fam} declares no endpoint",
                )
                # Audio truth model: "audio" means a generate_audio TOGGLE
                # exists; "audio_native" means audio is ALWAYS ON with no
                # toggle. A family may have neither (silent model) but
                # never both — that would be contradictory.
                self.assertFalse(
                    meta.get("audio") and meta.get("audio_native"),
                    f"{fam}: audio toggle and always-on are mutually "
                    "exclusive",
                )

    def test_always_on_audio_surfaces_in_description_not_param(self):
        """Maintainer catch (H3 has audio!): families whose audio is
        always-on (no API toggle) must TELL the model about the audio in
        the description rather than advertise a dead `audio` param."""
        schema = TestDynamicParamGating._schema_with(
            TestDynamicParamGating(), {
                "modalities": ["text", "image"],
                "supports_audio": False, "audio_always_on": True,
                "supports_negative_prompt": False, "supports_seed": False,
                "supports_upscale": False, "max_reference_images": 0,
            })
        self.assertNotIn("audio", schema["parameters"]["properties"])
        self.assertIn("always on", schema["description"])

    def test_declaration_matches_implementation(self):
        """supports_seed / supports_upscale: declaration ⇄ implementation,
        source-level, both directions, every in-tree plugin.

        deepinfra inherits generate() from OpenAICompatibleVideoGenProvider
        (agent/video_gen_provider.py), so its implementation source is the
        base class file."""
        import pathlib

        base_src = (pathlib.Path(__file__).resolve().parents[2]
                    / "agent" / "video_gen_provider.py").read_text(encoding="utf-8")
        for name, src in _plugin_sources().items():
            with self.subTest(provider=name):
                impl_src = src if "def generate" in src else src + base_src
                declares_upscale = '"supports_upscale": True' in src
                implements_upscale = ("_upscale_video" in impl_src
                                      or "UPSCALER_ENDPOINT" in impl_src)
                self.assertEqual(
                    declares_upscale, implements_upscale,
                    f"{name}: supports_upscale declaration "
                    f"({declares_upscale}) != implementation "
                    f"({implements_upscale})",
                )
                declares_seed = '"supports_seed": True' in src
                # Merely accepting ``seed`` in generate() is part of the ABC
                # contract, not proof that the backend implements it. Providers
                # can mark that contract-only parameter explicitly.
                implements_seed = ("payload[\"seed\"]" in impl_src
                                   or "\"seed\": seed" in impl_src
                                   or ("seed: Optional[int]" in impl_src
                                       and "_IGNORES_SEED = True" not in impl_src))
                self.assertEqual(
                    declares_seed, implements_seed,
                    f"{name}: supports_seed declaration ({declares_seed}) "
                    f"!= implementation ({implements_seed})",
                )


class TestDynamicParamGating(unittest.TestCase):
    def _schema_with(self, caps, model_meta=None):
        class _Prov:
            name = "fake"
            display_name = "Fake"
            def capabilities(self):
                return caps
            def list_models(self):
                return [dict({"id": "m1"}, **(model_meta or {}))]
            def default_model(self):
                return "m1"
        with patch.object(vt, "_resolve_active_provider",
                          return_value=_Prov()), \
             patch.object(vt, "_read_configured_video_model",
                          return_value="m1"):
            return _build_dynamic_video_schema()

    def test_full_featured_backend_gets_all_params(self):
        schema = self._schema_with({
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9"], "resolutions": ["720p"],
            "min_duration": 2, "max_duration": 12,
            "supports_audio": True, "supports_negative_prompt": True,
            "supports_seed": True, "supports_upscale": True,
            "max_reference_images": 7,
        })
        props = schema["parameters"]["properties"]
        for p in ("image_url", "reference_image_urls", "negative_prompt",
                  "audio", "seed", "upscale"):
            self.assertIn(p, props, p)
        self.assertEqual(props["reference_image_urls"]["maxItems"], 7)
        self.assertEqual(props["duration"]["minimum"], 2)
        self.assertEqual(props["duration"]["maximum"], 12)
        self.assertEqual(props["aspect_ratio"]["enum"], ["16:9"])

    def test_minimal_backend_gets_bare_params(self):
        schema = self._schema_with({
            "modalities": ["text"],
            "supports_audio": False, "supports_negative_prompt": False,
            "supports_seed": False, "supports_upscale": False,
            "max_reference_images": 0,
        })
        props = schema["parameters"]["properties"]
        for p in ("image_url", "reference_image_urls", "negative_prompt",
                  "audio", "seed", "upscale"):
            self.assertNotIn(p, props, p)
        self.assertIn("text-to-video only", schema["description"])

    def test_i2v_only_model_overrides_backend_union(self):
        # gemini-omni-flash case: dual-modality backend, i2v-only model.
        schema = self._schema_with(
            {"modalities": ["text", "image"], "max_reference_images": 0,
             "supports_audio": False, "supports_negative_prompt": False,
             "supports_seed": False, "supports_upscale": False},
            model_meta={"modalities": ["image"]},
        )
        self.assertIn("image_url", schema["parameters"]["properties"])
        self.assertIn("image-to-video only", schema["description"])

    def test_no_provider_serves_prompt_only(self):
        with patch.object(vt, "_resolve_active_provider", return_value=None):
            schema = _build_dynamic_video_schema()
        self.assertEqual(sorted(schema["parameters"]["properties"]), ["prompt"])

    def test_static_schema_carries_no_capability_args(self):
        props = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
        self.assertEqual(
            sorted(props),
            ["aspect_ratio", "duration", "model", "prompt", "resolution"],
        )


if __name__ == "__main__":
    unittest.main()
