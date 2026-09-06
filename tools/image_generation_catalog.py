"""FAL image model catalog + upscaler constants for ``tools.image_generation_tool``.

Each entry translates the unified inputs (prompt + aspect_ratio) into the model's native
payload. ``size_style``: ``"image_size_preset"`` (FAL preset enum), ``"aspect_ratio"`` (ratio
enum), ``"gpt_literal"`` (literal "WxH"). ``supports`` / ``edit_supports`` are whitelists —
other keys are stripped so models never receive rejected parameters. ``upscale`` is False
everywhere: Clarity redraws content (creativity 0.35) and degraded text/CJK/faces when
default-on, so upscaling is strictly per-call opt-in. Pricing strings may drift.
"""

from typing import Any, Dict, Optional

_PRESET_SIZES = {"landscape": "landscape_16_9", "square": "square_hd", "portrait": "portrait_16_9"}
_ASPECT_SIZES = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
_DEFAULT_SIZES = {"image_size_preset": _PRESET_SIZES, "aspect_ratio": _ASPECT_SIZES}


def _model(
    display: str, speed: str, strengths: str, price: str, *, style: str = "image_size_preset",
    sizes: Optional[Dict[str, Any]] = None, defaults: Dict[str, Any], supports: set,
    edit_endpoint: Optional[str] = None, edit_supports: Optional[set] = None,
    max_reference_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one catalog entry; edit keys are present only for edit-capable models."""
    entry: Dict[str, Any] = {
        "display": display, "speed": speed, "strengths": strengths, "price": price,
        "size_style": style, "sizes": sizes if sizes is not None else _DEFAULT_SIZES[style],
        "defaults": defaults, "supports": supports, "upscale": False,
    }
    if edit_endpoint:
        entry["edit_endpoint"] = edit_endpoint
        entry["edit_supports"] = edit_supports
        entry["max_reference_images"] = max_reference_images
    return entry


FAL_MODELS: Dict[str, Dict[str, Any]] = {
    "fal-ai/flux-2/klein/9b": _model(
        "FLUX 2 Klein 9B", "<1s", "Fast, crisp text", "$0.006/MP",
        defaults={
            "num_inference_steps": 4, "output_format": "png", "enable_safety_checker": False,
        },
        supports={
            "prompt", "image_size", "num_inference_steps", "seed", "output_format", "enable_safety_checker",
        },
        edit_endpoint="fal-ai/flux-2/klein/9b/edit",
        edit_supports={
            "prompt", "image_urls", "num_inference_steps", "seed", "output_format", "enable_safety_checker",
        },
        max_reference_images=9,
    ),
    "fal-ai/flux-2-pro": _model(
        "FLUX 2 Pro", "~6s", "Studio photorealism", "$0.03/MP",
        defaults={
            "num_inference_steps": 50, "guidance_scale": 4.5, "num_images": 1,
            "output_format": "png", "enable_safety_checker": False, "safety_tolerance": "5",
            "sync_mode": True,
        },
        supports={
            "prompt", "image_size", "num_inference_steps", "guidance_scale", "num_images", "output_format",
            "enable_safety_checker", "safety_tolerance", "sync_mode", "seed",
        },
        edit_endpoint="fal-ai/flux-2-pro/edit",
        edit_supports={
            "prompt", "image_urls", "num_inference_steps", "guidance_scale", "num_images", "output_format",
            "enable_safety_checker", "safety_tolerance", "sync_mode", "seed",
        },
        max_reference_images=9,
    ),
    "fal-ai/z-image/turbo": _model(
        "Z-Image Turbo", "~2s", "Bilingual EN/CN, 6B", "$0.005/MP",
        defaults={  # prompt expansion off: avoids the extra per-request charge
            "num_inference_steps": 8, "num_images": 1, "output_format": "png",
            "enable_safety_checker": False, "enable_prompt_expansion": False,
        },
        supports={
            "prompt", "image_size", "num_inference_steps", "num_images", "seed", "output_format",
            "enable_safety_checker", "enable_prompt_expansion",
        },
    ),
    "fal-ai/nano-banana-pro": _model(
        "Nano Banana Pro (Gemini 3 Pro Image)", "~8s", "Gemini 3 Pro, reasoning depth, text rendering", "$0.15/image (1K)",
        style="aspect_ratio",
        # "1K" is the cheapest tier; 4K doubles the per-image cost (Nous Subscription billing).
        defaults={
            "num_images": 1, "output_format": "png", "safety_tolerance": "5",
            "resolution": "1K",
        },
        supports={
            "prompt", "aspect_ratio", "num_images", "output_format", "safety_tolerance", "seed", "sync_mode",
            "resolution", "enable_web_search", "limit_generations",
        },
        edit_endpoint="fal-ai/nano-banana-pro/edit",
        edit_supports={
            "prompt", "image_urls", "aspect_ratio", "num_images", "output_format", "safety_tolerance", "seed",
            "sync_mode", "resolution", "enable_web_search", "limit_generations",
        },
        max_reference_images=2,
    ),
    "fal-ai/nano-banana-2": _model(
        "Nano Banana 2 (Gemini 3.1 Flash Image)", "~3s", "Fast reasoning, multilingual text, infographics", "Lower-cost Flash tier",
        style="aspect_ratio",
        defaults={
            "num_images": 1, "output_format": "png", "safety_tolerance": "4",
            "resolution": "1K", "limit_generations": True,
        },
        supports={
            "prompt", "aspect_ratio", "num_images", "output_format", "safety_tolerance", "seed", "sync_mode",
            "system_prompt", "resolution", "enable_web_search", "limit_generations", "thinking_level",
        },
        edit_endpoint="fal-ai/nano-banana-2/edit",
        edit_supports={
            "prompt", "image_urls", "aspect_ratio", "num_images", "output_format", "safety_tolerance", "seed",
            "sync_mode", "system_prompt", "resolution", "enable_web_search", "limit_generations",
            "thinking_level",
        },
        max_reference_images=14,
    ),
    "fal-ai/gpt-image-1.5": _model(
        "GPT Image 1.5", "~15s", "Prompt adherence", "$0.034/image",
        style="gpt_literal", sizes={
            "landscape": "1536x1024", "square": "1024x1024", "portrait": "1024x1536",
        },
        # quality pinned to medium (also for gpt-image-2) so portal billing stays
        # predictable: low is too rough, high is 3-6x the per-image cost.
        defaults={"quality": "medium", "num_images": 1, "output_format": "png"},
        supports={
            "prompt", "image_size", "quality", "num_images", "output_format", "background", "sync_mode",
        },
        edit_endpoint="fal-ai/gpt-image-1.5/edit",
        edit_supports={
            "prompt", "image_urls", "image_size", "quality", "num_images", "output_format", "sync_mode",
        },
        max_reference_images=16,
    ),
    # GPT Image 2 uses FAL's preset enum (unlike 1.5's literal dims) mapped to the
    # 4:3 variants: the 16:9 presets (1024x576) fall below its 655,360 min-pixel
    # requirement. openai_api_key (BYOK) is deliberately not in `supports` — all
    # users go through the shared FAL billing path. Its edit endpoint lives under
    # the OpenAI namespace (NOT fal-ai/) and auto-infers size, so no image_size.
    "fal-ai/gpt-image-2": _model(
        "GPT Image 2", "~20s", "SOTA text rendering + CJK, world-aware photorealism", "$0.04–0.06/image",
        style="image_size_preset", sizes={
            "landscape": "landscape_4_3", "square": "square_hd", "portrait": "portrait_4_3",
        },
        defaults={"quality": "medium", "num_images": 1, "output_format": "png"},
        supports={
            "prompt", "image_size", "quality", "num_images", "output_format", "sync_mode",
        },
        edit_endpoint="openai/gpt-image-2/edit",
        edit_supports={
            "prompt", "image_urls", "quality", "num_images", "output_format", "sync_mode", "mask_image_url",
        },
        max_reference_images=16,
    ),
    "fal-ai/ideogram/v3": _model(
        "Ideogram V3", "~5s", "Best typography", "$0.03-0.09/image",
        defaults={"rendering_speed": "BALANCED", "expand_prompt": True, "style": "AUTO"},
        supports={
            "prompt", "image_size", "rendering_speed", "expand_prompt", "style", "seed",
        },
        edit_endpoint="fal-ai/ideogram/v3/edit",
        edit_supports={
            "prompt", "image_urls", "rendering_speed", "expand_prompt", "style", "seed",
        },
        max_reference_images=1,
    ),
    "fal-ai/recraft/v4/pro/text-to-image": _model(
        "Recraft V4 Pro", "~8s", "Design, brand systems, production-ready", "$0.25/image",
        defaults={"enable_safety_checker": False},  # V4 Pro dropped V3's required `style` enum
        supports={
            "prompt", "image_size", "enable_safety_checker", "colors", "background_color",
        },
    ),
    "fal-ai/qwen-image": _model(
        "Qwen Image", "~12s", "LLM-based, complex text", "$0.02/MP",
        defaults={
            "num_inference_steps": 30, "guidance_scale": 2.5, "num_images": 1,
            "output_format": "png", "acceleration": "regular",
        },
        supports={
            "prompt", "image_size", "num_inference_steps", "guidance_scale", "num_images", "output_format",
            "acceleration", "seed", "sync_mode",
        },
        edit_endpoint="fal-ai/qwen-image-2/pro/edit",
        edit_supports={
            "prompt", "image_urls", "num_inference_steps", "guidance_scale", "num_images", "output_format",
            "acceleration", "seed", "sync_mode",
        },
        max_reference_images=3,
    ),
    # Krea 2 on FAL — same family as ``plugins/image_gen/krea`` but billed through
    # FAL / the FAL managed gateway. Native ``krea-2-*`` ids route to the plugin.
    "fal-ai/krea/v2/medium/text-to-image": _model(
        "Krea 2 Medium", "~15-25s", "Illustration, anime, painting, expressive/artistic styles", "$0.030 (text) / $0.035 (style refs)",
        style="aspect_ratio",
        defaults={"creativity": "medium"},
        supports={
            "prompt", "aspect_ratio", "creativity", "seed", "image_style_references",
        },
    ),
    "fal-ai/krea/v2/large/text-to-image": _model(
        "Krea 2 Large", "~25-60s", "Photorealism, raw textured looks (motion blur, grain, film)", "$0.060 (text) / $0.065 (style refs)",
        style="aspect_ratio",
        defaults={"creativity": "medium"},
        supports={
            "prompt", "aspect_ratio", "creativity", "seed", "image_style_references",
        },
    ),
    # Entries below take endpoint ids, `supports` whitelists and enum defaults from
    # each model's FAL OpenAPI schema; paired `/edit` apps hang off their
    # text-to-image entry rather than appearing as separate picker rows.
    # Seedream Pro requires total pixels between 1024² and 2048² — explicit
    # ImageSize dicts keep every aspect inside that window.
    "bytedance/seedream/v5/pro/text-to-image": _model(
        "Seedream 5.0 Pro", "~10s", "ByteDance flagship, dense layouts, native text in 14 languages", "$0.0675/image (≤1536²)",
        style="image_size_preset", sizes={
            "landscape": {"width": 2048, "height": 1152},
            "square": {"width": 1536, "height": 1536},
            "portrait": {"width": 1152, "height": 2048},
        },
        defaults={
            "num_images": 1, "output_format": "png", "enable_safety_checker": False,
        },
        supports={
            "prompt", "image_size", "num_images", "output_format", "sync_mode", "enable_safety_checker",
        },
        edit_endpoint="bytedance/seedream/v5/pro/edit",
        edit_supports={
            "prompt", "image_urls", "image_size", "num_images", "output_format", "sync_mode",
            "enable_safety_checker",
        },
        max_reference_images=10,
    ),
    # Lite wants 2560x1440..4096x4096 total pixels: use the documented presets (FAL
    # auto-scales under the floor) rather than hand-rolled dicts that drift.
    "bytedance/seedream/v5/lite/text-to-image": _model(
        "Seedream 5.0 Lite", "~5s", "Fast/cheap Seedream tier, high-res output", "$0.035/image",
        defaults={"num_images": 1, "enable_safety_checker": False},
        supports={
            "prompt", "image_size", "num_images", "max_images", "sync_mode", "enable_safety_checker",
        },
    ),
    "ideogram/v4/instant": _model(
        "Ideogram V4 (Instant)", "<1s", "Latest Ideogram typography, posters/logos, instant", "$0.0075/MP",
        defaults={
            "expansion_model": "Medium", "output_format": "png",
            "enable_safety_checker": False,
        },
        supports={
            "prompt", "image_size", "expansion_model", "num_images", "seed", "sync_mode",
            "enable_safety_checker", "output_format",
        },
    ),
    "ideogram/v4/fast": _model(
        "Ideogram V4 (Fast)", "~1s", "Ideogram V4 quality tiers via rendering_speed", "$0.005-0.018/MP",
        defaults={"expansion_model": "Medium", "rendering_speed": "BALANCED"},
        supports={
            "prompt", "image_size", "expansion_model", "rendering_speed", "num_images", "seed", "sync_mode",
        },
    ),
    "alibaba/qwen-image-3/text-to-image": _model(
        "Qwen Image 3", "~8s", "Complex CN/EN text rendering, prompt-guided resolution", "$0.04 (1K) / $0.075 (2K) per image",
        defaults={
            "num_images": 1, "output_format": "png", "enable_prompt_expansion": False,
            "enable_safety_checker": False,
        },
        supports={
            "prompt", "negative_prompt", "image_size", "num_images", "seed", "sync_mode", "output_format",
            "enable_prompt_expansion", "enable_safety_checker",
        },
        edit_endpoint="alibaba/qwen-image-3/edit",
        edit_supports={
            "prompt", "image_urls", "negative_prompt", "num_images", "seed", "sync_mode", "output_format",
            "enable_prompt_expansion", "enable_safety_checker",
        },
        max_reference_images=3,
    ),
    "microsoft/mai-image-2.5-pro": _model(
        "MAI Image 2.5 Pro", "~10s", "Microsoft flagship, hero imagery, precise typography", "~$0.17/image",
        style="aspect_ratio",
        defaults={"num_images": 1, "output_format": "png"},
        supports={"prompt", "aspect_ratio", "num_images", "output_format", "sync_mode"},
    ),
    "google/nano-banana-2-lite": _model(
        "Nano Banana 2 Lite", "<2s", "Gemini image family, sub-2s, 14 aspect ratios incl. extreme", "~$0.04/image (1K fixed)",
        style="aspect_ratio",
        defaults={"num_images": 1, "output_format": "png", "safety_tolerance": "5"},
        supports={
            "prompt", "aspect_ratio", "num_images", "seed", "output_format", "safety_tolerance", "sync_mode",
            "system_prompt", "limit_generations", "thinking_level",
        },
        edit_endpoint="google/nano-banana-2-lite/edit",
        edit_supports={
            "prompt", "image_urls", "aspect_ratio", "num_images", "seed", "output_format", "safety_tolerance",
            "sync_mode", "system_prompt",
        },
        max_reference_images=4,
    ),
    "fal-ai/recraft/v4.1/text-to-image": _model(
        "Recraft V4.1", "~8s", "Design-first raster, brand systems, editorial", "$0.035/image",
        defaults={"enable_safety_checker": False},
        supports={
            "prompt", "image_size", "enable_safety_checker", "colors", "background_color",
        },
    ),
    "xai/grok-imagine-image/v2.0/text-to-image": _model(
        "Grok Imagine Image 2.0", "~5s", "xAI. Design-grade typography/layout, instruction following", "$0.06/image (1K medium)",
        style="aspect_ratio",
        # 1k + medium is the cheapest sensible tier; 2k is roughly +33%/image. 1k native
        # is sub-2MP — pass upscale=true per call when needed. Edits omit aspect_ratio
        # (defaults to "auto", following the first input image).
        defaults={
            "num_images": 1, "output_format": "png", "resolution": "1k", "quality": "medium",
        },
        supports={
            "prompt", "aspect_ratio", "num_images", "output_format", "resolution", "quality", "sync_mode",
        },
        edit_endpoint="xai/grok-imagine-image/v2.0/edit",
        edit_supports={
            "prompt", "image_urls", "num_images", "output_format", "resolution", "quality", "sync_mode",
        },
        max_reference_images=3,
    ),
}


# Fastest reasonable option; cheap and sub-1s.
DEFAULT_MODEL = "fal-ai/flux-2/klein/9b"

DEFAULT_ASPECT_RATIO = "landscape"
VALID_ASPECT_RATIOS = ("landscape", "square", "portrait")

# Clarity Upscaler settings.
UPSCALER_MODEL = "fal-ai/clarity-upscaler"
UPSCALER_FACTOR = 2
UPSCALER_SAFETY_CHECKER = False
UPSCALER_DEFAULT_PROMPT = "masterpiece, best quality, highres"
UPSCALER_NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2)"
UPSCALER_CREATIVITY = 0.35
UPSCALER_RESEMBLANCE = 0.6
UPSCALER_GUIDANCE_SCALE = 4
UPSCALER_NUM_INFERENCE_STEPS = 18
