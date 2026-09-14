"""Image-generation JSON-RPC handler (ws twin of the image_generate tool) for UI surfaces
(avatar pickers, artifact panes). The result is a data URL: a remote desktop can't read a
gateway file path and hosted URLs are often CORS-opaque to a renderer canvas. Bodies are
rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from .contracts.config_free_tier_control import ImageGenerateParams, ImageGenerateResult
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def _image_to_data_url(ref: str, cap: int):
    """Fetch a URL or read a local path into a data URL; None when missing, over *cap*, or failing."""
    import base64
    import mimetypes
    import os
    try:
        if ref.startswith(("http://", "https://")):
            import urllib.request
            req = urllib.request.Request(ref, headers={"User-Agent": "hermes-agent"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.length is not None and resp.length > cap:
                    return None
                data = resp.read(cap + 1)
                mime = resp.headers.get_content_type() or "image/png"
        elif os.path.isfile(ref):
            if os.path.getsize(ref) > cap:
                return None
            with open(ref, "rb") as fh:
                data = fh.read(cap + 1)
            mime = mimetypes.guess_type(ref)[0] or "image/png"
        else:
            return None
        if len(data) > cap:
            return None
        mime = mime if mime.startswith("image/") else "image/png"
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return None


@method("image.generate")
def _(rid, params: ImageGenerateParams) -> ImageGenerateResult | dict:
    """Generate an image or report provider availability."""
    try:
        from tools.image_generation_tool import check_image_generation_requirements
        available = bool(check_image_generation_requirements())
    except Exception:
        available = False
    if is_truthy_value(params.probe):
        return ImageGenerateResult(available=available)
    if not available:
        return ImageGenerateResult(
            available=False, success=False,
            error="No image generation backend configured (run `hermes tools` to enable one).")
    prompt = (params.prompt or "").strip()
    if not prompt:
        return _err(rid, 4071, "prompt required")
    aspect = (params.aspect_ratio or "square").strip().lower()
    cap = min(params.max_bytes or 8_000_000, 16_000_000)
    try:
        from tools.image_generation_tool import _handle_image_generate
        result = json.loads(_handle_image_generate({"prompt": prompt, "aspect_ratio": aspect}))
    except Exception as e:
        return _err(rid, 5071, str(e))
    if not result.get("success"):
        return ImageGenerateResult(available=True, success=False,
                                   error=str(result.get("error") or "generation failed"))
    image_ref = str(result.get("image") or "")
    data_url = _image_to_data_url(image_ref, cap) if image_ref else None
    return ImageGenerateResult(available=True, success=True, image=image_ref, image_data=data_url)


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
