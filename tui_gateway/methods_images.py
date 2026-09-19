"""Image-generation JSON-RPC handler (ws twin of the image_generate tool) for UI surfaces
(avatar pickers, artifact panes). The result is a data URL: a remote desktop can't read a
gateway file path and hosted URLs are often CORS-opaque to a renderer canvas.
"""

from .contracts.config_free_tier_control import ImageGenerateParams, ImageGenerateResult
from .method_ctx import HandlerRegistry, bind_module
from utils import is_truthy_value
import json

_registry = HandlerRegistry()
method = _registry.method


def _image_to_data_url(ref: str, cap: int):
    """Fetch a URL or read a local path into a data URL; None when missing, over *cap*, or failing."""
    import base64
    import mimetypes
    import os
    try:
        if ref.startswith(("http://", "https://")):
            # Provider-result URLs are remote-party-controlled — same SSRF guard as
            # agent/provider_media.save_url (which may have fallen back to the bare URL).
            from tools.url_safety import create_ssrf_safe_client, is_safe_url
            if not is_safe_url(ref):
                return None
            with create_ssrf_safe_client(timeout=60, follow_redirects=True) as client, \
                    client.stream("GET", ref, headers={"User-Agent": "hermes-agent"}) as resp:
                resp.raise_for_status()
                if resp.headers.get("content-length") and int(resp.headers["content-length"]) > cap:
                    return None
                chunks, total = [], 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > cap:
                        return None
                    chunks.append(chunk)
                data = b"".join(chunks)
                mime = (resp.headers.get("content-type") or "image/png").split(";", 1)[0].strip()
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
        return srv._err(rid, 4071, "prompt required")
    aspect = (params.aspect_ratio or "square").strip().lower()
    cap = min(params.max_bytes or 8_000_000, 16_000_000)
    try:
        from tools.image_generation_tool import _handle_image_generate
        result = json.loads(_handle_image_generate({"prompt": prompt, "aspect_ratio": aspect}))
    except Exception as e:
        return srv._err(rid, 5071, str(e))
    if not result.get("success"):
        return ImageGenerateResult(available=True, success=False,
                                   error=str(result.get("error") or "generation failed"))
    image_ref = str(result.get("image") or "")
    data_url = srv._image_to_data_url(image_ref, cap) if image_ref else None
    return ImageGenerateResult(available=True, success=True, image=image_ref, image_data=data_url)


def register(server) -> None:
    bind_module(globals(), server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
