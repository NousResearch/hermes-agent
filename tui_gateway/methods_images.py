"""Image-generation JSON-RPC handler (ws twin of the image_generate tool) for UI surfaces
(avatar pickers, artifact panes). The result is a data URL: a remote desktop can't read a
gateway file path and hosted URLs are often CORS-opaque to a renderer canvas. Bodies are
rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

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
def _(rid, params: dict) -> dict:
    """Params: ``prompt`` (required unless ``probe``), ``aspect_ratio``
    (landscape|square|portrait), ``probe`` (availability only), ``max_bytes`` (cap
    on the data URL, default 8MB, max 16MB). Result: ``{available, success, image,
    image_data, error}`` — ``image_data`` is omitted when the download fails, so
    callers fall back to ``image`` (the backend's URL/path)."""
    try:
        from tools.image_generation_tool import check_image_generation_requirements
        available = bool(check_image_generation_requirements())
    except Exception:
        available = False
    if is_truthy_value(params.get("probe", False)):
        return _ok(rid, {"available": available})
    if not available:
        return _ok(rid, {
            "available": False, "success": False,
            "error": "No image generation backend configured (run `hermes tools` to enable one)."})
    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        return _err(rid, 4071, "prompt required")
    aspect = str(params.get("aspect_ratio") or "square").strip().lower()
    try:
        cap = min(int(params.get("max_bytes", 8_000_000) or 8_000_000), 16_000_000)
    except (TypeError, ValueError):
        cap = 8_000_000
    try:
        # Full provider dispatcher — same path as the model tool (source-image confinement,
        # plugin providers, managed routing, FAL fallback); the FAL leaf bypassed providers.
        from tools.image_generation_tool import _handle_image_generate
        result = json.loads(_handle_image_generate({"prompt": prompt, "aspect_ratio": aspect}))
    except Exception as e:
        return _err(rid, 5071, str(e))
    if not result.get("success"):
        return _ok(rid, {"available": True, "success": False,
                         "error": str(result.get("error") or "generation failed")})
    image_ref = str(result.get("image") or "")
    data_url = _image_to_data_url(image_ref, cap) if image_ref else None
    return _ok(rid, {"available": True, "success": True, "image": image_ref,
                     **({"image_data": data_url} if data_url else {})})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
