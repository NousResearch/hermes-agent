"""Image-generation JSON-RPC handler (ws twin of the image_generate tool).

Desktop plugins reach the backend only through ws JSON-RPC; the image
generation capability existed solely as a model tool. ``image.generate``
lets UI surfaces (avatar pickers, artifact panes) generate directly.

The result image is returned as a data URL (``image_data``): a remote
desktop client cannot read a file path on the gateway host, and hosted
result URLs are often CORS-opaque to a renderer canvas. Data URLs work
identically over local and remote gateways.

Handlers are rebound onto server.py's globals at install time (see
method_ctx.py) — helpers must stay nested inside the handler body.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("image.generate")
def _(rid, params: dict) -> dict:
    """Generate an image with the configured backend.

    Params: ``prompt`` (required unless ``probe``), ``aspect_ratio``
    (landscape|square|portrait), ``probe`` (return availability only),
    ``max_bytes`` (cap on the returned data URL payload, default 8MB).

    Result: ``{available, success, image, image_data, error}`` where
    ``image`` is the backend's URL/path and ``image_data`` is a data URL
    of the downloaded bytes (omitted when the download fails — callers
    should fall back to ``image``).
    """

    def _availability() -> bool:
        try:
            from tools.image_generation_tool import check_image_generation_requirements

            return bool(check_image_generation_requirements())
        except Exception:
            return False

    def _to_data_url(ref: str, cap: int):
        """Fetch a URL or read a local path into a data URL, size-capped."""
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
            if not mime.startswith("image/"):
                mime = "image/png"
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        except Exception:
            return None

    available = _availability()
    if is_truthy_value(params.get("probe", False)):
        return _ok(rid, {"available": available})
    if not available:
        return _ok(
            rid,
            {
                "available": False,
                "success": False,
                "error": "No image generation backend configured (run `hermes tools` to enable one).",
            },
        )

    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        return _err(rid, 4071, "prompt required")

    aspect = str(params.get("aspect_ratio") or "square").strip().lower()
    try:
        cap = min(int(params.get("max_bytes", 8_000_000) or 8_000_000), 16_000_000)
    except (TypeError, ValueError):
        cap = 8_000_000

    try:
        from tools.image_generation_tool import image_generate_tool

        raw = image_generate_tool(prompt=prompt, aspect_ratio=aspect)
        result = json.loads(raw)
    except Exception as e:
        return _err(rid, 5071, str(e))

    if not result.get("success"):
        return _ok(
            rid,
            {
                "available": True,
                "success": False,
                "error": str(result.get("error") or "generation failed"),
            },
        )

    image_ref = str(result.get("image") or "")
    payload = {"available": True, "success": True, "image": image_ref}
    data_url = _to_data_url(image_ref, cap) if image_ref else None
    if data_url:
        payload["image_data"] = data_url
    return _ok(rid, payload)


def register(server) -> None:
    _registry.install(server)
