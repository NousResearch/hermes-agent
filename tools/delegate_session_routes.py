"""Remember route identity without retaining credentials or freezing provider config."""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def _endpoint_fingerprint(base_url) -> str:
    """Only the digest is persisted; credentials embedded in URL userinfo are omitted."""
    parsed = urlsplit(str(base_url or "").strip())
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("This unnamed delegation endpoint has no stable URL; configure a named provider before retaining it.")
    scheme, host = parsed.scheme.lower(), parsed.hostname.lower()
    port = parsed.port
    host = f"[{host}]" if ":" in host else host
    if port is not None and (scheme, port) not in {("https", 443), ("http", 80)}:
        host += f":{port}"
    normalized = urlunsplit((scheme, host, parsed.path.rstrip("/"), parsed.query, ""))
    return hashlib.sha256(normalized.encode()).hexdigest()


def capture_route(child, *, routing_cfg) -> dict:
    """Capture the selector used at spawn, not an endpoint/credential bundle.

    Named providers are resolved anew by name. Anonymous endpoints additionally
    require the selected configuration (or inherited parent) to keep its identity.
    """
    from tools.delegate_tool_config import _NATIVE_SDK_PROVIDERS
    route = routing_cfg
    provider = str(getattr(child, "requested_provider", None) or child.provider).strip().lower()
    if route.get("base_url") and str(route.get("provider") or "").lower() not in _NATIVE_SDK_PROVIDERS:
        descriptor = {"kind": "default"}
    elif provider not in {"", "auto", "custom"}:
        return {"kind": "named", "provider": provider}
    elif route.get("provider"):
        # An explicitly selected generic custom provider is still anonymous:
        # its configured endpoint must match before new credentials are adopted.
        descriptor = {"kind": "named", "provider": provider}
    else:
        descriptor = {"kind": "parent"}
    descriptor["endpoint_sha256"] = _endpoint_fingerprint(child.base_url)
    return descriptor


def _require_endpoint(descriptor, base_url) -> None:
    expected = descriptor.get("endpoint_sha256")
    if not expected or _endpoint_fingerprint(base_url) != expected:
        raise ValueError(
            "The retained child's unnamed provider endpoint no longer matches its original route. "
            "Restore that route or start a new delegation with a named provider; this conversation was not resumed.")


def resume_route(parent, record, model=None) -> dict:
    """Reconstruct the saved route for host-owned activation, bypassing spawn policy.

    Resolve credentials anew while preserving the named provider or verified endpoint.
    """
    from tools.delegate_tool_config import (
        _inherit_parent_endpoint, _load_config, _resolve_delegation_credentials,
    )
    selected_model = record["model"] if model is None else model
    if not isinstance(selected_model, str) or not selected_model.strip():
        raise ValueError("Resume model must be a non-empty string.")
    task = {"model": selected_model.strip()}
    descriptor = record.get("route_descriptor") or {}
    kind = descriptor.get("kind")
    provider = record["provider"]
    if not kind:
        if provider in {"", "auto", "custom"}:
            raise ValueError("This older unnamed child has no saved route identity; start a new delegation with a named provider.")
        kind = "named"
    cfg = _load_config()
    if kind == "parent":
        current_provider = str(getattr(parent, "requested_provider", None) or
                               getattr(parent, "provider", "")).strip().lower()
        if current_provider != str(provider).strip().lower():
            raise ValueError("The retained child's inherited provider has changed; restore the original provider or start a new delegation.")
        endpoint, _key = _inherit_parent_endpoint(parent, getattr(parent, "base_url", None),
                                                 getattr(parent, "api_key", None))
        _require_endpoint(descriptor, endpoint)
        # A model-only route inherits the live parent's endpoint and credentials;
        # never fall through to an unrelated current delegation default.
        return {**task, **{k: cfg[k] for k in ("reasoning_effort",) if k in cfg}}
    if kind == "named":
        # A saved named provider owns its credentials and transport; current defaults
        # may now describe an unrelated endpoint. Preserve only non-identity tuning.
        cfg = {k: cfg[k] for k in ("reasoning_effort",) if k in cfg}
        task["provider"] = provider
        task["fallback_providers"] = []
    elif kind != "default":
        raise ValueError("The retained child has an unsupported route identity.")
    route = {**cfg, **task}
    if kind != "named" or descriptor.get("endpoint_sha256"):
        credentials = _resolve_delegation_credentials(route, parent)
        _require_endpoint(descriptor, credentials.get("base_url"))
        if str(credentials.get("provider") or "").strip().lower() != str(provider).strip().lower():
            raise ValueError("The retained child's provider has changed on its saved endpoint; restore the original route or start a new delegation.")
    return route
