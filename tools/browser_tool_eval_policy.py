"""Shared browser private-page guards and fixed, internal URL probes."""
from typing import Optional

from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_session as _session

_CURRENT_PAGE_URL_EXPRESSION = "window.location.href"


def _eval_ssrf_guard_active(effective_task_id: str) -> bool:
    """Whether ordinary private pages must be guarded for this backend."""
    _bt = _origin()
    return not _cloud._is_local_backend() and not _bt._is_local_sidecar_key(effective_task_id) and not _cloud._allow_private_urls()


def _url_blocked(_bt, url: str, *, include_private: bool = True) -> bool:
    """True for the metadata floor, or ordinary private URLs when requested."""
    return _bt._is_always_blocked_url(url) or (include_private and not _bt._is_safe_url(url))


def _read_current_page_url(effective_task_id: str) -> Optional[str]:
    """Read only the location URL through a fixed internal browser command."""
    result = _session._run_browser_command(
        effective_task_id, "eval", [_CURRENT_PAGE_URL_EXPRESSION], timeout=5, _engine_override="auto"
    )
    if not result.get("success"):
        return None
    value = result.get("data", {}).get("result", "")
    return value.strip().strip('"').strip("'") if isinstance(value, str) else None


def _current_page_blocked_url(effective_task_id: str, *, include_private: bool) -> Optional[str]:
    """Return a blocked current-page URL; a failed fixed probe fails open."""
    _bt = _origin()
    try:
        current_url = _read_current_page_url(effective_task_id)
        if current_url and _url_blocked(_bt, current_url, include_private=include_private):
            return current_url
    except Exception as exc:
        _bt.logger.debug("current-page URL probe failed (%s)", exc)
    return None


def _current_page_private_url(effective_task_id: str) -> Optional[str]:
    """Compatibility wrapper for callers that require ordinary-private blocking."""
    return _current_page_blocked_url(effective_task_id, include_private=True)


def _read_camofox_current_page_url(tab_id: str, user_id: str) -> Optional[str]:
    """Read only the location URL through Camofox's fixed internal endpoint call."""
    from tools.browser_camofox import _post

    data = _post(
        f"/tabs/{tab_id}/evaluate", body={"expression": _CURRENT_PAGE_URL_EXPRESSION, "userId": user_id}
    )
    value = data.get("result") if isinstance(data, dict) else data
    return str(value).strip().strip('"').strip("'") if value is not None else None


def _camofox_current_page_blocked_url(tab_id: str, user_id: str, *, include_private: bool) -> Optional[str]:
    """Camofox counterpart to :func:`_current_page_blocked_url`."""
    _bt = _origin()
    try:
        current_url = _read_camofox_current_page_url(tab_id, user_id)
        if current_url and _url_blocked(_bt, current_url, include_private=include_private):
            return current_url
    except Exception as exc:
        _bt.logger.debug("Camofox current-page URL probe failed (%s)", exc)
    return None


def _camofox_current_page_private_url(tab_id: str, user_id: str) -> Optional[str]:
    """Compatibility wrapper for callers that require ordinary-private blocking."""
    return _camofox_current_page_blocked_url(tab_id, user_id, include_private=True)
