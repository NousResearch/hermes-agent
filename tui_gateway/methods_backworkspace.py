"""Back-workspace JSON-RPC handlers — the Desktop's door to the page on the back of the window.

The Desktop's titlebar flips the window over to a blank page; its text is stored by
``tui_gateway/backworkspace.py``. Contracts:

- ``backworkspace.open`` → ``{page: {id, content, path} | null}``, the most recent page.
- ``backworkspace.save`` → writes ``content`` to page ``id`` (a new page when ``id`` is
  omitted) and returns ``{id, path}``. The client keeps the returned id for later saves.
- ``backworkspace.attach`` → stores a pasted image beside the pages and returns its path plus
  the relative ``href`` the page links it by.

``path`` is the page's file on the backend host: the client can hand it to an agent, which
reads the page with its own file tools instead of the text travelling back over the wire.

Every handler honours ``params.profile``: the requested profile's HERMES_HOME is bound around
the body, so each profile keeps its own pages.

Handlers are rebound onto server.py's globals at install time (see method_ctx.py) and may
reference server module globals (``_ok``, ``_err``).
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

# JSON-RPC error code 5097 = back-workspace failure. Kept as a literal inside handler bodies:
# handlers are rebound onto server.py's globals, so module-level constants are not reachable.


@method("backworkspace.open")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """The most recent page, or ``{page: null}`` before the first save."""
    from tui_gateway.backworkspace import latest_page

    try:
        return _ok(rid, {"page": latest_page()})
    except Exception as e:
        return _err(rid, 5097, str(e))


@method("backworkspace.save")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Write ``content`` to page ``id`` (a new page when omitted). Result: ``{id, path}``."""
    from tui_gateway.backworkspace import page_path, save_page

    content = params.get("content")
    if not isinstance(content, str):
        return _err(rid, 5097, "content must be a string")
    try:
        page_id = save_page(str(params.get("id") or "") or None, content)
        return _ok(rid, {"id": page_id, "path": str(page_path(page_id))})
    except Exception as e:
        return _err(rid, 5097, str(e))


@method("backworkspace.attach")
@_profile_scoped
def _(rid, params: dict) -> dict:
    """Store a pasted image beside the pages. Result: ``{path, href}`` — the link the page uses."""
    from tui_gateway.backworkspace import attach_image

    try:
        return _ok(rid, attach_image(str(params.get("name") or ""), str(params.get("data") or "")))
    except Exception as e:
        return _err(rid, 5097, str(e))


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
