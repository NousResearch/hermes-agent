"""The desktop's built-in tours (the ``tour`` server request's ``preset``; the app starts the quick one itself).

A leaf module with no ``registry.register``: the wire contract
(``tui_gateway/contracts/server_requests.py``) imports this enum, and importing
``tools/tour_tool.py`` there would register ``gui_tour`` as a side effect.
"""

from enum import StrEnum


class TourPreset(StrEnum):
    """Which built-in tour ``start`` without steps runs."""

    quick = "quick"
    full = "full"
