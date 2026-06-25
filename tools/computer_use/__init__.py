"""Native Hermes Computer Use package.

The public model-facing surface is registered by ``tools.computer_use_tool`` as
explicit ``computer_use_*`` tools. This package contains shared policy,
backend, dispatch, and response-shaping code.
"""

from __future__ import annotations

from tools.computer_use.tool import (  # noqa: F401
    check_computer_use_requirements,
    handle_computer_use,
    set_approval_callback,
)
