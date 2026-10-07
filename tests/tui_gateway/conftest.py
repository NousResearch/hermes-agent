"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"rabbit_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``rabbit_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``rabbit_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

import rabbit_bootstrap  # noqa: F401
