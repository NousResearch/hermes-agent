"""Wintermute pulse — entry point run by the Hermes cron job.

Hermes only runs cron scripts that live in $HERMES_HOME/scripts/, and it runs them
with its own Python (the shebang is ignored). The logic lives in the engine package
next to Wintermute's state in $HERMES_HOME/wintermute/.

stdout is injected into the cron prompt. A last line of {"wakeAgent": false} makes
Hermes skip the agent turn for this tick.

Manual check:  python3 ~/.hermes/scripts/wintermute_pulse.py
"""

import os
import sys
from pathlib import Path

_home = Path(os.environ.get("HERMES_HOME", "").strip() or Path.home() / ".hermes").expanduser()
sys.path.insert(0, str(_home / "wintermute"))

from wintermute_engine.pulse import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
