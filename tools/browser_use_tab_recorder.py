"""Record the tabs a ``browser_exec`` call creates, from inside the browser-harness process.

browser_exec installs this over ``browser_harness.helpers.cdp`` (see ``browser_use_shot.harness_command``)
before ``browser_harness.run`` star-imports the helpers, so the model's code, ``new_tab()`` and
``close_tab()`` all go through the wrapper. Each successful ``Target.createTarget`` reply's ``targetId``
is written to the per-call 0600 report file named by ``_HERMES_BU_TAB_REPORT`` (set by browser_exec only
on a shared browser; unset, nothing is installed), and a successful ``Target.closeTarget`` takes it out.
The report is rewritten after every event, so a call that is killed or times out still says what it
opened. Hermes merges the report into ``tools.browser_session_owned_tabs`` after the call and closes
only those tabs when the session ends: a tab the call merely switched to is never recorded.

The variable is taken out of the environment when the recorder installs, so neither a subprocess the
model's code starts nor the harness daemon inherits it (a second recorder would overwrite this call's
report). Recording is best-effort: a report that cannot be written never fails the CDP call, and the last
report that was written stays.

Loaded by file path inside the harness interpreter, which has no Hermes checkout on ``sys.path``: this
module imports only the standard library.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
from typing import Any

REPORT_ENV = "_HERMES_BU_TAB_REPORT"
TMP_SUFFIX = ".tmp"  # the half-written report next to it; Hermes removes a leftover one with the report


def _persist(path: str, state: dict[str, list[str]]) -> None:
    pending = path + TMP_SUFFIX
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(state, stream)
    os.replace(pending, path)


def install(helpers: Any) -> None:
    """Wrap ``helpers.cdp`` so created and closed target ids land in ``$_HERMES_BU_TAB_REPORT``."""
    report_path = os.environ.pop(REPORT_ENV, None)
    if not report_path:
        return
    original = helpers.cdp
    state: dict[str, list[str]] = {"created": [], "closed": []}
    lock = threading.Lock()  # the model's code may call cdp() from several threads
    warned = []

    def record(target: str, created: bool) -> None:
        with lock:
            if created:
                state["created"].append(target)
            elif target in state["created"]:
                state["created"].remove(target)
            else:
                state["closed"].append(target)  # opened by an earlier call of this session
            try:
                _persist(report_path, state)
            except OSError as exc:
                if not warned:
                    warned.append(exc)
                    print(f"hermes: cannot record the tabs this call opens, they may stay open: {exc}",
                          file=sys.stderr)

    @functools.wraps(original)
    def cdp(method, *args, **kwargs):
        result = original(method, *args, **kwargs)  # a CDP error raises: nothing is recorded
        if method == "Target.createTarget":
            target = result.get("targetId") if isinstance(result, dict) else None
        elif method == "Target.closeTarget":
            target = kwargs.get("targetId")
        else:
            return result
        if isinstance(target, str) and target:
            record(target, created=method == "Target.createTarget")
        return result

    helpers.cdp = cdp
