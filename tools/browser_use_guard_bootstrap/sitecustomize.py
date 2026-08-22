"""sitecustomize bootstrap for the browser_exec egress guard (Region B/E H4).

Loaded at interpreter boot via PYTHONPATH injection (the guard dir is
prepended by ``tools/browser_exec_egress_guard._install_egress_guard``),
BEFORE any harness or model code imports. Delegates to the sibling
``browser_exec_egress_guard`` module (stdlib-only) which performs the actual
socket interposition. When the guard is disabled (env flag != "1") this is a
no-op so the CLI runs untouched.
"""

import os


def _install() -> None:
    if os.environ.get("HERMES_BROWSER_EXEC_EGRESS_GUARD") != "1":
        return
    try:
        import browser_exec_egress_guard as _guard

        _guard.install()
    except Exception:
        # Never break the CLI on guard import failure — the parent fails
        # closed on the missing/``:disabled:`` marker instead.
        try:
            os.write(2, b"__HERMES_EGRESS_GUARD__:disabled:import-failure\n")
        except Exception:
            pass


_install()
