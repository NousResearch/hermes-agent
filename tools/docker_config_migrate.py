#!/usr/bin/env python3
"""Docker container-boot config migration.

Runs the same non-interactive config-schema migration that ``hermes update``
performs for non-Docker installs. ``hermes update`` intentionally exits early
for Docker installs, and the container-boot hook only seeds missing files, so
a persistent ``$HERMES_HOME`` volume can otherwise carry a stale/unversioned
raw config schema across image updates and never have version-gated
migrations applied (#35406).

Invoked from docker/stage2-hook.sh as the ``hermes`` user against the active
``$HERMES_HOME``. Honors ``HERMES_SKIP_CONFIG_MIGRATION=1`` as an operator
opt-out (handled inside ``run_boot_config_migration``). Always exits 0 so a
migration hiccup never blocks container start — the gateway's own startup
validation still surfaces a broken config.
"""

import logging
import sys

from hermes_cli.config import run_boot_config_migration

logging.basicConfig(level=logging.WARNING, format="[config-migrate] %(message)s")


def main() -> int:
    try:
        outcome = run_boot_config_migration(quiet=False)
    except Exception:  # defensive: run_boot_config_migration shouldn't raise
        logging.getLogger(__name__).warning(
            "container-boot config migration crashed; continuing", exc_info=True
        )
        return 0

    status = outcome.get("status")
    if status not in {"up_to_date", "no_config", "skipped_optout"}:
        print(f"[config-migrate] done (status={status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
