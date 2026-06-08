#!/usr/bin/env python3
"""Generate the Cron Recipes catalog JSON for the docs site.

Mirrors ``extract-skills.py``: imports the single-source-of-truth recipe
definitions from ``cron/recipe_catalog.py`` and emits a flat JSON array the
docs page renders into cards (description, schedule, copy-paste slash command,
and a ``hermes://`` "Send to App" deep-link).

Output: ``website/static/api/cron-recipes-index.json`` (served at
``/docs/api/cron-recipes-index.json``). Run automatically by
``website/scripts/prebuild.mjs`` before ``npm start`` / ``npm run build``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root = two levels up from website/scripts/.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT = REPO_ROOT / "website" / "static" / "api" / "cron-recipes-index.json"


def build_index() -> list:
    from cron.recipe_catalog import CATALOG, recipe_catalog_entry

    return [recipe_catalog_entry(r) for r in CATALOG]


def main() -> int:
    try:
        index = build_index()
    except Exception as e:  # pragma: no cover - import/build failure
        # Match extract-skills.py's resilience: write an empty array so the
        # docs build never hard-fails on a generator hiccup.
        sys.stderr.write(f"extract-cron-recipes: {e}; writing empty index\n")
        index = []

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    sys.stderr.write(f"extract-cron-recipes: wrote {len(index)} recipes -> {OUTPUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
