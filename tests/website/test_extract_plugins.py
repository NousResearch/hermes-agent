"""Tests for website/scripts/extract-plugins.py.

Behavioral contracts for the /docs/plugins catalog extractor:

1. Reads ``plugin-catalog/*.yaml`` entries (skipping ``removed.yaml``) and
   emits ``plugins.json`` rows carrying name/repo/sha/tier/capabilities plus
   a synthesized ``hermes plugins install <name>`` command.
2. Entries missing any of name/repo/sha are skipped (logged, not fatal).
3. A missing ``plugin-catalog/`` directory degrades gracefully: empty
   catalog list, zero counts in the meta sidecar, exit 0 — the docs build
   must stay green before the catalog directory lands on main.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT = REPO_ROOT / "website" / "scripts" / "extract-plugins.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_plugins", EXTRACT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_entry(catalog_dir: Path, name: str, **overrides) -> Path:
    import yaml

    entry = {
        "name": name,
        "repo": f"https://github.com/example/{name}",
        "sha": "38fe0fb53eff98d477f807432e965429e665ca33",
        "description": f"{name} does things.",
        "maintainer": "Example",
        "tier": "community",
    }
    entry.update(overrides)
    # Drop keys explicitly set to None so tests can simulate missing fields.
    entry = {k: v for k, v in entry.items() if v is not None}
    path = catalog_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(entry), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Entry loading + validation
# --------------------------------------------------------------------------

def test_valid_entry_is_extracted_with_install_command(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(
        catalog,
        "example-plugin",
        tier="official",
        docs_url="https://example.com/docs",
        requires_hermes=">=0.19",
        platforms=["linux"],
        capabilities={
            "provides_tools": ["do_thing"],
            "provides_hooks": ["on_start"],
            "provides_middleware": [],
            "requires_env": ["EXAMPLE_TOKEN"],
        },
    )

    entries = mod.load_catalog_entries(catalog)

    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "example-plugin"
    assert e["repo"] == "https://github.com/example/example-plugin"
    assert e["sha"] == "38fe0fb53eff98d477f807432e965429e665ca33"
    assert e["shaShort"] == "38fe0fb"
    assert e["tier"] == "official"
    assert e["maintainer"] == "Example"
    assert e["requiresHermes"] == ">=0.19"
    assert e["platforms"] == ["linux"]
    assert e["docsUrl"] == "https://example.com/docs"
    assert e["capabilities"]["providesTools"] == ["do_thing"]
    assert e["capabilities"]["providesHooks"] == ["on_start"]
    assert e["capabilities"]["requiresEnv"] == ["EXAMPLE_TOKEN"]
    assert e["installCommand"] == "hermes plugins install example-plugin"


def test_entries_missing_required_fields_are_skipped(mod, tmp_path, capsys):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "good-plugin")
    _write_entry(catalog, "no-sha", sha=None)
    _write_entry(catalog, "no-repo", repo=None)

    entries = mod.load_catalog_entries(catalog)

    assert [e["name"] for e in entries] == ["good-plugin"]
    err = capsys.readouterr().err
    assert "no-sha" in err
    assert "no-repo" in err


def test_removed_yaml_is_not_treated_as_an_entry(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "kept-plugin")
    (catalog / "removed.yaml").write_text(
        "removed:\n  - name: evil-plugin\n    repo: https://github.com/evil/x\n"
        '    reason: "bad"\n    date: "2026-07-02"\n',
        encoding="utf-8",
    )

    entries = mod.load_catalog_entries(catalog)
    assert [e["name"] for e in entries] == ["kept-plugin"]
    assert mod.count_removed(catalog) == 1


def test_unknown_tier_normalizes_to_community(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "weird-tier", tier="platinum")

    entries = mod.load_catalog_entries(catalog)
    assert entries[0]["tier"] == "community"


# --------------------------------------------------------------------------
# Full run: outputs + graceful degradation
# --------------------------------------------------------------------------

def test_main_writes_catalog_and_meta(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "alpha", tier="official")
    _write_entry(catalog, "beta")
    (catalog / "removed.yaml").write_text(
        "removed:\n  - name: gone\n", encoding="utf-8"
    )
    out_dir = tmp_path / "api"

    rc = mod.main(catalog_dir=catalog, output_dir=out_dir)

    assert rc == 0
    plugins = json.loads((out_dir / "plugins.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "plugins-meta.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in plugins] == ["alpha", "beta"]
    assert meta["total"] == 2
    assert meta["byTier"] == {"official": 1, "community": 1}
    assert meta["removedCount"] == 1
    assert meta["generatedAt"]


def test_missing_catalog_dir_degrades_to_empty_outputs_exit_zero(mod, tmp_path):
    out_dir = tmp_path / "api"

    rc = mod.main(catalog_dir=tmp_path / "does-not-exist", output_dir=out_dir)

    assert rc == 0
    plugins = json.loads((out_dir / "plugins.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "plugins-meta.json").read_text(encoding="utf-8"))
    assert plugins == []
    assert meta["total"] == 0
    assert meta["byTier"] == {"official": 0, "community": 0}
    assert meta["removedCount"] == 0


def test_script_exits_zero_as_subprocess_when_catalog_missing(tmp_path):
    """CLI contract: the deploy step runs the script hard (no `|| true`);
    it must exit 0 even when plugin-catalog/ hasn't landed yet."""
    out_dir = tmp_path / "api"
    result = subprocess.run(
        [
            sys.executable,
            str(EXTRACT),
            "--catalog-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "plugins.json").exists()
    assert (out_dir / "plugins-meta.json").exists()
