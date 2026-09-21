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


def test_version_and_image_are_emitted_and_offhost_image_is_dropped_not_fatal(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "labelled", version="1.4.0", image="https://raw.githubusercontent.com/owner/repo/38fe0fb53eff98d477f807432e965429e665ca33/banner.png")
    _write_entry(catalog, "offhost", version="1.4.0", image="https://cdn.example.com/banner.png")

    entries = {e["name"]: e for e in mod.load_catalog_entries(catalog)}
    assert entries["labelled"]["version"] == "1.4.0" and entries["labelled"]["image"] == "https://raw.githubusercontent.com/owner/repo/38fe0fb53eff98d477f807432e965429e665ca33/banner.png"
    assert entries["offhost"]["image"] == "" and entries["offhost"]["version"] == "1.4.0"


# --------------------------------------------------------------------------
# Full run: outputs + graceful degradation
# --------------------------------------------------------------------------

def test_main_writes_catalog_and_meta(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "alpha", tier="official", category="memory")
    _write_entry(catalog, "beta")  # no category → default "desktop" shelf
    _write_entry(catalog, "gamma")
    # Star cache from fetch-plugin-stars.py: ranking is stars only; the official tier gets no boost.
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "plugin-stars.json").write_text(json.dumps({
        "fetched_at": "2026-09-15T00:00:00+00:00",
        "stars": {"example/gamma": 50, "example/beta": 3}}), encoding="utf-8")
    (catalog / "removed.yaml").write_text(
        "removed:\n  - name: gone\n", encoding="utf-8"
    )
    out_dir = tmp_path / "api"

    rc = mod.main(catalog_dir=catalog, output_dir=out_dir)

    assert rc == 0
    plugins = json.loads((out_dir / "plugins.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "plugins-meta.json").read_text(encoding="utf-8"))
    assert [p["name"] for p in plugins] == ["gamma", "beta", "alpha"]  # stars desc, unknown stars last
    assert {p["name"]: p["stars"] for p in plugins} == {"alpha": None, "gamma": 50, "beta": 3}
    assert meta["total"] == 3
    assert meta["byTier"] == {"official": 1, "community": 2}
    assert {p["name"]: p["category"] for p in plugins} == {"alpha": "memory", "beta": "desktop", "gamma": "desktop"}
    assert meta["byCategory"] == {"desktop": 2, "memory": 1}
    assert meta["starsFetchedAt"] == "2026-09-15T00:00:00+00:00"
    assert meta["removedCount"] == 1
    assert meta["generatedAt"]
    # The live-refresh document consumed by installed clients: loader-schema entries + the kill list.
    from hermes_cli.plugin_catalog import entry_from_mapping
    live = json.loads((out_dir / "plugin-catalog.json").read_text(encoding="utf-8"))
    assert [entry_from_mapping(raw, "live").name for raw in live["entries"]] == ["alpha", "beta", "gamma"]
    assert live["removed"] == [{"name": "gone"}]


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


# --------------------------------------------------------------------------
# addedAt / updatedAt from git history
# --------------------------------------------------------------------------

def _git(repo: Path, *args: str, date: str) -> None:
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "GIT_COMMITTER_DATE": date,
        "HOME": str(repo), "PATH": __import__("os").environ["PATH"],
    }
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


def test_git_dates_added_is_first_commit_updated_is_last_and_renames_keep_added(mod, tmp_path):
    repo = tmp_path / "repo"
    catalog = repo / "plugin-catalog"
    catalog.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main", date="2026-01-01T00:00:00+00:00")
    _write_entry(catalog, "alpha")
    _write_entry(catalog, "old-name")
    _git(repo, "add", ".", date="2026-01-01T00:00:00+00:00")
    _git(repo, "commit", "-q", "-m", "add", date="2026-01-01T00:00:00+00:00")
    # Pin bump on alpha only.
    _write_entry(catalog, "alpha", sha="a" * 40)
    _git(repo, "commit", "-q", "-am", "bump alpha", date="2026-02-01T00:00:00+00:00")
    # Rename old-name → new-name (content unchanged so git detects the rename).
    _git(repo, "mv", "plugin-catalog/old-name.yaml", "plugin-catalog/new-name.yaml", date="2026-03-01T00:00:00+00:00")
    _git(repo, "commit", "-q", "-m", "rename", date="2026-03-01T00:00:00+00:00")

    dates = mod.load_git_dates(catalog)

    assert dates["alpha.yaml"] == {"addedAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-02-01T00:00:00Z"}
    assert dates["new-name.yaml"] == {"addedAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-03-01T00:00:00Z"}
    entries = mod.load_catalog_entries(catalog, dates=dates)
    by_name = {e["name"]: e for e in entries}
    assert by_name["alpha"]["addedAt"] == "2026-01-01T00:00:00Z"
    assert by_name["alpha"]["updatedAt"] == "2026-02-01T00:00:00Z"


def test_git_dates_are_null_outside_a_repository(mod, tmp_path):
    catalog = tmp_path / "plugin-catalog"
    catalog.mkdir()
    _write_entry(catalog, "alpha")
    assert mod.load_git_dates(catalog) == {}
    entries = mod.load_catalog_entries(catalog)
    assert entries[0]["addedAt"] is None and entries[0]["updatedAt"] is None
