import importlib.util
import os
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "profiles" / "torben" / "scripts" / "torben_gtm_radar.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("torben_gtm_radar_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_magnus_radar_command_refreshes_json_without_delivery_context():
    script = _load_script_module()

    command = script._magnus_radar_command(
        hours_back=48,
        scanner_max_items=40,
        min_score=80,
        newsletter_files=[Path("/tmp/security-news.json"), Path("/tmp/ai-news.json")],
        dry_run=True,
    )

    assert command[:4] == ["uv", "run", "python", "scripts/cron_gtm_intelligence_radar.py"]
    assert "--json" in command
    assert "--no-delivery-context" in command
    assert command[command.index("--hours-back") + 1] == "48"
    assert command[command.index("--max-items") + 1] == "40"
    assert command[command.index("--min-score") + 1] == "80"
    assert command.count("--newsletter-file") == 2
    assert "--dry-run" in command
    assert "--include-seen" in command
    assert "--no-mark-seen" in command
    assert "--no-persist" in command


def test_torben_gtm_refresh_skips_preview_by_default(monkeypatch):
    script = _load_script_module()
    monkeypatch.delenv("TORBEN_GTM_REFRESH_MAGNUS", raising=False)
    monkeypatch.delenv("TORBEN_GTM_REFRESH_MAGNUS_PREVIEW", raising=False)

    assert script._refresh_magnus_enabled(preview=False) is True
    assert script._refresh_magnus_enabled(preview=True) is False

    monkeypatch.setenv("TORBEN_GTM_REFRESH_MAGNUS_PREVIEW", "1")
    assert script._refresh_magnus_enabled(preview=True) is True

    monkeypatch.setenv("TORBEN_GTM_REFRESH_MAGNUS", "0")
    assert script._refresh_magnus_enabled(preview=False) is False


def test_torben_gtm_newsletter_env_splits_and_dedupes(monkeypatch):
    script = _load_script_module()
    monkeypatch.setenv("TORBEN_GTM_INCLUDE_DEFAULT_NEWSLETTER", "0")
    monkeypatch.setenv("TORBEN_GTM_NEWSLETTER_FILE", "/tmp/a.json")
    monkeypatch.setenv("TORBEN_GTM_NEWSLETTER_FILES", os.pathsep.join(["/tmp/b.json", "/tmp/a.json", ""]))

    assert script._newsletter_files_from_env() == [Path("/tmp/a.json"), Path("/tmp/b.json")]


def test_torben_gtm_newsletter_includes_default_morning_artifact(monkeypatch, tmp_path):
    script = _load_script_module()
    home = tmp_path / "torben-home"
    default_file = home / "state" / "torben-morning-brief-inbox-context-latest.json"
    default_file.parent.mkdir(parents=True)
    default_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TORBEN_GTM_NEWSLETTER_FILE", raising=False)
    monkeypatch.delenv("TORBEN_GTM_NEWSLETTER_FILES", raising=False)
    monkeypatch.delenv("TORBEN_GTM_INCLUDE_DEFAULT_NEWSLETTER", raising=False)

    assert script._newsletter_files_from_env() == [default_file]


def test_torben_gtm_extracts_json_from_noisy_stdout():
    script = _load_script_module()

    payload = script._extract_json_object('scanner warmup\n{"success": true, "scanned_count": 12}\n')

    assert payload == {"success": True, "scanned_count": 12}
