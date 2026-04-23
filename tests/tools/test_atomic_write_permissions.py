import os
import stat
from unittest.mock import patch

from tools.skill_manager_tool import _atomic_write_text
from tools.skills_sync import _write_manifest


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_atomic_write_text_preserves_existing_file_mode(tmp_path, monkeypatch):
    target = tmp_path / "demo-skill" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("old content", encoding="utf-8")
    os.chmod(target, 0o664)

    original_replace = os.replace
    seen = {}

    def capture_replace(src, dst):
        seen["temp_mode"] = _mode(src)
        return original_replace(src, dst)

    monkeypatch.setattr("tools.skill_manager_tool.os.replace", capture_replace)

    _atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "new content"
    assert seen["temp_mode"] == 0o664
    assert _mode(target) == 0o664


def test_write_manifest_preserves_existing_file_mode(tmp_path, monkeypatch):
    manifest = tmp_path / ".bundled_manifest"
    manifest.write_text("old:hash\n", encoding="utf-8")
    os.chmod(manifest, 0o664)

    original_replace = os.replace
    seen = {}

    def capture_replace(src, dst):
        seen["temp_mode"] = _mode(src)
        return original_replace(src, dst)

    monkeypatch.setattr("tools.skills_sync.os.replace", capture_replace)

    with patch("tools.skills_sync.MANIFEST_FILE", manifest):
        _write_manifest({"skill-a": "abc123"})

    assert manifest.read_text(encoding="utf-8") == "skill-a:abc123\n"
    assert seen["temp_mode"] == 0o664
    assert _mode(manifest) == 0o664
