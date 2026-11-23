"""load_godmode finds a hub install under a custom category.

The installer records that category in the lock file and resolves it with
``_resolve_lock_install_path``. The loader used to look only at
``skills/security/godmode``.
"""

from __future__ import annotations

from pathlib import Path

from tools.skills_hub import HubLockFile

_LOADER = Path("optional-skills/security/godmode/scripts/load_godmode.py")
_SCRIPTS = ("parseltongue.py", "godmode_race.py", "auto_jailbreak.py")


def _stub(where: str) -> str:
    return f"def godmode_install_marker():\n    return {where!r}\n"


def _write_scripts(home: Path, category: str, where: str) -> None:
    scripts = home / "skills" / category / "godmode" / "scripts"
    scripts.mkdir(parents=True)
    for name in _SCRIPTS:
        (scripts / name).write_text(_stub(where), encoding="utf-8")


def _record(category: str) -> None:
    HubLockFile().record_install(
        name="godmode",
        source="official",
        identifier=f"official/{category}/godmode",
        trust_level="builtin",
        scan_verdict="safe",
        skill_hash="sha256:fixture",
        install_path=f"{category}/godmode",
        files=[f"scripts/{name}" for name in _SCRIPTS],
    )


def _load():
    namespace = {"__name__": "_godmode_loader_test"}
    exec(compile(_LOADER.read_text(encoding="utf-8"), "load_godmode.py", "exec"), namespace)
    return namespace["godmode_install_marker"]()


def test_custom_category_install_is_found_via_the_installer_lock(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_scripts(home, "custom-lab", "custom-lab")
    _record("custom-lab")

    assert _load() == "custom-lab"


def test_security_copy_remains_the_fallback_without_a_hub_install(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_scripts(home, "security", "security")

    assert _load() == "security"


def test_custom_category_wins_over_the_security_fallback(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_scripts(home, "custom-lab", "custom-lab")
    _write_scripts(home, "security", "security")
    _record("custom-lab")

    assert _load() == "custom-lab"


def test_missing_custom_install_falls_back_to_security(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_scripts(home, "security", "security")
    _record("custom-lab")

    assert _load() == "security"
