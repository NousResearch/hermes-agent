"""Bot deliveries must not pin the caller's obsolete dependency generation."""
from pathlib import Path
import sys

import pytest

from tools import bot_relay


def _delivery_argv(profile, query_file):
    # Bot-chat delivery runs over the gateway here; the DM path (tools/bot_mode_dm.py) still spawns
    # ``_hermes_cli() -p <profile> ...``, so the launcher it picks is the invariant under test.
    return [bot_relay._hermes_cli(), "-p", profile, "--query-file", query_file]


@pytest.fixture
def launchers(tmp_path, monkeypatch):
    root = tmp_path / "source install"
    module = root / "tools" / "bot_relay.py"
    module.parent.mkdir(parents=True)
    module.touch()
    monkeypatch.setattr(bot_relay, "__file__", str(module))
    old_bin = tmp_path / "old generation" / "bin"
    old_bin.mkdir(parents=True)
    name = "hermes.exe" if sys.platform == "win32" else "hermes"
    sibling = old_bin / name
    sibling.touch()
    monkeypatch.setattr(sys, "executable", str(old_bin / "python"))
    published = root / ".hermes" / "bin" / name
    published.parent.mkdir(parents=True)
    return published, sibling


def test_delivery_prefers_install_launcher_over_old_generation(launchers):
    published, sibling = launchers
    published.touch()
    argv = _delivery_argv("researcher", "message with spaces.txt")
    assert argv[0] == str(published)
    assert str(sibling) not in argv


def test_unpublished_install_retains_interpreter_sibling(launchers):
    _, sibling = launchers
    assert _delivery_argv("default", "body.txt")[0] == str(sibling)


@pytest.mark.platforms("windows")
def test_windows_delivery_does_not_select_batch_shims(launchers):
    published, sibling = launchers
    published.with_suffix(".cmd").touch()
    assert _delivery_argv("default", "body.txt")[0] == str(sibling)


def test_path_then_bare_fallback_remain_available(launchers, monkeypatch):
    _, sibling = launchers
    sibling.unlink()
    monkeypatch.setattr(bot_relay.shutil, "which", lambda name: "external-hermes")
    assert bot_relay._hermes_cli() == "external-hermes"
    monkeypatch.setattr(bot_relay.shutil, "which", lambda name: None)
    assert bot_relay._hermes_cli() == "hermes"


@pytest.mark.platforms("windows")
def test_real_delivery_launcher_imports_new_generation(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    from hermes_cli import _launchers
    from pm.environments import runtime_facts_path, site_packages

    real_python = Path(sys.executable)
    real_root = Path(bot_relay.__file__).resolve().parents[1]
    root = tmp_path / "install with spaces"
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    # Load the checkout's constants before the launcher's bootstrap import;
    # an editable test interpreter may also expose an older installed checkout.
    import shutil
    shutil.copyfile(real_root / "hermes_constants.py", root / "hermes_constants.py")
    (package / "__init__.py").write_text(
        f"__path__.append({str(real_root / 'hermes_cli')!r})\n", encoding="utf-8")
    (package / "main.py").write_text(
        "import json, sys\n"
        "def main():\n"
        "    import plugin_generation_probe\n"
        "    print(json.dumps([plugin_generation_probe.VALUE, sys.argv[1:]]))\n"
        "    return 0\n", encoding="utf-8")
    # Keep the entry point local and non-networked, but run real PM selection.
    (root / "hermes_bootstrap.py").write_text(
        f"import sys\nsys.path.append({str(real_root)!r})\n"
        "from pathlib import Path\nfrom pm.environments import activate_dependencies\n"
        "activate_dependencies(Path(__file__).resolve().parent)\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    out = root / ".hermes" / "bin"
    out.mkdir(parents=True)
    published = _launchers.mint_launcher("hermes", root, out, real_python, None)
    assert published is not None
    record = runtime_facts_path(root)
    record.parent.mkdir(parents=True, exist_ok=True)
    old_bin = tmp_path / "old" / "Scripts"
    old_bin.mkdir(parents=True)
    # Both choices are executable: the old console script reports missing deps.
    old_package = tmp_path / "old source" / "hermes_cli"
    old_package.mkdir(parents=True)
    (old_package / "__init__.py").write_text("", encoding="utf-8")
    (old_package / "main.py").write_text(
        "def main():\n    print('old-generation')\n    return 0\n", encoding="utf-8")
    (old_package.parent / "hermes_bootstrap.py").write_text("", encoding="utf-8")
    assert _launchers.mint_launcher("hermes", old_package.parent, old_bin, real_python, None)
    monkeypatch.setattr(bot_relay, "__file__", str(root / "tools" / "bot_relay.py"))
    monkeypatch.setattr(sys, "executable", str(old_bin / "python.exe"))
    argv = _delivery_argv("researcher", str(tmp_path / "message&extra.txt"))
    for generation in ("first", "new-plugin"):
        selected = record.parent / "environments" / generation / "venv"
        packages = site_packages(selected)
        packages.mkdir(parents=True)
        (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        (packages / "plugin_generation_probe.py").write_text(
            f"VALUE = {generation!r}\n", encoding="utf-8")
        record.write_text(json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
        result = subprocess.run(argv, cwd=tmp_path, env=dict(os.environ),
                                capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == json.dumps([generation, argv[1:]])
    # A broken selected generation must fail, never silently reuse the old CLI.
    record.write_text(json.dumps({"packages": {"venv": {"environment": str(record.parent / 'missing')}}}), encoding="utf-8")
    result = subprocess.run(argv, cwd=tmp_path, env=dict(os.environ),
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode != 0
    assert "dependency environment" in result.stderr
    assert "old-generation" not in result.stdout


@pytest.mark.parametrize("name", ["hermes", "hermes.exe"])
def test_launcher_shape_preserves_profile_and_lock(tmp_path, monkeypatch, name):
    import contextlib
    from tools import bot_mode_dm, bot_mode_probe

    home = tmp_path / "home"
    target = home / "profiles" / "researcher"
    monkeypatch.setattr(bot_mode_dm, "_default_home", lambda: str(home))
    monkeypatch.setattr(bot_mode_probe, "_hermes_root", lambda path: home)
    monkeypatch.setattr(bot_mode_probe, "_roster", lambda root: [("researcher", target)])
    locked = []
    @contextlib.contextmanager
    def lock(root, profile):
        locked.append((root, profile))
        yield
    monkeypatch.setattr(bot_relay, "acquire_turn_lock", lock)
    argv = [str(tmp_path / name), "-p", "researcher", "chat"]
    assert bot_mode_dm._local_delivery_home(argv) == target
    with bot_mode_dm._delivery_lock(argv, stdin_file=False):
        assert locked == [(home, "researcher")]
    with bot_mode_dm._delivery_lock(argv, stdin_file=True):
        assert locked == [(home, "researcher")]
