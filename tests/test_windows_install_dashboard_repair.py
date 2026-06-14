from pathlib import Path


INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _install_source() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_windows_installer_builds_dashboard_web_dist():
    source = _install_source()

    assert (
        "function _Run-NpmBuild"
        "([string]$label, [string]$buildDir, [string]$logPath, [string]$npmPath)"
    ) in source
    assert '$webDir = "$InstallDir\\web"' in source
    assert '& $npmPath run build' in source
    assert '"Dashboard web UI" $webDir $webLog $npmExe' in source


def test_windows_installer_repairs_missing_command_launchers():
    source = _install_source()

    assert '$InstallDir\\venv\\Scripts\\hermes.exe' in source
    assert '$InstallDir\\venv\\Scripts\\hermes-gateway.exe' in source
    assert (
        "$missingLaunchers = @($launcherPaths | Where-Object { -not (Test-Path $_) })"
    ) in source
    assert '& $UvCmd pip install -e .' in source
