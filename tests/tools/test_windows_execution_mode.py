from tools.environments.windows_execution_mode import (
    distribution_from_wsl_path,
    normalize_mode,
    parse_wsl_distributions,
    resolve_windows_execution_mode,
    windows_to_wsl_path,
    wsl_to_windows_path,
)


def test_normalize_mode_defaults_to_smart():
    assert normalize_mode("wsl2") == "wsl2"
    assert normalize_mode("unknown") == "smart"


def test_parse_utf16_wsl_output():
    raw = "Ubuntu\r\nDebian\r\n".encode("utf-16-le")
    assert parse_wsl_distributions(raw) == ["Ubuntu", "Debian"]


def test_smart_mode_uses_wsl_unc_project():
    result = resolve_windows_execution_mode(
        r"\\wsl.localhost\Ubuntu\home\shay\project",
        configured="smart",
        distributions=["Ubuntu"],
        is_windows=True,
    )
    assert result.resolved == "wsl2"
    assert result.distribution == "Ubuntu"


def test_smart_mode_keeps_drive_project_native():
    result = resolve_windows_execution_mode(
        r"C:\Users\shay\project",
        configured="smart",
        distributions=["Ubuntu"],
        is_windows=True,
    )
    assert result.resolved == "windows-native"


def test_explicit_wsl_uses_first_distribution():
    result = resolve_windows_execution_mode(
        r"C:\Users\shay\project",
        configured="wsl2",
        distributions=["Ubuntu", "Debian"],
        is_windows=True,
    )
    assert result.resolved == "wsl2"
    assert result.distribution == "Ubuntu"


def test_path_bridges():
    assert windows_to_wsl_path(r"C:\Users\shay\project", "Ubuntu") == "/mnt/c/Users/shay/project"
    assert windows_to_wsl_path(r"\\wsl$\Ubuntu\home\shay", "Ubuntu") == "/home/shay"
    assert wsl_to_windows_path("/mnt/c/Users/shay", "Ubuntu") == r"C:\Users\shay"
    assert wsl_to_windows_path("/home/shay", "Ubuntu") == r"\\wsl.localhost\Ubuntu\home\shay"
    assert distribution_from_wsl_path(r"\\wsl.localhost\Ubuntu\home") == "Ubuntu"
