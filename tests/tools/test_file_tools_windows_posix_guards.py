"""POSIX safety namespaces remain protected when Hermes runs on Windows."""

import ntpath
from unittest.mock import patch


def _windows_path_semantics(file_tools):
    return patch.multiple(file_tools.os, path=ntpath, sep="\\")


def test_device_and_proc_paths_stay_blocked_with_windows_path_semantics():
    from tools import file_tools

    blocked = (
        "/dev/zero",
        "/dev/stdin",
        "/proc/self/fd/0",
        "/proc/123/environ",
        "/proc/123/task/456/maps",
    )
    with _windows_path_semantics(file_tools):
        assert all(file_tools._is_blocked_device_path(path) for path in blocked)


def test_sensitive_posix_write_paths_stay_blocked_on_windows_hosts(monkeypatch):
    from tools import file_tools

    monkeypatch.setattr(file_tools, "_get_hermes_config_resolved", lambda: None)
    blocked = (
        "/etc/hosts",
        "/boot/loader.conf",
        "/usr/lib/systemd/system/demo.service",
        "/var/run/docker.sock",
    )
    with _windows_path_semantics(file_tools):
        assert all(file_tools._check_sensitive_path(path) for path in blocked)


def test_windows_native_paths_do_not_match_posix_guards(monkeypatch):
    from tools import file_tools

    monkeypatch.setattr(file_tools, "_get_hermes_config_resolved", lambda: None)
    with _windows_path_semantics(file_tools):
        assert file_tools._is_blocked_device_path(r"C:\\dev\\zero") is False
        assert file_tools._check_sensitive_path(r"C:\\etc\\hosts") is None
