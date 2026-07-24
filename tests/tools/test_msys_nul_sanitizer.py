import pytest
from tools.environments.local import _sanitize_msys_nul_redirection

def test_sanitize_nul_stderr_redirection():
    cmd = "uv pip list 2>nul || python -m pip list 2>nul || true"
    expected = "uv pip list 2>/dev/null || python -m pip list 2>/dev/null || true"
    assert _sanitize_msys_nul_redirection(cmd) == expected

def test_sanitize_nul_stdout_redirection():
    cmd = "taskkill /f /im hermes.exe >nul 2>&1"
    expected = "taskkill /f /im hermes.exe >/dev/null 2>&1"
    assert _sanitize_msys_nul_redirection(cmd) == expected

def test_sanitize_nul_with_spaces():
    cmd = "timeout /t 1 /nobreak > nul"
    expected = "timeout /t 1 /nobreak > /dev/null"
    assert _sanitize_msys_nul_redirection(cmd) == expected

def test_preserve_normal_file_or_param():
    cmd = "python script.py --param nul > log.txt"
    assert _sanitize_msys_nul_redirection(cmd) == cmd

def test_preserve_cat_command():
    cmd = "cat nul_data.json"
    assert _sanitize_msys_nul_redirection(cmd) == cmd
