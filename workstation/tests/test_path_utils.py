from __future__ import annotations

import pytest

from workstation.path_utils import PathSyntax, classify_path, is_sensitive_path, path_is_within


@pytest.mark.parametrize(
    ("value", "syntax"),
    [
        (r"C:\Windows\System32\calc.exe", PathSyntax.WINDOWS_ABSOLUTE),
        ("C:/clean", PathSyntax.WINDOWS_ABSOLUTE),
        (r"D:\workspace\file.txt", PathSyntax.WINDOWS_ABSOLUTE),
        (r"\\server\share\file.txt", PathSyntax.WINDOWS_ABSOLUTE),
        ("//server/share/file.txt", PathSyntax.WINDOWS_ABSOLUTE),
        ("/etc/hosts", PathSyntax.POSIX_ABSOLUTE),
        ("relative/file.txt", PathSyntax.RELATIVE),
        ("C:relative.txt", PathSyntax.RELATIVE),
    ],
)
def test_classify_path_does_not_use_host_path_semantics(value, syntax):
    result = classify_path(value)

    assert result.syntax is syntax
    assert result.is_absolute is (syntax is not PathSyntax.RELATIVE)


@pytest.mark.parametrize(
    ("parent", "child", "expected"),
    [
        (r"C:\Workspace", r"c:/workspace\src\main.py", True),
        (r"C:\Workspace", r"C:\Workspace-old\main.py", False),
        (r"C:\Workspace", r"D:\Workspace\main.py", None),
        (r"\\Server\Share\Workspace", r"//server/share/workspace\x.txt", True),
        ("/tmp/workspace", "/tmp/workspace/src/main.py", True),
        ("/tmp/workspace", "/tmp/workspace-old/main.py", False),
        (r"C:\Workspace", "/tmp/workspace/main.py", None),
        ("relative", "/tmp/workspace/main.py", None),
    ],
)
def test_path_containment_is_lexical_and_fail_closed_for_incompatible_syntax(parent, child, expected):
    assert path_is_within(parent, child) is expected


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Windows\System32\calc.exe",
        "c:/WINDOWS/system32/calc.exe",
        r"D:\Program Files\Hermes\bin.exe",
        r"\\server\c$\Windows\System32\calc.exe",
        "/etc/hosts",
        "/usr/bin/python",
    ],
)
def test_sensitive_paths_are_host_independent(value):
    assert is_sensitive_path(value) is True


@pytest.mark.parametrize("value", ["relative.txt", "C:relative.txt", "/tmp/workspace/file.txt"])
def test_non_sensitive_and_relative_paths_are_not_misclassified(value):
    assert is_sensitive_path(value) is False
