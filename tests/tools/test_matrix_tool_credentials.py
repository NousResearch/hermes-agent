"""Supported Matrix login methods remain available in cached discovery."""
import pytest
from tools import matrix_tool

@pytest.mark.parametrize("values,expected", [
    ({"MATRIX_HOMESERVER":"https://matrix.example.org", "MATRIX_ACCESS_TOKEN":"token"}, True),
    ({"MATRIX_HOMESERVER":"https://matrix.example.org", "MATRIX_USER_ID":"@user:example.org", "MATRIX_PASSWORD":"password"}, True),
    ({"MATRIX_HOMESERVER":"https://matrix.example.org", "MATRIX_USER_ID":"@user:example.org"}, False),
    ({"MATRIX_HOMESERVER":"https://matrix.example.org", "MATRIX_PASSWORD":"password"}, False),
    ({"MATRIX_ACCESS_TOKEN":"token"}, False),
    ({}, False),
])
def test_supported_login_credentials(monkeypatch, values, expected):
    monkeypatch.setattr(matrix_tool, "get_secret", values.get)
    assert matrix_tool.check_matrix_tool_requirements() is expected
