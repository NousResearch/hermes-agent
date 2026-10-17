"""execute_code's ``credentials=`` binding: fail loudly, never echo, register with the redactor."""

import pytest

import agent.credential_store as cs
from tools.code_execution_tool import _inject_credentials


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "get_hermes_home", lambda: tmp_path)
    cs.reset_store()
    yield
    cs.reset_store()


def test_no_credentials_leaves_code_untouched():
    code = "print(1)"
    assert _inject_credentials(code, None) == (code, None)
    assert _inject_credentials(code, []) == (code, None)


def test_a_bare_string_is_rejected_not_iterated():
    """`credentials=\"name\"` must not silently iterate the string's characters."""
    _code, error = _inject_credentials("print(1)", "supabase_service_key")
    assert error and "array of credential names" in error


def test_missing_credential_fails_before_running():
    code, error = _inject_credentials("print(1)", ["not_stored_name"])
    assert error and "not stored" in error
    assert code == "print(1)"  # original code returned unchanged


def test_unbindable_name_is_rejected():
    _code, error = _inject_credentials("print(1)", ["not-an-identifier"])
    assert error and "identifiers" in error


def test_binds_value_registers_redactor_and_keeps_code_last():
    cs.get_store().set("supabase_service_key", "sk-live-Value42")
    code, error = _inject_credentials("print('body')", ["supabase_service_key"])
    assert error is None
    assert code.endswith("print('body')")
    assert "supabase_service_key = 'sk-live-Value42'" in code

    # Bound values are registered: any output path masks them.
    from agent.redact import redact_sensitive_text

    assert "sk-live-Value42" not in redact_sensitive_text(
        f"stdout was: {code.splitlines()[1]}", force=True)


def test_corrupt_store_is_reported_not_ignored(tmp_path):
    cs.get_store().set("alpha_token_one", "value1")
    cs.get_store()._store_path.write_bytes(b"\x00broken")
    _code, error = _inject_credentials("print(1)", ["alpha_token_one"])
    assert error and "unavailable" in error