"""The core bridge applies plugin decisions without owning domain policy."""

import pytest

from tools import write_boundary as boundary


def _install(monkeypatch, result):
    import hermes_cli.plugins as plugins

    calls = []

    monkeypatch.setattr(plugins, "has_hook", lambda name: name == "write_boundary_provider")

    def invoke(name, **kwargs):
        calls.append((name, kwargs))
        return result(kwargs) if callable(result) else result

    monkeypatch.setattr(plugins, "invoke_hook", invoke)
    return calls


def _value(value):
    return [{"contract": "hermes.write-boundary/v1", "value": value}]


def test_absent_provider_leaves_an_unrequired_profile_unchanged(monkeypatch, tmp_path):
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, "has_hook", lambda _name: False)
    monkeypatch.setattr(boundary, "_require_loaded", lambda _name: None)
    home = str(tmp_path)
    assert boundary.protected_basenames(home) == frozenset()
    assert boundary.refuse_command("printf x > note.txt", home) is None
    assert boundary.guard_command("echo hi", env_type="local", home=home) == "echo hi"
    assert boundary.wrap_code("print(1)", home) == "print(1)"
    assert boundary.guard_process_argv(["python3"], home=home) == ["python3"]


def test_provider_decisions_are_scoped_and_applied(monkeypatch, tmp_path):
    home = tmp_path / "big"
    calls = _install(monkeypatch, lambda kwargs: _value({
        "protected_basenames": ["facts.json"],
        "refuse_paths": "blocked",
        "refuse_command": "blocked",
        "guard_command": "wrapped command",
        "wrap_code": "wrapped code",
        "guard_process_argv": ["sandbox", "python3"],
        "guard_remote_process_command": "sandbox python3",
    }[kwargs["operation"]]))
    assert boundary.protected_basenames(str(home)) == frozenset({"facts.json"})
    assert boundary.refuse_paths(["facts.json"], str(home)) == "blocked"
    assert boundary.refuse_command("touch facts.json", str(home)) == "blocked"
    assert boundary.guard_command("echo hi", env_type="local", home=str(home)) == "wrapped command"
    assert boundary.wrap_code("print(1)", str(home)) == "wrapped code"
    assert boundary.guard_process_argv(["python3"], home=str(home)) == ["sandbox", "python3"]
    assert boundary.guard_remote_process_command(["python3"], home=str(home)) == "sandbox python3"
    assert all(kwargs["hermes_home"] == str(home.resolve()) for _, kwargs in calls)


@pytest.mark.parametrize("result", [
    [],
    [{"action": "block", "message": "timed out"}],
    [{"contract": "wrong", "value": "unsafe"}],
    _value("a") + _value("b"),
])
def test_invalid_or_failed_provider_does_not_allow_execution(monkeypatch, tmp_path, result):
    _install(monkeypatch, result)
    home = str(tmp_path)
    assert boundary.guard_command("echo hi", env_type="local", home=home) is None
    with pytest.raises(PermissionError):
        boundary.wrap_code("print(1)", home)
    assert boundary.refuse_command("touch note.txt", home)


def test_required_plugin_absence_blocks_even_when_no_hook_is_registered(monkeypatch, tmp_path):
    import hermes_cli.plugins as plugins
    monkeypatch.setattr(plugins, "has_hook", lambda _name: False)

    def require(_name):
        raise PermissionError("required plugin missing")

    monkeypatch.setattr(boundary, "_require_loaded", require)
    assert boundary.guard_command("echo hi", env_type="local", home=str(tmp_path)) is None
    assert boundary.refuse_command("touch note.txt", str(tmp_path)) == "required plugin missing"
    with pytest.raises(PermissionError):
        boundary.guard_process_argv(["python3"], home=str(tmp_path))
