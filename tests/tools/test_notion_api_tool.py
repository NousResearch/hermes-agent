"""Unit tests for the notion_api tool. ntn is not installed in CI, so the
subprocess and binary lookup are mocked; these tests pin argv construction,
validation, env bridging, and result shaping."""
import json
import subprocess

import pytest

import tools.notion_api_tool as nt


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def ntn_ok(monkeypatch):
    """Make the binary present and the token resolvable for happy-path tests."""
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: "/usr/local/bin/ntn")
    monkeypatch.setattr(nt, "_ntn_env", lambda: {"NOTION_API_TOKEN": "tok"})


def _call(args):
    return json.loads(nt._notion_api_handler(args))


def test_registered_under_notion_toolset():
    from tools.registry import registry
    assert registry.get_toolset_for_tool("notion_api") == "notion"
    schema = registry.get_schema("notion_api")
    assert schema and schema["name"] == "notion_api"


def test_get_builds_argv_and_parses_json(monkeypatch, ntn_ok):
    captured = {}

    def fake_run(argv, input=None, **kw):
        captured["argv"] = argv
        captured["input"] = input
        return _FakeProc(0, '{"object": "list", "results": []}', "")

    monkeypatch.setattr(nt.subprocess, "run", fake_run)
    out = _call({"path": "v1/data_sources/abc"})
    assert out["ok"] is True
    assert out["data"] == {"object": "list", "results": []}
    assert captured["argv"] == ["ntn", "api", "v1/data_sources/abc", "-X", "GET"]
    assert captured["input"] is None


def test_post_serializes_body_to_stdin(monkeypatch, ntn_ok):
    captured = {}

    def fake_run(argv, input=None, **kw):
        captured["argv"] = argv
        captured["input"] = input
        return _FakeProc(0, '{"id": "p1"}', "")

    monkeypatch.setattr(nt.subprocess, "run", fake_run)
    out = _call({"path": "v1/pages", "method": "POST",
                 "body": {"parent": {"database_id": "d"}}})
    assert out["ok"] is True and out["data"] == {"id": "p1"}
    argv = captured["argv"]
    assert argv == ["ntn", "api", "v1/pages", "-X", "POST"]
    # ntn 0.15.x reads the request body from stdin — there is NO --json flag.
    assert "--json" not in argv
    assert json.loads(captured["input"]) == {"parent": {"database_id": "d"}}


def test_rejects_unknown_method(monkeypatch):
    monkeypatch.setattr(nt.subprocess, "run",
                        lambda *a, **k: pytest.fail("subprocess must not run"))
    out = _call({"path": "v1/pages", "method": "PUT"})
    assert "error" in out


def test_rejects_non_v1_path(monkeypatch):
    monkeypatch.setattr(nt.subprocess, "run",
                        lambda *a, **k: pytest.fail("subprocess must not run"))
    assert "error" in _call({"path": "config"})
    assert "error" in _call({"path": "-rf"})


def test_nonzero_exit_returns_structured_error(monkeypatch, ntn_ok):
    monkeypatch.setattr(nt.subprocess, "run",
                        lambda *a, **k: _FakeProc(1, "", "401 unauthorized"))
    out = _call({"path": "v1/pages/x"})
    assert "error" in out and "401" in out["error"]


def test_non_json_stdout_returned_as_text(monkeypatch, ntn_ok):
    monkeypatch.setattr(nt.subprocess, "run",
                        lambda *a, **k: _FakeProc(0, "## Heading\n\nbody", ""))
    out = _call({"path": "v1/pages/x/markdown"})
    assert out["ok"] is True and out["text"] == "## Heading\n\nbody"


def test_missing_ntn_binary(monkeypatch):
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: None)
    out = _call({"path": "v1/pages"})
    assert "error" in out


def test_env_bridges_api_key_to_token(monkeypatch):
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.setenv("NOTION_API_KEY", "secret")
    env = nt._ntn_env()
    assert env["NOTION_API_TOKEN"] == "secret"
    assert env["NOTION_KEYRING"] == "0"


def test_check_fn_follows_ntn_presence(monkeypatch):
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: "/usr/local/bin/ntn")
    assert nt._check_notion_api() is True
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: None)
    assert nt._check_notion_api() is False


def test_rejects_path_with_whitespace(monkeypatch):
    monkeypatch.setattr(nt.subprocess, "run",
                        lambda *a, **k: pytest.fail("subprocess must not run"))
    out = _call({"path": "v1/pages\n-X DELETE"})
    assert "error" in out


def test_timeout_returns_structured_error(monkeypatch, ntn_ok):
    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(["ntn"], 30)
    monkeypatch.setattr(nt.subprocess, "run", raise_timeout)
    out = _call({"path": "v1/pages/x"})
    assert "error" in out and "timed out" in out["error"]


def test_missing_token_returns_structured_error(monkeypatch):
    monkeypatch.setattr(nt.shutil, "which", lambda cmd: "/usr/local/bin/ntn")
    monkeypatch.setattr(nt, "_ntn_env", lambda: {"NOTION_KEYRING": "0"})  # no token
    out = _call({"path": "v1/pages"})
    assert "error" in out and "token" in out["error"].lower()


def test_notion_toolset_declared_and_maps_to_tool():
    import toolsets
    from tools.registry import registry
    assert "notion" in toolsets.TOOLSETS
    assert toolsets.TOOLSETS["notion"]["tools"] == ["notion_api"]
    assert registry.get_tool_names_for_toolset("notion") == ["notion_api"]


def test_schema_describes_the_verified_markdown_shapes():
    """POST v1/pages takes no markdown; the markdown endpoint takes a command
    body (verified live on ntn 0.15.1). The old {'markdown': …} advice was wrong."""
    from tools.notion_api_tool import NOTION_API_SCHEMA
    desc = NOTION_API_SCHEMA["description"]
    for shape in ("insert_content", "update_content", "replace_content", "NO markdown"):
        assert shape in desc
    assert "{'markdown': '...'}" not in desc
