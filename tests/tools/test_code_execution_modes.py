"""Real interpreter, environment and RPC contracts for both execution modes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.tools._child_env_fixtures import child_env, project_python, run_code  # noqa: F401
from tools import code_execution_env as ce
from tools.code_execution_tool import build_execute_code_schema


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("mode", ["strict", "project"])
@pytest.mark.parametrize("activation", ["VIRTUAL_ENV", "CONDA_PREFIX"])
def test_selected_interpreter_environment_and_real_rpc(child_env, project_python, monkeypatch, mode, activation):
    python, prefix = project_python
    monkeypatch.setenv(activation, str(prefix))
    repo = Path(__file__).resolve().parents[2]
    site = Path(sys.prefix) / ("Lib/site-packages" if os.name == "nt" else
                              f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    user_lib = child_env / "user-lib"
    user_lib.mkdir()
    (user_lib / "user_probe.py").write_text("VALUE = 'user → 雪'\n", encoding="utf-8")
    witness = child_env / "rpc-witness.txt"
    witness.write_text("RPC → 雪\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, [repo, site, repo, user_lib, user_lib])))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-provider-secret")
    before = dict(os.environ)
    result = run_code(f'''
import importlib.util, json, os, sys
import hermes_tools, user_probe
from hermes_tools import read_file
print(json.dumps({{
    "executable": sys.executable, "prefix": sys.prefix, "cwd": os.getcwd(),
    "pythonpath": os.environ["PYTHONPATH"].split(os.pathsep),
    "sys_path": sys.path,
    "staging": os.path.dirname(hermes_tools.__file__),
    "encoding": [os.environ.get("PYTHONIOENCODING"), os.environ.get("PYTHONUTF8")],
    "essentials": {{k: os.environ.get(k) for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")}},
    "secret": os.environ.get("OPENAI_API_KEY"),
    "project": (__import__("project_only_probe").VALUE
                if importlib.util.find_spec("project_only_probe") else None),
    "user": user_probe.VALUE, "rpc": read_file({str(witness)!r}),
}}, ensure_ascii=False))
''', mode)
    assert result["encoding"] == ["utf-8", "1"]
    if os.name == "nt":
        assert result["essentials"] == {k: before.get(k) for k in result["essentials"]}
    assert result["secret"] is None
    assert result["user"] == "user → 雪"
    assert "RPC → 雪" in result["rpc"]["content"]
    assert result["rpc"]["total_lines"] == 1
    expected_python, expected_prefix = (python, prefix) if mode == "project" else (Path(sys.executable), Path(sys.prefix))
    assert Path(result["executable"]) == expected_python
    assert Path(result["prefix"]).resolve() == expected_prefix.resolve()
    assert result["project"] == ("project-only → 雪" if mode == "project" else None)
    if mode == "project":
        assert os.path.normcase(str(site)) not in list(map(os.path.normcase, result["sys_path"]))
    expected_cwd = child_env if mode == "project" else Path(result["staging"])
    assert Path(result["cwd"]).resolve() == expected_cwd.resolve()
    controlled = [result["staging"]] + ([str(repo)] if mode == "strict" else [])
    assert list(map(os.path.normcase, result["pythonpath"])) == list(map(os.path.normcase, controlled + [str(user_lib)] * 2))
    assert dict(os.environ) == before


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("mode", ["strict", "project"])
def test_credential_policy_and_whitelist_in_real_child(child_env, monkeypatch, mode):
    from tools.env_passthrough import register_env_passthrough
    blocked = (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "MY_SECRET",
        "DB_PASSWORD", "VAULT_CREDENTIAL", "LDAP_PASSWD", "AUTH_TOKEN",
        "SENTRY_DSN", "SLACK_WEBHOOK", "HOME_APIKEY", "USER_CREDS", "TERM_BEARER",
        "HERMES_BASE_URL", "HERMES_INTERACTIVE", "BUZZ_PRIVATE_KEY", "RANDOM_UNKNOWN",
    )
    allowed = ("HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV", "LC_ENV_TEST", "TENOR_API_KEY")
    for name in blocked + allowed:
        monkeypatch.setenv(name, "fake-" + name)
    # Registration is real: a skill cannot tunnel a provider or Buzz credential.
    register_env_passthrough(["TENOR_API_KEY", "OPENAI_API_KEY", "BUZZ_PRIVATE_KEY"])
    result = run_code(f'''
import json, os, hermes_tools
print(json.dumps({{"env": {{k: os.environ.get(k) for k in {blocked + allowed!r}}},
                  "recursive": hasattr(hermes_tools, "execute_code"),
                  "delegation": hasattr(hermes_tools, "delegate_task")}}))
''', mode, enabled_tools=("read_file", "execute_code", "delegate_task"))
    assert result["env"] == {**dict.fromkeys(blocked), **{k: "fake-" + k for k in allowed}}
    assert result["recursive"] is False
    assert result["delegation"] is False


@pytest.mark.parametrize("helper,cache,failed,success", [
    (ce._is_usable_python, ce._usable_python_cache, False, True),
    (ce._python_environment_prefix, ce._python_prefix_cache, "", os.path.realpath("recovered-prefix")),
])
def test_probe_retries_transient_failure_and_caches_success(helper, cache, failed, success):
    cache.clear()
    try:
        with patch("subprocess.run", side_effect=[
            subprocess.TimeoutExpired(cmd=[], timeout=5),
            subprocess.CompletedProcess([], 0, "recovered-prefix\n"),
        ]) as run:
            assert helper("probe-python") == failed
            assert helper("probe-python") == success
            assert helper("probe-python") == success
        assert run.call_count == 2
    finally:
        cache.clear()


@pytest.mark.parametrize("outcome", [
    OSError("missing interpreter"), subprocess.CompletedProcess([], 1, ""),
    subprocess.CompletedProcess([], 0, " \n"),
])
def test_unknown_prefix_is_not_cached(outcome):
    ce._python_prefix_cache.clear()
    with patch("subprocess.run", side_effect=outcome if isinstance(outcome, Exception) else None,
               return_value=outcome) as run:
        assert ce._python_environment_prefix("bad-python") == ""
        assert ce._python_environment_prefix("bad-python") == ""
    assert run.call_count == 2
    assert "bad-python" not in ce._python_prefix_cache


def test_current_interpreter_needs_no_probe():
    with patch("subprocess.run", side_effect=AssertionError("unexpected probe")):
        assert ce._uses_hermes_python_environment(sys.executable)


def test_project_without_active_venv_falls_back(child_env):
    assert ce._resolve_child_python("project") == sys.executable


def test_project_stale_cwd_falls_through_then_uses_process_cwd(child_env, monkeypatch):
    from tools import terminal_tool
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    terminal_tool.register_task_env_overrides("stale", {"cwd": str(child_env)})
    terminal_tool.record_session_cwd("stale", str(child_env / "deleted"))
    monkeypatch.setenv("TERMINAL_CWD", str(child_env / "missing"))
    assert ce._resolve_child_cwd("project", "staging", task_id="stale") == str(child_env)
    assert ce._resolve_child_cwd("project", "staging") == os.getcwd()


@pytest.mark.parametrize("mode,description", [("strict", "temp dir"), ("project", "session")])
def test_mode_schema_matches_config_without_claiming_isolation(mode, description):
    with patch("tools.code_execution_tool._load_config", return_value={"mode": mode}):
        text = build_execute_code_schema()["description"].lower()
    assert description in text
    assert not any(word in text for word in ("sandbox", "isolated", "cloud"))
