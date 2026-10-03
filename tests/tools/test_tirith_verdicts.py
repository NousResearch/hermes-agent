"""Independent verdict rows; PM security-consumers owns acquisition/provenance."""
import json
import subprocess

import pytest

from tools import tirith_security as tirith


@pytest.fixture
def scan(monkeypatch):
    config = {"tirith_enabled": True, "tirith_path": "/fixture/tirith",
              "tirith_timeout": 5, "tirith_fail_open": True}
    monkeypatch.setattr(tirith, "_load_security_config", lambda: config)
    monkeypatch.setattr(tirith, "_resolve_tirith_path", lambda path: path)
    monkeypatch.setattr(tirith, "_crash_count", 0)
    monkeypatch.setattr(tirith, "_circuit_open", False)
    monkeypatch.setattr(tirith, "_warned_messages", set())
    return config


@pytest.mark.parametrize("code,valid,fail_open,action", [
    (0, True, True, "allow"), (1, True, True, "block"), (2, True, True, "warn"),
    (0, False, True, "allow"), (1, False, True, "block"), (2, False, True, "warn"),
    (99, False, True, "allow"), (99, False, False, "block"),
])
def test_exit_code_is_authority(scan, monkeypatch, code, valid, fail_open, action):
    scan["tirith_fail_open"] = fail_open
    body = json.dumps({"findings": [{"rule_id": "homograph"}], "summary": "diagnostic"}) if valid else "NOT JSON"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, code, body, ""))
    result = tirith.check_command_security("curl https://example.test")
    assert result["action"] == action
    if valid:
        assert result["findings"] == [{"rule_id": "homograph"}]
        assert result["summary"] == "diagnostic"
    else:
        assert result["findings"] == []
        if code == 99:
            assert "exit code 99" in result["summary"]
        elif code:
            assert "details unavailable" in result["summary"]


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), subprocess.TimeoutExpired("tirith", 5)])
@pytest.mark.parametrize("fail_open,action", [(True, "allow"), (False, "block")])
def test_operational_error_policy(scan, monkeypatch, error, fail_open, action):
    scan["tirith_fail_open"] = fail_open
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(subprocess, "run", fail)
    result = tirith.check_command_security("echo safe")
    assert result["action"] == action
    assert ("fail-closed" in result["summary"]) is (not fail_open)


def test_circuit_dedup_and_success_reset(scan, monkeypatch, caplog):
    calls = []
    def fail(*a, **kw):
        calls.append(a)
        raise FileNotFoundError("missing")
    monkeypatch.setattr(subprocess, "run", fail)
    with caplog.at_level("WARNING", logger=tirith.__name__):
        tirith.check_command_security("one")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "{}", ""))
        tirith.check_command_security("healthy")
        assert tirith._crash_count == 0
        monkeypatch.setattr(subprocess, "run", fail)
        for _ in range(tirith._CRASH_LIMIT):
            assert "unavailable" in tirith.check_command_security("fail")["summary"]
        scan["tirith_fail_open"] = False
        assert tirith.check_command_security("after") == {
            "action": "allow", "findings": [], "summary": "tirith disabled (circuit breaker)"}
    assert len(calls) == 1 + tirith._CRASH_LIMIT
    assert sum("tirith spawn failed" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("code,findings,expected", [
    (2, [{"rule_id": "lookalike_tld", "value": ".app"}], "allow"),
    (2, [{"rule_id": "lookalike_tld", "value": ".APP"}], "allow"),
    (2, [{"rule_id": "lookalike_tld", "message": "Domain uses '.app' TLD"}], "allow"),
    (2, [{"rule_id": "lookalike_tld", "value": ".DEV"}], "allow"),
    (2, [{"rule_id": "lookalike_tld", "value": ".app"}, {"rule_id": "lookalike_tld", "value": ".dev"}], "allow"),
    (2, [{"rule_id": "shortened_url", "value": ".app"}], "warn"),
    (2, [{"rule_id": "lookalike_tld", "value": ".zip"}], "warn"),
    (2, [{"rule_id": "lookalike_tld", "value": ".app"}, {"rule_id": "shortened_url"}], "warn"),
    (1, [{"rule_id": "lookalike_tld", "value": ".app"}], "block"),
    (2, [{"rule_id": "lookalike_tld", "value": ".dev"}, {"rule_id": "shortened_url"}], "warn"),
    (1, [{"rule_id": "lookalike_tld", "value": ".dev"}], "block"),
])
def test_app_warning_exception_never_downgrades_block(scan, monkeypatch, code, findings, expected):
    body = json.dumps({"findings": findings, "summary": "finding"})
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, code, body, ""))
    result = tirith.check_command_security("curl https://example.app")
    assert result == {"action": expected, "findings": [] if expected == "allow" else findings,
                      "summary": "" if expected == "allow" else "finding"}


def test_caps_disabled_and_programming_errors(scan, monkeypatch):
    body = json.dumps({"findings": [{"rule_id": "other"}] * 100, "summary": "x" * 1000})
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 2, body, ""))
    result = tirith.check_command_security("echo safe")
    assert len(result["findings"]) == 50 and len(result["summary"]) == 500
    def bug(*a, **kw):
        raise AttributeError("programming error")
    monkeypatch.setattr(subprocess, "run", bug)
    with pytest.raises(AttributeError, match="programming error"):
        tirith.check_command_security("echo safe")
    scan["tirith_enabled"] = False
    assert tirith.check_command_security("anything")["action"] == "allow"


@pytest.mark.platforms("windows")
def test_windows_default_never_scans_but_override_does(scan, monkeypatch):
    scan["tirith_path"] = "tirith"
    def forbidden(*a, **kw):
        pytest.fail("unsupported default must not scan or install")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(tirith.threading.Thread, "start", forbidden)
    assert tirith.check_command_security("echo safe")["action"] == "allow"
    assert tirith.ensure_installed() is None
    scan["tirith_path"] = r"C:\fixture\tirith.exe"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "{}", ""))
    assert tirith.check_command_security("command")["action"] == "block"
