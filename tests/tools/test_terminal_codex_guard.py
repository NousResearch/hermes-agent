import json

from tools.terminal_tool import _raw_codex_guard, terminal_tool


def test_raw_codex_exec_is_blocked(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_RAW_CODEX", raising=False)

    err = _raw_codex_guard(
        "export HOME=/Users/agent1/Operator\n"
        "codex exec --skip-git-repo-check 'review this'"
    )

    assert err is not None
    assert "codex-run.sh" in err


def test_npx_codex_is_blocked(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_RAW_CODEX", raising=False)

    err = _raw_codex_guard("npx --yes @openai/codex exec 'review this'")

    assert err is not None
    assert "CODEX_HOME=/Users/agent1/.codex" in err


def test_npm_exec_codex_is_blocked(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_RAW_CODEX", raising=False)

    err = _raw_codex_guard("npm exec @openai/codex -- exec 'review this'")

    assert err is not None
    assert "codex-run.sh" in err


def test_codex_wrapper_is_allowed(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_RAW_CODEX", raising=False)

    assert _raw_codex_guard(
        "/Users/agent1/Operator/scripts/codex-run.sh exec 'review this'"
    ) is None


def test_raw_codex_can_be_explicitly_allowed(monkeypatch):
    monkeypatch.setenv("HERMES_ALLOW_RAW_CODEX", "1")

    assert _raw_codex_guard("codex exec 'review this'") is None


def test_quoted_codex_reference_is_allowed(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_RAW_CODEX", raising=False)

    assert _raw_codex_guard("printf '%s\\n' 'codex exec is documented here'") is None


def test_terminal_tool_blocks_raw_codex_before_execution(monkeypatch):
    monkeypatch.delenv("HERMES_ALLOW_RAW_CODEX", raising=False)

    result = json.loads(terminal_tool("codex exec 'review this'"))

    assert result["status"] == "blocked"
    assert result["exit_code"] == -1
    assert "codex-run.sh" in result["error"]
