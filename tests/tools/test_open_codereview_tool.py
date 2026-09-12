"""``open_codereview`` tool: the review CLI is a stub script, so no key and no network are needed.

The behaviours under test are (a) a review's report comes back through the tool, and (b) the
provider/model Hermes is using are mirrored into the review command before that review runs — the
hand-configuration of open-codereview.ai that #108639 asks Hermes to remove.
"""

import json
import sys

import pytest
import yaml

from hermes_constants import get_hermes_home
from tools.open_codereview_tool import (
    check_open_codereview_requirements,
    hermes_assignment,
    open_codereview_tool,
)

# Stub review CLI: records every invocation (argv + whether the key reached its env) to
# $OCR_STUB_LOG, then answers like a `config ...` or `review ...` command would.
STUB_SOURCE = '''
import json, os, sys

args = sys.argv[1:]
entry = {
    "args": args,
    "key_present": bool(os.environ.get("OPEN_CODEREVIEW_API_KEY")),
    "alt_key_present": bool(os.environ.get("OCR_ALT_KEY")),
}
with open(os.environ["OCR_STUB_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(entry) + chr(10))

if os.environ.get("OCR_STUB_SLEEP"):
    import time
    time.sleep(float(os.environ["OCR_STUB_SLEEP"]))
if os.environ.get("OCR_STUB_EXIT"):
    sys.stderr.write("stub: cannot review" + chr(10))
    sys.exit(int(os.environ["OCR_STUB_EXIT"]))

if args[:1] == ["config"]:
    print("configured " + ".".join(args[1:]))
elif args[:1] == ["review"]:
    print("REVIEW REPORT: " + (args[1] if len(args) > 1 else ""))
else:
    sys.exit(2)
'''


class StubCLI:
    """The stub review command plus the log of everything it was invoked with."""

    def __init__(self, script, log):
        self.script = script
        self.log = log

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines() if line]

    def args(self):
        return [call["args"] for call in self.calls()]


@pytest.fixture
def stub(tmp_path, monkeypatch):
    script = tmp_path / "ocr_stub.py"
    script.write_text(STUB_SOURCE, encoding="utf-8")
    log = tmp_path / "ocr.log"
    monkeypatch.setenv("OCR_STUB_LOG", str(log))
    return StubCLI(script, log)


def write_config(*, provider="zai", model="glm-5.2", stub=None, **ocr_overrides):
    """Write a real config.yaml into the isolated HERMES_HOME (the tool reads it via load_config).

    ``command`` is the test interpreter and every argv template starts with the stub script, so the
    stub receives exactly the arguments the real review command would.
    """
    section = {
        "command": sys.executable,
        "provider_args": [str(stub.script), "config", "provider", "{provider}"],
        "model_args": [str(stub.script), "config", "model", "{model}"],
        "review_args": [str(stub.script), "review", "{target}"],
        "timeout": 30,
    }
    section.update(ocr_overrides)
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": provider, "default": model},
        "open_codereview": section,
    }), encoding="utf-8")


def payload(result):
    assert isinstance(result, str), "tool handlers must return JSON strings"
    return json.loads(result)


# ---- the integration the issue asks for ------------------------------------------------------


def test_review_returns_the_report_and_mirrors_hermes_assignment(stub):
    """A review reaches the chat, and the review tool was told which provider/model Hermes uses."""
    write_config(stub=stub)

    result = payload(open_codereview_tool("review", target="HEAD~1..HEAD"))

    assert result["success"] is True
    assert "REVIEW REPORT: HEAD~1..HEAD" in result["output"]
    assert stub.args() == [
        ["config", "provider", "zai"],
        ["config", "model", "glm-5.2"],
        ["review", "HEAD~1..HEAD"],
    ]


def test_switching_models_in_hermes_resyncs_the_review_tool(stub):
    """The #108639 workflow: switch model in Hermes, no terminal reconfiguration of the review tool."""
    write_config(stub=stub)
    open_codereview_tool("review", target="HEAD~1..HEAD")

    write_config(provider="anthropic", model="claude-opus-4.6-long", stub=stub)
    open_codereview_tool("review", target="HEAD~1..HEAD")

    assert stub.args() == [
        ["config", "provider", "zai"],
        ["config", "model", "glm-5.2"],
        ["review", "HEAD~1..HEAD"],
        ["config", "provider", "anthropic"],
        ["config", "model", "claude-opus-4.6-long"],
        ["review", "HEAD~1..HEAD"],
    ]


def test_unchanged_assignment_is_not_pushed_again(stub):
    """The mirrored values are remembered, so a steady model costs one review call, not three."""
    write_config(stub=stub)
    open_codereview_tool("review", target="HEAD~1..HEAD")
    open_codereview_tool("review", target="HEAD~1..HEAD")

    assert stub.args() == [
        ["config", "provider", "zai"],
        ["config", "model", "glm-5.2"],
        ["review", "HEAD~1..HEAD"],
        ["review", "HEAD~1..HEAD"],
    ]


def test_manual_mode_only_syncs_when_asked(stub):
    """``mode: manual`` keeps the review tool untouched until the user runs the sync action."""
    write_config(stub=stub, mode="manual")

    open_codereview_tool("review", target="HEAD")
    assert stub.args() == [["review", "HEAD"]]

    synced = payload(open_codereview_tool("sync"))
    assert synced["synced"] is True
    assert [entry["field"] for entry in synced["applied"]] == ["provider", "model"]


def test_status_reports_both_sides_and_a_pending_sync(stub):
    write_config(stub=stub)

    status = payload(open_codereview_tool("status"))

    assert status["hermes"] == {"provider": "zai", "model": "glm-5.2"}
    assert status["cli"] == {"provider": "", "model": "", "synced_at": None}
    assert status["sync_pending"] is True
    assert stub.calls() == []          # status answers from config, it runs no command
    assert status["api_key_present"] is False


def test_explicit_provider_and_model_override_the_hermes_assignment(stub):
    write_config(stub=stub)

    payload(open_codereview_tool("sync", provider="openrouter", model="z-ai/glm-5.2"))

    assert stub.args() == [
        ["config", "provider", "openrouter"],
        ["config", "model", "z-ai/glm-5.2"],
    ]


def test_auto_provider_resolves_through_the_auth_resolver(stub, monkeypatch):
    """``model.provider: auto`` reaches the review tool as a concrete provider, never as "auto"."""
    write_config(stub=stub, provider="auto")
    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda *a, **kw: "nous")

    assert hermes_assignment() == ("nous", "glm-5.2")
    payload(open_codereview_tool("sync"))
    assert stub.args()[0] == ["config", "provider", "nous"]


# ---- credentials -----------------------------------------------------------------------------


def test_api_key_reaches_the_child_env_and_never_a_command_line(stub, monkeypatch):
    write_config(stub=stub)
    monkeypatch.setenv("OPEN_CODEREVIEW_API_KEY", "stub-secret-value")

    open_codereview_tool("review", target="HEAD~1..HEAD")

    assert all(call["key_present"] for call in stub.calls())
    assert "stub-secret-value" not in json.dumps(stub.args())


def test_configured_api_key_env_is_honoured(stub, monkeypatch):
    write_config(stub=stub, api_key_env="OCR_ALT_KEY", env_passthrough=["OCR_ALT_KEY"])
    monkeypatch.setenv("OCR_ALT_KEY", "stub-secret-value")

    open_codereview_tool("review", target="HEAD~1..HEAD")

    assert all(call["alt_key_present"] for call in stub.calls())


# ---- error handling --------------------------------------------------------------------------


def test_review_without_a_target_is_rejected(stub):
    write_config(stub=stub)

    result = payload(open_codereview_tool("review"))

    assert "target" in result["error"]
    assert stub.calls() == []


def test_unknown_action_is_rejected(stub):
    write_config(stub=stub)

    result = payload(open_codereview_tool("publish"))

    assert "Unknown action" in result["error"]
    assert stub.calls() == []


def test_missing_command_is_actionable(stub):
    write_config(stub=stub, command="definitely-not-an-ocr-binary")

    assert check_open_codereview_requirements() is False
    result = payload(open_codereview_tool("status"))
    assert "definitely-not-an-ocr-binary" in result["error"]
    assert "open_codereview.command" in result["error"]


def test_check_fn_is_true_for_a_resolvable_command(stub):
    write_config(stub=stub)

    assert check_open_codereview_requirements() is True


def test_failing_command_reports_exit_code_and_stderr(stub, monkeypatch):
    write_config(stub=stub)
    monkeypatch.setenv("OCR_STUB_EXIT", "3")

    result = payload(open_codereview_tool("review", target="HEAD~1..HEAD"))

    assert "code 3" in result["error"]
    assert "stub: cannot review" in result["error"]


def test_timeout_is_reported_instead_of_hanging(stub, monkeypatch):
    write_config(stub=stub, timeout=1)
    monkeypatch.setenv("OCR_STUB_SLEEP", "30")

    result = payload(open_codereview_tool("review", target="HEAD~1..HEAD"))

    assert "timed out" in result["error"]
