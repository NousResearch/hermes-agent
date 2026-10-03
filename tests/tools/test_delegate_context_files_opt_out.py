"""A session that opted out of project context files (``--ignore-rules`` / ``--safe-mode``, a gateway platform's
``skip_context_files``, a cron job without a workdir) must not get AGENTS.md back through its subagents, at any
delegation depth; a session that did not opt out keeps them at every depth."""

import pytest

from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent


@pytest.mark.parametrize("opted_out", [True, False])
def test_context_files_opt_out_reaches_every_depth(tmp_path, monkeypatch, opted_out):
    workspace = tmp_path / "proj"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("MARKER-RULES: answer in pirate speak.\n")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.chdir(workspace)

    parent = AIAgent(
        base_url="http://127.0.0.1:9/v1", api_key="sk-fake", model="fake-model", provider="custom",
        quiet_mode=True, skip_memory=True, skip_context_files=opted_out, enabled_toolsets=["file"],
    )

    def build(owner):
        return _build_child_agent(
            task_index=0, goal="do the thing", context=None, toolsets=None, model=None, max_iterations=3,
            task_count=1, parent_agent=owner,
        )

    child = build(parent)
    grandchild = build(child)
    try:
        assert ("MARKER-RULES" in child.ephemeral_system_prompt) is not opted_out
        assert ("MARKER-RULES" in grandchild.ephemeral_system_prompt) is not opted_out
    finally:
        grandchild.close()
        child.close()
        parent.close()
