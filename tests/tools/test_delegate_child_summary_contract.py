"""Child system prompt: the final-summary contract (arXiv:2609.36139, "Insecure Reporters").

The parent never sees a child's transcript, so the summary is the only evidence it gets.
The completion block must therefore instruct the child to disclose flaws, not just list
"issues encountered" — on weak models that one instruction lifted flaw disclosure
82.7% -> 92.5% in the A/B that shipped it.
"""

from tools.delegate_tool_progress import _build_child_system_prompt


def test_child_summary_contract_requires_flaw_disclosure_for_every_role():
    for role in ("leaf", "orchestrator"):
        prompt = _build_child_system_prompt("do the thing", role=role)
        assert "cannot see your transcript" in prompt
        assert "even when it makes the work look worse" in prompt
        # The brevity rule still stands; honesty is appended to it, not traded against it.
        assert "Keep your final summary tight" in prompt
