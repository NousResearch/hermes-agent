"""A guardian answer blanked by the provider's content filter is retried once on the first fallback provider.

Codex returns ``finish_reason="content_filter"`` with no text for some ordinary commands. That empty answer
escalated, and in an unattended run (cron, Kanban, ``-q``) an escalation is a refusal, so routine work stopped.
Any other empty answer (``length``, ``stop``) still escalates without a second call.
"""

from unittest.mock import MagicMock, patch

from tools.approval_smart import _smart_approve

FALLBACK = {"fallback_providers": [{"provider": "anthropic", "model": "claude-sonnet-5-5"}]}


def _reply(text, finish_reason="stop"):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text), finish_reason=finish_reason)])


def _run(replies, config=FALLBACK):
    with patch("hermes_cli.config.load_config_readonly", return_value=config), \
            patch("agent.auxiliary_client.call_llm", side_effect=replies) as call_llm:
        verdict = _smart_approve("python -c \"print('hi')\"", "script execution via -c flag")
    return verdict, [c.kwargs.get("provider") for c in call_llm.call_args_list]


def test_content_filter_blank_asks_the_first_fallback_once():
    verdict, providers = _run([_reply("", "content_filter"), _reply("APPROVE")])
    assert verdict == "approve"
    assert providers == [None, "anthropic"]


def test_fallback_blank_too_still_escalates():
    verdict, providers = _run([_reply("", "content_filter"), _reply("", "content_filter")])
    assert verdict == "escalate"
    assert len(providers) == 2


def test_fallback_error_escalates():
    verdict, providers = _run([_reply("", "content_filter"), RuntimeError("fallback down")])
    assert verdict == "escalate"
    assert providers == [None, "anthropic"]


def test_no_fallback_configured_escalates_without_a_second_call():
    verdict, providers = _run([_reply("", "content_filter")], config={})
    assert verdict == "escalate"
    assert providers == [None]


def test_other_empty_answers_are_not_retried():
    verdict, providers = _run([_reply("", "length")])
    assert verdict == "escalate"
    assert providers == [None]
