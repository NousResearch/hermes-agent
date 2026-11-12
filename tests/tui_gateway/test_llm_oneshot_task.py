"""The TUI gateway's ``llm.oneshot`` hands the client's ``task`` to the model call as the last
release did: any key, blank meaning ``title_generation``, never refused for being unregistered."""

from hermes_agent.tui_gateway import server


def test_llm_oneshot_passes_any_task_key_through(monkeypatch):
    seen = []

    def fake_oneshot(**kwargs):
        seen.append(kwargs["task"])
        return "t"

    monkeypatch.setattr("agent.oneshot.run_oneshot", fake_oneshot)
    replies = [server._methods["llm.oneshot"]("r", {"input": "x", **({"task": task} if task else {})})
               for task in ("my_digest", "  ", None)]
    assert [reply.get("result") for reply in replies] == [{"text": "t"}] * 3, replies
    assert seen == ["my_digest", "title_generation", "title_generation"]
