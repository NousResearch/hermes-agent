"""A staged memory change immediately attempts its own review prompt."""


def test_stage_write_prompts_immediately(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sent = []
    import tools.write_approval as wa

    monkeypatch.setattr(wa, "_prompt_for_record", lambda sub, rec: sent.append(rec["id"]))
    rec = wa.stage_write(
        "memory", {"action": "add", "target": "user", "content": "A new memory"},
        summary="s", origin="pytest",
    )
    assert sent == [rec["id"]], "exactly one prompt per staged record"
