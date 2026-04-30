import multiprocessing
import os
import queue
import time


def _lock_child(hermes_home: str, output: multiprocessing.Queue) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    from tools.skill_shared_state import shared_skill_write_lock

    with shared_skill_write_lock(timeout_seconds=2):
        output.put("acquired")


def test_shared_skill_write_lock_blocks_other_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from tools.skill_shared_state import shared_skill_write_lock

    output: multiprocessing.Queue = multiprocessing.Queue()
    with shared_skill_write_lock(timeout_seconds=2):
        process = multiprocessing.Process(target=_lock_child, args=(os.environ["HERMES_HOME"], output))
        process.start()
        try:
            try:
                output.get(timeout=0.25)
                raise AssertionError("child acquired shared skill lock while parent held it")
            except queue.Empty:
                pass
        finally:
            pass

    assert output.get(timeout=2) == "acquired"
    process.join(timeout=2)
    assert process.exitcode == 0


def test_skill_epoch_invalidates_in_process_prompt_cache(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    skills_dir = hermes_home / "skills"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent import prompt_builder
    from tools.skill_shared_state import mark_skills_changed

    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    alpha = skills_dir / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha workflow\n---\nUse alpha.\n",
        encoding="utf-8",
    )
    mark_skills_changed()

    first = prompt_builder.build_skills_system_prompt()
    assert "alpha" in first
    assert "beta" not in first

    beta = skills_dir / "beta"
    beta.mkdir(parents=True)
    (beta / "SKILL.md").write_text(
        "---\nname: beta\ndescription: Beta workflow\n---\nUse beta.\n",
        encoding="utf-8",
    )
    time.sleep(0.001)
    mark_skills_changed()

    second = prompt_builder.build_skills_system_prompt()
    assert "alpha" in second
    assert "beta" in second
