"""Tests for tools/learning_triage.py — deterministic MVP learning proposal tool."""

import json

from tools.memory_tool import ENTRY_DELIMITER


def _first_non_ignore(candidates):
    for candidate in candidates:
        if candidate["target"] != "ignore":
            return candidate
    raise AssertionError(f"expected a non-ignore candidate, got {candidates!r}")


class TestLearningTriageClassification:
    def test_stable_user_preference_becomes_user_memory_candidate(self):
        from tools.learning_triage import classify_learning_candidates

        candidates = classify_learning_candidates([
            "User correction: please keep responses concise and terminal-readable; do not end with generic offers."
        ])

        candidate = _first_non_ignore(candidates)
        assert candidate["target"] == "user"
        assert candidate["action"] == "add"
        assert "concise" in candidate["candidate"].lower()
        assert "rationale" not in candidate
        assert candidate["reason"]
        assert candidate["risk"] in {"low", "medium", "high"}
        assert candidate["confidence"] == "high"
        assert candidate["old_text"] is None
        assert candidate["skill_name"] is None

    def test_transient_task_progress_is_ignored(self):
        from tools.learning_triage import classify_learning_candidates

        candidates = classify_learning_candidates([
            "Done: implemented PR #123, tests pass, next step is merge tomorrow and update the ticket."
        ])

        assert candidates
        assert all(candidate["target"] == "ignore" for candidate in candidates)
        assert all(candidate["action"] == "ignore" for candidate in candidates)
        assert "task progress" in candidates[0]["reason"].lower()

    def test_recurring_procedural_workflow_becomes_skill_patch_candidate(self):
        from tools.learning_triage import classify_learning_candidates

        candidates = classify_learning_candidates([
            "Recurring workflow: brainstorm first, run five neutral claude -p adversarial review rounds, then write the implementation plan and execute it. Prefer patching umbrella skills over creating narrow one-off skills."
        ])

        candidate = _first_non_ignore(candidates)
        assert candidate["target"] == "skills"
        assert candidate["action"] == "patch_skill"
        assert candidate["skill_name"] is not None
        assert "umbrella" in candidate["reason"].lower()
        assert candidate["confidence"] in {"medium", "high"}

    def test_stable_environment_fact_becomes_agent_memory_candidate(self):
        from tools.learning_triage import classify_learning_candidates

        candidates = classify_learning_candidates([
            "Environment fact: Hermes repo tests must use scripts/run_tests.sh because direct pytest can diverge from CI."
        ])

        candidate = _first_non_ignore(candidates)
        assert candidate["target"] == "memory"
        assert candidate["action"] == "add"
        assert "scripts/run_tests.sh" in candidate["candidate"]
        assert candidate["confidence"] in {"medium", "high"}

    def test_target_filter_limits_non_matching_candidates(self):
        from tools.learning_triage import classify_learning_candidates

        candidates = classify_learning_candidates(
            [
                "User correction: prefer concise terminal-readable responses.",
                "Environment fact: this project uses scripts/run_tests.sh for test parity.",
            ],
            target="user",
        )

        assert candidates
        assert all(candidate["target"] in {"user", "ignore"} for candidate in candidates)
        assert any(candidate["target"] == "user" for candidate in candidates)
        assert not any(candidate["target"] == "memory" for candidate in candidates)


class TestLearningTriageTool:
    def test_memory_usage_reporting_uses_profile_safe_hermes_home(self, tmp_path, monkeypatch):
        from tools.learning_triage import learning_triage

        hermes_home = tmp_path / "profile-home"
        memories = hermes_home / "memories"
        memories.mkdir(parents=True)
        (memories / "USER.md").write_text(
            ENTRY_DELIMITER.join(["User prefers concise responses", "User works in terminals"]),
            encoding="utf-8",
        )
        (memories / "MEMORY.md").write_text("Machine has RTX 2070", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        result = json.loads(learning_triage(scope="recent", limit=3))

        assert result["success"] is True
        assert result["memory_usage"]["user"]["current_chars"] == len(
            ENTRY_DELIMITER.join(["User prefers concise responses", "User works in terminals"])
        )
        assert result["memory_usage"]["memory"]["current_chars"] == len("Machine has RTX 2070")
        assert result["memory_usage"]["user"]["limit_chars"] > 0
        assert result["memory_usage"]["memory"]["limit_chars"] > 0

    def test_apply_mode_without_confirmation_does_not_mutate_memory(self, tmp_path, monkeypatch):
        from tools.learning_triage import learning_triage

        hermes_home = tmp_path / "profile-home"
        memories = hermes_home / "memories"
        memories.mkdir(parents=True)
        user_file = memories / "USER.md"
        memory_file = memories / "MEMORY.md"
        user_file.write_text("User prefers concise responses", encoding="utf-8")
        memory_file.write_text("Machine fact", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        before_user = user_file.read_text(encoding="utf-8")
        before_memory = memory_file.read_text(encoding="utf-8")

        result = json.loads(learning_triage(mode="apply", scope="recent", limit=3))

        assert result["success"] is True
        assert result["mode"] == "apply"
        assert "applied" not in result
        assert any("not implemented" in note.lower() for note in result["notes"])
        assert user_file.read_text(encoding="utf-8") == before_user
        assert memory_file.read_text(encoding="utf-8") == before_memory

    def test_apply_mode_with_confirmed_candidate_remains_proposal_only(self, tmp_path, monkeypatch):
        from hermes_state import SessionDB
        from tools.learning_triage import learning_triage

        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s-apply-user", "cli", model="test-model")
        db.append_message("s-apply-user", "user", "User correction: prefer terse terminal-readable answers.")

        result = json.loads(
            learning_triage(
                session_id="s-apply-user",
                scope="session",
                mode="apply",
                db=db,
            )
        )

        assert result["success"] is True
        assert result["mode"] == "apply"
        assert any("not implemented" in note.lower() for note in result["notes"])
        assert any("proposal-only" in note.lower() for note in result["notes"])
        assert not (hermes_home / "memories" / "USER.md").exists()
        db.close()

    def test_apply_mode_replaces_similar_existing_memory(self, tmp_path, monkeypatch):
        from hermes_state import SessionDB
        from tools.learning_triage import learning_triage

        hermes_home = tmp_path / "hermes-home"
        memories = hermes_home / "memories"
        memories.mkdir(parents=True)
        memory_file = memories / "MEMORY.md"
        memory_file.write_text("Hermes repo tests use pytest directly", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s-apply-replace", "cli", model="test-model")
        db.append_message(
            "s-apply-replace",
            "user",
            "Environment fact: Hermes repo tests must use scripts/run_tests.sh because direct pytest diverges from CI.",
        )

        proposal = json.loads(learning_triage(session_id="s-apply-replace", scope="session", db=db))
        candidate = _first_non_ignore(proposal["candidates"])
        assert candidate["target"] == "memory"
        assert candidate["action"] == "replace"
        assert candidate["old_text"] == "Hermes repo tests use pytest directly"

        before = memory_file.read_text(encoding="utf-8")
        result = json.loads(
            learning_triage(
                session_id="s-apply-replace",
                scope="session",
                mode="apply",
                db=db,
            )
        )

        assert any("not implemented" in note.lower() for note in result["notes"])
        assert memory_file.read_text(encoding="utf-8") == before
        db.close()

    def test_apply_mode_skips_skill_candidates_without_mutating_skills(self, tmp_path, monkeypatch):
        from hermes_state import SessionDB
        from tools.learning_triage import learning_triage

        hermes_home = tmp_path / "hermes-home"
        skills_dir = hermes_home / "skills"
        skills_dir.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s-apply-skill", "cli", model="test-model")
        db.append_message(
            "s-apply-skill",
            "user",
            "Recurring workflow: when adding a stateful tool, add it to _AGENT_LOOP_TOOLS and route it in run_agent.py, then verify discovery.",
        )

        result = json.loads(
            learning_triage(
                session_id="s-apply-skill",
                scope="session",
                mode="apply",
                db=db,
            )
        )

        assert result["candidates"][0]["target"] == "skills"
        assert any("proposal-only" in note.lower() for note in result["notes"])
        assert list(skills_dir.iterdir()) == []
        db.close()

    def test_session_scope_loads_requested_session_from_db(self, tmp_path, monkeypatch):
        from hermes_state import SessionDB
        from tools.learning_triage import learning_triage

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s-learning", "cli", model="test-model")
        db.append_message("s-learning", "user", "User correction: prefer concise terminal-readable answers.")
        db.append_message("s-learning", "assistant", "Acknowledged.")

        result = json.loads(learning_triage(session_id="s-learning", scope="session", db=db, limit=5))

        assert result["success"] is True
        assert result["source"]["session_id"] == "s-learning"
        assert result["source"]["message_count"] == 2
        assert any(candidate["target"] == "user" for candidate in result["candidates"])
        db.close()

    def test_tool_registers_and_schema_is_strict_backend_friendly(self):
        import tools.learning_triage  # noqa: F401 - import triggers registry registration
        from tools.registry import registry
        from tools.learning_triage import LEARNING_TRIAGE_SCHEMA
        from toolsets import TOOLSETS

        entry = registry.get_entry("learning_triage")
        assert entry is not None
        assert entry.toolset == "learning"
        assert "learning_triage" in TOOLSETS["learning"]["tools"]
        assert "learning_triage" in TOOLSETS["hermes-cli"]["tools"]

        params = LEARNING_TRIAGE_SCHEMA["parameters"]
        assert params["type"] == "object"
        for forbidden in ("allOf", "anyOf", "oneOf", "not"):
            assert forbidden not in params
        assert "candidate_indexes" not in params["properties"]
        assert "confirm_apply" not in params["properties"]
        json.dumps(LEARNING_TRIAGE_SCHEMA)
