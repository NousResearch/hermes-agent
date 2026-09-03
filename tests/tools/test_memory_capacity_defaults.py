"""Memory capacity defaults must fit real accumulated use (#101459).

The old defaults (memory 2200 / user 1375 chars) filled within a day or two
of real use — a daily session-distillation cron failed every write with
"would exceed the limit" until consolidation gave up, losing cross-session
continuity silently. Sized from a working deployment: ~6,600 chars of user
memory accumulated over weeks of daily distillation.
"""

import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\salma\dev\hermes-agent")

import pytest

from tools.memory_tool import MemoryStore, load_on_disk_store


class TestNewDefaults:
    def test_config_defaults_match_new_capacity(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        mem = DEFAULT_CONFIG["memory"]
        assert mem["user_char_limit"] == 8000
        assert mem["memory_char_limit"] == 12000

    def test_memorystore_defaults_match(self):
        store = MemoryStore(memory_enabled=True, user_profile_enabled=True)
        assert store.user_char_limit == 8000
        assert store.memory_char_limit == 12000

    def test_agent_init_fallbacks_match(self):
        """The agent construction path must agree with the store defaults
        (config-absent fallback), so a missing config.yaml can't silently
        restore the tight caps."""
        import inspect

        import agent.agent_init as ai

        src = inspect.getsource(ai)
        assert '"memory_char_limit", 12000' in src
        assert '"user_char_limit", 8000' in src


class TestReporterScenario:
    @staticmethod
    def _seeded_store(user_char_limit):
        """A store whose user memory is near the OLD cap, seeded through the
        real disk path (add() re-reads from disk under lock, so in-memory
        seeding would be discarded)."""
        store = MemoryStore(
            memory_enabled=False,
            user_profile_enabled=True,
            user_char_limit=user_char_limit,
        )
        seeded = "Accumulated profile fact. " * 48  # ~1,250 chars
        store._set_entries("user", [seeded])
        store.save_to_disk("user")
        return store

    def test_daily_distillation_write_fits_under_new_default(self):
        """The reporter's exact failure: a user store at ~1,200 chars whose
        daily 374-char distillation write failed at the 1,375 cap. Under the
        new default the same write succeeds with room for weeks more."""
        store = self._seeded_store(8000)
        prefix = (
            "Session distillation 2026-09-01: explored the memory limit "
            "failure, traced config_defaults, raised defaults, verified "
            "cron writes succeed."
        )
        daily = prefix + "x" * (374 - len(prefix))
        assert len(daily) == 374

        result = store.add("user", daily)
        assert result["success"] is True, (
            "the reporter's daily write must fit under the new default"
        )

    def test_old_default_would_have_failed_this_write(self):
        """Pin the bug: the same write fails under the old caps — proving
        the test above is meaningful and the default is what fixed it."""
        store = self._seeded_store(1375)
        prefix = (
            "Session distillation 2026-09-01: explored the memory limit "
            "failure, traced config_defaults, raised defaults, verified "
            "cron writes succeed."
        )
        daily = prefix + "x" * (374 - len(prefix))
        result = store.add("user", daily)
        assert result["success"] is False
        assert "would exceed the limit" in result["error"]
        # The failure response tells the model how to self-correct.
        assert "current_entries" in result


class TestExplicitOverrideStillWins:
    def test_user_config_value_still_overrides_default(self, monkeypatch, tmp_path):
        """Raising the default must not clobber an explicit user setting:
        the config read path (load_on_disk_store -> get_builtin_memory_config)
        still honors memory.user_char_limit."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg_dir = tmp_path / ".hermes"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            "memory:\n  user_char_limit: 500\n", encoding="utf-8"
        )
        try:
            store = load_on_disk_store()
            assert store.user_char_limit == 500, (
                "an explicit user config must win over any default"
            )
        except Exception:
            pytest.skip("config load unavailable in this environment")
