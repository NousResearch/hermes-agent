import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.delegate_personas import (
    apply_persona_to_task,
    list_delegate_personas,
    persona_can_write,
    resolve_persona,
)
from tools.delegate_tool import delegate_task


def _write_persona(root: Path, name: str, body: str, description: str = "") -> Path:
    path = root / f"{name}.md"
    path.write_text(
        "---\n"
        f"description: {description}\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _make_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._memory_manager = None
    return parent


class TestDelegatePersonas(unittest.TestCase):
    def test_lists_personas_from_configured_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            personas.mkdir()
            _write_persona(personas, "code-reviewer", "# Role\nReview code.", "Reviews code")

            result = list_delegate_personas(
                cfg={"persona_dirs": {"project": str(personas)}},
                mode="signatures",
            )

            self.assertEqual(result["count"], 1)
            self.assertEqual(result["personas"][0]["name"], "code-reviewer")
            self.assertEqual(result["personas"][0]["pool"], "project")

    def test_configured_dir_precedence_dedupes_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            _write_persona(first, "verifier", "# First\nUse me.")
            _write_persona(second, "verifier", "# Second\nDo not use me.")

            persona = resolve_persona(
                "verifier",
                {"persona_dirs": {"first": str(first), "second": str(second)}},
            )

            self.assertEqual(persona["pool"], "first")
            self.assertEqual(persona["path"], str(first / "verifier.md"))

    def test_symlinked_persona_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            personas.mkdir()
            target = Path(tmp) / "secret.md"
            target.write_text("not a persona", encoding="utf-8")
            (personas / "linked.md").symlink_to(target)

            result = list_delegate_personas(
                cfg={"persona_dirs": {"project": str(personas)}},
                mode="full",
            )

            self.assertEqual(result["count"], 0)

    def test_write_persona_detection_uses_name_boundaries(self):
        self.assertTrue(persona_can_write("implementer"))
        self.assertTrue(persona_can_write("quick-fix"))
        self.assertFalse(persona_can_write("preimplementer-notes"))
        self.assertFalse(persona_can_write("latest-validator"))

    def test_apply_persona_builds_safe_cursor_bridge_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            workdir = Path(tmp) / "repo"
            personas.mkdir()
            workdir.mkdir()
            _write_persona(personas, "code-reviewer", "# Role\nReview safely.")

            task = apply_persona_to_task(
                {"goal": "Review this change", "persona": "code-reviewer"},
                cfg={"persona_dirs": {"project": str(personas)}},
                top_level_provider="cursor-agent",
                top_level_workdir=str(workdir),
            )

            self.assertEqual(task["acp_command"], "cursor-agent")
            self.assertEqual(task["transport"], "bridge")
            self.assertFalse(task["unsafe_allow_writes"])
            self.assertIn("--mode", task["acp_args"])
            self.assertIn("plan", task["acp_args"])
            self.assertIn("# DELEGATED PERSONA: code-reviewer", task["context"])
            self.assertIn(str(workdir), task["context"])

    def test_apply_persona_treats_empty_top_level_acp_args_as_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            workdir = Path(tmp) / "repo"
            personas.mkdir()
            workdir.mkdir()
            _write_persona(personas, "verifier", "# Role\nVerify safely.")

            task = apply_persona_to_task(
                {"goal": "Verify", "persona": "verifier"},
                cfg={"persona_dirs": {"project": str(personas)}},
                top_level_provider="cursor-agent",
                top_level_workdir=str(workdir),
                top_level_acp_args=[],
            )

            self.assertEqual(task["acp_args"], ["-p", "--output-format", "text", "--model", "gpt-5.5-extra-high", "--mode", "plan"])

    def test_apply_persona_builds_safe_claude_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            workdir = Path(tmp) / "repo"
            personas.mkdir()
            workdir.mkdir()
            _write_persona(personas, "verifier", "# Role\nVerify safely.")

            task = apply_persona_to_task(
                {"goal": "Verify", "persona": "verifier"},
                cfg={"persona_dirs": {"project": str(personas)}},
                top_level_provider="claude",
                top_level_workdir=str(workdir),
            )

            self.assertEqual(task["acp_command"], "claude")
            self.assertIn("--permission-mode", task["acp_args"])
            self.assertIn("plan", task["acp_args"])
            self.assertIn("--add-dir", task["acp_args"])
            self.assertIn(str(workdir), task["acp_args"])

    def test_apply_persona_write_persona_marks_unsafe_but_does_not_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            personas.mkdir()
            _write_persona(personas, "implementer", "# Role\nImplement changes.")

            task = apply_persona_to_task(
                {"goal": "Implement fix", "persona": "implementer"},
                cfg={"persona_dirs": {"project": str(personas)}},
                top_level_provider="cursor-agent",
                top_level_workdir=tmp,
            )

            self.assertTrue(task["unsafe_allow_writes"])
            self.assertIn("--yolo", task["acp_args"])


class TestDelegateTaskPersonaIntegration(unittest.TestCase):
    @patch("tools.delegate_tool.spawn_bridge_session")
    @patch("tools.delegate_tool._load_config")
    def test_delegate_task_persona_defaults_to_bridge_without_credentials(self, mock_cfg, mock_spawn):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            workdir = Path(tmp) / "repo"
            personas.mkdir()
            workdir.mkdir()
            _write_persona(personas, "verifier", "# Role\nVerify completion.")
            mock_cfg.return_value = {
                "max_iterations": 45,
                "persona_dirs": {"project": str(personas)},
            }
            mock_spawn.return_value = {
                "session_id": "hermes-verifier",
                "status": "waiting_for_reply",
                "worker_type": "cursor-agent",
                "model": "gpt-5.5-extra-high",
                "pid": 123,
                "bridge_ready": True,
                "pending": {"message": "ready"},
            }

            result = json.loads(
                delegate_task(
                    goal="Verify this change",
                    persona="verifier",
                    persona_provider="cursor-agent",
                    workdir=str(workdir),
                    parent_agent=_make_parent(),
                )
            )

            self.assertEqual(result["transport"], "bridge")
            self.assertEqual(result["results"][0]["bridge_session_id"], "hermes-verifier")
            _, kwargs = mock_spawn.call_args
            self.assertEqual(kwargs["acp_command"], "cursor-agent")
            self.assertIn("--mode", kwargs["acp_args"])
            self.assertIn("plan", kwargs["acp_args"])
            self.assertIn("# DELEGATED PERSONA: verifier", kwargs["context"])

    @patch("tools.delegate_tool.spawn_bridge_session")
    @patch("tools.delegate_tool._load_config")
    def test_delegate_task_write_persona_still_requires_operator_gate(self, mock_cfg, mock_spawn):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            personas.mkdir()
            _write_persona(personas, "implementer", "# Role\nImplement changes.")
            mock_cfg.return_value = {
                "max_iterations": 45,
                "persona_dirs": {"project": str(personas)},
            }

            result = json.loads(
                delegate_task(
                    goal="Implement this change",
                    persona="implementer",
                    persona_provider="cursor-agent",
                    workdir=tmp,
                    parent_agent=_make_parent(),
                )
            )

            self.assertIn("error", result)
            self.assertIn("unsafe_allow_writes requires operator approval", result["error"])
            mock_spawn.assert_not_called()

    @patch("tools.delegate_tool._load_config")
    def test_delegate_task_unknown_persona_returns_tool_error(self, mock_cfg):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            personas.mkdir()
            mock_cfg.return_value = {
                "max_iterations": 45,
                "persona_dirs": {"project": str(personas)},
            }

            result = json.loads(
                delegate_task(
                    goal="Verify this change",
                    persona="missing-persona",
                    parent_agent=_make_parent(),
                )
            )

            self.assertIn("error", result)
            self.assertIn("delegation persona not found", result["error"])

    @patch("tools.delegate_tool._load_config")
    def test_delegate_task_rejects_mixed_persona_bridge_and_embedded_batch(self, mock_cfg):
        with tempfile.TemporaryDirectory() as tmp:
            personas = Path(tmp) / "personas"
            personas.mkdir()
            _write_persona(personas, "verifier", "# Role\nVerify safely.")
            mock_cfg.return_value = {
                "max_iterations": 45,
                "persona_dirs": {"project": str(personas)},
            }

            result = json.loads(
                delegate_task(
                    tasks=[
                        {"goal": "Verify this change", "persona": "verifier"},
                        {"goal": "Summarize normally"},
                    ],
                    parent_agent=_make_parent(),
                )
            )

            self.assertIn("error", result)
            self.assertIn("Cannot mix bridge transport tasks", result["error"])

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    def test_delegate_task_without_persona_keeps_embedded_path(self, mock_cfg, mock_creds):
        mock_cfg.return_value = {"max_iterations": 45}
        mock_creds.return_value = {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

        with patch("tools.delegate_tool._build_child_agent") as mock_build, patch(
            "tools.delegate_tool._run_single_child"
        ) as mock_run:
            child = MagicMock()
            child._delegate_saved_tool_names = []
            child._delegate_role = "leaf"
            child._credential_pool = None
            mock_build.return_value = child
            mock_run.return_value = {
                "task_index": 0,
                "status": "completed",
                "summary": "done",
                "api_calls": 1,
                "duration_seconds": 0.1,
                "_child_role": "leaf",
            }

            result = json.loads(delegate_task(goal="Do normal embedded work", parent_agent=_make_parent()))

        self.assertEqual(result["results"][0]["status"], "completed")
        _, kwargs = mock_build.call_args
        self.assertIsNone(kwargs.get("override_acp_command"))
        self.assertIsNone(kwargs.get("override_acp_args"))


if __name__ == "__main__":
    unittest.main()
