import importlib.util
import pathlib
import tempfile
import unittest
from typing import Any


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "workflow_lab_analyze.py"
SPEC = importlib.util.spec_from_file_location("workflow_lab_analyze", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load analyzer module")
MODULE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WorkflowLabAnalyzerTests(unittest.TestCase):
    def test_redacts_url_userinfo(self):
        value = MODULE.redact("git clone https://user:password@example.test/repo.git")
        self.assertNotIn("password", value)
        self.assertIn("https://[REDACTED]example.test", value)

    def test_normalizes_millisecond_timestamps(self):
        value = MODULE.unix_timestamp(1_725_000_000_000)
        self.assertTrue(value.startswith("2024-"))

    def test_state_ingestion_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as directory:
            state = pathlib.Path(directory) / "state.db"
            import sqlite3

            con = sqlite3.connect(state)
            con.execute("create table sessions (id text, source text, model text, started_at real, message_count integer, tool_call_count integer, cwd text, title text)")
            con.execute("insert into sessions values ('id', 'cli', 'model', 1725000000, 3, 2, '/tmp/repo', 'safe title')")
            con.commit()
            original_home = MODULE.HOME
            MODULE.HOME = pathlib.Path(directory)
            try:
                (pathlib.Path(directory) / ".hermes").mkdir()
                state.rename(pathlib.Path(directory) / ".hermes" / "state.db")
                rows = []
                MODULE.ingest_hermes_state(rows)
            finally:
                MODULE.HOME = original_home
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "session")
            self.assertNotIn("system_prompt", rows[0]["text"])


if __name__ == "__main__":
    unittest.main()