"""Peer delivery admission tests exercising the gateway queue helper."""
import ast
import pathlib
import threading
import unittest
from unittest.mock import Mock

ROOT = pathlib.Path(__file__).resolve().parents[2]


def method_handler(namespace):
    path = ROOT / "tui_gateway" / "methods_session.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and any(isinstance(d, ast.Call) and d.args and isinstance(d.args[0], ast.Constant)
                        and d.args[0].value == "session.peer_deliver" for d in n.decorator_list))
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def real_enqueue():
    path = ROOT / "tui_gateway" / "session_auto_continue.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_enqueue_prompt")
    namespace = {"_drop_queued_duplicates_of_inflight_user": lambda session: None,
                 "_ac_inflight_original": lambda session: ""}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_enqueue_prompt"]


class CapturedThreading:
    def __init__(self, starts): self.starts = starts
    def Thread(self, **kwargs):
        starts = self.starts
        class Thread:
            def start(self): starts.append(kwargs)
        return Thread()


class PeerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.transport = object()
        self.source = {"session_key": "arbitrary:source", "history_lock": threading.RLock()}
        self.target = {"session_key": "arbitrary:target", "history_lock": threading.RLock(),
                       "running": True, "transport": self.transport}
        self.started = []
        self.persist = Mock()
        self.ns = {"_sessions_lock": threading.RLock(), "_sessions": {"s1": self.source, "s2": self.target},
                   "_enqueue_prompt": real_enqueue(), "_persist_queued_user_row": self.persist,
                   "_drain_queued_prompt": Mock(), "_ok": lambda rid, result: {"result": result},
                   "_err": lambda rid, code, msg: {"error": {"code": code, "message": msg}},
                   "threading": CapturedThreading(self.started), "time": __import__("time"), "logger": Mock()}
        self.deliver = method_handler(self.ns)
        self.params = {"source": "arbitrary:source", "target": "arbitrary:target", "content": "note"}

    def test_busy_exact_queue_preserves_transport_and_author(self):
        response = self.deliver(1, self.params)
        self.assertEqual(response["result"], {"accepted": True, "status": "queued"})
        self.assertIs(self.target["transport"], self.transport)
        self.assertEqual(self.target["queued_prompt"]["transport"], None)
        self.assertEqual(self.target["queued_prompt"]["turn_author"]["label"], "arbitrary:source")
        self.assertEqual(self.started, [])
        self.persist.assert_called_once()

    def test_missing_or_ambiguous_source_rejected_without_queue(self):
        self.ns["_sessions"].pop("s1")
        self.assertIn("error", self.deliver(1, self.params))
        self.ns["_sessions"]["s1"] = self.source
        self.ns["_sessions"]["s3"] = dict(self.source)
        self.assertIn("error", self.deliver(1, self.params))
        self.assertNotIn("queued_prompt", self.target)

    def test_capacity_counts_head_plus_tail(self):
        self.target["queued_prompt"] = {"text": "head"}
        self.target["queued_prompts"] = [{} for _ in range(99)]
        self.assertEqual(self.deliver(1, self.params)["error"]["code"], 5030)
        self.assertEqual(len(self.target["queued_prompts"]), 99)

    def test_persistence_failure_does_not_falsely_reject_accepted(self):
        self.persist.side_effect = OSError("storage unavailable")
        self.assertTrue(self.deliver(1, self.params)["result"]["accepted"])
        self.assertEqual(self.target["queued_prompt"]["text"], "note")

    def test_concurrent_idle_deliveries_share_single_drain(self):
        self.target["running"] = False
        barrier = threading.Barrier(3)
        errors = []
        def send():
            try:
                barrier.wait()
                self.deliver(1, self.params)
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=send) for _ in range(2)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.target["queued_prompt"]["turn_author"]["kind"], "peer")
        self.assertEqual(len(self.target.get("queued_prompts", [])), 1)

    def test_idle_drain_marker_clears_after_worker_finishes(self):
        self.target["running"] = False
        self.assertTrue(self.deliver(1, self.params)["result"]["accepted"])
        self.assertEqual(len(self.started), 1)
        self.started[0]["target"]()
        self.assertFalse(self.target.get("_peer_drain_scheduled"))
        self.assertTrue(self.deliver(2, self.params)["result"]["accepted"])
        self.assertEqual(len(self.started), 2)

    def test_idle_drain_thread_start_failure_dispatches_inline(self):
        self.target["running"] = False
        class FailingThreading:
            class Thread:
                def __init__(self, **kwargs): pass
                def start(self): raise RuntimeError("cannot start worker")
        self.ns["threading"] = FailingThreading
        result = self.deliver(1, self.params)
        self.assertTrue(result["result"]["accepted"])
        self.ns["_drain_queued_prompt"].assert_called_once_with(1, "s2", self.target)
        self.assertFalse(self.target.get("_peer_drain_scheduled"))

if __name__ == "__main__":
    unittest.main()
