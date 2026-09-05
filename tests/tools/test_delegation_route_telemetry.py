"""T6 — route provenance telemetry on result entries, progress payloads, and lifecycle handles.

Contract: profile-run result entries (and SubagentHandle) carry requested_profile /
resolved_provider / resolved_model / fallback_policy. Legacy (profile-less) runs OMIT all four so
their entries stay byte-identical to main; profile runs carry the SPAWN-TIME route. The
pre-existing "model" key stays the child's LIVE model, so
resolved_model != model is the requested-vs-effective divergence signal when a fallback fires.
"""

import unittest

from agent.subagent_lifecycle import PUBLIC_CONTRACT_VERSION, SubagentHandle
from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


class _FakeChild:
    """Minimal child double; route attrs stamped per-test (None == unstamped/legacy)."""

    _route_requested_profile: "str | None" = None
    _route_resolved_provider: "str | None" = None
    _route_resolved_model: "str | None" = None
    _route_fallback_policy: "str | None" = None
    model = "live-model"
    _delegate_role = "leaf"
    session_prompt_tokens = 10
    session_completion_tokens = 5
    session_estimated_cost_usd = 0.0
    session_cost_status = "estimated"


def _entry(child):
    result = {"final_response": "done", "completed": True, "messages": []}
    return _build_result_entry(child, result, 0, 1.5, _SchemaOutcome(None, None, [], 0))


ROUTE_KEYS = ("requested_profile", "resolved_provider", "resolved_model", "fallback_policy")

# Today's legacy result-entry key set on main — EXACTLY this, no provenance keys. Snapshot of the
# KEY SET only (values are contract-tested elsewhere) so legacy payload shape stays byte-identical.
LEGACY_ENTRY_KEYS = {
    "task_index", "status", "summary", "api_calls", "duration_seconds", "model", "exit_reason",
    "truncated", "tokens", "tool_trace", "_child_role", "_child_cost_usd", "cost_usd", "cost_status",
}


class TestResultEntryRouteTelemetry(unittest.TestCase):
    def test_legacy_run_omits_route_keys_and_key_set_matches_main(self):
        entry = _entry(_FakeChild())
        for key in ROUTE_KEYS:
            self.assertNotIn(key, entry)
        self.assertEqual(set(entry.keys()), LEGACY_ENTRY_KEYS)
        self.assertEqual(entry["model"], "live-model")

    def test_profile_run_carries_all_four_and_model(self):
        child = _FakeChild()
        child._route_requested_profile = "fast"
        child._route_resolved_provider = "openrouter"
        child._route_resolved_model = "live-model"
        child._route_fallback_policy = "none"
        entry = _entry(child)
        self.assertEqual(entry["requested_profile"], "fast")
        self.assertEqual(entry["resolved_provider"], "openrouter")
        self.assertEqual(entry["resolved_model"], "live-model")
        self.assertEqual(entry["fallback_policy"], "none")
        self.assertEqual(entry["model"], "live-model")

    def test_fallback_divergence_survives_verbatim(self):
        """resolved_model is the spawn-time route; "model" is the child's live model. When a
        fallback fires mid-run they diverge — both must survive verbatim as the signal."""
        child = _FakeChild()
        child.model = "fallback/effective-model"
        child._route_requested_profile = "fast"
        child._route_resolved_provider = "openrouter"
        child._route_resolved_model = "primary/spawn-model"
        child._route_fallback_policy = "profile:fast"
        entry = _entry(child)
        self.assertEqual(entry["resolved_model"], "primary/spawn-model")
        self.assertEqual(entry["model"], "fallback/effective-model")
        self.assertNotEqual(entry["model"], entry["resolved_model"])
        self.assertEqual(entry["fallback_policy"], "profile:fast")


class TestChildRouteStamp(unittest.TestCase):
    def test_stamp_helper_empty_for_unstamped_child(self):
        from tools.delegate_tool_child_run import _route_telemetry
        self.assertEqual(_route_telemetry(object()), {})


class TestHandleRouteTelemetry(unittest.TestCase):
    def _base(self):
        return {
            "contract_version": PUBLIC_CONTRACT_VERSION, "subagent_id": "sa-1", "parent_session_id": "p",
            "correlation_id": None, "created_at": 1.0, "provider": "openai", "model": "gpt-x",
            "role": "leaf", "depth": 1, "capability": "cap",
        }

    def test_from_dict_without_new_keys_defaults_none(self):
        handle = SubagentHandle.from_dict(self._base())
        for key in ROUTE_KEYS:
            self.assertIsNone(getattr(handle, key))

    def test_from_dict_round_trip_with_new_keys(self):
        payload = {
            **self._base(), "requested_profile": "fast", "resolved_provider": "openrouter",
            "resolved_model": "m1", "fallback_policy": "profile:fast",
        }
        handle = SubagentHandle.from_dict(payload)
        self.assertEqual(handle.requested_profile, "fast")
        self.assertEqual(handle.resolved_provider, "openrouter")
        self.assertEqual(handle.resolved_model, "m1")
        self.assertEqual(handle.fallback_policy, "profile:fast")
        self.assertEqual(SubagentHandle.from_dict(handle.to_dict()), handle)


if __name__ == "__main__":
    unittest.main()
