"""End-to-end Hermes ↔ real cascadeflow integration tests.

Unlike test_delegate.py::TestCascadeFlowDelegationRouting (which stubs out
cascadeflow.integrations.hermes), these tests exercise the real installed
package so the full classification → route-selection → decision-annotation
pipeline runs through Hermes' delegation resolver.

If cascadeflow is not installed, all tests in this module are skipped.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import cascadeflow.integrations.hermes  # noqa: F401
    CASCADEFLOW_AVAILABLE = True
except ImportError:
    CASCADEFLOW_AVAILABLE = False

from tools.delegate_tool import _resolve_task_delegation_route


def _make_mock_parent():
    class _Parent:
        provider = "openai"
        model = "gpt-4o"
        base_url = None
        api_key = "sk-test"
        api_mode = None
        fallback_chain = ()
    return _Parent()


BASE_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
}


@unittest.skipUnless(CASCADEFLOW_AVAILABLE, "cascadeflow package not installed")
class TestRealCascadeFlowEndToEnd(unittest.TestCase):
    """Hermes resolver wired up to the real cascadeflow.integrations.hermes router."""

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_code_domain_task_picks_code_route(self, mock_resolve):
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "cascadeflow_model_routing": {
                "enabled": True,
                "mode": "route",
                "routes": {
                    "code": {
                        "provider": "nous",
                        "model": "nous/hermes-4.1",
                        "reasoning_effort": "high",
                    }
                },
            }
        }

        _, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "Implement the typed python API client with unit tests"},
            None,
            "leaf",
        )

        # Real classifier identified "code" → matched route → applied
        self.assertEqual(effort, "high")
        self.assertIsNotNone(routing)
        self.assertTrue(routing["metadata"]["hermes_applied"])
        self.assertEqual(routing.get("domain"), "code")

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_unmatched_domain_does_not_apply(self, mock_resolve):
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "cascadeflow_model_routing": {
                "enabled": True,
                "mode": "route",
                "routes": {
                    "research": {
                        "provider": "nous",
                        "model": "nous/hermes-4.1",
                    }
                },
            }
        }

        _, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "Implement the typed python API client with unit tests"},
            None,
            "leaf",
        )

        # Code task with only research route → no apply
        self.assertIsNone(effort)
        self.assertIsNotNone(routing)
        self.assertFalse(routing["metadata"]["hermes_applied"])

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_high_stakes_legal_domain_inherits_when_unconfigured(self, mock_resolve):
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "cascadeflow_model_routing": {
                "enabled": True,
                "mode": "route",
                "routes": {
                    "code": {"provider": "nous", "model": "nous/hermes-4.1"},
                },
            }
        }

        _, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "Review this contract for legal privacy compliance terms"},
            None,
            "leaf",
        )

        self.assertIsNone(effort)
        self.assertEqual(routing.get("domain"), "legal")
        self.assertFalse(routing["metadata"]["hermes_applied"])

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_disabled_config_does_not_call_cascadeflow(self, mock_resolve):
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "cascadeflow_model_routing": {"enabled": False},
        }

        creds, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "any task"},
            None,
            "leaf",
        )

        # Disabled → no routing decision at all
        self.assertEqual(creds, BASE_CREDS)
        self.assertIsNone(effort)
        self.assertIsNone(routing)

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_observe_mode_records_decision_without_applying(self, mock_resolve):
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "cascadeflow_model_routing": {
                "enabled": True,
                "mode": "observe",
                "routes": {
                    "code": {"provider": "nous", "model": "nous/hermes-4.1"},
                },
            }
        }

        _, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "Implement the python API"},
            None,
            "leaf",
        )

        # Observe mode → audit recorded but not applied
        self.assertIsNone(effort)
        self.assertIsNotNone(routing)
        self.assertFalse(routing["metadata"]["hermes_applied"])
        self.assertEqual(
            routing["metadata"].get("hermes_reason"), "observe_or_inherit"
        )

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_min_confidence_gate_blocks_low_confidence_route(self, mock_resolve):
        # Cascadeflow router enforces min_confidence internally and returns
        # action="inherit" with reason="confidence_below_threshold". Hermes
        # then sees action != "route" and records its own "observe_or_inherit"
        # reason. Hermes' separate confidence_below_minimum branch is therefore
        # defensive only — it never fires when cascadeflow is the decision source.
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "cascadeflow_model_routing": {
                "enabled": True,
                "mode": "route",
                "min_confidence": 0.99,
                "routes": {
                    "code": {"provider": "nous", "model": "nous/hermes-4.1"},
                },
            }
        }

        _, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "Implement the python API"},
            None,
            "leaf",
        )

        self.assertIsNone(effort)
        self.assertFalse(routing["metadata"]["hermes_applied"])
        self.assertEqual(
            routing["metadata"].get("hermes_reason"), "observe_or_inherit"
        )
        # Cascadeflow's original reason is preserved on the top-level decision
        self.assertEqual(routing.get("reason"), "confidence_below_threshold")

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    def test_static_delegation_config_wins_over_cascadeflow(self, mock_resolve):
        mock_resolve.return_value = BASE_CREDS
        cfg = {
            "provider": "openai",  # static config present
            "model": "gpt-4o",
            "cascadeflow_model_routing": {
                "enabled": True,
                "mode": "route",
                "routes": {
                    "code": {"provider": "nous", "model": "nous/hermes-4.1"},
                },
            },
        }

        _, effort, routing = _resolve_task_delegation_route(
            cfg,
            _make_mock_parent(),
            {"goal": "Implement the python API"},
            None,
            "leaf",
        )

        self.assertIsNone(effort)
        self.assertFalse(routing["metadata"]["hermes_applied"])
        self.assertEqual(
            routing["metadata"].get("hermes_reason"),
            "static_delegation_config_wins",
        )


if __name__ == "__main__":
    unittest.main()
