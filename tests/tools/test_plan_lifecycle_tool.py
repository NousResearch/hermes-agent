"""Tests for tools/plan_lifecycle_tool.py.

Coverage:
  - _spec_coverage: each of the 5 dimensions present / absent
  - _next_question: correct priority ordering + None when complete
  - plan_lifecycle (tool entrypoint):
      • incomplete spec → returns question JSON, no POST
      • complete + no budget → returns budget question, no POST
      • complete + budget → POSTs, asserts URL / Linear-Signature / payload shape
      • 401 from sensor → surfaces as error string
      • 200 from sensor → surfaces created_issue_ids
  - HMAC correctness: test verifies the signature in the mock's received request
    against a recomputed reference using a known test secret
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tools.plan_lifecycle_tool import (
    _next_question,
    _sign_body,
    _spec_coverage,
    plan_lifecycle,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_SPEC: dict[str, Any] = {
    "operator": "blake",
    "budget_usd": 15.0,
    "repos": ["blakeaber/agentic-hub"],
    "success_criteria": ["CI is green"],
    "constraints": ["none"],
}

_TEST_SECRET = "test-secret-abc123"


def _build_args(**overrides: Any) -> dict[str, Any]:
    """Build a full spec kwargs dict, optionally overriding fields."""
    base = dict(_FULL_SPEC)
    base.update(overrides)
    return base


def _recompute_sig(raw_body: bytes) -> str:
    return hmac.new(_TEST_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# _spec_coverage — pure unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestSpecCoverage:
    def test_all_absent_on_empty(self):
        cov = _spec_coverage({})
        assert cov == {
            "operator": False,
            "budget_usd": False,
            "repos": False,
            "success_criteria": False,
            "constraints": False,
        }

    def test_all_present_on_full_spec(self):
        cov = _spec_coverage(_FULL_SPEC)
        assert all(cov.values()), f"Expected all True, got {cov}"

    # --- operator ---

    def test_operator_present(self):
        assert _spec_coverage({"operator": "blake"})["operator"] is True

    def test_operator_absent_blank(self):
        assert _spec_coverage({"operator": ""})["operator"] is False

    def test_operator_absent_whitespace(self):
        assert _spec_coverage({"operator": "   "})["operator"] is False

    def test_operator_absent_none(self):
        assert _spec_coverage({"operator": None})["operator"] is False

    # --- budget_usd ---

    def test_budget_present_zero(self):
        # Zero is a valid budget (means "approved, $0 ceiling")
        assert _spec_coverage({"budget_usd": 0})["budget_usd"] is True

    def test_budget_present_float(self):
        assert _spec_coverage({"budget_usd": 9.99})["budget_usd"] is True

    def test_budget_present_int(self):
        assert _spec_coverage({"budget_usd": 10})["budget_usd"] is True

    def test_budget_absent_none(self):
        assert _spec_coverage({"budget_usd": None})["budget_usd"] is False

    def test_budget_absent_missing(self):
        assert _spec_coverage({})["budget_usd"] is False

    def test_budget_absent_negative(self):
        assert _spec_coverage({"budget_usd": -1})["budget_usd"] is False

    def test_budget_absent_bool_true(self):
        # bool is a subclass of int — must be excluded explicitly
        assert _spec_coverage({"budget_usd": True})["budget_usd"] is False

    # --- repos ---

    def test_repos_present(self):
        assert _spec_coverage({"repos": ["blakeaber/hub"]})["repos"] is True

    def test_repos_absent_empty_list(self):
        assert _spec_coverage({"repos": []})["repos"] is False

    def test_repos_absent_none(self):
        assert _spec_coverage({"repos": None})["repos"] is False

    # --- success_criteria ---

    def test_success_criteria_present_none_sentinel(self):
        # ["none"] counts as present (user explicitly said "no criteria")
        assert _spec_coverage({"success_criteria": ["none"]})["success_criteria"] is True

    def test_success_criteria_present_real(self):
        assert _spec_coverage({"success_criteria": ["CI green"]})["success_criteria"] is True

    def test_success_criteria_absent_empty(self):
        assert _spec_coverage({"success_criteria": []})["success_criteria"] is False

    def test_success_criteria_absent_none(self):
        assert _spec_coverage({"success_criteria": None})["success_criteria"] is False

    # --- constraints ---

    def test_constraints_present_none_sentinel(self):
        assert _spec_coverage({"constraints": ["none"]})["constraints"] is True

    def test_constraints_present_real(self):
        assert _spec_coverage({"constraints": ["no new AWS resources"]})["constraints"] is True

    def test_constraints_absent_empty(self):
        assert _spec_coverage({"constraints": []})["constraints"] is False

    def test_constraints_absent_none(self):
        assert _spec_coverage({"constraints": None})["constraints"] is False


# ---------------------------------------------------------------------------
# _next_question — priority ordering, pure unit tests
# ---------------------------------------------------------------------------


class TestNextQuestion:
    def _all_present(self) -> dict[str, bool]:
        return {d: True for d in ("operator", "budget_usd", "repos", "success_criteria", "constraints")}

    def test_none_when_all_present(self):
        assert _next_question(self._all_present()) is None

    def test_operator_is_first_priority(self):
        cov = self._all_present()
        cov["operator"] = False
        q = _next_question(cov)
        assert q is not None
        assert "operator" in q.lower() or "responsible" in q.lower()

    def test_success_criteria_second(self):
        cov = self._all_present()
        cov["success_criteria"] = False
        # operator is still True, so success_criteria should be next
        q = _next_question(cov)
        assert q is not None
        assert "success" in q.lower() or "criteria" in q.lower()

    def test_repos_third(self):
        cov = self._all_present()
        cov["repos"] = False
        q = _next_question(cov)
        assert q is not None
        assert "repo" in q.lower()

    def test_budget_fourth(self):
        cov = self._all_present()
        cov["budget_usd"] = False
        q = _next_question(cov)
        assert q is not None
        assert "budget" in q.lower() or "usd" in q.lower() or "spend" in q.lower()

    def test_constraints_fifth(self):
        cov = self._all_present()
        cov["constraints"] = False
        q = _next_question(cov)
        assert q is not None
        assert "constraint" in q.lower()

    def test_operator_wins_over_all_others_absent(self):
        # When multiple are absent, operator (highest priority) wins
        cov = {d: False for d in ("operator", "budget_usd", "repos", "success_criteria", "constraints")}
        q = _next_question(cov)
        assert q is not None
        # Must be asking for operator, not any other dimension
        assert "operator" in q.lower() or "responsible" in q.lower() or "team" in q.lower()


# ---------------------------------------------------------------------------
# plan_lifecycle — integration-style tests (mock HTTP + env)
# ---------------------------------------------------------------------------


def _mock_200_response(payload: dict[str, Any] = None) -> MagicMock:
    """Return a mock httpx.Response with status 200."""
    if payload is None:
        payload = {
            "created_issue_ids": ["HUB-101", "HUB-102"],
            "phase_ids": ["phase-A", "phase-B"],
            "signaled": True,
        }
    resp = MagicMock()
    resp.status_code = 200
    resp.is_success = True
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _mock_error_response(status_code: int, text: str = "error") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = False
    resp.text = text
    resp.json.return_value = {"detail": text}
    return resp


class TestPlanLifecycleIncompleteSpec:
    """Incomplete spec → returns question JSON, NO POST made."""

    def test_missing_operator_returns_question(self):
        result = json.loads(
            plan_lifecycle(
                goal="Ship the thing",
                # operator not provided
                budget_usd=10,
                repos=["blakeaber/hub"],
                success_criteria=["CI green"],
                constraints=["none"],
            )
        )
        assert "question" in result
        assert "operator" in result["question"].lower() or "responsible" in result["question"].lower()

    def test_missing_repos_returns_question(self):
        result = json.loads(
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                budget_usd=10,
                # repos not provided
                success_criteria=["CI green"],
                constraints=["none"],
            )
        )
        assert "question" in result
        assert "repo" in result["question"].lower()

    def test_missing_success_criteria_returns_question(self):
        result = json.loads(
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                budget_usd=10,
                repos=["blakeaber/hub"],
                # success_criteria not provided
                constraints=["none"],
            )
        )
        assert "question" in result

    def test_missing_constraints_returns_question(self):
        result = json.loads(
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                budget_usd=10,
                repos=["blakeaber/hub"],
                success_criteria=["CI green"],
                # constraints not provided
            )
        )
        assert "question" in result

    def test_no_post_on_incomplete_spec(self):
        """Verify httpx.post is never called when spec is incomplete."""
        with patch("tools.plan_lifecycle_tool.httpx.post") as mock_post:
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                # budget_usd missing
                repos=["blakeaber/hub"],
                success_criteria=["CI green"],
                constraints=["none"],
            )
        mock_post.assert_not_called()


class TestPlanLifecycleNoBudget:
    """Complete spec but no budget → prompts for budget, NO POST."""

    def test_complete_no_budget_returns_budget_question(self):
        result = json.loads(
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                # budget_usd deliberately omitted
                repos=["blakeaber/hub"],
                success_criteria=["CI green"],
                constraints=["none"],
            )
        )
        assert "question" in result
        q = result["question"].lower()
        assert "budget" in q or "usd" in q or "spend" in q

    def test_no_post_on_missing_budget(self):
        with patch("tools.plan_lifecycle_tool.httpx.post") as mock_post:
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                repos=["blakeaber/hub"],
                success_criteria=["CI green"],
                constraints=["none"],
            )
        mock_post.assert_not_called()

    def test_budget_none_explicit_returns_budget_question(self):
        result = json.loads(
            plan_lifecycle(
                goal="Ship the thing",
                operator="blake",
                budget_usd=None,
                repos=["blakeaber/hub"],
                success_criteria=["CI green"],
                constraints=["none"],
            )
        )
        assert "question" in result


class TestPlanLifecyclePost:
    """Complete spec + budget → POST made; assert URL, sig, payload, and response handling."""

    def _call_with_full_spec(self, **kwargs: Any) -> str:
        """Helper: call plan_lifecycle with a complete spec."""
        return plan_lifecycle(
            goal="Ship the thing",
            operator="blake",
            budget_usd=15.0,
            repos=["blakeaber/agentic-hub"],
            success_criteria=["CI green", "feature behind flag"],
            constraints=["none"],
            plan_id="hub-058",
            ref="main",
            **kwargs,
        )

    def test_posts_to_correct_url(self):
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET, "ORCHESTRATOR_SENSOR_URL": "http://sensor.test:8000"}),
        ):
            self._call_with_full_spec()

        mock_post.assert_called_once()
        url_used = mock_post.call_args[0][0]
        assert url_used == "http://sensor.test:8000/plan/decompose"

    def test_linear_signature_header_is_correct_hmac(self):
        """The Linear-Signature must be hmac-sha256(secret, raw_body) in hex."""
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET, "ORCHESTRATOR_SENSOR_URL": "http://sensor.test:8000"}),
        ):
            self._call_with_full_spec()

        call_kwargs = mock_post.call_args[1]
        sent_body: bytes = call_kwargs["content"]
        sent_headers: dict = call_kwargs["headers"]
        sent_sig: str = sent_headers["Linear-Signature"]

        # Recompute expected signature from the EXACT body that was sent
        expected_sig = _recompute_sig(sent_body)
        assert hmac.compare_digest(sent_sig, expected_sig), (
            f"Signature mismatch.\n  sent:     {sent_sig}\n  expected: {expected_sig}"
        )

    def test_payload_shape(self):
        """POST body must contain all required + optional orchestrator fields."""
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET, "ORCHESTRATOR_SENSOR_URL": "http://sensor.test:8000"}),
        ):
            self._call_with_full_spec()

        sent_body = json.loads(mock_post.call_args[1]["content"])

        # Required fields for the orchestrator (422 if missing)
        assert sent_body["goal"] == "Ship the thing"
        assert sent_body["plan_id"] == "hub-058"
        assert sent_body["repo"] == "blakeaber/agentic-hub"  # defaults to repos[0]
        assert sent_body["ref"] == "main"

        # Optional but forwarded
        assert sent_body["budget_usd"] == 15.0
        assert sent_body["repos"] == ["blakeaber/agentic-hub"]
        assert sent_body["success_criteria"] == ["CI green", "feature behind flag"]
        assert sent_body["constraints"] == ["none"]
        assert sent_body["operator"] == "blake"

    def test_200_response_surfaces_issue_ids(self):
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()),
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            raw = self._call_with_full_spec()

        result = json.loads(raw)
        assert result["success"] is True
        assert result["created_issue_ids"] == ["HUB-101", "HUB-102"]
        assert result["plan_id"] == "hub-058"
        assert result["signaled"] is True

    def test_401_response_surfaces_as_error(self):
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_error_response(401, "invalid signature")),
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            raw = self._call_with_full_spec()

        result = json.loads(raw)
        assert "error" in result
        assert "401" in result["error"] or "Unauthorized" in result["error"] or "secret" in result["error"].lower()

    def test_422_response_surfaces_as_error(self):
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_error_response(422, "missing required field")),
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            raw = self._call_with_full_spec()

        result = json.loads(raw)
        assert "error" in result
        assert "422" in result["error"] or "Unprocessable" in result["error"]

    def test_500_response_surfaces_as_error(self):
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_error_response(500, "internal error")),
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            raw = self._call_with_full_spec()

        result = json.loads(raw)
        assert "error" in result

    def test_missing_secret_returns_error_without_posting(self):
        """If LINEAR_WEBHOOK_SECRET is unset, return error immediately — no POST."""
        env_without_secret = {k: v for k, v in __import__("os").environ.items() if k != "LINEAR_WEBHOOK_SECRET"}
        with (
            patch("tools.plan_lifecycle_tool.httpx.post") as mock_post,
            patch.dict("os.environ", env_without_secret, clear=True),
        ):
            # Unset the secret
            import os as _os
            _os.environ.pop("LINEAR_WEBHOOK_SECRET", None)
            raw = self._call_with_full_spec()

        mock_post.assert_not_called()
        result = json.loads(raw)
        assert "error" in result
        assert "LINEAR_WEBHOOK_SECRET" in result["error"]

    def test_repo_defaults_to_first_repos_entry(self):
        """When ``repo`` is not supplied, default to repos[0]."""
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            plan_lifecycle(
                goal="Ship it",
                operator="blake",
                budget_usd=5.0,
                repos=["blakeaber/agentic-hub"],
                success_criteria=["done"],
                constraints=["none"],
                # repo intentionally omitted
            )

        sent_body = json.loads(mock_post.call_args[1]["content"])
        assert sent_body["repo"] == "blakeaber/agentic-hub"

    def test_plan_id_auto_generated_when_omitted(self):
        """When ``plan_id`` is omitted a 'plan-<uuid8>' id is generated."""
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            plan_lifecycle(
                goal="Ship it",
                operator="blake",
                budget_usd=5.0,
                repos=["blakeaber/agentic-hub"],
                success_criteria=["done"],
                constraints=["none"],
                # plan_id intentionally omitted
            )

        sent_body = json.loads(mock_post.call_args[1]["content"])
        assert sent_body["plan_id"].startswith("plan-")
        assert len(sent_body["plan_id"]) == len("plan-") + 8

    def test_ref_defaults_to_main(self):
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            plan_lifecycle(
                goal="Ship it",
                operator="blake",
                budget_usd=5.0,
                repos=["blakeaber/agentic-hub"],
                success_criteria=["done"],
                constraints=["none"],
                # ref intentionally omitted
            )

        sent_body = json.loads(mock_post.call_args[1]["content"])
        assert sent_body["ref"] == "main"

    def test_goal_required(self):
        """Empty goal returns an error without POSTing."""
        with patch("tools.plan_lifecycle_tool.httpx.post") as mock_post:
            raw = plan_lifecycle(goal="")
        mock_post.assert_not_called()
        result = json.loads(raw)
        assert "error" in result

    def test_content_type_header_sent(self):
        """POST must send Content-Type: application/json."""
        with (
            patch("tools.plan_lifecycle_tool.httpx.post", return_value=_mock_200_response()) as mock_post,
            patch.dict("os.environ", {"LINEAR_WEBHOOK_SECRET": _TEST_SECRET}),
        ):
            self._call_with_full_spec()

        headers = mock_post.call_args[1]["headers"]
        assert headers.get("Content-Type") == "application/json"


class TestSignBody:
    """Unit tests for the _sign_body helper (pure, no I/O)."""

    def test_produces_hex_string(self):
        sig = _sign_body("secret", b"body")
        assert isinstance(sig, str)
        assert all(c in "0123456789abcdef" for c in sig)

    def test_length_is_64_chars(self):
        # sha256 hex digest is always 64 chars
        assert len(_sign_body("secret", b"anything")) == 64

    def test_deterministic(self):
        assert _sign_body("s", b"b") == _sign_body("s", b"b")

    def test_different_body_different_sig(self):
        assert _sign_body("s", b"body1") != _sign_body("s", b"body2")

    def test_different_secret_different_sig(self):
        assert _sign_body("secret1", b"body") != _sign_body("secret2", b"body")

    def test_matches_manual_recomputation(self):
        secret = "test-key"
        body = b'{"plan_id": "hub-001"}'
        expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        assert _sign_body(secret, body) == expected
