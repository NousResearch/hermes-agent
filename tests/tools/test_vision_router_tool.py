"""Tests for the Vision Router wrapper tool (Stage 1: registered but hidden).

The minimal-enablement gate keeps the wrapper model-invisible until a session
explicitly enables the local Vision Router.
"""
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from tools import vision_router_tool as vrt  # noqa: E402
from tools.vision_orchestrator import analyze_image  # noqa: E402  (ensure import works)
# Ensure the local Vision tools are registered in the registry (registration
# happens at import time; get_tool_definitions does not import them).
import tools.vision_tools  # noqa: E402,F401
import tools.vision_ocr_page  # noqa: E402,F401


def _mk_result(**over):
    base = {
        "request_id": "REQ-1",
        "task": "UI_READ",
        "execution_status": "SUCCESS",
        "quality_decision": "PASS",
        "initial_model_slot": "PRECISION_VLM",
        "final_model_slot": "PRECISION_VLM",
        "actual_model": "qwen3.6:27b",
        "logical_model_calls": 1,
        "human_review_required": False,
        "recommended_next_slot": None,
        "structured": {"observed_text": ["ok"]},
        "trace": [{"total_ms": 9000, "done_reason": "stop",
                   "latency_ms": 9000}],
    }
    base.update(over)
    return base


class TestSchema:
    def test_schema_shape(self):
        s = vrt.VISION_ROUTER_SCHEMA
        assert s["name"] == "vision_router_analyze"
        props = s["parameters"]["properties"]
        assert "source_handle" in props
        assert props["source_handle"]["type"] == "string"
        assert s["parameters"]["required"] == ["source_handle"]
        # model may not control runtime/activation parameters
        for forbidden in ("enabled", "transport", "model", "base_url",
                          "num_ctx", "timeout", "criticality"):
            assert forbidden not in props, forbidden
        # task is a narrow enum
        assert set(props["task"]["enum"]) == {
            "UI_READ", "SCENE_DESCRIBE", "EVIDENCE_VERIFY", "EXACT_OCR"}

    def test_toolset_registered(self):
        from tools.registry import registry
        m = registry.get_tool_to_toolset_map()
        assert m.get("vision_router_analyze") == "vision"


class TestSourceAuthorization:
    def setup_method(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()

    pytestmark = pytest.mark.asyncio

    async def test_empty_handle_denied_zero_calls(self):
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock()) as m:
            out = json.loads(await vrt._handle_vision_router({"source_handle": ""}))
        m.assert_not_awaited()
        assert out["execution_status"] == "POLICY_BLOCKED"
        assert out["logical_model_calls"] == 0
        assert out["error"].startswith("SOURCE_DENIED")

    async def test_unknown_handle_denied_zero_calls(self):
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock()) as m:
            out = json.loads(await vrt._handle_vision_router(
                {"source_handle": "not-a-handle"}))
        m.assert_not_awaited()
        assert out["execution_status"] == "POLICY_BLOCKED"
        assert out["logical_model_calls"] == 0

    async def test_attachment_handle_resolves(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        assert vrt._resolve_source(key) == "/tmp/synthetic-vision.png"
        assert vrt._resolve_source("attachment://sess/other") is None

    async def test_authorized_handle_invokes_analyze(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock(return_value=_mk_result())) as m:
            out = json.loads(await vrt._handle_vision_router(
                {"source_handle": key, "task": "UI_READ"}))
        m.assert_awaited_once()
        assert out["execution_status"] == "SUCCESS"
        assert out["quality_decision"] == "PASS"


class TestCriticalityDerivation:
    def setup_method(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()

    def test_policy_derived(self):
        assert vrt._criticality_for("UI_READ") == "HIGH"
        assert vrt._criticality_for("EXACT_OCR") == "HIGH"
        assert vrt._criticality_for("SCENE_DESCRIBE") == "NORMAL"
        assert vrt._criticality_for("EVIDENCE_VERIFY") == "NORMAL"
        assert vrt._criticality_for("BOGUS") == "NORMAL"

    @pytest.mark.asyncio
    async def test_request_criticality_not_model_controlled(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        captured = {}

        async def fake(request, **kw):
            captured["criticality"] = request.criticality.value
            captured["task"] = request.task.value
            return _mk_result()

        with patch("tools.vision_orchestrator.analyze_image",
                   new=fake):
            await vrt._handle_vision_router(
                {"source_handle": key, "task": "UI_READ"})
        assert captured["criticality"] == "HIGH"
        # even a weird task value falls back to a bounded set
        await vrt._handle_vision_router(
            {"source_handle": key, "task": "!!!UNKNOWN!!!"})
        assert captured["task"] == "UI_READ"


class TestEnvelope:
    def setup_method(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()

    pytestmark = pytest.mark.asyncio

    async def test_envelope_is_safe_and_bounded(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock(return_value=_mk_result())):
            out = json.loads(await vrt._handle_vision_router(
                {"source_handle": key, "task": "UI_READ"}))
        text = json.dumps(out, ensure_ascii=False)
        for banned in ("thinking", "raw_text", "base64,", "/mnt/", "/Users/",
                       "api_key"):
            assert banned not in text, banned
        assert out["source_handle"] == key
        assert out["logical_model_calls"] == 1

    async def test_ocr_excerpt_truncates_long_text(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        long_text = "字" * 6000
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock(return_value=_mk_result(
                       structured={"observed_text": [long_text]}))):
            out = json.loads(await vrt._handle_vision_router(
                {"source_handle": key, "task": "EXACT_OCR"}))
        assert out["ocr_meta"] is not None
        assert out["ocr_meta"]["truncated"] is True
        assert out["ocr_meta"]["total_chars"] == 6000
        assert out["ocr_meta"]["returned_chars"] <= 4000
        assert len(out["observed_text"][0]) <= 4000
        assert out["ocr_meta"]["full_text_policy"] == "explicit_followup_required"

    async def test_short_ocr_not_truncated(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock(return_value=_mk_result(
                       structured={"observed_text": ["short"]}))):
            out = json.loads(await vrt._handle_vision_router(
                {"source_handle": key, "task": "EXACT_OCR"}))
        assert out["ocr_meta"] is None
        assert out["observed_text"] == ["short"]

    async def test_fail_closed_on_orchestrator_exception(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/synthetic", "/tmp/synthetic-vision.png")
        key = "attachment://sess/synthetic"
        with patch("tools.vision_orchestrator.analyze_image",
                   new=AsyncMock(side_effect=RuntimeError("boom"))):
            out = json.loads(await vrt._handle_vision_router(
                {"source_handle": key}))
        assert out["execution_status"] == "INVALID_RESPONSE"
        assert out["quality_decision"] == "NOT_EVALUATED"
        assert out["logical_model_calls"] == 0


class TestVisibilityGate:
    def setup_method(self):
        # The effective-flag fingerprint includes the in-memory session flag;
        # reset it so these tests exercise the server flag exclusively.
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(False)

    def test_flag_default_false(self):
        assert vrt._config_value(None, "ocr_excerpt_chars", 1) == 1

    def test_vision_router_tool_visible_default_false(self):
        from toolsets import vision_router_tool_visible
        assert vision_router_tool_visible(None) is False
        assert vision_router_tool_visible({}) is False
        assert vision_router_tool_visible(
            {"vision_router": {"enabled": True}}) is True
        assert vision_router_tool_visible(
            {"vision_router": {"enabled": False}}) is False

    def test_get_tool_definitions_filters_wrapper_when_disabled(self):
        from model_tools import get_tool_definitions
        with patch("hermes_cli.config.load_config",
                   return_value={"vision_router": {"enabled": False}}):
            defs = get_tool_definitions(
                enabled_toolsets=["vision"], quiet_mode=True)
        names = [d["function"]["name"] for d in defs]
        assert "vision_router_analyze" not in names
        # legacy vision_analyze stays (status quo) when the Router is off

    def test_get_tool_definitions_includes_wrapper_when_enabled(self):
        from model_tools import get_tool_definitions
        with patch("hermes_cli.config.load_config",
                   return_value={"vision_router": {"enabled": True}}):
            defs = get_tool_definitions(
                enabled_toolsets=["vision"], quiet_mode=True)
        names = [d["function"]["name"] for d in defs]
        assert "vision_router_analyze" in names

    def test_router_on_replaces_legacy_vision_analyze(self):
        # ChatGPT review finding: when the Router is on, the model must see
        # exactly ONE vision entry point — the wrapper replaces the legacy
        # auxiliary vision_analyze (design §7 "replacing the legacy role").
        from model_tools import get_tool_definitions
        with patch("hermes_cli.config.load_config",
                   return_value={"vision_router": {"enabled": True}}):
            defs = get_tool_definitions(
                enabled_toolsets=["coding", "vision"], quiet_mode=True)
        names = [d["function"]["name"] for d in defs]
        assert "vision_router_analyze" in names
        assert "vision_analyze" not in names

    def test_config_defaults_added(self):
        import hermes_cli.config_defaults as cd
        vr = cd.DEFAULT_CONFIG["auxiliary"]["vision_router"]
        assert vr["ocr_excerpt_chars"] == 4000
        assert vr["ocr_page_chars"] == 65536
        assert vr["per_workflow_max_calls"] == 20
        assert vr["enabled"] is False


class TestSessionState:
    """In-memory session state (Stage-3): flag / budgets / allowlist / OCR."""

    def setup_method(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(False)

    def test_toggle_on_off(self):
        from tools.vision_session_state import vision_session_state as s
        s.set_enabled(True)
        assert s.enabled is True
        s.set_enabled(False)
        assert s.enabled is False

    def test_off_revokes_all_sources(self):
        from tools.vision_session_state import vision_session_state as s
        s.set_enabled(True)
        s.register_attachment("attachment://sess/a", "/tmp/a.png")
        s.register_ocr_result("ocr://r", "/tmp/r.txt")
        s.set_enabled(False)
        assert s.resolve_attachment("attachment://sess/a") is None
        assert s.resolve_ocr_result("ocr://r") is None

    def test_budget_turn_and_session(self):
        from tools.vision_session_state import vision_session_state as s
        s.set_enabled(True)
        assert s.consume_call(1, 5) is None
        # second consume while the first is still in flight -> BUSY
        assert s.consume_call(1, 5) == "VISION_BUSY_IN_FLIGHT"
        s.finish_call()
        # same turn, second finished call -> turn budget exhausted
        assert s.consume_call(1, 5) == "TURN_BUDGET_EXHAUSTED"
        s.fail_call()
        s.begin_turn()
        assert s.consume_call(1, 5) is None
        for _ in range(3):
            s.finish_call()
            s.begin_turn()
            assert s.consume_call(1, 5) is None
        s.finish_call()
        s.begin_turn()
        assert s.consume_call(1, 5) == "SESSION_BUDGET_EXHAUSTED"
        s.fail_call()

    def test_in_flight_blocks(self):
        from tools.vision_session_state import vision_session_state as s
        s.set_enabled(True)
        assert s.consume_call(1, 5) is None
        assert s.consume_call(1, 5) == "VISION_BUSY_IN_FLIGHT"
        s.fail_call()

    def test_attachment_register_resolve_revoke(self):
        from tools.vision_session_state import vision_session_state as s
        s.set_enabled(True)
        s.register_attachment("attachment://sess/a", "/tmp/a.png")
        assert s.resolve_attachment("attachment://sess/a") == "/tmp/a.png"
        s.revoke_attachment("attachment://sess/a")
        assert s.resolve_attachment("attachment://sess/a") is None

    def test_dedupe_and_authorization(self):
        from tools.vision_session_state import vision_session_state as s
        s.set_enabled(True)
        assert s.needs_authorization("h", "UI_READ") is False
        s.record_call("h", "UI_READ")
        assert s.needs_authorization("h", "UI_READ") is True
        s.authorize_source_task("h", "UI_READ")
        assert s.needs_authorization("h", "UI_READ") is False


class TestStage3SessionGates:
    """Wrapper gates: SESSION_DISABLED / budgets / dedupe / attachment://."""

    pytestmark = pytest.mark.asyncio

    def setup_method(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/taobao-0003", "/tmp/synthetic-a.png")
        vision_session_state.register_attachment(
            "attachment://sess/taobao-0004", "/tmp/synthetic-b.png")

    @staticmethod
    def _real_cfg():
        import copy
        from hermes_cli.config import load_config
        cfg = copy.deepcopy(load_config())
        cfg["auxiliary"]["vision_router"]["enabled"] = True
        return cfg

    async def test_session_disabled_rejects_zero_calls(self):
        from tools.registry import registry
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(False)
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        called = {"n": 0}

        async def fake_ai(*a, **k):
            called["n"] += 1
            return {"execution_status": "SUCCESS"}

        with patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = await handler({"source_handle": "attachment://sess/taobao-0003",
                                 "task": "UI_READ"})
        env = json.loads(raw)
        assert env["error"] == "SESSION_DISABLED"
        assert called["n"] == 0

    async def test_turn_budget_exhausted(self):
        from tools.registry import registry
        from tools.vision_session_state import vision_session_state
        vision_session_state.begin_turn()
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None

        async def fake_ai(*a, **k):
            return {"execution_status": "SUCCESS", "quality_decision": "PASS",
                    "initial_model_slot": "PRECISION_VLM",
                    "final_model_slot": "PRECISION_VLM",
                    "actual_model": "qwen3.6:27b",
                    "logical_model_calls": 1,
                    "structured": {"observed_text": ["ok"]},
                    "trace": []}

        with patch("hermes_cli.config.load_config",
                   return_value=self._real_cfg()), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            env1 = json.loads(await handler(
                {"source_handle": "attachment://sess/taobao-0003", "task": "UI_READ"}))
            # second call in same turn -> turn budget exhausted
            env2 = json.loads(await handler(
                {"source_handle": "attachment://sess/taobao-0004", "task": "UI_READ"}))
        assert env1["execution_status"] == "SUCCESS"
        assert env2["error"] == "TURN_BUDGET_EXHAUSTED"
        vision_session_state.set_enabled(False)

    async def test_same_source_task_needs_authorization(self):
        from tools.registry import registry
        from tools.vision_session_state import vision_session_state
        vision_session_state.begin_turn()
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None

        async def fake_ai(*a, **k):
            return {"execution_status": "SUCCESS", "quality_decision": "PASS",
                    "initial_model_slot": "PRECISION_VLM",
                    "final_model_slot": "PRECISION_VLM",
                    "actual_model": "qwen3.6:27b",
                    "logical_model_calls": 1,
                    "structured": {"observed_text": ["ok"]},
                    "trace": []}

        with patch("hermes_cli.config.load_config",
                   return_value=self._real_cfg()), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            env1 = json.loads(await handler(
                {"source_handle": "attachment://sess/taobao-0003", "task": "UI_READ"}))
            vision_session_state.begin_turn()  # new turn (fresh budget)
            env2 = json.loads(await handler(
                {"source_handle": "attachment://sess/taobao-0003", "task": "UI_READ"}))
        assert env1["execution_status"] == "SUCCESS"
        assert env2["error"].startswith("NEEDS_AUTHORIZATION")
        vision_session_state.set_enabled(False)

    async def test_attachment_handle_authorized(self):
        from tools.registry import registry
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/a", "/tmp/stage3-att.png")
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        received = {}

        async def fake_ai(request, **kw):
            received["source"] = request.image_source
            return {"execution_status": "SUCCESS", "quality_decision": "PASS",
                    "initial_model_slot": "PRECISION_VLM",
                    "final_model_slot": "PRECISION_VLM",
                    "actual_model": "qwen3.6:27b",
                    "logical_model_calls": 1,
                    "structured": {"observed_text": ["ok"]},
                    "trace": []}

        with patch("hermes_cli.config.load_config",
                   return_value=self._real_cfg()), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = await handler({"source_handle": "attachment://sess/a",
                                 "task": "UI_READ"})
        env = json.loads(raw)
        assert env["execution_status"] == "SUCCESS"
        assert received["source"] == "/tmp/stage3-att.png"
        vision_session_state.set_enabled(False)

    async def test_unknown_attachment_denied_zero_calls(self):
        from tools.registry import registry
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        called = {"n": 0}

        async def fake_ai(*a, **k):
            called["n"] += 1
            return {"execution_status": "SUCCESS"}

        with patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = await handler({"source_handle": "attachment://sess/nope",
                                 "task": "UI_READ"})
        env = json.loads(raw)
        assert env["error"] == "SOURCE_DENIED: source handle is not authorized"
        assert called["n"] == 0
        vision_session_state.set_enabled(False)


class TestOcrPageTool:
    """vision_ocr_page: bounded pagination, session binding, zero calls."""

    @staticmethod
    def _mk_handle(text):
        from tools.vision_session_state import vision_session_state
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                        encoding="utf-8")
        f.write(text)
        f.close()
        vision_session_state.set_enabled(True)
        vision_session_state.register_ocr_result("ocr://page-test", f.name)
        return f.name

    def test_first_page_bounded(self):
        from tools.vision_ocr_page import _handle_vision_ocr_page
        long = "页" * 70000
        self._mk_handle(long)
        raw = _handle_vision_ocr_page({"handle": "ocr://page-test", "page": 1})
        env = json.loads(raw)
        assert env["execution_status"] == "SUCCESS"
        assert len(env["page_text"]) == 65536
        assert env["total_chars"] == 70000
        assert env["remaining_chars"] == 70000 - 65536
        assert env["truncated"] is True
        assert env["sha256"]
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(False)

    def test_second_page(self):
        from tools.vision_ocr_page import _handle_vision_ocr_page
        long = "行" * 70000
        self._mk_handle(long)
        raw = _handle_vision_ocr_page({"handle": "ocr://page-test", "page": 2})
        env = json.loads(raw)
        assert env["execution_status"] == "SUCCESS"
        assert env["remaining_chars"] == 0
        assert env["page_chars"] == 70000 - 65536
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(False)

    def test_unknown_handle_denied(self):
        from tools.vision_ocr_page import _handle_vision_ocr_page
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        raw = _handle_vision_ocr_page({"handle": "ocr://missing", "page": 1})
        env = json.loads(raw)
        assert env["error"].startswith("SOURCE_DENIED")
        vision_session_state.set_enabled(False)

    def test_session_disabled_denied(self):
        from tools.vision_ocr_page import _handle_vision_ocr_page
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(False)
        raw = _handle_vision_ocr_page({"handle": "ocr://x", "page": 1})
        env = json.loads(raw)
        assert env["error"] == "SESSION_DISABLED"

    def test_schema_no_path_fields(self):
        import tools.vision_ocr_page as vop
        props = vop.VISION_OCR_PAGE_SCHEMA["parameters"]["properties"]
        for banned in ("path", "base_url", "endpoint"):
            assert banned not in props

    def test_wrapper_registers_retrievable_ocr_handle(self):
        """End-to-end (mocked analyze_image): long OCR result through the
        registry wrapper must mint a NON-EMPTY private handle, persist the
        full text, and be retrievable via vision_ocr_page (zero calls)."""
        import asyncio
        from tools.registry import registry
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://uat-test/att", "/tmp/ocr-e2e.png")
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        long_text = "行" * 9000

        async def fake_ai(*a, **k):
            return {"execution_status": "SUCCESS",
                    "quality_decision": "PASS",
                    "initial_model_slot": "OCR",
                    "final_model_slot": "OCR",
                    "actual_model": "glm-ocr",
                    "logical_model_calls": 1,
                    "structured": {"observed_text": [long_text]},
                    "trace": []}

        import tools.vision_router_tool as vrt
        with patch("hermes_cli.config.load_config",
                   return_value=self._cfg()), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = asyncio.run(handler(
                {"source_handle": "attachment://uat-test/att",
                 "task": "EXACT_OCR"}))
        env = json.loads(raw)
        ocr_meta = env.get("ocr_meta") or {}
        handle = ocr_meta.get("private_handle")
        assert handle, "private_handle must be non-empty"
        assert ocr_meta.get("truncated") is True
        assert ocr_meta.get("returned_chars", 0) <= 4000
        # full text persisted + registered
        assert vision_session_state.resolve_ocr_result(f"ocr://{handle}")
        # retrievable through vision_ocr_page (zero calls)
        from tools.vision_ocr_page import _handle_vision_ocr_page
        page = json.loads(_handle_vision_ocr_page(
            {"handle": f"ocr://{handle}", "page": 0}))
        assert page["execution_status"] == "SUCCESS"
        assert page["page_chars"] > 0
        assert page["total_chars"] == 9000
        assert page["logical_model_calls"] == 0
        vision_session_state.set_enabled(False)

    @staticmethod
    def _cfg():
        import copy
        from hermes_cli.config import load_config
        cfg = copy.deepcopy(load_config())
        cfg["auxiliary"]["vision_router"]["enabled"] = True
        return cfg



class TestConfigPathAlignment:
    """Config-path alignment repair (HERMES_VISION_ROUTER_FLAG_CONFIG_PATH_
    ALIGNMENT_V0_1): canonical path config["auxiliary"]["vision_router"],
    legacy top-level fallback only when nested is absent, nested wins on
    conflict, malformed values fail closed. Real-config shape tests included.
    """

    pytestmark = pytest.mark.asyncio

    @staticmethod
    def _real_shape(**vr_overrides):
        """Config dict in the exact shape produced by the real loader."""
        import copy
        from hermes_cli.config import load_config
        cfg = copy.deepcopy(load_config())
        for k, v in vr_overrides.items():
            cfg["auxiliary"]["vision_router"][k] = v
        return cfg

    # -- resolver unit behavior ---------------------------------------------
    def test_nested_false_resolves_false(self):
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = self._real_shape(enabled=False)
        assert resolve_vision_router_enabled(cfg) is False

    def test_nested_true_resolves_true(self):
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = self._real_shape(enabled=True)
        assert resolve_vision_router_enabled(cfg) is True

    def test_legacy_top_level_only_accepted(self):
        # LEGACY COMPATIBILITY: accepted only when the nested mapping is absent.
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = {"vision_router": {"enabled": True}}
        assert resolve_vision_router_enabled(cfg) is True

    def test_nested_true_overrides_legacy_false(self):
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = {"auxiliary": {"vision_router": {"enabled": True}},
               "vision_router": {"enabled": False}}
        assert resolve_vision_router_enabled(cfg) is True

    def test_nested_false_overrides_legacy_true(self):
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = {"auxiliary": {"vision_router": {"enabled": False}},
               "vision_router": {"enabled": True}}
        assert resolve_vision_router_enabled(cfg) is False

    def test_malformed_nested_mapping_fails_closed(self):
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = {"auxiliary": {"vision_router": "not-a-mapping"}}
        assert resolve_vision_router_enabled(cfg) is False

    def test_malformed_enabled_fails_closed(self):
        from tools.vision_policy import resolve_vision_router_enabled
        cfg = {"auxiliary": {"vision_router": {"enabled": "yes"}}}
        assert resolve_vision_router_enabled(cfg) is False

    def test_missing_config_fails_closed(self):
        from tools.vision_policy import resolve_vision_router_enabled
        assert resolve_vision_router_enabled(None) is False
        assert resolve_vision_router_enabled({}) is False

    # -- visibility + orchestrator share the effective flag -----------------
    def test_tool_visibility_nested_true(self):
        from toolsets import vision_router_tool_visible
        assert vision_router_tool_visible(self._real_shape(enabled=True)) is True

    def test_orchestrator_gate_nested_true(self):
        from tools.vision_orchestrator import vision_router_enabled
        assert vision_router_enabled(self._real_shape(enabled=True)) is True

    def test_visibility_and_orchestrator_same_value(self):
        from toolsets import vision_router_tool_visible
        from tools.vision_orchestrator import vision_router_enabled
        for enabled in (True, False):
            cfg = self._real_shape(enabled=enabled)
            assert vision_router_tool_visible(cfg) == vision_router_enabled(cfg) == enabled

    # -- cache transitions through the REAL nested structure -----------------
    def _defs_nested(self, enabled):
        from model_tools import get_tool_definitions
        # The official legacy vision_analyze check_fn resolves the real
        # auxiliary.vision client; the sandboxed test HERMES_HOME has none,
        # so force the resolver to return a mock client. This keeps the
        # kill-switch visibility semantics (not the availability probe) as
        # the behavior under test. The registry TTL-caches check_fn results,
        # so the cache must be invalidated between calls.
        from tools.registry import invalidate_check_fn_cache

        invalidate_check_fn_cache()
        with patch("agent.auxiliary_client.resolve_vision_provider_client",
                   return_value=("mock", object(), "mock-model")), \
             patch("hermes_cli.config.load_config",
                   return_value=self._real_shape(enabled=enabled)):
            return [d["function"]["name"]
                    for d in get_tool_definitions(
                        enabled_toolsets=["vision"], quiet_mode=True)]

    def test_false_to_true_cache_transition_nested(self):
        names_false = self._defs_nested(False)
        assert "vision_router_analyze" not in names_false
        # Public compatibility: official legacy vision_analyze stays visible
        # while the local Router is off.
        assert "vision_analyze" in names_false
        names_true = self._defs_nested(True)
        assert "vision_router_analyze" in names_true
        assert "vision_analyze" not in names_true

    def test_true_to_false_cache_transition_nested(self):
        names_true = self._defs_nested(True)
        assert "vision_router_analyze" in names_true
        names_false = self._defs_nested(False)
        assert "vision_router_analyze" not in names_false
        # Public compatibility: legacy vision_analyze restored when off.
        assert "vision_analyze" in names_false

    # -- analyze_image(enabled=None) policy gate with real nested flag -------
    @staticmethod
    def _vr_request():
        from tools.vision_policy import (
            VisionCriticality, VisionMode, VisionRequest, VisionTask,
        )
        return VisionRequest(
            request_id="vr-req-1",
            task=VisionTask.UI_READ,
            mode=VisionMode.AUTO,
            criticality=VisionCriticality.NORMAL,
            image_source="opaque://test-image",
            question="read visible text",
        )

    async def test_analyze_image_gate_allowed_nested_true(self):
        from tools.vision_orchestrator import analyze_image

        calls = {"native": 0, "prepare": 0, "openai": 0}

        async def fake_prepare(*a, **k):
            calls["prepare"] += 1
            return ("data:image/png;base64,QUJD", 10, 10, "image/png",
                    "sha", {"transport_image_sha256": "sha", "transport_mime_type": "image/png"})

        async def fake_native(*a, **k):
            calls["native"] += 1
            return {
                "execution_status": "SUCCESS",
                "response": '{"observed_text": ["测试可见文本"]}',
                "content_source": "response",
                "thinking_fallback_used": False,
                "response_character_count": 40,
                "thinking_character_count": 0,
                "done_reason": "stop",
                "total_duration_ms": 5000,
                "load_duration_ms": 100,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }

        async def fake_openai(*a, **k):
            calls["openai"] += 1
            return {
                "execution_status": "SUCCESS",
                "content": '{"observed_text": ["测试可见文本"]}',
                "done_reason": "stop",
                "total_duration_ms": 5000,
            }

        with patch("hermes_cli.config.load_config",
                   return_value=self._real_shape(enabled=True)), \
             patch("tools.vision_orchestrator.prepare_image", new=fake_prepare), \
             patch("tools.vision_orchestrator.invoke_native_generate", new=fake_native), \
             patch("tools.vision_orchestrator.invoke_vision_model", new=fake_openai):
            result = await analyze_image(self._vr_request(), enabled=None)
        assert result["execution_status"] != "POLICY_BLOCKED"
        # the Router gate passed and exactly one provider path was executed
        assert calls["native"] + calls["openai"] == 1
        assert calls["prepare"] == 1

    async def test_analyze_image_gate_blocked_nested_false(self):
        from tools.vision_orchestrator import analyze_image

        calls = {"native": 0, "prepare": 0}

        async def fake_prepare(*a, **k):
            calls["prepare"] += 1
            raise AssertionError("must not prepare when router=false")

        async def fake_native(*a, **k):
            calls["native"] += 1
            raise AssertionError("must not invoke provider when router=false")

        with patch("hermes_cli.config.load_config",
                   return_value=self._real_shape(enabled=False)), \
             patch("tools.vision_orchestrator.prepare_image", new=fake_prepare), \
             patch("tools.vision_orchestrator.invoke_native_generate", new=fake_native):
            result = await analyze_image(self._vr_request(), enabled=None)
        assert result["execution_status"] == "POLICY_BLOCKED"
        assert calls["native"] == 0
        assert calls["prepare"] == 0


class TestOllamaEndpointAlignment:
    """Trusted Ollama endpoint wiring (HERMES_VISION_ROUTER_NATIVE_BASE_URL_
    ALIGNMENT_V0_1): one shared resolver; wrapper passes the resolved native
    root to analyze_image; base_url stays model-invisible.
    """
    def setup_method(self):
        from tools.vision_session_state import vision_session_state
        vision_session_state.set_enabled(True)
        vision_session_state.begin_turn()
        vision_session_state.register_attachment(
            "attachment://sess/taobao-0003", "/tmp/synthetic-a.png")
        vision_session_state.register_attachment(
            "attachment://sess/taobao-0004", "/tmp/synthetic-b.png")


    pytestmark = pytest.mark.asyncio

    @staticmethod
    def _cfg(base_url="http://ollama.internal:11434/v1", enabled=True):
        import copy
        from hermes_cli.config import load_config
        cfg = copy.deepcopy(load_config())
        cfg["auxiliary"]["vision_router"]["enabled"] = enabled
        cfg["auxiliary"]["vision"]["base_url"] = base_url
        return cfg

    # -- resolver rules ------------------------------------------------------
    def test_real_loaded_config_resolves_trusted_url(self):
        # real loaded-config SHAPE (load_config deepcopy) with an explicit
        # trusted endpoint resolves to the native root. Note: under pytest the
        # loader is sandboxed to a default config (HERMES_HOME isolation), so
        # the endpoint value is supplied explicitly while the shape is real.
        from tools.vision_policy import resolve_ollama_base_url
        b = resolve_ollama_base_url(self._cfg())
        assert b == "http://ollama.internal:11434"

    def test_default_isolated_config_fails_closed(self):
        # under pytest isolation the loaded default config has no endpoint;
        # fail-closed behavior must hold (no crash, None).
        from hermes_cli.config import load_config
        from tools.vision_policy import resolve_ollama_base_url
        b = resolve_ollama_base_url(load_config())
        assert b is None or b.startswith("http")

    def test_ollama_host_fallback_when_config_absent(self):
        from tools.vision_policy import resolve_ollama_base_url
        cfg = {"auxiliary": {"vision": {"base_url": ""}}}
        with patch.dict(os.environ, {"OLLAMA_HOST": "ollama.local:11434"}, clear=False):
            assert resolve_ollama_base_url(cfg) == "http://ollama.local:11434"

    def test_explicit_config_wins_over_environment(self):
        from tools.vision_policy import resolve_ollama_base_url
        cfg = {"auxiliary": {"vision": {"base_url": "http://cfg.host:11434/v1"}}}
        with patch.dict(os.environ, {"OLLAMA_HOST": "env.host:11434"}):
            assert resolve_ollama_base_url(cfg) == "http://cfg.host:11434"

    def test_empty_value_fails_closed(self):
        from tools.vision_policy import resolve_ollama_base_url
        assert resolve_ollama_base_url({"auxiliary": {"vision": {"base_url": "  "}}}) is None
        assert resolve_ollama_base_url({}) is None
        assert resolve_ollama_base_url(None) is None

    def test_unsupported_scheme_fails_closed(self):
        from tools.vision_policy import resolve_ollama_base_url
        for bad in ("ftp://ollama:11434", "file:///etc/hosts", "ws://ollama:11434"):
            assert resolve_ollama_base_url(
                {"auxiliary": {"vision": {"base_url": bad}}}) is None

    def test_file_path_fails_closed(self):
        from tools.vision_policy import resolve_ollama_base_url
        assert resolve_ollama_base_url(
            {"auxiliary": {"vision": {"base_url": "/var/run/ollama.sock"}}}) is None

    def test_embedded_credentials_fails_closed(self):
        from tools.vision_policy import resolve_ollama_base_url
        assert resolve_ollama_base_url(
            {"auxiliary": {"vision": {"base_url": "http://user:pass@ollama:11434"}}}) is None

    def test_trailing_slash_and_v1_normalization(self):
        from tools.vision_policy import resolve_ollama_base_url
        for raw, want in (
            ("http://ollama.internal:11434/v1", "http://ollama.internal:11434"),
            ("http://ollama.internal:11434/", "http://ollama.internal:11434"),
            ("http://ollama.internal:11434", "http://ollama.internal:11434"),
            ("ollama.internal:11434", "http://ollama.internal:11434"),
            ("https://ollama.internal", "https://ollama.internal:443"),
        ):
            assert resolve_ollama_base_url(
                {"auxiliary": {"vision": {"base_url": raw}}}) == want

    # -- wrapper wiring ------------------------------------------------------
    def test_wrapper_schema_has_no_base_url(self):
        props = vrt.VISION_ROUTER_SCHEMA["parameters"]["properties"]
        for banned in ("base_url", "endpoint", "host", "port", "transport"):
            assert banned not in props

    async def test_registry_wrapper_passes_trusted_base_url(self):
        # integration-style: registry handler → real wrapper → mocked
        # analyze_image; assert the exact trusted base_url received.
        from tools.registry import registry
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        cfg = self._cfg()
        received = {}

        async def fake_ai(request, **kw):
            received.update(kw)
            return {
                "request_id": "vr-ok",
                "task": "UI_READ",
                "execution_status": "SUCCESS",
                "quality_decision": "PASS",
                "initial_model_slot": "PRECISION_VLM",
                "final_model_slot": "PRECISION_VLM",
                "actual_model": "qwen3.6:27b",
                "recommended_next_slot": None,
                "human_review_required": False,
                "logical_model_calls": 1,
                "structured": {"observed_text": ["ok"]},
                "trace": [{"total_ms": 1, "done_reason": "stop"}],
            }

        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = await handler({"source_handle": "attachment://sess/taobao-0003",
                                 "task": "UI_READ", "mode": "AUTO"})
        env = json.loads(raw)
        assert received.get("base_url") == "http://ollama.internal:11434"
        # enabled carries the in-memory session flag (limited-use session
        # mode user toggle); model-visible flag resolution stays server-side.
        assert received.get("enabled") is True
        assert env["execution_status"] == "SUCCESS"
        assert "ollama.internal" not in json.dumps(env)  # endpoint not leaked

    async def test_wrapper_ignores_model_supplied_endpoint(self):
        from tools.registry import registry
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        cfg = self._cfg()
        received = {}

        async def fake_ai(request, **kw):
            received.update(kw)
            return {"execution_status": "SUCCESS", "quality_decision": "PASS",
                    "initial_model_slot": "PRECISION_VLM",
                    "final_model_slot": "PRECISION_VLM",
                    "actual_model": "qwen3.6:27b",
                    "logical_model_calls": 1,
                    "structured": {"observed_text": ["ok"]},
                    "trace": []}

        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = await handler({"source_handle": "attachment://sess/taobao-0003",
                                 "task": "UI_READ",
                                 "base_url": "http://evil.example:9999"})
        env = json.loads(raw)
        # model-supplied endpoint is not accepted; trusted config value wins
        assert received.get("base_url") == "http://ollama.internal:11434"
        assert env["execution_status"] == "SUCCESS"

    async def test_authorized_path_reaches_mocked_native_without_value_error(self):
        # the prior canary blocker: OLLAMA_NATIVE_GENERATE requires base_url.
        # With wiring in place, a mocked native transport is reached.
        from tools.registry import registry
        handler = registry.get_entry("vision_router_analyze").handler
        assert handler is not None
        cfg = self._cfg()
        reached = {"native": False}

        async def fake_ai(request, **kw):
            # analyze_image native branch would raise without base_url; here
            # the wrapper already supplies it, so execution may proceed.
            assert kw.get("base_url"), "base_url must be wired"
            reached["native"] = True
            return {"execution_status": "SUCCESS", "quality_decision": "PASS",
                    "initial_model_slot": "PRECISION_VLM",
                    "final_model_slot": "PRECISION_VLM",
                    "actual_model": "qwen3.6:27b",
                    "logical_model_calls": 1,
                    "structured": {"observed_text": ["ok"]},
                    "trace": []}

        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("tools.vision_orchestrator.analyze_image", new=fake_ai):
            raw = await handler({"source_handle": "attachment://sess/taobao-0003",
                                 "task": "UI_READ"})
        env = json.loads(raw)
        assert reached["native"] is True
        assert env["execution_status"] == "SUCCESS"
