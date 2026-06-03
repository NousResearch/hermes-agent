"""Tests for delegate_task model_hint parameter (OMO P0 borrowing, 2026-06-03).

Source: multi-agent-harness skill 第十一节路线图, 借鉴 OMO 工具 (lazyclaudecode)
的 model 分层模式：opus 决策 / haiku 侦察。
"""
import inspect
import sys
from unittest.mock import patch, MagicMock

import pytest

from tools.delegate_tool import (
    _resolve_model_hint,
    _is_full_model_id,
    _MODEL_HINT_SHORT_NAMES,
    delegate_task,
)


# ---------------------------------------------------------------------------
# _is_full_model_id
# ---------------------------------------------------------------------------

class TestIsFullModelId:
    def test_with_slash_provider_prefix(self):
        """带 provider 前缀的 id（"openai/gpt-4o"）算完整 id"""
        assert _is_full_model_id("openai/gpt-4o") is True
        assert _is_full_model_id("anthropic/claude-opus-4") is True

    def test_with_version_date(self):
        """带版本日期的 id 算完整 id"""
        assert _is_full_model_id("claude-3-5-sonnet-20241022") is True
        assert _is_full_model_id("gpt-4o-2024-08-06") is True

    def test_with_dot_version(self):
        """含点号的 id（"claude-3.5-sonnet"）算完整 id"""
        assert _is_full_model_id("claude-3.5-sonnet") is True

    def test_short_names_not_full(self):
        """短名（"haiku" / "opus"）不算完整 id"""
        for name in _MODEL_HINT_SHORT_NAMES.keys():
            assert _is_full_model_id(name) is False, f"{name} should be a short name"

    def test_empty_string(self):
        assert _is_full_model_id("") is False

    def test_user_main_model_id(self):
        """用户主 model: minimax-m2.7-highspeed 算完整 id"""
        assert _is_full_model_id("minimax-m2.7-highspeed") is True


# ---------------------------------------------------------------------------
# _resolve_model_hint — 行为矩阵
# ---------------------------------------------------------------------------

class TestResolveModelHint:
    def test_none_returns_none(self):
        """hint=None → 继承 parent model"""
        assert _resolve_model_hint(None) is None
        assert _resolve_model_hint(None, parent_model="claude-opus-4") is None

    def test_empty_string_returns_none(self):
        assert _resolve_model_hint("") is None
        assert _resolve_model_hint("   ") is None

    def test_full_id_passes_through(self):
        """完整 model id 原样返回"""
        assert _resolve_model_hint("minimax-m2.7-highspeed") == "minimax-m2.7-highspeed"
        assert _resolve_model_hint("openai/gpt-4o") == "openai/gpt-4o"
        assert _resolve_model_hint("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5-20250929"

    def test_short_name_haiku(self):
        """short name 'haiku' 返回原样，让 provider 解析到该 family 最新版"""
        assert _resolve_model_hint("haiku") == "haiku"
        assert _resolve_model_hint("haiku", parent_model="claude-sonnet-4-5") == "haiku"

    def test_short_name_opus(self):
        """short name 'opus' 同上"""
        assert _resolve_model_hint("opus") == "opus"
        assert _resolve_model_hint("opus", parent_model="claude-sonnet-4-5") == "opus"

    def test_short_name_case_insensitive(self):
        """short name 大小写不敏感"""
        assert _resolve_model_hint("HAIKU") == "HAIKU"
        assert _resolve_model_hint("Opus") == "Opus"

    def test_unknown_short_name_passes_through(self):
        """未注册的 hint 原样返回（让 provider 报 'unknown model'）"""
        assert _resolve_model_hint("xyz-unknown") == "xyz-unknown"
        assert _resolve_model_hint("totally-made-up-model") == "totally-made-up-model"

    def test_strips_whitespace(self):
        assert _resolve_model_hint("  haiku  ") == "haiku"


# ---------------------------------------------------------------------------
# delegate_task 签名：model_hint 参数
# ---------------------------------------------------------------------------

class TestDelegateTaskSignature:
    def test_has_model_hint_parameter(self):
        """delegate_task 必须有 model_hint 参数"""
        sig = inspect.signature(delegate_task)
        assert "model_hint" in sig.parameters
        assert sig.parameters["model_hint"].default is None

    def test_docstring_mentions_model_hint(self):
        """docstring 提到 model_hint 和 OMO 借鉴出处"""
        doc = delegate_task.__doc__ or ""
        assert "model_hint" in doc
        assert "OMO" in doc

    def test_position_after_role(self):
        """model_hint 在 role 之后、parent_agent 之前"""
        params = list(inspect.signature(delegate_task).parameters.keys())
        assert params.index("model_hint") > params.index("role")
        assert params.index("model_hint") < params.index("parent_agent")


# ---------------------------------------------------------------------------
# delegate_task 主体逻辑：model_hint 实际生效
# ---------------------------------------------------------------------------

class TestDelegateTaskModelResolution:
    """Mock 掉 _build_child_agent，验证 model_hint 实际下传到了子 agent。"""

    def test_top_level_model_hint_overrides_creds_model(self):
        """top-level model_hint 会覆盖 _resolve_delegation_credentials 返回的 model"""
        captured = []

        def fake_build_child_agent(**kwargs):
            captured.append(kwargs)
            mock_child = MagicMock()
            mock_child._delegate_saved_tool_names = []
            return mock_child

        with patch("tools.delegate_tool._build_child_agent", side_effect=fake_build_child_agent), \
             patch("tools.delegate_tool._resolve_delegation_credentials",
                   return_value={"model": "claude-opus-4", "provider": None,
                                 "base_url": None, "api_key": None, "api_mode": None,
                                 "command": None, "args": None}), \
             patch("tools.delegate_tool._load_config", return_value={}), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=3), \
             patch("tools.delegate_tool._run_single_child",
                   return_value={"task_index": 0, "status": "completed",
                                 "summary": "ok", "api_calls": 0, "duration_seconds": 0.1}), \
             patch("tools.delegate_tool._normalize_role", return_value="leaf"):

            parent = MagicMock()
            parent.model = "claude-opus-4"
            parent._delegate_depth = 0
            parent._delegate_spinner = None
            parent.tool_progress_callback = None

            import json
            result = json.loads(delegate_task(
                goal="test",
                model_hint="haiku",
                parent_agent=parent,
            ))

        # 子 agent 收到 model=haiku（而不是 creds 默认的 opus）
        assert len(captured) == 1
        assert captured[0]["model"] == "haiku"

    def test_per_task_model_hint_beats_top_level(self):
        """per-task model_hint 覆盖 top-level model_hint"""
        captured = []

        def fake_build_child_agent(**kwargs):
            captured.append(kwargs)
            mock_child = MagicMock()
            mock_child._delegate_saved_tool_names = []
            return mock_child

        with patch("tools.delegate_tool._build_child_agent", side_effect=fake_build_child_agent), \
             patch("tools.delegate_tool._resolve_delegation_credentials",
                   return_value={"model": "claude-opus-4", "provider": None,
                                 "base_url": None, "api_key": None, "api_mode": None,
                                 "command": None, "args": None}), \
             patch("tools.delegate_tool._load_config", return_value={}), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=3), \
             patch("tools.delegate_tool._run_single_child",
                   return_value={"task_index": 0, "status": "completed",
                                 "summary": "ok", "api_calls": 0, "duration_seconds": 0.1}), \
             patch("tools.delegate_tool._normalize_role", return_value="leaf"):

            parent = MagicMock()
            parent.model = "claude-opus-4"
            parent._delegate_depth = 0
            parent._delegate_spinner = None
            parent.tool_progress_callback = None

            import json
            # top-level 用 haiku, per-task 用 opus → per-task 赢
            delegate_task(
                goal="test",
                model_hint="haiku",
                tasks=[
                    {"goal": "task A", "model_hint": "opus"},
                ],
                parent_agent=parent,
            )

        # per-task 覆盖了 top-level
        assert len(captured) == 1
        assert captured[0]["model"] == "opus"

    def test_no_model_hint_uses_creds_default(self):
        """不传 model_hint → 用 creds 的 model（继承 parent 或 delegation 配置）"""
        captured = []

        def fake_build_child_agent(**kwargs):
            captured.append(kwargs)
            mock_child = MagicMock()
            mock_child._delegate_saved_tool_names = []
            return mock_child

        with patch("tools.delegate_tool._build_child_agent", side_effect=fake_build_child_agent), \
             patch("tools.delegate_tool._resolve_delegation_credentials",
                   return_value={"model": "claude-sonnet-4-5", "provider": None,
                                 "base_url": None, "api_key": None, "api_mode": None,
                                 "command": None, "args": None}), \
             patch("tools.delegate_tool._load_config", return_value={}), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=3), \
             patch("tools.delegate_tool._run_single_child",
                   return_value={"task_index": 0, "status": "completed",
                                 "summary": "ok", "api_calls": 0, "duration_seconds": 0.1}), \
             patch("tools.delegate_tool._normalize_role", return_value="leaf"):

            parent = MagicMock()
            parent.model = "claude-sonnet-4-5"
            parent._delegate_depth = 0
            parent._delegate_spinner = None
            parent.tool_progress_callback = None

            delegate_task(goal="test", parent_agent=parent)

        # 不传 model_hint → 用 creds['model']
        assert len(captured) == 1
        assert captured[0]["model"] == "claude-sonnet-4-5"
