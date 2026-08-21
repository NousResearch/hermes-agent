"""Safe, opt-in per-task model/reasoning selection for delegate_task."""

from unittest.mock import patch

import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _SELECTION_UNSET,
    _build_dynamic_schema_overrides,
    _resolve_task_execution_overrides,
    _strip_model_hidden_task_fields,
)


BASE_CREDS = {
    "model": "gpt-5.6-luna",
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_key": "",
    "api_mode": "codex_responses",
    "command": None,
    "args": None,
}


class TestSafeSelectionSchema:
    def test_feature_is_disabled_by_default(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG["delegation"]
        assert cfg["allow_model_selection"] is False
        assert cfg["allowed_models"] == []
        assert cfg["allowed_reasoning_efforts"] == []

    def test_fields_are_hidden_when_selection_is_disabled(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            params = _build_dynamic_schema_overrides()["parameters"]

        assert "model" not in params["properties"]
        assert "reasoning_effort" not in params["properties"]
        task_props = params["properties"]["tasks"]["items"]["properties"]
        assert "model" not in task_props
        assert "reasoning_effort" not in task_props

    def test_fields_use_operator_allowlists_when_enabled(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["medium", "high", "xhigh"],
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            params = _build_dynamic_schema_overrides()["parameters"]

        assert params["properties"]["model"]["enum"] == cfg["allowed_models"]
        assert params["properties"]["reasoning_effort"]["enum"] == cfg["allowed_reasoning_efforts"]
        task_props = params["properties"]["tasks"]["items"]["properties"]
        assert task_props["model"]["enum"] == cfg["allowed_models"]
        assert task_props["reasoning_effort"]["enum"] == cfg["allowed_reasoning_efforts"]

    def test_malformed_allowlist_config_exposes_no_fields(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": "gpt-5.6-sol",
            "allowed_reasoning_efforts": {"high": True},
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            params = _build_dynamic_schema_overrides()["parameters"]

        assert "model" not in params["properties"]
        assert "reasoning_effort" not in params["properties"]
        task_props = params["properties"]["tasks"]["items"]["properties"]
        assert "model" not in task_props
        assert "reasoning_effort" not in task_props

    def test_static_schema_is_never_mutated(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna"],
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            _build_dynamic_schema_overrides()

        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "model" not in props
        assert "reasoning_effort" not in props
        assert "model" not in props["tasks"]["items"]["properties"]

    def test_empty_allowlist_does_not_expose_an_empty_enum(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna"],
            "allowed_reasoning_efforts": [],
        }
        with patch("tools.delegate_tool._load_config", return_value=cfg):
            params = _build_dynamic_schema_overrides()["parameters"]

        assert params["properties"]["model"]["enum"] == ["gpt-5.6-luna"]
        assert "reasoning_effort" not in params["properties"]
        task_props = params["properties"]["tasks"]["items"]["properties"]
        assert task_props["model"]["enum"] == ["gpt-5.6-luna"]
        assert "reasoning_effort" not in task_props


class TestModelFacingSanitizer:
    def test_keeps_approved_compute_fields_and_strips_transport_fields(self):
        tasks = [
            {
                "goal": "review",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "acp_command": "forbidden",
                "acp_args": ["--forbidden"],
            }
        ]

        stripped = _strip_model_hidden_task_fields(tasks)

        assert stripped == [
            {
                "goal": "review",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
            }
        ]


class TestSafeSelectionValidation:
    def test_allowed_model_only_changes_model_not_transport(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna", "gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["high", "xhigh"],
        }
        creds, reasoning = _resolve_task_execution_overrides(
            "gpt-5.6-sol", "high", cfg, BASE_CREDS
        )

        assert creds["model"] == "gpt-5.6-sol"
        for key in ("provider", "base_url", "api_key", "api_mode", "command", "args"):
            assert creds[key] == BASE_CREDS[key]
        assert reasoning == {"enabled": True, "effort": "high"}
        assert BASE_CREDS["model"] == "gpt-5.6-luna"

    def test_allowlisted_model_pins_inherited_parent_api_mode(self):
        inherited = dict(BASE_CREDS, api_mode=None)
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["anthropic/claude-sonnet-4.6"],
        }

        creds, _ = _resolve_task_execution_overrides(
            "anthropic/claude-sonnet-4.6",
            _SELECTION_UNSET,
            cfg,
            inherited,
            parent_api_mode="chat_completions",
        )

        assert creds["api_mode"] == "chat_completions"
        assert inherited["api_mode"] is None

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            (" gpt-5.6-sol", "high"),
            ("gpt-5.6-sol ", "high"),
            ("gpt-5.6-sol", "HIGH"),
        ],
    )
    def test_allowlists_use_exact_string_membership(self, model, effort):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["high"],
        }
        with pytest.raises(ValueError, match="not allowed"):
            _resolve_task_execution_overrides(model, effort, cfg, BASE_CREDS)

    @pytest.mark.parametrize("model", ["gpt-5.6-sol-evil", "openrouter/other", ""])
    def test_unlisted_or_empty_explicit_model_is_rejected(self, model):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna", "gpt-5.6-sol"],
        }
        if model == "":
            creds, reasoning = _resolve_task_execution_overrides(
                _SELECTION_UNSET, _SELECTION_UNSET, cfg, BASE_CREDS
            )
            assert creds is BASE_CREDS
            assert reasoning is None
        else:
            with pytest.raises(ValueError, match="not allowed"):
                _resolve_task_execution_overrides(model, _SELECTION_UNSET, cfg, BASE_CREDS)

    def test_selection_is_rejected_when_operator_flag_is_off(self):
        with pytest.raises(ValueError, match="disabled"):
            _resolve_task_execution_overrides("gpt-5.6-sol", _SELECTION_UNSET, {}, BASE_CREDS)

    @pytest.mark.parametrize("value", [None, "", "   ", False, 0, [], {}])
    def test_explicit_malformed_model_is_rejected(self, value):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-sol"],
        }
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            _resolve_task_execution_overrides(value, None, cfg, BASE_CREDS)

    @pytest.mark.parametrize("value", [None, "", "   ", False, 0, [], {}])
    def test_explicit_malformed_reasoning_effort_is_rejected(self, value):
        cfg = {
            "allow_model_selection": True,
            "allowed_reasoning_efforts": ["high"],
        }
        with pytest.raises(
            ValueError, match="reasoning effort must be a non-empty string"
        ):
            _resolve_task_execution_overrides(_SELECTION_UNSET, value, cfg, BASE_CREDS)

    def test_unlisted_reasoning_effort_is_rejected(self):
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["xhigh"],
        }
        with pytest.raises(ValueError, match="reasoning effort.*not allowed"):
            _resolve_task_execution_overrides("gpt-5.6-sol", "ultra", cfg, BASE_CREDS)


class TestPerTaskSelectionIntegration:
    def test_single_goal_builds_requested_child_compute(self):
        from unittest.mock import MagicMock

        from tools.delegate_tool import delegate_task

        parent = MagicMock()
        parent._delegate_depth = 0
        parent._interrupt_requested = False
        parent._active_children = []
        parent._active_children_lock = None
        parent.session_id = "parent-test"
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna", "gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["high", "xhigh"],
            "max_concurrent_children": 2,
            "max_iterations": 50,
        }
        captured = {}

        def capture_child(**kwargs):
            captured.update(kwargs)
            child = MagicMock()
            child._delegate_role = "leaf"
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value=BASE_CREDS,
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=capture_child,
            ),
            patch(
                "tools.delegate_tool._run_single_child",
                return_value={
                    "task_index": 0,
                    "status": "success",
                    "summary": "ok",
                    "api_calls": 0,
                    "duration_seconds": 0,
                },
            ),
        ):
            delegate_task(
                goal="Review the release blocker",
                context="raw\n- multiline\ncontext",
                model="gpt-5.6-sol",
                reasoning_effort="high",
                background=False,
                parent_agent=parent,
            )

        assert captured["model"] == "gpt-5.6-sol"
        assert captured["context"] == "raw\n- multiline\ncontext"
        assert captured["override_reasoning_config"] == {
            "enabled": True,
            "effort": "high",
        }

    def test_omitted_fields_preserve_delegation_defaults(self):
        creds, reasoning = _resolve_task_execution_overrides(
            _SELECTION_UNSET,
            _SELECTION_UNSET,
            {"allow_model_selection": True},
            BASE_CREDS,
        )

        assert creds is BASE_CREDS
        assert creds["model"] == "gpt-5.6-luna"
        assert reasoning is None

    def test_parallel_tasks_build_children_with_independent_compute(self):
        from unittest.mock import MagicMock

        from tools.delegate_tool import delegate_task

        parent = MagicMock()
        parent._delegate_depth = 0
        parent._interrupt_requested = False
        parent._active_children = []
        parent._active_children_lock = None
        parent.session_id = "parent-test"
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna", "gpt-5.6-sol"],
            "allowed_reasoning_efforts": ["high", "xhigh"],
            "max_concurrent_children": 2,
            "max_iterations": 50,
        }
        built = []

        def capture_child(**kwargs):
            child = MagicMock()
            child.model = kwargs["model"]
            child.reasoning_config = kwargs["override_reasoning_config"]
            child._delegate_role = "leaf"
            built.append(kwargs)
            return child

        tasks = [
            {"goal": "Review module alpha carefully", "model": "gpt-5.6-luna", "reasoning_effort": "xhigh"},
            {"goal": "Review module beta carefully", "model": "gpt-5.6-sol", "reasoning_effort": "high"},
        ]
        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch("tools.delegate_tool._resolve_delegation_credentials", return_value=BASE_CREDS),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=capture_child),
            patch("tools.delegate_tool.create_live_transcripts", create=True, return_value=("", [], [])),
            patch(
                "tools.delegate_tool._run_single_child",
                side_effect=lambda task_index, *args, **kwargs: {
                    "task_index": task_index,
                    "status": "success",
                    "summary": "ok",
                    "api_calls": 0,
                    "duration_seconds": 0,
                },
            ),
        ):
            # Force synchronous execution so this test can inspect both builds.
            delegate_task(tasks=tasks, background=False, parent_agent=parent)

        assert [x["model"] for x in built] == ["gpt-5.6-luna", "gpt-5.6-sol"]
        assert [x["override_reasoning_config"] for x in built] == [
            {"enabled": True, "effort": "xhigh"},
            {"enabled": True, "effort": "high"},
        ]
        assert all(x["override_provider"] == "openai-codex" for x in built)
        assert all(x["override_base_url"] == BASE_CREDS["base_url"] for x in built)

    def test_invalid_later_task_fails_before_any_child_is_built(self):
        from unittest.mock import MagicMock

        from tools.delegate_tool import delegate_task

        parent = MagicMock()
        parent._delegate_depth = 0
        cfg = {
            "allow_model_selection": True,
            "allowed_models": ["gpt-5.6-luna"],
            "max_concurrent_children": 2,
            "max_iterations": 50,
        }
        tasks = [
            {"goal": "Review module alpha carefully", "model": "gpt-5.6-luna"},
            {"goal": "Review module beta carefully", "model": "not-allowed"},
        ]
        with (
            patch("tools.delegate_tool._load_config", return_value=cfg),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value=BASE_CREDS,
            ),
            patch("tools.delegate_tool._build_child_preserving_parent_tools") as build,
        ):
            result = delegate_task(tasks=tasks, background=False, parent_agent=parent)

        assert "not allowed" in result
        build.assert_not_called()
