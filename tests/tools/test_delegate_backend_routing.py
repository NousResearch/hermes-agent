"""Tests for per-task backend routing via delegation.backends.

Verifies that child subagents can target named backends from config.yaml,
enabling automatic task distribution across multiple Ollama instances or
providers (issue #344).

Coverage:
- _get_backends() loads / validates / rejects backend specs
- _resolve_backend() returns overrides for known names, None for unknown/empty
- _build_children() merges per-task backend credentials into child agent
- Task validation rejects non-string backend values
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure project root on path (matches tests/conftest.py)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.delegate_tool_config import _resolve_backend, _get_backends
from tools.delegate_tool_tasks import _normalize_task_list


# ── Helpers ────────────────────────────────────────────────────────────────

def _config_with_backends(backends: dict) -> dict:
    """Return a minimal delegation config carrying the given backends map."""
    return {
        "backends": backends,
    }


# ── _get_backends ─────────────────────────────────────────────────────────


class TestGetBackends:
    def test_returns_empty_when_no_backends_key(self):
        with patch("tools.delegate_tool_config._load_config", return_value={}):
            assert _get_backends() == {}

    def test_returns_empty_when_backends_not_dict(self):
        with patch("tools.delegate_tool_config._load_config", return_value={"backends": "oops"}):
            assert _get_backends() == {}

    def test_loads_valid_backend_specs(self):
        cfg = _config_with_backends({
            "machine_a": {"base_url": "http://10.0.0.1:8443/v1", "model": "gemma4:27b"},
            "machine_b": {"base_url": "http://10.0.0.2:8443/v1", "model": "qwen3:32b"},
        })
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            result = _get_backends()
        assert "machine_a" in result
        assert "machine_b" in result
        assert result["machine_a"]["base_url"] == "http://10.0.0.1:8443/v1"
        assert result["machine_b"]["model"] == "qwen3:32b"

    def test_skips_backend_missing_base_url(self):
        cfg = _config_with_backends({
            "good": {"base_url": "http://ok/v1"},
            "bad": {"model": "gemma4:27b"},  # no base_url
        })
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            result = _get_backends()
        assert "good" in result
        assert "bad" not in result


# ── _resolve_backend ──────────────────────────────────────────────────────


class TestResolveBackend:
    def test_returns_none_for_empty_name(self):
        assert _resolve_backend(None) is None
        assert _resolve_backend("") is None

    def test_returns_none_for_unknown_name(self):
        cfg = _config_with_backends({"known": {"base_url": "http://x/v1"}})
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            result = _resolve_backend("not_registered")
        assert result is None

    def test_returns_overrides_for_known_name(self):
        cfg = _config_with_backends({
            "machine_b": {
                "base_url": "http://10.0.0.2:8443/v1",
                "model": "gemma4:27b",
                "provider": "custom",
            },
        })
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            result = _resolve_backend("machine_b")
        assert result is not None
        assert result["override_base_url"] == "http://10.0.0.2:8443/v1"
        assert result["model"] == "gemma4:27b"
        assert result["override_provider"] == "custom"

    def test_defaults_provider_to_custom_when_absent(self):
        cfg = _config_with_backends({
            "minimal": {"base_url": "http://x/v1"},
        })
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            result = _resolve_backend("minimal")
        assert result["override_provider"] == "custom"


# ── Task validation ───────────────────────────────────────────────────────


class TestTaskValidation:
    def test_accepts_string_backend(self):
        tasks = [{"goal": "do something", "backend": "machine_a"}]
        task_list, err = _normalize_task_list(
            goal=None, context=None, tasks=tasks, output_schema=None,
            top_role="worker", max_children=10,
        )
        assert err is None
        assert task_list[0]["backend"] == "machine_a"

    def test_rejects_non_string_backend(self):
        tasks = [{"goal": "do something", "backend": 42}]
        _, err = _normalize_task_list(
            goal=None, context=None, tasks=tasks, output_schema=None,
            top_role="worker", max_children=10,
        )
        assert err is not None
        assert "backend" in err.lower()

    def test_rejects_dict_backend(self):
        tasks = [{"goal": "do something", "backend": {"name": "x"}}]
        _, err = _normalize_task_list(
            goal=None, context=None, tasks=tasks, output_schema=None,
            top_role="worker", max_children=10,
        )
        assert err is not None

    def test_omitted_backend_is_fine(self):
        tasks = [{"goal": "do something"}]
        task_list, err = _normalize_task_list(
            goal=None, context=None, tasks=tasks, output_schema=None,
            top_role="worker", max_children=10,
        )
        assert err is None


# ── _build_children integration ───────────────────────────────────────────

class TestBuildChildrenBackendRouting:
    def test_child_receives_backend_overrides(self):
        """When a task specifies backend='machine_b', the child agent gets that
        backend's base_url and model instead of the parent's."""
        cfg = _config_with_backends({
            "machine_b": {
                "base_url": "http://10.0.0.2:8443/v1",
                "model": "gemma4:27b",
            },
        })
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            # Patch the child builder to capture what it receives
            captured_kwargs = {}

            def fake_build_child(**kwargs):
                captured_kwargs.update(kwargs)
                child = MagicMock()
                child._delegate_output_schema = None
                return child

            from tools.delegate_tool import _build_children

            task_list = [{"goal": "research", "backend": "machine_b"}]
            creds = {
                "model": "default-model",
                "provider": "custom",
                "base_url": "http://localhost:8443/v1",
                "api_key": None,
                "api_mode": None,
            }

            with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=fake_build_child):
                children, err = _build_children(
                    task_list=task_list,
                    task_schemas=[None],
                    creds=creds,
                    top_role="worker",
                    max_iterations=20,
                    parent_agent=MagicMock(),
                    routing_cfg={},
                    live_deleg_id=None,
                    live_writers=[None],
                )

            assert err is None
            assert len(children) == 1
            # Child should have received the backend's model and base_url
            assert captured_kwargs["model"] == "gemma4:27b"
            assert captured_kwargs["override_base_url"] == "http://10.0.0.2:8443/v1"

    def test_child_without_backend_inherits_parent_creds(self):
        """Tasks without a backend key keep the parent's credentials."""
        with patch("tools.delegate_tool_config._load_config", return_value={}):
            captured_kwargs = {}

            def fake_build_child(**kwargs):
                captured_kwargs.update(kwargs)
                child = MagicMock()
                return child

            from tools.delegate_tool import _build_children

            task_list = [{"goal": "research"}]
            creds = {
                "model": "parent-model",
                "provider": "custom",
                "base_url": "http://localhost:8443/v1",
                "api_key": None,
                "api_mode": None,
            }

            with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=fake_build_child):
                children, err = _build_children(
                    task_list=task_list,
                    task_schemas=[None],
                    creds=creds,
                    top_role="worker",
                    max_iterations=20,
                    parent_agent=MagicMock(),
                    routing_cfg={},
                    live_deleg_id=None,
                    live_writers=[None],
                )

            assert err is None
            assert captured_kwargs["model"] == "parent-model"
            assert captured_kwargs["override_base_url"] == "http://localhost:8443/v1"

    def test_mixed_backends_in_single_batch(self):
        """Different tasks in the same batch can target different backends."""
        cfg = _config_with_backends({
            "machine_a": {"base_url": "http://10.0.0.1:8443/v1", "model": "gemma4:27b"},
            "machine_b": {"base_url": "http://10.0.0.2:8443/v1", "model": "qwen3:32b"},
        })
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            captured = []

            def fake_build_child(**kwargs):
                captured.append(dict(kwargs))
                child = MagicMock()
                return child

            from tools.delegate_tool import _build_children

            task_list = [
                {"goal": "research", "backend": "machine_a"},
                {"goal": "implement", "backend": "machine_b"},
            ]
            creds = {
                "model": "default-model",
                "provider": "custom",
                "base_url": "http://localhost:8443/v1",
                "api_key": None,
                "api_mode": None,
            }

            with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=fake_build_child):
                children, err = _build_children(
                    task_list=task_list,
                    task_schemas=[None, None],
                    creds=creds,
                    top_role="worker",
                    max_iterations=20,
                    parent_agent=MagicMock(),
                    routing_cfg={},
                    live_deleg_id=None,
                    live_writers=[None, None],
                )

            assert err is None
            assert len(captured) == 2
            # First task -> machine_a
            assert captured[0]["model"] == "gemma4:27b"
            assert captured[0]["override_base_url"] == "http://10.0.0.1:8443/v1"
            # Second task -> machine_b
            assert captured[1]["model"] == "qwen3:32b"
            assert captured[1]["override_base_url"] == "http://10.0.0.2:8443/v1"

    def test_unknown_backend_falls_back_to_parent(self):
        """If a task names an unregistered backend, it falls back to parent creds."""
        cfg = _config_with_backends({"known": {"base_url": "http://x/v1", "model": "m"}})
        with patch("tools.delegate_tool_config._load_config", return_value=cfg):
            captured_kwargs = {}

            def fake_build_child(**kwargs):
                captured_kwargs.update(kwargs)
                child = MagicMock()
                return child

            from tools.delegate_tool import _build_children

            task_list = [{"goal": "research", "backend": "nonexistent"}]
            creds = {
                "model": "parent-model",
                "provider": "custom",
                "base_url": "http://localhost:8443/v1",
                "api_key": None,
                "api_mode": None,
            }

            with patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=fake_build_child):
                children, err = _build_children(
                    task_list=task_list,
                    task_schemas=[None],
                    creds=creds,
                    top_role="worker",
                    max_iterations=20,
                    parent_agent=MagicMock(),
                    routing_cfg={},
                    live_deleg_id=None,
                    live_writers=[None],
                )

            assert err is None
            # Should have inherited parent credentials
            assert captured_kwargs["model"] == "parent-model"
            assert captured_kwargs["override_base_url"] == "http://localhost:8443/v1"
