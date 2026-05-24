from __future__ import annotations

import types


class TestDelegateRouteMetadata:
    def test_reasoning_effort_label_handles_disabled_and_missing_configs(self):
        from tools.delegate_tool import _reasoning_effort_label

        assert _reasoning_effort_label({"enabled": True, "effort": "xhigh"}) == "xhigh"
        assert _reasoning_effort_label({"enabled": False, "effort": "high"}) == "none"
        assert _reasoning_effort_label(None) == ""

    def test_progress_callback_relays_route_metadata(self):
        from tools.delegate_tool import _build_child_progress_callback

        calls = []

        def parent_cb(event_type, tool_name=None, preview=None, args=None, **kwargs):
            calls.append((event_type, tool_name, preview, args, kwargs))

        parent = types.SimpleNamespace(tool_progress_callback=parent_cb)
        callback = _build_child_progress_callback(
            task_index=0,
            goal="inspect repo",
            parent_agent=parent,
            task_count=1,
            subagent_id="sa-route",
            parent_id="sa-parent",
            depth=1,
            model="deepseek-v4-pro",
            provider="deepseek",
            reasoning_effort="low",
            role="leaf",
            execution_mode="delegate_task",
            route_reason="delegation provider override",
            toolsets=["terminal", "file"],
        )

        assert callback is not None
        callback("subagent.start", preview="inspect repo")

        assert calls == [
            (
                "subagent.start",
                None,
                "inspect repo",
                None,
                {
                    "task_index": 0,
                    "task_count": 1,
                    "goal": "inspect repo",
                    "subagent_id": "sa-route",
                    "parent_id": "sa-parent",
                    "depth": 1,
                    "model": "deepseek-v4-pro",
                    "provider": "deepseek",
                    "reasoning_effort": "low",
                    "role": "leaf",
                    "execution_mode": "delegate_task",
                    "route_reason": "delegation provider override",
                    "toolsets": ["terminal", "file"],
                    "tool_count": 0,
                },
            )
        ]

    def test_active_subagent_snapshot_includes_route_metadata_without_agent_object(self):
        from tools import delegate_tool

        record = {
            "subagent_id": "sa-live",
            "parent_id": None,
            "depth": 0,
            "goal": "live child",
            "model": "deepseek-v4-pro",
            "provider": "deepseek",
            "reasoning_effort": "low",
            "role": "leaf",
            "execution_mode": "delegate_task",
            "route_reason": "delegation provider override",
            "status": "running",
            "tool_count": 0,
            "agent": object(),
        }

        try:
            delegate_tool._register_subagent(record)
            snapshot = delegate_tool.list_active_subagents()
        finally:
            delegate_tool._unregister_subagent("sa-live")

        assert snapshot == [
            {
                "subagent_id": "sa-live",
                "parent_id": None,
                "depth": 0,
                "goal": "live child",
                "model": "deepseek-v4-pro",
                "provider": "deepseek",
                "reasoning_effort": "low",
                "role": "leaf",
                "execution_mode": "delegate_task",
                "route_reason": "delegation provider override",
                "status": "running",
                "tool_count": 0,
            }
        ]
