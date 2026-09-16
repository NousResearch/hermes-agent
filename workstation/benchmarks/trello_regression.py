"""Controlled ACIRV/Trello-like replay through AIAgent; never calls real APIs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import os


def run(root: Path, n: int = 12):
    from run_agent import AIAgent
    from tools.registry import registry
    from tools.effects import ToolEffect
    from workstation.task_compiler import execute_compiled_work
    from workstation.artifacts import ArtifactStore
    from workstation.reference_plane import schema_projection
    effects = {"trello_resolve_board": ToolEffect.DISCOVERY, "trello_resolve_list": ToolEffect.DISCOVERY,
               "trello_find_existing": ToolEffect.DISCOVERY, "trello_create_card": ToolEffect.MUTATION,
               "trello_get_card": ToolEffect.PURE_READ, "trello_collect_results": ToolEffect.PURE_READ}
    schemas = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
               for name in ["work_execute", "read_file", "tool_describe", *effects]]
    cards, counts = {}, {"tool_calls": 0, "discovery_calls": 0, "mutations": 0,
                         "setup_calls": 0, "replayed_mutations": 0, "raw_bytes": 0}
    req = {"operation_key": "acirv-trello", "title": "ACIRV 12 cards", "items": [{"index": i} for i in range(n)],
           "setup_steps": [
               {"id": "board", "tool": "trello_resolve_board", "args": {"name": "ACIRV"}},
               {"id": "list", "tool": "trello_resolve_list", "args": {"board": "$setup.board.board_id"}},
               {"id": "existing", "tool": "trello_find_existing", "args": {"list": "$setup.list.list_id"}}],
           "steps": [
               {"id": "schema", "tool": "tool_describe", "args": {"name": "trello_create_card"}},
               {"id": "template", "depends_on": ["schema"], "tool": "read_file", "args": {"path": "template.md"}},
               {"id": "create", "depends_on": ["template", "existing"], "tool": "trello_create_card",
                "args": {"list": "$setup.list.list_id", "index": "$item.index"}, "expect": {"ok": True}},
               {"id": "verify", "tool": "trello_get_card", "args": {"card": "$steps.create.card_id"}, "expect": {"ok": True}}],
           "finalize_steps": [{"id": "collect", "depends_on": ["verify"], "tool": "trello_collect_results", "args": {"items": "$items_ref"}}]}
    old = dict(registry._tools)
    with patch.dict(os.environ, {"HERMES_HOME": str(root)}):
        for name, effect in effects.items():
            registry.register(name, "test_replay", {"name": name}, lambda **kw: "{}", effect=effect)
        artifacts = ArtifactStore()
        def handler(name, args, task, **kwargs):
            if name == "work_execute":
                return execute_compiled_work(args, task_id=task)
            counts["tool_calls"] += 1
            if name in effects and effects[name] == ToolEffect.DISCOVERY:
                counts["discovery_calls"] += 1
                counts["setup_calls"] += 1
            if name == "trello_resolve_board":
                raw = {"board_id": "board-acirv"}
            elif name == "trello_resolve_list":
                assert args["board"] == "board-acirv"
                raw = {"list_id": "list-backlog"}
            elif name == "trello_find_existing":
                raw = {"cards": [], "history": "prior board state " * 10000}
            elif name == "trello_create_card":
                counts["mutations"] += 1
                counts["replayed_mutations"] += args["index"] in cards
                cards[args["index"]] = f"card-{args['index']}"
                raw = {"ok": True, "card_id": cards[args["index"]], "description": "card data " * 8000}
            elif name == "trello_get_card":
                assert args["card"] in cards.values()
                raw = {"ok": True, "card_id": args["card"]}
            elif name == "tool_describe":
                counts["discovery_calls"] += 1
                schema = {"name": "trello_create_card", "description": "schema detail " * 5000, "parameters": {}}
                raw = schema_projection(artifacts, "replay-schema", schema, "trusted-fake-v1")
            elif name == "read_file":
                raw = {"template": "standard text " * 8000}
            else:
                assert len(cards) == n
                raw = {"ok": True, "completed": n, "results_ref": args["items"]}
            text = json.dumps(raw)
            counts["raw_bytes"] += len(text.encode())
            return text
        def response(content, tools, finish):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tools), finish_reason=finish)], model="fake/provider", usage=None)
        try:
            with patch("run_agent.get_tool_definitions", return_value=schemas), patch("run_agent.check_toolset_requirements", return_value={}), \
                 patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}), patch("run_agent.OpenAI"):
                agent = AIAgent(api_key="fake", base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_memory=True, skip_context_files=True)
            agent.client = MagicMock()
            agent._cached_system_prompt = "Execute verified work."
            agent._use_prompt_caching = False
            agent.compression_enabled = False
            agent.save_trajectories = False
            compiled = SimpleNamespace(id="compiled", type="function", function=SimpleNamespace(name="work_execute", arguments=json.dumps(req)))
            agent.client.chat.completions.create.side_effect = [response("", [compiled], "tool_calls"), response(f"{n} cards verified", None, "stop")]
            with patch("run_agent.handle_function_call", side_effect=handler):
                result = agent.run_conversation(f"Crie {n} cards no board ACIRV.", task_id="trello-task")
            tool_message = next(m for m in result["messages"] if m["role"] == "tool")
            envelope = json.loads(tool_message["content"])
            return {"items": n, "baseline_modeled": {"provider_calls": n + 2, "LLM_interventions": n + 2,
                    "setup_calls": 3 * n, "inline_context_bytes": counts["raw_bytes"]},
                    "durable_measured": {"provider_calls": agent.client.chat.completions.create.call_count,
                    "LLM_interventions": agent.client.chat.completions.create.call_count, **{k: v for k, v in counts.items() if k != "raw_bytes"},
                    "completed_items": envelope["completed"], "compactions": envelope["metrics"]["compactions"],
                    "inline_context_bytes": len(tool_message["content"].encode()), "artifact_bytes": envelope["metrics"]["artifact_bytes"],
                    "bytes_avoided_by_refs": envelope["metrics"]["bytes_avoided_by_refs"], "cache_hits": envelope["metrics"]["cache_hits"],
                    "token_count": None, "usage_status": envelope["metrics"]["usage_status"]}}
        finally:
            with registry._lock:
                registry._tools = old
                registry._generation += 1


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        print(json.dumps(run(Path(root)), indent=2))
