"""Controlled ACIRV/Trello-like replay through AIAgent; never calls real APIs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import os


def run(root: Path, n: int = 12, scenario="corrected", capability="description-content-area"):
    from run_agent import AIAgent
    from tools.registry import registry
    from tools.effects import ToolEffect
    from workstation.task_compiler import execute_compiled_work
    from workstation.artifacts import ArtifactStore
    from workstation.reference_plane import schema_projection
    effects = {"trello_resolve_board": ToolEffect.DISCOVERY, "trello_resolve_list": ToolEffect.DISCOVERY,
               "trello_find_existing": ToolEffect.DISCOVERY, "trello_create_card": ToolEffect.MUTATION,
               "trello_get_card": ToolEffect.PURE_READ, "trello_collect_results": ToolEffect.PURE_READ,
               "trello_capabilities": ToolEffect.DISCOVERY}
    schemas = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
               for name in ["work_execute", "read_file", "tool_describe", *effects]]
    cards, counts = {}, {"tool_calls": 0, "discovery_calls": 0, "mutations": 0,
                         "setup_calls": 0, "replayed_mutations": 0, "raw_bytes": 0}
    req = {"operation_key": "acirv-trello-" + scenario, "recipe_key": "acirv.trello.cards.v1",
           "recipe_scope": {"route": "tool.trello_create_card", "host": "trello.test", "path_family": "/c/:card"},
           "preflight": [{"tool": "trello_capabilities", "args": {}, "expect": {"description_testid": capability}}],
           "title": "ACIRV 12 cards", "items": [{"index": i} for i in range(n)],
           "setup_steps": [
               {"id": "board", "tool": "trello_resolve_board", "args": {"name": "ACIRV"}},
               {"id": "list", "tool": "trello_resolve_list", "args": {"board": "$setup.board.board_id"}},
               {"id": "existing", "tool": "trello_find_existing", "args": {"list": "$setup.list.list_id"}}],
           "steps": [
               {"id": "schema", "tool": "tool_describe", "args": {"name": "trello_create_card"}},
               {"id": "template", "depends_on": ["schema"], "tool": "read_file", "args": {"path": "template.md"}},
               {"id": "create", "depends_on": ["template", "existing"], "tool": "trello_create_card",
                "args": {"list": "$setup.list.list_id", "index": "$item.index"}, "expect": {"ok": True}},
               {"id": "verify", "tool": "trello_get_card", "verifies": ["create"], "args": {"card": "$steps.create.card_id"}, "expect": {"persisted": True, "card_id": "$steps.create.card_id"}}],
           "finalize_steps": [{"id": "collect", "depends_on": ["verify"], "tool": "trello_collect_results", "args": {"items": "$items_ref"}}]}
    if scenario.startswith("known") or scenario == "stale":
        req = {"operation_key": "acirv-trello-" + scenario, "recipe_key": req["recipe_key"], "items": [{"index": n+i} for i in range(n)]}
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
                counts["setup_calls"] += name in {"trello_resolve_board", "trello_resolve_list", "trello_find_existing"}
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
                raw = {"persisted": not scenario.startswith("broken"), "card_id": args["card"]}
            elif name == "trello_capabilities":
                raw = {"description_testid": capability}
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
            def provider(**kwargs):
                if agent.client.chat.completions.create.call_count == 1:
                    return response("", [compiled], "tool_calls")
                return response(f"{n} cards verified" if len(cards) == n and not scenario.startswith("broken") else "Workflow blocked; review exceptions", None, "stop")
            agent.client.chat.completions.create.side_effect = provider
            with patch("run_agent.handle_function_call", side_effect=handler):
                result = agent.run_conversation(f"Crie {n} cards no board ACIRV.", task_id="trello-task")
            tool_message = next(m for m in result["messages"] if m["role"] == "tool")
            envelope = json.loads(tool_message["content"])
            from workstation.continuation import build_durable_handoff
            from workstation.durable_tasks import DurableTaskStore
            from agent.conversation_compression import compress_context
            store = DurableTaskStore()
            try:
                handoff = build_durable_handoff(store, envelope["plan_id"])
            finally:
                store.close()
            # Run the real automatic durable fast path twice. No LLM summary,
            # no session rotation, no duplicate handoff in the persisted list.
            before = json.dumps(result["messages"], default=str)
            for _ in range(2):
                returned, _ = compress_context(agent, result["messages"], agent._cached_system_prompt)
                assert returned is result["messages"]
            assert json.dumps(result["messages"], default=str) == before
            context_metrics = agent._durable_context_metrics
            metrics = envelope["metrics"]
            uncertain = envelope["ledger"].get("uncertain", 0)
            return {"items": n, "baseline_modeled": {"provider_calls": n + 2, "LLM_interventions": n + 2,
                    "setup_calls": 3 * n, "inline_context_bytes": counts["raw_bytes"]},
                    "durable_measured": {"provider_calls": agent.client.chat.completions.create.call_count,
                    "LLM_interventions": agent.client.chat.completions.create.call_count, **{k: v for k, v in counts.items() if k != "raw_bytes"},
                    "completed_items": envelope["completed"], "compactions": envelope["metrics"]["compactions"],
                    "inline_context_bytes": len(tool_message["content"].encode()), "artifact_bytes": envelope["metrics"]["artifact_bytes"],
                    "bytes_avoided_by_refs": envelope["metrics"]["bytes_avoided_by_refs"], "cache_hits": envelope["metrics"]["cache_hits"],
                    "token_count": None, "usage_status": metrics["usage_status"],
                    "planner_calls": agent.client.chat.completions.create.call_count, "executor_llm_calls": 0,
                    "uncertain_items": uncertain, "failed_fanout_items": metrics["failed_fanout_items"],
                    **{k: metrics[k] for k in ("canary_attempts", "canary_successes", "canary_failures", "recipe_cache_hits", "recipe_cache_misses", "recipe_invalidations")},
                    "handoff_bytes": len(json.dumps(handoff, separators=(",", ":")).encode()),
                    "compaction_bytes": context_metrics["compaction_bytes"],
                    "duplicate_compaction_bytes_suppressed": context_metrics["duplicate_compaction_bytes_suppressed"],
                    "provider_context_bytes": len(json.dumps(agent.client.chat.completions.create.call_args.kwargs["messages"]).encode()),
                    "projection_source_bytes": context_metrics["projection_source_bytes"],
                    **dict.fromkeys(("input_tokens", "uncached_input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"), None)}}
        finally:
            with registry._lock:
                registry._tools = old
                registry._generation += 1


def run_scenarios(root: Path, n=12):
    broken = run(root, n, "broken")
    corrected = run(root, n, "corrected")
    known = run(root, n, "known")
    stale = run(root, n, "stale", "description-content-area-v2")
    broken_after_stale = run(root, n, "broken_after_stale", "description-content-area-v2")
    corrected_v2 = run(root, n, "corrected_v2", "description-content-area-v2")
    known_v2 = run(root, n, "known_v2", "description-content-area-v2")
    bad, good, hit = (r["durable_measured"] for r in (broken, corrected, known))
    assert bad["canary_attempts"] == bad["canary_failures"] == 1
    assert bad["mutations"] <= 1 and bad["failed_fanout_items"] == 0
    assert good["completed_items"] == hit["completed_items"] == n
    assert good["replayed_mutations"] == hit["replayed_mutations"] == 0
    assert hit["recipe_cache_hits"] == 1 and hit["canary_attempts"] == 0
    assert all(r["durable_measured"]["executor_llm_calls"] == 0 for r in (broken, corrected, known))
    assert stale["durable_measured"]["recipe_invalidations"] == 1 and stale["durable_measured"]["mutations"] == 0
    assert broken_after_stale["durable_measured"]["mutations"] == 1 and broken_after_stale["durable_measured"]["canary_failures"] == 1
    assert corrected_v2["durable_measured"]["completed_items"] == known_v2["durable_measured"]["completed_items"] == n
    assert known_v2["durable_measured"]["recipe_cache_hits"] == 1
    return {"broken": broken, "corrected": corrected, "known_after_restart": known,
            "stale": stale, "broken_after_stale": broken_after_stale, "corrected_v2": corrected_v2, "known_v2": known_v2}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        print(json.dumps(run_scenarios(Path(root)), indent=2))
