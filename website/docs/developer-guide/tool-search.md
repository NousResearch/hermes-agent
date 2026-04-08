# Tool Search: Progressive Disclosure for Tool Definitions

Tool definitions consume significant context tokens — often 10-15% of a 32K-token model's context window before the user says a word. Tool search replaces the full JSON schemas with a compact text catalog and two meta-tools that load schemas on demand.

## How it Works

1. **Startup**: Hermes estimates the token cost of tool definitions vs. the model's context window. If tool tokens exceed the threshold (default 10%), deferred mode activates.

2. **System prompt**: Instead of full schemas, the model sees a compact catalog listing each tool's name, toolset, and one-line description — roughly 5-8 tokens per tool vs. 200-400 tokens per full schema.

3. **Meta-tools**: Two tools are always loaded:
   - `tool_search(query)` — keyword search across the catalog; returns full schemas for matching tools
   - `tool_details(name)` — loads one tool's full schema by exact name

4. **Dynamic loading**: When the model calls `tool_search` or `tool_details`, the returned schemas are injected into the API's `tools` parameter for subsequent turns.

5. **Eviction**: When context compression fires, tools that haven't been called in the last N turns (default 10) are evicted from `self.tools`. They remain in the catalog and can be re-loaded via `tool_search` if needed later. Pinned tools and the meta-tools are never evicted.

## Configuration

```yaml
tool_search:
  mode: auto              # auto | always | never
  threshold: 0.10         # activate when tool tokens > 10% of context
  pinned_tools: []        # tools always loaded with full schemas
  evict_after_turns: 10   # evict unused tools after this many API turns
```

### Pinned tools

Tools listed in `pinned_tools` always have their full schemas loaded, bypassing tool search. Use this for tools the model needs on nearly every turn:

```yaml
tool_search:
  mode: auto
  pinned_tools:
    - read_file
    - terminal
    - write_file
```

## Auto-activation

With `mode: auto`, Hermes estimates tool tokens as `sum(len(json.dumps(schema)) / 4)` across all enabled tools. If this estimate exceeds `threshold × context_length`, deferred mode activates.

On large-context models (128K+), tool definitions are a negligible fraction and tool search stays off. On 32K models with ~60 tools, it typically activates and saves ~12K tokens.

## Interaction with Context Compression

Dynamically loaded tool schemas live on the agent's `self.tools` list, not in the message history. Context compression only touches messages, so loaded tools survive compression by default.

However, to prevent token creep from accumulated tool schemas, **eviction** runs as part of the compression cycle. When compression fires, any dynamically loaded tool that hasn't been *called* (not just searched for) in the last `evict_after_turns` API turns is removed from `self.tools`. The tool remains in the catalog, so the model can re-discover and re-load it with one extra turn of latency if needed.

This mirrors how both Anthropic and OpenAI approach the problem: loaded tools persist until explicitly removed. The difference is that on large-context models (200K+) neither platform needs automatic eviction, while on 32K models every token counts, so we evict proactively.

Pinned tools and the `tool_search`/`tool_details` meta-tools are never evicted.

## Key Files

| File | Role |
|---|---|
| `tools/tool_search.py` | Meta-tool handlers and keyword matching |
| `tools/registry.py` | `get_catalog()` and `get_single_definition()` |
| `agent/prompt_builder.py` | `build_tool_catalog_prompt()` |
| `model_tools.py` | `should_defer_tools()`, deferred mode in `get_tool_definitions()` |
| `run_agent.py` | Activation logic and dynamic schema injection |
