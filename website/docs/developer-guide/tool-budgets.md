# Tool Result Budgets: Context-Aware Output Limits

Tool results can be enormous — a `ps aux` on a busy system, a `cat` of a large file, a recursive `grep` across a codebase. On frontier models with 200K+ context, this is manageable. On 32K models, a single oversized tool result can exceed the entire context window.

Tool result budgets prevent this by dynamically limiting how much output any single tool can return, based on the model's actual context window and current usage.

## How it Works

1. **After each tool call**, the budget layer estimates how much context is available.
2. **Budget calculation**: `max(floor, min(baseline, available))` where:
   - `baseline` = `context_length × result_pct` (default 25%)
   - `available` = remaining tokens × 4 chars/token
   - `floor` = minimum useful result (default 2000 tokens)
3. **If the result fits**: pass through unchanged. On large-context models, this is almost always the case.
4. **If context is tight**: attempt conversation compaction first to free up room.
5. **If the result still exceeds budget**: store the full output to disk, return a preview with pagination instructions.
6. **The model pages through** stored results using `read_file` with `offset`/`limit`.

## Configuration

```yaml
tool_budgets:
  result_pct: 0.25           # max single result as fraction of context
  turn_pct: 0.50             # max all results in one turn as fraction of context
  floor_tokens: 2000         # minimum useful result size in tokens
  compact_before_spill: true # try compaction before accepting tiny budget
```

## Scaling

| Model | Context | Baseline | Behavior |
|---|---|---|---|
| Gemma 4 31B | 32K | ~8K tokens | Active — large results paginated |
| GPT-5.4 | 128K | ~32K tokens | Rarely triggers |
| Claude 4 Opus | 200K | ~50K tokens | Almost never triggers |
| Gemini 2.5 | 1M | ~250K tokens | Invisible |

## Interaction with Existing Systems

- **Tool result persistence** (`tool_result_storage.py`): The budget layer runs first. `maybe_persist_tool_result()` becomes a secondary safety net.
- **Context compression**: Budget-triggered compaction uses the same `_compress_context()` as normal compression. No new compression logic.
- **Prompt caching**: Budget enforcement happens at insertion time (before the result enters messages), so cached prefixes are never invalidated.
- **Tool search** (`tool_search` feature): Independent — tool search controls which tools are *available*, budgets control how big their *results* can be.

## Key Files

| File | Role |
|---|---|
| `agent/tool_budget.py` | Budget calculation, spill logic |
| `run_agent.py` | Wiring: init, `_apply_tool_budget()`, dispatch interception |
| `tools/budget_config.py` | Default constants (used by persistence layer) |
| `cli-config.yaml.example` | User configuration |
