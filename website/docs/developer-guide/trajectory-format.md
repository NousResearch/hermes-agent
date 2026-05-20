# Trajectory Format

Hermes Agent saves conversation trajectories in ShareGPT-compatible JSONL format
for use as training data, debugging artifacts, and reinforcement learning datasets.

Source files: `agent/trajectory.py`, `run_agent.py` (search for `_save_trajectory`), `batch_runner.py`


## File Naming Convention

Trajectories are written to files in the current working directory:

| File | When |
|------|------|
| `trajectory_samples.jsonl` | Conversations that completed successfully (`completed=True`) |
| `failed_trajectories.jsonl` | Conversations that failed or were interrupted (`completed=False`) |

The batch runner (`batch_runner.py`) writes to a custom output file per batch
(e.g., `batch_001_output.jsonl`) with additional metadata fields.

You can override the filename via the `filename` parameter in `save_trajectory()`.


## JSONL Entry Format

Each line in the file is a self-contained JSON object. There are two variants:

### CLI/Interactive Format (from `_save_trajectory`)

```json
{
  "conversations": [ ... ],
  "timestamp": "2026-03-30T14:22:31.456789",
  "model": "anthropic/claude-sonnet-4.6",
  "completed": true
}
```

### Batch Runner Format (from `batch_runner.py`)

```json
{
  "prompt_index": 42,
  "conversations": [ ... ],
  "metadata": { "prompt_source": "gsm8k", "difficulty": "hard" },
  "completed": true,
  "partial": false,
  "api_calls": 7,
  "toolsets_used": ["code_tools", "file_tools"],
  "tool_stats": {
    "terminal": {"count": 3, "success": 3, "failure": 0},
    "read_file": {"count": 2, "success": 2, "failure": 0},
    "write_file": {"count": 0, "success": 0, "failure": 0}
  },
  "tool_error_counts": {
    "terminal": 0,
    "read_file": 0,
    "write_file": 0
  }
}
```

The `tool_stats` and `tool_error_counts` dictionaries are normalized to include
ALL possible tools (from `model_tools.TOOL_TO_TOOLSET_MAP`) with zero defaults,
ensuring consistent schema across entries for HuggingFace dataset loading.


## Conversations Array (ShareGPT Format)

The `conversations` array uses ShareGPT role conventions:

| API Role | ShareGPT `from` |
|----------|-----------------|
| system | `"system"` |
| user | `"human"` |
| assistant | `"gpt"` |
| tool | `"tool"` |

### Complete Example

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. If available tools are not relevant in assisting with user query, just respond in natural conversational language. Don't make assumptions about what values to plug into functions. After calling & executing the functions, you will be provided with function results within <tool_response> </tool_response> XML tags. Here are the available tools:\n<tools>\n[{\"name\": \"terminal\", \"description\": \"Execute shell commands\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}}, \"required\": null}]\n</tools>\nFor each function call return a JSON object, with the following pydantic model json schema for each:\n{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\nEach function call should be enclosed within <tool_call> </tool_call> XML tags.\nExample:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
    },
    {
      "from": "human",
      "value": "What Python version is installed?"
    },
    {
      "from": "gpt",
      "value": "<think>\nThe user wants to know the Python version. I should run python3 --version.\n</think>\n<tool_call>\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"python3 --version\"}}\n</tool_call>"
    },
    {
      "from": "tool",
      "value": "<tool_response>\n{\"tool_call_id\": \"call_abc123\", \"name\": \"terminal\", \"content\": \"Python 3.11.6\"}\n</tool_response>"
    },
    {
      "from": "gpt",
      "value": "<think>\nGot the version. I can now answer the user.\n</think>\nPython 3.11.6 is installed on this system."
    }
  ],
  "timestamp": "2026-03-30T14:22:31.456789",
  "model": "anthropic/claude-sonnet-4.6",
  "completed": true
}
```


## Normalization Rules

### Reasoning Content Markup

The trajectory converter normalizes ALL reasoning into `<think>` tags, regardless
of how the model originally produced it:

1. **Native thinking tokens** (`msg["reasoning"]` field from providers like
   Anthropic, OpenAI o-series): Wrapped as `<think>\n{reasoning}\n</think>\n`
   and prepended before the content.

2. **REASONING_SCRATCHPAD XML** (when native thinking is disabled and the model
   reasons via system-prompt-instructed XML): `<REASONING_SCRATCHPAD>` tags are
   converted to `<think>` via `convert_scratchpad_to_think()`.

3. **Empty think blocks**: Every `gpt` turn is guaranteed to have a `<think>`
   block. If no reasoning was produced, an empty block is inserted:
   `<think>\n</think>\n` — this ensures consistent format for training data.

### Tool Call Normalization

Tool calls from the API format (with `tool_call_id`, function name, arguments as
JSON string) are converted to XML-wrapped JSON:

```
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>
```

- Arguments are parsed from JSON strings back to objects (not double-encoded)
- If JSON parsing fails (shouldn't happen — validated during conversation),
  an empty `{}` is used with a warning logged
- Multiple tool calls in one assistant turn produce multiple `<tool_call>` blocks
  in a single `gpt` message

### Tool Response Normalization

All tool results following an assistant message are grouped into a single `tool`
turn with XML-wrapped JSON responses:

```
<tool_response>
{"tool_call_id": "call_abc123", "name": "terminal", "content": "output here"}
</tool_response>
```

- If tool content looks like JSON (starts with `{` or `[`), it's parsed so the
  content field contains a JSON object/array rather than a string
- Multiple tool results are joined with newlines in one message
- The tool name is matched by position against the parent assistant's `tool_calls`
  array

### System Message

The system message is generated at save time (not taken from the conversation).
It follows the Hermes function-calling prompt template with:

- Preamble explaining the function-calling protocol
- `<tools>` XML block containing the JSON tool definitions
- Schema reference for `FunctionCall` objects
- `<tool_call>` example

Tool definitions include `name`, `description`, `parameters`, and `required`
(set to `null` to match the canonical format).


## Loading Trajectories

Trajectories are standard JSONL — load with any JSON-lines reader:

```python
import json

def load_trajectories(path: str):
    """Load trajectory entries from a JSONL file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

# Filter to successful completions only
successful = [e for e in load_trajectories("trajectory_samples.jsonl")
              if e.get("completed")]

# Extract just the conversations for training
training_data = [e["conversations"] for e in successful]
```

### Loading for HuggingFace Datasets

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="trajectory_samples.jsonl")
```

The normalized `tool_stats` schema ensures all entries have the same columns,
preventing Arrow schema mismatch errors during dataset loading.


## Controlling Trajectory Saving

In the CLI, trajectory saving is controlled by:

```yaml
# config.yaml
agent:
  save_trajectories: true  # default: false
```

Or via the `--save-trajectories` flag. When the agent initializes with
`save_trajectories=True`, the `_save_trajectory()` method is called at the end
of each conversation turn.

The batch runner always saves trajectories (that's its primary purpose).

Samples with zero reasoning across all turns are automatically discarded by the
batch runner to avoid polluting training data with non-reasoning examples.
