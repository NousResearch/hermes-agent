# Tools Subtree Instructions

This file scopes tool-specific guidance to `tools/` work. Root `AGENTS.md` still contains the non-negotiable project rules.

## Adding or changing built-in tools

- Built-in/core tools require two pieces: a `tools/<name>.py` module with `registry.register(...)`, and a toolset entry in `toolsets.py`.
- Prefer plugins, CLI commands, MCP servers, or skills over new core tools unless the Footprint Ladder clearly says the capability belongs in core.
- Every tool handler must return a JSON string.
- Gate optional/external capabilities with `check_fn` and `requires_env` so unavailable tools do not appear in schemas.
- Use `get_hermes_home()` for persistent state and `display_hermes_home()` for user-facing paths; never hardcode `~/.hermes`.
- Do not mention tools from other toolsets in schema descriptions. If a cross-tool hint is needed, add it dynamically in `model_tools.py` where availability is known.
- Agent-level tools such as todo/memory are intercepted in `run_agent.py` before normal `handle_function_call()` dispatch; follow existing patterns.

## Verification

- Add/update focused tests for changed tool behavior.
- Run tests with `scripts/run_tests.sh ...`, never direct `pytest`.
