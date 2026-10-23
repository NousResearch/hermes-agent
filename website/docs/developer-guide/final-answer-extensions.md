---
sidebar_position: 40
title: Final-answer extensions
---

# Final-answer extensions

A plugin can look at the agent's would-be final answer and, once, ask for one more model call with
a private instruction. Typical uses: a style or completeness lint that should fix the answer
silently instead of showing a half-finished draft to the user.

```python
from agent.final_answer_extensions import FinalNudge

class MyLint:
    def before_final(self, context, text):
        turn = context.identity["_current_turn_id"]
        if context.state.get("turn") == turn or "TODO" not in text:
            return None                      # let the answer through
        context.state["turn"] = turn         # at most one nudge per turn
        return FinalNudge("Finish the TODO items before answering.", "todo_lint", "_todo_lint_synthetic")

def register(ctx):
    ctx.register_final_answer_provider("todo-lint", MyLint())
```

## Contract

- `context.identity` is a read-only mapping (`platform`, `_user_id`, `_chat_type`, `session_id`,
  `_current_turn_id`, `_interrupt_requested`). `context.messages` is a copy. `context.state` is the
  provider's own per-session dict; the host keeps it only after a valid return.
- Return `None` or a `FinalNudge`. The flag must start with `_` and end with `_synthetic`.
- The draft answer and the nudge are marked with the host flag `_final_answer_extension_synthetic`
  and your own flag. They are never written to the session transcript or replayed on resume.
- Each call is bounded (0.25 s). A missing, slow, raising or invalid provider lets the answer
  through. Providers never get the agent, a send handle or tools.
- Registration is load-owned: unloading the plugin removes the provider.
