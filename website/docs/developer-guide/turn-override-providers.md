---
sidebar_position: 41
title: Turn-override providers
---

# Turn-override providers

A plugin can route one admitted user turn to a different model (for example a mixture-of-agents
preset for high-stakes questions) without ever owning a persistent model switch.

```python
from gateway.turn_overrides import TurnOverride

class CarefulReview:
    def decide(self, request):
        if request.chat_type != "dm" or "approve" not in request.text.lower():
            return None
        return TurnOverride({"provider": "moa", "model": "default", "base_url": "moa://local",
                             "api_key": "moa-virtual-provider", "api_mode": "chat_completions"}, "approval")

def register(ctx):
    ctx.register_turn_override_provider("careful-review", CarefulReview())
```

## Contract

- `decide(request)` runs once per admitted, claimed, non-consumed turn, before the agent runs.
  `request` is frozen: text, platform, chat type/id, user id, bot/internal/command flags, session key
  and a detached copy of the gateway configuration (when it is a mapping).
- Return `None` or `TurnOverride(override, reason)`. `override` must have exactly the keys `provider`,
  `model`, `base_url`, `api_key`, `api_mode` (string values), the same shape as `/model --once`.
- The host arms it through the canonical one-turn owner: the earliest pending snapshot is kept, the
  cached agent is evicted and the generation-guarded `finally` restores the previous model on every
  exit (success, error, interrupt, stale generation). Providers never see the conversation or runner.
- An explicit `/moa` or `/model --once` for the same turn wins; providers are not consulted.
- Optional and fail-open: a missing, raising, slow (0.25 s bound, own capacity) or invalid provider
  means no override. Registration is load-owned.
