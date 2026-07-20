# Brain Networks

Hermes can run a lightweight **Brain Networks** stack that models four cognitive
systems for reflection, task focus, emotional salience, and offline consolidation.

| Network | Role |
|---------|------|
| **DMN** (Default Mode Network) | Idle / reflective prompts → short reflections (SQLite) |
| **ECN** (Executive Control Network) | Standing task focus across turns (persistent) |
| **Limbic** | Salience + tone hints from user language |
| **Dream Engine** | Offline episode consolidation (`/dream`, cron, `dream_pass` tool) |

## Enable

```yaml
# ~/.hermes/config.yaml
brain_networks:
  enabled: true
  dmn_reflection_chance: 0.3
  dream_idle_threshold_seconds: 300
  ecn_max_task_stack: 10
  use_llm_for_reflection: true
```

When enabled, `AIAgent` constructs a `BrainNetworkOrchestrator` at init and binds
it to the session. Turn hooks inject a **volatile** `<brain-networks>` block into
the user message via `pre_llm_call` — they never mutate the stable system prompt
(prompt-cache safe).

## `/focus` — persistent ECN task focus

```text
/focus ship the Model Desk PR     # pin a standing goal for this session
/focus show                       # show current focus + level
/focus clear                      # clear pinned focus
/focus status                     # alias for show
```

Focus is stored under `$HERMES_HOME/brain_networks/orchestrator.db` so it
survives process restarts for the same session id. Pinned focus resists casual
topic drift until you clear it or use explicit switch language.

Works on CLI and messaging gateway (same command registry).

## `/dream` — consolidation pass

```text
/dream            # run now
/dream status     # enabled? idle threshold?
```

Dream narratives prefer the auxiliary LLM when configured; otherwise templates.
Emotional tone is derived from limbic keyword analysis of source episodes
(deterministic — not random). Dreams are persisted to the same SQLite store.

## Doctor

```bash
hermes doctor
# includes brain_networks check + persistence smoke
```

## Design notes

- **Footprint**: `dream_pass` / skill tools stay gated on `brain_networks.enabled`.
- **Cache**: all turn signals are volatile user-context, not system-prompt edits.
- **Herens**: when `herens.enabled` is on, DMN/ECN also surface through Herens
  turn hooks using the shared orchestrator singleton (no fresh ECN per turn).
