# Subagent Profiles

Hermes supports profile-aware skill routing for spawned child sessions.

A `subagent_profile` is a compact role label such as:
- `builder`
- `reviewer`
- `operator`
- `browser_scout`
- `researcher`
- `security_reviewer`
- `performance_reviewer`
- `maintainability_reviewer`
- `red_team`

The profile affects two things:
1. child prompt guidance in `tools/delegate_tool.py`
2. skill recommendation ranking in `agent/prompt_builder.py` and `tools/skills_tool.py`

## Source of truth

Primary files:
- `agent/subagent_profiles.py` — built-in profiles and gstack affinity
- `agent/skill_utils.py` — metadata extraction and ranking helpers
- `tools/skills_tool.py` — runtime recommendation helper
- `agent/prompt_builder.py` — prompt rendering and cache keying
- `tools/delegate_tool.py` — child-profile resolution and prompt injection
- `agent/review_mesh.py` — specialist -> profile mapping for review mesh
- `scripts/audit_subagent_skill_routing.py` — routing audit

## How routing works

1. `delegate_task` resolves a profile from:
   - explicit `subagent_profile`
   - review-mesh specialist mapping
   - config default
   - fallback inference from goal/context/toolsets
2. Hermes builds a child prompt with profile-specific guidance.
3. The skills system prepends a `<recommended_skills>` block for that profile.
4. The full `<available_skills>` catalog still appears underneath.

## G-Stack

Active gstack skills are the imported local skills under `~/.hermes/skills/gstack/gstack-*`.
Treat those as the runtime surface.
Do not build a second gstack-specific loader for subagents.

Examples:
- reviewer -> `gstack-review`, `gstack-plan-eng-review`, `gstack-qa`
- operator -> `gstack-ship`, `gstack-land-and-deploy`, `gstack-document-release`
- browser_scout -> `gstack-browse`, `gstack-canary`, `gstack-setup-browser-cookies`

## Extending profiles safely

Config knobs under `delegation:` in `~/.hermes/config.yaml`:
- `default_profile` — fallback role for child sessions when no explicit profile is passed
- `profile_overrides` — per-profile metadata overrides such as prompt preamble or preferred skills
- `gstack_mode` — `auto`, `prefer`, or `off` for gstack hint emphasis in child prompts

When adding a new subagent role:
1. add/update the profile in `agent/subagent_profiles.py`
2. give matching skills strong metadata:
   - `metadata.hermes.tags`
   - `metadata.hermes.requires_toolsets`
   - `metadata.hermes.requires_tools`
3. update or confirm ranking logic in `agent/skill_utils.py`
4. run:

```bash
python scripts/audit_subagent_skill_routing.py
pytest tests/agent/test_subagent_profiles.py tests/agent/test_prompt_builder.py tests/tools/test_skills_tool.py tests/agent/test_review_mesh.py -q
```

## Failure modes to watch

- prompt bloat from dumping too many recommended skills
- cache bleed between profiles if the prompt-builder cache key forgets profile identity
- weak metadata causing random-looking recommendations
- fake gstack mappings that point to skills not present in `~/.hermes/skills/gstack`

Audit the routing before you trust it.
