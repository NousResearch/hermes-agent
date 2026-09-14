Phase 3 per-model OpenRouter routing result

Status: implementation evidence complete; audit pending
Base endpoint: 8c8c737461871d0c47f99636d5bf0d46309b3e14
Implementation scope: provider_routing.models.<model-id> only

Adopted/adapted upstream
- 12871bd01ede1c6eaf3bf456066d1e010034185f
- Added sparse model-entry resolution beside reasoning overrides using the existing canonical model variants.
- Updated _provider_preferences_for_agent as the shared resolution chokepoint.
- Updated turn-recovery messaging to inspect effective per-model only restrictions.
- Added Kensei-specific tests for profile-home routing, fallback model re-resolution, direct/Nous exclusion and separation from fallback providers.
- Updated cli-config.yaml.example and provider-routing documentation.

Rejected upstream hunk
- OpenRouter endpoint-pin precedence was not applied: KenseiAgent has no OPENROUTER_ENDPOINT_PINS seam. Introducing one would be unrelated scope.

Verified behaviour
- Sparse overlay over flat defaults.
- Dot/dash and openrouter-prefix model matching.
- /model/fallback model changes re-resolve at the shared helper.
- Profile context selects target routing config rather than parent config.
- Provider payload is emitted only for OpenRouter; direct and Nous Portal payloads exclude it.
- Per-model routing remains separate from Hermes fallback_providers and delegation provider/model pins.

Tests
- Routing/provider suite: 44 passed.
- Integrated owned Phases 1–3 set: 98 passed, 1 skipped, 3 deselected.
- Collection: 66 tests collected.
- Deferred completion/fallback subset: 6 assertion-level failures, 2 passed, 8 deselected.

Boundaries
- No completion-unit or aggregator implementation.
- No cross-provider fallback changes.
- No merge, push, restart, activation or deployment.
