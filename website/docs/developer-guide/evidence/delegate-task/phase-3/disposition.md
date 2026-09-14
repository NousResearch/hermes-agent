Phase 3 upstream disposition

Headline upstream commit: 12871bd01ede1c6eaf3bf456066d1e010034185f

ADOPT/ADAPT
- agent/chat_completion_helpers.py: shared _provider_preferences_for_agent sparse overlay.
- hermes_constants.py: resolve_per_model_provider_routing using _canonical_model_variants.
- agent/turn_recovery.py: effective per-model only restriction in tool-use failure notice.
- cli-config.yaml.example and website/docs/user-guide/features/provider-routing.md: user-facing schema/documentation.

REJECT
- plugins/model-providers/openrouter/__init__.py endpoint-pin precedence hunk. KenseiAgent's plugin has sticky/Pareto behaviour but no OPENROUTER_ENDPOINT_PINS map; adding that unrelated seam would violate phase scope.

PRESERVE
- Kensei profile-scoped runtime context from Phase 2.
- Hermes fallback_providers and delegation provider/model overrides as separate layers.
- direct provider and Nous Portal payload exclusion.
- Kensei request override and transport parity tests.

Evidence source: /tmp/phase3-upstream.patch, generated from the local upstream ref; its hash is recorded in the Phase 3 manifest.
