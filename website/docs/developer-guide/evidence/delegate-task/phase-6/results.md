Phase 6 optional memory experiment result

Decision: NO-GO (allowed by approved plan)
Base endpoint: eb359803a12d33acdf7c35d7e3e9f6609a807343

Evidence
- 32/32 memory guard tests passed.
- Memory remains off by default for delegated children.
- No child durable-memory write path was introduced.
- The repository exposes provider-specific memory tools/providers, not a stable backend-neutral inline read-only adapter for this delegation path.
- No frozen comparable task corpus, judge, cost budget or approved provider-neutral implementation exists in this phase.

Why NO-GO
A controlled quality comparison cannot be run honestly without adding the very provider coupling and production hooks this optional phase forbids. A synthetic experiment would not prove value and would create trap/leakage risk.

Disposition
- Retain memoryless children.
- Do not add memory configuration, adapter, provider wiring or feature flag.
- Reopen only with separate approval, frozen corpus/judge/thresholds and backend-neutral adapter contract from protocol.md.
- Main reliability work is not blocked; Phase 7 must retain the memoryless-child assertion.
