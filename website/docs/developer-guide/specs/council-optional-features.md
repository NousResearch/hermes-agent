# Kensei LLM Council optional features

Kensei's council remains pipeline-specific: it reviews `prd.md` and `spec.md`, writes the existing top-level verdict contract, and participates in the external revision loop. It is separate from the public free-form council skill.

## Feature flags

All flags default off in source so upstream-compatible behaviour remains stable until explicitly activated in the live config.

```yaml
council:
  compose: true
  cross_examination: true
  cascade_breaker: true
  minority_report: true
  evidence_labels: true
  html_report: true
  protocol: deliberate
  adaptive_stopping: true
```

- `compose`: chairman creates task-specific personas from fenced PRD/spec data.
- `cross_examination`: anonymised reviewers revise their positions.
- `cascade_breaker`: independently reviews the PRD/spec without panel signals.
- `minority_report`: lowest-confidence successful originating member preserves dissent.
- `evidence_labels`: requests evidence-type tags in Phase 1.
- `html_report`: writes escaped `council-report.html`.
- `protocol`: `deliberate`, `vote`, or `synthesize`.
- `adaptive_stopping`: on `deliberate`, skips ranking when initial and revised confidence distributions converge.

## Protocols

| Protocol | Path | Closure |
|---|---|---|
| `deliberate` | compose → independent → cross-exam → ranking → cascade → chairman | chairman verdict |
| `vote` | compose → independent → ranking | deterministic majority; ties revise |
| `synthesize` | compose → independent → chairman | chairman synthesis |

Minority and HTML outputs remain independent optional post-processing steps.

## Safety invariants

- PRD/spec content is fenced as untrusted data.
- Compose is task-specific.
- Cascade-breaking receives first-party documents, never panel critiques.
- Phase 1 results retain original panel indices despite parallel completion.
- The token cap is checked after every provider call and phase.
- Every phase receives only the remaining overall deadline.
- Unsupported protocols fail before model work.

## Verification

```bash
python -m pytest tests/test_phase_b_council.py tests/test_council_additive_features.py -q
python -m py_compile hermes_cli/council.py hermes_cli/config_defaults.py
```

A mocked test pass proves execution and failure semantics. A live canary is a separate activation proof and should use a disposable task artefact directory.

## Rollback

1. Set the seven boolean flags to `false` and `protocol` to `deliberate` in the live root config.
2. Validate with `hermes config check`.
3. Restart the fleet because gateways import council code/config at process start.
4. Confirm the root gateway imports `hermes_cli.council` from the canonical KenseiAgent checkout.
5. Preserve canary and failed-run artefacts for diagnosis.

Code rollback is the reviewed council-hardening commit revert. Configuration rollback alone restores the pre-feature execution path because all additions are gated.
