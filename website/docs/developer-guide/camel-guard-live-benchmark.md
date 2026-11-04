# Reproducing CaMeL Guard benchmarks

This document describes the harness. Generated model responses, local machine
identifiers and historical results are not committed as validation evidence.

## Offline native-dispatch controls

```sh
python scripts/camel_guard_offline_benchmark.py --iterations 3
```

The classifier response is a deterministic fixture. Real PluginManager
discovery, host structured parsing, lifecycle hooks, `read_file` and scratch
`write_file` handlers run. Every repetition covers off/monitor/enforce with
unauthorized and authorized mutations. Enforce must prevent the unrequested
write and allow the authorized write. Monitor must dispatch and record its
would-block decision. Off must dispatch with zero classifier calls and no
trace state. Latency here measures fixture overhead, not live-model latency.

## Live classifier and agent controls

```sh
python scripts/camel_guard_live_benchmark.py --live \
  --provider openai-codex --model YOUR_MODEL --phase all \
  --auth-home /path/to/your/hermes-profile
```

`--phase policy` isolates six prospective terminal/file/memory deny/allow
classifications. `--phase agent` records twelve full-agent attack and authorized
controls. Sensitive sinks are replaced with recorders. The agent chooses
whether to request a tool; a model refusing an attack in off mode is not a
CaMeL block. The generated interpretation is computed from actual outcomes.
Provider failures and fallback plans fail live utility gates, even when they
safely deny an attack. Model, source commit, dirty-tree flag, call counts,
tokens, latency and outcomes appear in the ignored JSON report.

Authentication remains in Hermes. The optional auth-home scopes only host
provider resolution/agent initialization to an already-configured Hermes
profile. The harness never reads token files or handles token values. The explicit
`--adopt-external-logins` flag asks Hermes's native login importer to reuse the
operator's Codex login in disposable homes (never in the default profile).
This may share a rotating token family; a separate Hermes sign-in is preferable
for ongoing operation. A fresh scratch
home without authentication cannot supply a valid live allow control; either
configure a Hermes auth profile or use your provider's supported API-key route.
No retired local-model runtime is started by the harness.

## Artifacts and interpretation

Default reports go to `benchmarks/camel_guard/artifacts/` (gitignored).
`--json-output` and the live runner's `--markdown-output` choose another path.
CI uploads generated offline reports and logs. Review the scripts, rerun them,
and inspect the artifacts associated with the tested commit. Do not treat a
committed result file or old PR numbers as fresh execution proof.

The focused integration and information-flow suites additionally exercise the
capability cross-product, malformed output, confinement, concurrent single
classification, native tool-result framing and trace non-disclosure. This is
bounded behavioral evidence, not a statistical robustness estimate or formal
proof against arbitrary prompt injection.
