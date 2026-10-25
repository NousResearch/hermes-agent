# Artifact review completion contract

`artifact-review-v1` is an opt-in Kanban completion contract for work whose
deliverable must be bound to independent review and evidence before dependent
tasks can start.

The verifier and evidence policy are accepted only from the bootstrap-pinned
`/etc/hermes/config.yaml`. `HERMES_MANAGED_DIR`, user config and profile config
cannot select this trust boundary. The directory, config and verifier must be
root-owned and not group/world writable. The verifier is one absolute,
non-symlink executable kept beneath `/etc/hermes`; interpreter-plus-script and
inline-code argv forms are refused. Hermes invokes it without a shell, with an
empty environment, `/` as its working directory, and closed inherited file
descriptors.

```yaml
kanban:
  artifact_review_verifier_command:
    - /etc/hermes/bin/artifact-review-verifier
  artifact_review_verifier_timeout_seconds: 30
  artifact_review_evidence_policy:
    test_scopes: [targeted-worker]
    integration_scopes: [combined-revision]
```

The verifier reads JSON on stdin. Artifact acceptance requests use schema
`hermes.artifact-review-acceptance/v1` and carry board-derived task, run,
contract, implementer, reviewer, configured scope policy, and content digest
facts alongside an untrusted candidate. Protected archive dispositions use
schema `hermes.non-acceptance-disposition/v1`; the verifier must return the
trusted actor and exact task/run/contract bindings.

An absent or invalid managed verifier fails closed. The local filesystem CAS is
content-addressed, read-only, rehashed before the terminal transaction, and kept
outside per-task attachment directories. Hermes does not claim resistance to a
same-privilege process that can chmod or rewrite Hermes-owned files, or to an
arbitrary database writer. Deployments that run untrusted workers under the
same operating-system identity need a privileged verifier/CAS broker or OS
isolation for a stronger boundary.
