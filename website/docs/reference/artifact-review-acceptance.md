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

## Delivery evidence and isolated install rehearsal

After acceptance, an integration owner can call
`build_artifact_review_evidence` from
`hermes_cli.kanban_acceptance_evidence` to export one deterministic report that
links Kanban builder, reviewer, and repair runs to the revalidated acceptance
receipt. The exporter is read-only. Caller-supplied Roots ticket and CFO labels
are isolated as unverified external claims and the report is explicitly barred
from serving as an acceptance gate. Check Roots and the CFO ledger at their own
authenticated boundaries.

Before installing an exact candidate, rehearse its bytes and rollback entirely
inside a temporary directory:

```bash
python evals/artifact_review_install_rehearsal.py \
  --repo /path/to/reviewed/hermes-agent \
  --expected-commit <reviewed-commit> \
  --component hermes_cli/kanban_artifact_acceptance.py \
  --component hermes_cli/kanban_acceptance_evidence.py \
  --safe-root /path/to/operator-owned-rehearsal-root \
  --output-name hermes-install-rehearsal.json
```

The command requires a pre-existing operator-owned safe root that is not
group/world writable and is disjoint from both the checkout and every known
Hermes home. Temporary files and the new, exclusive output receipt stay under
that root; process `TMPDIR` and `TEMP` cannot redirect them. It refuses a dirty
or different checkout and unsafe, untracked, or non-regular component paths.
Every object query is pinned to the initially captured commit, followed by a
final HEAD and clean-state recheck.

Git subprocesses inherit no caller credential or object-store environment and
disable system/global configuration, replacement objects, hooks, filesystem
monitors, untracked cache, and optional index locks. Repository-local config is
still read under those overrides. The local-only object commands do not invoke
a credential helper; the receipt does not claim that local config contains no
credential settings. The rehearsal compares selected files with a Git archive
of the captured commit, performs an isolated swap, and restores the prior
sentinel bytes. It does not touch a live checkout or Hermes home, restart a
service, launch a model worker, or activate the candidate.
