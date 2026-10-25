# Local activation evidence (proposed)

Disabled by default. To expose the resolved webhook activation flags of profiles
served by this Gateway, set the following in the host profile's `config.yaml` and
restart the Gateway:

```yaml
gateway:
  activation_evidence_export: true
```

Only YAML booleans are accepted. Strings, integers, nulls and containers are
configuration errors. A top-level `activation_evidence_export` key takes
precedence by presence over its nested spelling. `false` or an absent key leaves
the existing local control verbs unchanged and does not register the new verb.
This option is read at startup; there is no environment-variable override.
If YAML loading fails while a legacy `gateway.json` enables this export, startup
is refused rather than silently falling back to that enabled legacy setting.

When enabled, a local control client may request `activation-evidence`. The
response uses `hermes.gateway-activation-evidence/v1` and contains only:

- `installDirectory`, `runtimeCommit`, `runtimePid`, `runtimeStartTime`,
  `observedAt` and `schemaVersion`;
- `profiles`, mapping served profile identifiers to `{webhookEnabled: boolean}`.

The value is the already-resolved activation flag, not a connection-health check.
No configuration is reloaded and no credential is traversed. Missing resolved
profiles, incomplete startup, reconciliation and ambiguous flags refuse the
request. Errors at the control boundary are sanitized. Snapshotting occurs on the
Gateway loop, without awaiting between profiles, including inline Windows pipe
dispatch. Requests from a different thread have a two-second bounded loop hop.

Use the optional client to publish a validated snapshot outside the home and
installation, to a new destination only:

```text
python -B -m gateway.activation_evidence --home <home> --output <new-directory>/activation-evidence.json
```

The client rejects unexpected fields, mismatched installation identity and
observations older than five minutes. Publication does not overwrite an existing
file. The control endpoint retains its existing local access boundary; no TCP
listener, background export, platform connection, model call or new model tool is
added. Consumers remain responsible for matching the PID birth identity and code
commit to the process they intend to audit. Local profile names and installation
paths can still identify a host layout; this is not anonymous telemetry.
