# Checkpoint Store GC Operations

This workflow covers Git garbage collection for Hermes' shared checkpoint store. Its owner is the Hermes runtime maintainer responsible for `tools/checkpoint_manager.py`.

## Trigger and scope

Checkpoint GC is triggered after any of these maintenance paths drops history:

- per-project `max_snapshots` enforcement;
- total-store size-cap enforcement inside `CheckpointManager`;
- stale/orphan pruning from `hermes checkpoints prune` or auto-prune;
- the size-cap pass inside stale/orphan pruning.

All four paths call the same internal policy. The command is:

```text
git gc --prune=2.hours.ago --quiet
```

The two-hour grace period protects fresh objects written by another session while GC is running.

## Logs and normal success

Checkpoint-manager messages use the standard Hermes logs under `~/.hermes/logs/`:

- `agent.log` for normal runtime and debug evidence;
- `errors.log` for warning/error evidence;
- `gateway.log` when the checkpoint operation came from the gateway.

Use `hermes logs --level debug` for live diagnosis. A normal GC exits 0, removes its own `gc.pid`, and leaves no checkpoint GC error. Verify the store with:

```bash
hermes checkpoints status
git --git-dir="$HOME/.hermes/checkpoints/store" fsck --no-dangling
```

## Concurrent, stale, and failed GC handling

- **Live GC owner:** Git exits 128 with its exact `gc is already running ... (use --force if not)` diagnostic. Hermes records a debug-level skip and does not retry. The active GC already owns maintenance.
- **Stale/dead PID marker:** do not remove `gc.pid` in Hermes. Run the ordinary command and let Git validate the PID, replace the stale marker, perform GC, and clean the marker.
- **Unrelated failure:** any other return code or stderr, including partial/lookalike lock text, remains an error. Capture the command, Git version, platform, store `fsck`, and relevant log excerpt before changing code or data.

Never add `--force`, kill the recorded process, unlink `gc.pid`, or immediately retry an active GC in application code.

## Failure threshold and escalation

Create or update one deduplicated runtime issue when any of these occurs:

- one unrelated GC failure risks checkpoint integrity or contains missing/bad-object output;
- the same non-benign GC error occurs twice in 24 hours;
- a live-owner skip persists beyond the owning process lifetime or blocks the next scheduled maintenance pass;
- checkpoint storage continues growing across two daily reviews despite successful GC.

The issue must include platform, `git --version`, reproduction command, redacted log path/timestamps, `fsck` result, and checkpoint-store size. Do not attach checkpoint object contents or user project data.

## Review cadence

The runtime maintainer reviews checkpoint GC errors after each reported incident and checks aggregate error/size behavior weekly while this policy is new. Reduce to release-cycle review after two releases without repeated failures. A live-owner debug skip alone is not an incident.

## Verification after a change

Run the focused policy tests and inspect the diff:

```bash
python -m pytest tests/tools/test_checkpoint_manager.py -k 'CheckpointGcPolicy or RealPruning'
git diff --check
```

For controlled runtime evidence, use an isolated temporary checkpoint base. Cover normal GC, a live current-host PID marker, a stale dead-PID marker, fresh-object grace, and an unrelated rc=128 error. Record the temporary log path and delete/allow cleanup of only that isolated store afterward; never experiment against `~/.hermes/checkpoints/store`.

## Rollback or disable

If the policy itself regresses checkpoint correctness, disable checkpoints through the existing `checkpoints.enabled: false` configuration or omit `--checkpoints`, preserve the store for investigation, and revert the policy commit through normal review. Do not change the grace period to `now` as an emergency workaround. Re-enable only after focused tests and isolated runtime evidence pass.
