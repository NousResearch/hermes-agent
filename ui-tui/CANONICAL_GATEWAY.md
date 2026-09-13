# Ink canonical local gateway connection

Normal local Ink startup now ensures the profile's ordinary gateway through
`ensure_gateway_runtime`, obtains an instance/profile-bound private control ticket,
and connects over authenticated WebSocket. The short Python bootstrap subprocess
is only a client; Ink no longer starts `tui_gateway.entry` or owns an agent loop.
The ticket travels through a private child pipe and WebSocket subprotocol, never
in an endpoint URL or discovery receipt. Explicit remote URLs remain remote: a
failed connection cannot fall back to a local daemon.

Closing Ink detaches its socket without stopping the gateway. Reconnection gets
a fresh single-use ticket using discovery only; it cannot resurrect an explicitly
stopped gateway. An initial ensure failure is visible and does not create a rival
owner. Local profile selection remains the launcher's exact effective HERMES_HOME.

Ink requires `runtime.describe.session_create` to advertise the `tui` source and
every requested creation parameter. It sends a fresh `request_id`, `source: tui`,
and explicit launcher options; an older CLI-only runtime returns a visible policy
compatibility failure rather than silently creating CLI-policy work. Attach to an
existing session does not apply the viewer's creation options.

Prepared submission IDs and captured destination fences remain authoritative.
The wire adapter maps `submission_id` to canonical `input_id` and translates the
receipt's `ref` into the existing journal acknowledgement shape. Resume/create
snapshots and events retain the authority epoch and execution generation. Stored
conversation `content` is translated to transcript display text, so a fully settled
reconnect does not depend on catching a final live event. Canonical readiness is
published once, after `runtime.describe`, rather than also forwarding the listener's
legacy readiness event. Shared
approval/clarify responses carry the actual prompt ID and generation; a resumed
snapshot restores its pending prompt cards. Stop includes the active generation.

## Input recovery and controls

The disconnected view keeps its destination. Enter writes a private fsync/rename
journal before clearing the composer; reconnect only resumes that destination.
Discovery keeps retrying without starting a stopped daemon. Confirmed admissions
are retired from the journal; ambiguous prompt retries preserve their original ID
and payload. Unknown executions stay paused: select the row with Up and use
Ctrl+X to discard it through the generation-bound authority resolver. A refused
Discard remains visible and reports the error.

Model, branch and compression use `session.mutate` with revision/generation
preconditions and a stable request ID. The session picker reads `session.list`.
Busy preference reads/writes are session scoped. Steer/redirect target the active
generation; these controls are not durable prompt admissions. An ambiguous control
reply is retained for inspection but never automatically or manually replayed;
check the transcript before submitting a new correction. Unsupported controls do
not silently become queued turns.

Image paths, startup images and clipboard images are staged on the Ink host.
Local-owner files go into the captured profile's private `cache/images`; explicit
remote connections upload bytes and require an owner-readable returned path.
Identified `prompt.submit` carries `{path,mime}` attachments, including journal
retries. Clipboard extraction never reads a remote owner's clipboard. Image busy
corrections are refused and retained: use queue mode for an image-bearing turn.

## Current gaps

- Fresh TUI creation requires an authority advertising TUI launch-policy support;
  older CLI-only runtimes are rejected. No CLI impersonation is used.
- Runtime descriptions and session metadata are intentionally narrow. Ink renders
  an explicit unavailable-inventory message, not fabricated tool/skill lists.
- Canonical busy controls/config projection and remote image uploads require the
  corresponding owner handlers. Older owners reject them visibly. The model
  chooser still requires the owner's `model.options` projection; explicit `/model`
  uses the canonical mutation path.
- Rich legacy RPCs outside these controls (wake, general config settings, subagent
  panels and some slash execution) are not claimed implemented by this change.
- A per-view session switch does not terminate the old shared session. The present
  authority removes subscriptions when the socket closes, so old subscriptions
  can remain until that detach; the renderer still filters by current session.
- Secret/sudo and batch clarification are not claimed supported.

## Native verification

`python ui-tui/scripts/native_canonical_probe.py <receipt-directory>` uses a
private temporary HOME/HERMES_HOME, an ordinary no-platform daemon, a loopback
OpenAI-compatible model, and real Ink processes on native Linux PTYs. It creates
an explicitly labelled seed session with the runtime's supported policy, resumes
it from two Ink viewers, submits through the real renderer, checks both rendered
replies, detaches both viewers, and resumes persisted history in a new Ink process.
The fixture does not replace renderer predicates or inject RPC results. It waits
for the real authority to settle and retains that snapshot **before** starting the
reconnected viewer; otherwise a late live delta can conceal broken history hydration.

Optional mode argument after the receipt directory:

- `text` (default): two viewers, committed journal acknowledgement, settled reconnect.
- `approval` / `clarify`: detach both viewers with a pending card, restore and answer
  natively, then verify the owned effect / exact tool response on the model wire.
- `startup`: absent owner → native Ink ensure → one fresh TUI-policy session and reply.
- `launcher`: the same fresh path through `python -m hermes_cli.main --tui chat -q ...`,
  using the supported prebuilt `HERMES_TUI_DIR` path (no dependency install).
- `stop` / `stop-control` / `stop-launcher`: hold the loopback inference stream before a harmless
  fixture-owned tool effect. `stop-launcher` starts the viewer through the Python
  launcher and discovers its one fresh TUI-policy session, without seeding one.
  Native Ctrl+C must receive the real generation-bearing
  acknowledgement and suppress that effect; the no-Stop control must execute it.
  Both then type a fresh prompt and require rendered and persisted completion.
  These modes run the ordinary daemon entry via `native_stop_probe.py`, whose
  observation-only dispatch wrapper records the unchanged real interrupt RPC
  request/response. No response or execution predicate is replaced.

Use the repository Python environment. Every mode keeps HOME/HERMES_HOME temporary,
uses only a loopback model, and cleans up its owned processes. Raw PTY output and
JSON receipts stay in the requested receipt directory; timeout is a failure, never
an invitation to rerun until green.
