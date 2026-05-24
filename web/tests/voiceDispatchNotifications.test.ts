import assert from "node:assert/strict";
import {
  buildVoiceDelegateStatusNotification,
  summarizeVoiceDelegateStatus,
} from "../src/lib/voiceDispatchNotifications.ts";

const running = buildVoiceDelegateStatusNotification({
  run_id: "voice_delegate_running",
  status: "running",
});
assert.equal(running, null, "running delegates should not trigger proactive voice updates");

const completed = buildVoiceDelegateStatusNotification({
  delegate_id: "voice_delegate_done",
  run_id: "voice_delegate_done",
  status: "completed",
  output: "Built the feature and all checks passed.",
});
assert.equal(completed?.key, "voice_delegate_done:completed");
assert.equal(completed?.kind, "run");
assert.equal(completed?.title, "Delegate finished");
assert.match(completed?.voicePrompt ?? "", /proactively now/i);
assert.match(completed?.voicePrompt ?? "", /Built the feature/);

const failed = buildVoiceDelegateStatusNotification({
  run_id: "voice_delegate_failed",
  status: "failed",
  error: "Command exited 1",
});
assert.equal(failed?.key, "voice_delegate_failed:failed");
assert.match(failed?.transcriptBody ?? "", /failed: Command exited 1/);

const approval = buildVoiceDelegateStatusNotification({
  delegate_id: "voice_delegate_approval",
  run_id: "voice_delegate_approval",
  status: "waiting_for_approval",
  last_event: "Tool approval required",
});
assert.equal(approval?.key, "voice_delegate_approval:waiting_for_approval");
assert.equal(approval?.kind, "approval");
assert.match(approval?.voicePrompt ?? "", /approval is required/i);
assert.match(approval?.voicePrompt ?? "", /explicit Hermes UI click/i);
assert.doesNotMatch(approval?.voicePrompt ?? "", /always approve/i);

const longSummary = summarizeVoiceDelegateStatus({
  run_id: "voice_delegate_long",
  status: "completed",
  output: "x".repeat(1200),
});
assert.ok(longSummary.length < 950, "voice summaries should be capped before sending to realtime");
assert.ok(longSummary.endsWith("…"), "truncated voice summaries should show truncation");

console.log("voiceDispatchNotifications tests passed");
