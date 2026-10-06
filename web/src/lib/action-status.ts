import type { ActionStatusResponse } from "./api";

/** Losing a Popen registry is not update completion. */
export function actionNeedsPolling(action: string, status: ActionStatusResponse, expectedId?: string): boolean {
  if (action !== "hermes-update") return status.running;
  if (expectedId && status.action_id !== expectedId) {
    // A bounded, identity-free abandonment is a stop-waiting notice, not a
    // terminal claim. An unrelated identified attempt is never ours.
    return Boolean(status.action_id) || status.state !== "abandoned";
  }
  return status.exit_code === null && status.state !== "abandoned" && status.state !== "superseded";
}
