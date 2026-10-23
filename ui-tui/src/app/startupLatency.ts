import { randomUUID } from 'node:crypto'

import { DASHBOARD_TUI_MODE } from '../config/env.js'
import type { GatewayClient } from '../gatewayClient.js'

// Module-level so a gateway reconnect/respawn (a fresh gateway.ready, or a remounted
// handler) never re-reports: the metric is launch -> first ready, once per TUI process.
let reported = false

// One TUI process is one launch; the backend latches on it too, so a re-sent report counts once.
const LAUNCH_ID = randomUUID()

/** Node process start is the earliest moment the TUI owns, so uptime at the first
 *  gateway.ready is the launch latency. The backend buckets it and drops it unless the
 *  user opted in; older backends lack the method, so errors are swallowed.
 *  A dashboard Chat tab spawns a TUI per terminal it opens: that is a tab opening, not a
 *  user launching Hermes, so it stays out of the startup distribution. */
export function reportStartupLatency(gw: Pick<GatewayClient, 'request'>): void {
  if (reported || DASHBOARD_TUI_MODE) {
    return
  }

  reported = true
  gw.request('shared_metrics.startup_latency', {
    elapsed_ms: Math.round(process.uptime() * 1000),
    launch_id: LAUNCH_ID,
    surface: 'tui'
  }).catch(() => undefined)
}
