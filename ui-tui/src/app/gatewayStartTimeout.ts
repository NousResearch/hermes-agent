/** `gateway.start_timeout`: the backend has not attached yet, or its bootstrap gave up for good. */
import { isTerminalBootstrapFailure } from '../localGatewayBootstrap.js'

import { turnController } from './turnController.js'
import {
  backendGaveUpActivity,
  backendSlowStart,
  backendSlowStartStatus,
  stderrLooksLikeProblem
} from './userMessages.js'

const STDERR_LINE_CAP = 120
const STDERR_LINES_MAX = 4

/** Still waiting (the ready timer does not give up): say so, and point at /logs for the
 *  interpreter/cwd/stderr detail instead of printing paths here. Only failure-looking stderr lines
 *  are echoed inline so "wrong python" / "missing dep" stay diagnosable at a glance.
 *
 *  gateway_bootstrap.py's terminal verdict is different: the client stops retrying, so "still
 *  waiting" would be false. Its own verdict lines (the cause, then the remedy) are the message. */
export function reportGatewayStartTimeout(stderrTail: string | undefined, setStatus: (status: string) => void) {
  const terminal = isTerminalBootstrapFailure(stderrTail ?? '')

  setStatus(terminal ? backendGaveUpActivity() : backendSlowStartStatus())

  if (!terminal) {
    turnController.pushActivity(backendSlowStart(), 'warn')
  }

  const tailLines = (stderrTail ?? '')
    .split('\n')
    .map(l => l.trim())
    .filter(l => l && stderrLooksLikeProblem(l))
    .slice(-STDERR_LINES_MAX)

  for (const line of tailLines) {
    turnController.pushActivity(line.slice(0, STDERR_LINE_CAP), 'error')
  }
}
