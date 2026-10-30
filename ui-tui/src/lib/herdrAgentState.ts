import { execFile } from 'node:child_process'

export type AgentActivity = 'idle' | 'working' | 'blocked' | 'unknown'
export interface AgentReport {
  state: AgentActivity
  sessionId?: string
}
type Run = (file: string, args: string[], env: NodeJS.ProcessEnv) => Promise<void>

const run: Run = (file, args, env) =>
  new Promise((resolve, reject) => {
    execFile(file, args, { env, timeout: 1000, maxBuffer: 64 * 1024, windowsHide: true }, error =>
      error ? reject(error) : resolve()
    )
  })

/** Public Herdr API, scoped to the inherited pane. Never infer the focused pane.
 * Single flight + latest-only coalescing prevents late idle reports replacing work. */
export function createHerdrReporter(env: NodeJS.ProcessEnv = process.env, execute: Run = run) {
  const enabled = env.HERDR_ENV === '1' && Boolean(env.HERDR_PANE_ID && env.HERDR_BIN_PATH && env.HERDR_SOCKET_PATH)
  const source = 'hermes-tui'
  let sequence = Date.now()
  let desired: AgentReport | undefined
  let acknowledged = ''
  let flight: Promise<void> | undefined
  let closing = false
  let closed: Promise<void> | undefined
  const nextSeq = () => String((sequence = Math.max(Date.now(), sequence + 1)))
  const args = () => [env.HERDR_PANE_ID!, '--source', source, '--agent', 'hermes', '--seq', nextSeq()]

  const drain = () => {
    if (!enabled || flight || closing || !desired) {
      return
    }

    const value = desired
    desired = undefined
    const key = JSON.stringify(value)

    if (key === acknowledged) {
      return
    }

    const command = ['pane', 'report-agent', ...args(), '--state', value.state]

    if (value.sessionId) {
      command.push('--agent-session-id', value.sessionId)
    }

    flight = execute(env.HERDR_BIN_PATH!, command, env)
      .then(
        () => {
          acknowledged = key
        },
        () => undefined
      )
      .then(() => {
        flight = undefined
        drain()
      })
  }

  return {
    update(value: AgentReport) {
      if (!enabled || closing) {
        return
      }

      desired = value
      drain()
    },
    close(): Promise<void> {
      if (closed) {
        return closed
      }

      closing = true
      desired = undefined
      closed = (flight || Promise.resolve()).then(async () => {
        if (enabled) {
          await execute(env.HERDR_BIN_PATH!, ['pane', 'release-agent', ...args()], env).catch(() => undefined)
        }
      })

      return closed
    }
  }
}
