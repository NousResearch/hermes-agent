import { type ChildProcess, spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { resolve } from 'node:path'
import { createInterface } from 'node:readline'

const MAX_GATEWAY_LOG_LINES = 200
const MAX_LOG_PREVIEW = 240

export interface GatewayEvent {
  type: string
  session_id?: string
  payload?: Record<string, unknown>
}

interface Pending {
  resolve: (v: unknown) => void
  reject: (e: Error) => void
}

export class GatewayClient extends EventEmitter {
  private proc: ChildProcess | null = null
  private reqId = 0
  private logs: string[] = []
  private pending = new Map<string, Pending>()

  start() {
    const root = process.env.HERMES_ROOT ?? resolve(import.meta.dirname, '../../')

    this.proc = spawn(process.env.HERMES_PYTHON ?? resolve(root, 'venv/bin/python'), ['-m', 'tui_gateway.entry'], {
      cwd: root,
      stdio: ['pipe', 'pipe', 'pipe']
    })

    createInterface({ input: this.proc.stdout! }).on('line', raw => {
      try {
        this.dispatch(JSON.parse(raw))
      } catch {
        const preview = raw.trim().slice(0, MAX_LOG_PREVIEW) || '(empty line)'
        this.pushLog(`[protocol] malformed stdout: ${preview}`)
        this.emit('event', { type: 'gateway.protocol_error', payload: { preview } } satisfies GatewayEvent)
      }
    })

    createInterface({ input: this.proc.stderr! }).on('line', raw => {
      const line = raw.trim()

      if (!line) {
        return
      }

      this.pushLog(line)
      this.emit('event', { type: 'gateway.stderr', payload: { line } } satisfies GatewayEvent)
    })

    this.proc.on('error', err => {
      this.pushLog(`[spawn] ${err.message}`)
      this.rejectPending(new Error(`gateway error: ${err.message}`))
      this.emit('event', { type: 'gateway.stderr', payload: { line: `[spawn] ${err.message}` } } satisfies GatewayEvent)
    })

    this.proc.on('exit', code => {
      this.rejectPending(new Error(`gateway exited${code === null ? '' : ` (${code})`}`))
      this.emit('exit', code)
    })
  }

  private dispatch(msg: Record<string, unknown>) {
    const id = msg.id as string | undefined
    const p = id ? this.pending.get(id) : undefined

    if (p) {
      this.pending.delete(id!)
      msg.error ? p.reject(new Error((msg.error as any).message)) : p.resolve(msg.result)

      return
    }

    if (msg.method === 'event') {
      this.emit('event', msg.params as GatewayEvent)
    }
  }

  private pushLog(line: string) {
    this.logs.push(line)

    if (this.logs.length > MAX_GATEWAY_LOG_LINES) {
      this.logs.splice(0, this.logs.length - MAX_GATEWAY_LOG_LINES)
    }
  }

  private rejectPending(err: Error) {
    for (const [id, pending] of this.pending) {
      this.pending.delete(id)
      pending.reject(err)
    }
  }

  getLogTail(limit = 20): string {
    return this.logs.slice(-Math.max(1, limit)).join('\n')
  }

  request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    if (!this.proc?.stdin) {
      return Promise.reject(new Error('gateway not running'))
    }

    const id = `r${++this.reqId}`

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.pending.delete(id)) {
          reject(new Error(`timeout: ${method}`))
        }
      }, 30_000)

      this.pending.set(id, {
        reject: e => {
          clearTimeout(timeout)
          reject(e)
        },
        resolve: v => {
          clearTimeout(timeout)
          resolve(v)
        }
      })

      try {
        this.proc!.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
      } catch (e) {
        clearTimeout(timeout)
        this.pending.delete(id)
        reject(e instanceof Error ? e : new Error(String(e)))
      }
    })
  }

  kill() {
    this.proc?.kill()
  }
}
