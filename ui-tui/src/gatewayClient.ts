import { type ChildProcess, spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { resolve } from 'node:path'
import { createInterface } from 'node:readline'

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
  private pending = new Map<string, Pending>()

  start() {
    const root = resolve(import.meta.dirname, '../../')

    this.proc = spawn(process.env.HERMES_PYTHON ?? resolve(root, 'venv/bin/python'), ['-m', 'tui_gateway.entry'], {
      cwd: root,
      stdio: ['pipe', 'pipe', 'inherit']
    })

    createInterface({ input: this.proc.stdout! }).on('line', raw => {
      try {
        this.dispatch(JSON.parse(raw))
      } catch {
        /* malformed line */
      }
    })

    this.proc.on('exit', code => this.emit('exit', code))
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

  request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const id = `r${++this.reqId}`

    this.proc!.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })

      setTimeout(() => {
        if (this.pending.delete(id)) {
          reject(new Error(`timeout: ${method}`))
        }
      }, 30_000)
    })
  }

  kill() {
    this.proc?.kill()
  }
}
