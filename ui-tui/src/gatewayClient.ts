import { type ChildProcess, spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync } from 'node:fs'
import { delimiter, resolve } from 'node:path'
import { createInterface } from 'node:readline'

import type { GatewayEvent } from './gatewayTypes.js'

const MAX_GATEWAY_LOG_LINES = 200
const MAX_LOG_LINE_BYTES = 4096
const MAX_BUFFERED_EVENTS = 2000
const MAX_LOG_PREVIEW = 240
const STARTUP_TIMEOUT_MS = Math.max(5000, parseInt(process.env.HERMES_TUI_STARTUP_TIMEOUT_MS ?? '15000', 10) || 15000)
const REQUEST_TIMEOUT_MS = Math.max(30000, parseInt(process.env.HERMES_TUI_RPC_TIMEOUT_MS ?? '120000', 10) || 120000)

const truncateLine = (line: string) =>
  line.length > MAX_LOG_LINE_BYTES ? `${line.slice(0, MAX_LOG_LINE_BYTES)}… [truncated ${line.length} bytes]` : line

const resolvePython = (root: string) => {
  const configured = process.env.HERMES_PYTHON?.trim() || process.env.PYTHON?.trim()

  if (configured) {
    return configured
  }

  const venv = process.env.VIRTUAL_ENV?.trim()

  const hit = [
    venv && resolve(venv, 'bin/python'),
    venv && resolve(venv, 'Scripts/python.exe'),
    resolve(root, '.venv/bin/python'),
    resolve(root, '.venv/bin/python3'),
    resolve(root, 'venv/bin/python'),
    resolve(root, 'venv/bin/python3')
  ].find(p => p && existsSync(p))

  return hit || (process.platform === 'win32' ? 'python' : 'python3')
}

const asGatewayEvent = (value: unknown): GatewayEvent | null =>
  value && typeof value === 'object' && !Array.isArray(value) && typeof (value as { type?: unknown }).type === 'string'
    ? (value as GatewayEvent)
    : null

interface Pending {
  reject: (e: Error) => void
  resolve: (v: unknown) => void
}

export class GatewayClient extends EventEmitter {
  private proc: ChildProcess | null = null
  private reqId = 0
  private logs: string[] = []
  private pending = new Map<string, Pending>()
  private bufferedEvents: GatewayEvent[] = []
  private pendingExit: number | null | undefined
  private ready = false
  private readyTimer: ReturnType<typeof setTimeout> | null = null
  private subscribed = false
  private stdoutRl: ReturnType<typeof createInterface> | null = null
  private stderrRl: ReturnType<typeof createInterface> | null = null

  private publish(ev: GatewayEvent) {
    if (ev.type === 'gateway.ready') {
      this.ready = true

      if (this.readyTimer) {
        clearTimeout(this.readyTimer)
        this.readyTimer = null
      }
    }

    if (this.subscribed) {
      return void this.emit('event', ev)
    }

    if (this.bufferedEvents.push(ev) > MAX_BUFFERED_EVENTS) {
      this.bufferedEvents.splice(0, this.bufferedEvents.length - MAX_BUFFERED_EVENTS)
    }
  }

  start() {
    const root = process.env.HERMES_PYTHON_SRC_ROOT ?? resolve(import.meta.dirname, '../../')
    const python = resolvePython(root)
    const cwd = process.env.HERMES_CWD || root
    const env = { ...process.env }
    const pyPath = env.PYTHONPATH?.trim()
    env.PYTHONPATH = pyPath ? `${root}${delimiter}${pyPath}` : root

    this.ready = false
    this.bufferedEvents = []
    this.pendingExit = undefined
    this.stdoutRl?.close()
    this.stderrRl?.close()
    this.stdoutRl = null
    this.stderrRl = null

    if (this.proc && !this.proc.killed && this.proc.exitCode === null) {
      this.proc.kill()
    }

    if (this.readyTimer) {
      clearTimeout(this.readyTimer)
    }

    this.readyTimer = setTimeout(() => {
      if (this.ready) {
        return
      }

      this.pushLog(`[startup] timed out waiting for gateway.ready (python=${python}, cwd=${cwd})`)
      this.publish({ type: 'gateway.start_timeout', payload: { cwd, python } })
    }, STARTUP_TIMEOUT_MS)

    this.proc = spawn(python, ['-m', 'tui_gateway.entry'], { cwd, env, stdio: ['pipe', 'pipe', 'pipe'] })

    this.stdoutRl = createInterface({ input: this.proc.stdout! })
    this.stdoutRl.on('line', raw => {
      try {
        this.dispatch(JSON.parse(raw))
      } catch {
        const preview = raw.trim().slice(0, MAX_LOG_PREVIEW) || '(empty line)'

        this.pushLog(`[protocol] malformed stdout: ${preview}`)
        this.publish({ type: 'gateway.protocol_error', payload: { preview } })
      }
    })

    this.stderrRl = createInterface({ input: this.proc.stderr! })
    this.stderrRl.on('line', raw => {
      const line = truncateLine(raw.trim())

      if (!line) {
        return
      }

      this.pushLog(line)
      this.publish({ type: 'gateway.stderr', payload: { line } })
    })

    this.proc.on('error', err => {
      this.pushLog(`[spawn] ${err.message}`)
      this.rejectPending(new Error(`gateway error: ${err.message}`))
      this.publish({ type: 'gateway.stderr', payload: { line: `[spawn] ${err.message}` } })
    })

    this.proc.on('exit', code => {
      if (this.readyTimer) {
        clearTimeout(this.readyTimer)
        this.readyTimer = null
      }

      this.rejectPending(new Error(`gateway exited${code === null ? '' : ` (${code})`}`))

      if (this.subscribed) {
        this.emit('exit', code)
      } else {
        this.pendingExit = code
      }
    })
  }

  private dispatch(msg: Record<string, unknown>) {
    const id = msg.id as string | undefined
    const p = id ? this.pending.get(id) : undefined

    if (p) {
      this.pending.delete(id!)

      if (msg.error) {
        const err = msg.error as { message?: unknown } | null | undefined

        p.reject(new Error(typeof err?.message === 'string' ? err.message : 'request failed'))
      } else {
        p.resolve(msg.result)
      }

      return
    }

    if (msg.method === 'event') {
      const ev = asGatewayEvent(msg.params)

      if (ev) {
        this.publish(ev)
      }
    }
  }

  private pushLog(line: string) {
    if (this.logs.push(truncateLine(line)) > MAX_GATEWAY_LOG_LINES) {
      this.logs.splice(0, this.logs.length - MAX_GATEWAY_LOG_LINES)
    }
  }

  private rejectPending(err: Error) {
    for (const p of this.pending.values()) {
      p.reject(err)
    }

    this.pending.clear()
  }

  drain() {
    this.subscribed = true

    for (const ev of this.bufferedEvents.splice(0)) {
      this.emit('event', ev)
    }

    if (this.pendingExit !== undefined) {
      const code = this.pendingExit

      this.pendingExit = undefined
      this.emit('exit', code)
    }
  }

  getLogTail(limit = 20): string {
    return this.logs.slice(-Math.max(1, limit)).join('\n')
  }

  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    if (!this.proc?.stdin || this.proc.killed || this.proc.exitCode !== null) {
      this.start()
    }

    if (!this.proc?.stdin) {
      return Promise.reject(new Error('gateway not running'))
    }

    const id = `r${++this.reqId}`

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.pending.delete(id)) {
          reject(new Error(`timeout: ${method}`))
        }
      }, REQUEST_TIMEOUT_MS)

      this.pending.set(id, {
        reject: e => {
          clearTimeout(timeout)
          reject(e)
        },
        resolve: v => {
          clearTimeout(timeout)
          resolve(v as T)
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
