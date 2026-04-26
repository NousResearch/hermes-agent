#!/usr/bin/env -S node --max-old-space-size=8192 --expose-gc
import { GatewayClient } from './gatewayClient.js'
import { setupGracefulExit } from './lib/gracefulExit.js'
import { formatBytes, type HeapDumpResult, performHeapDump } from './lib/memory.js'
import { type MemorySnapshot, startMemoryMonitor } from './lib/memoryMonitor.js'

if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY')
  process.exit(0)
}

const gw = new GatewayClient()

gw.start()

const dumpNotice = (snap: MemorySnapshot, dump: HeapDumpResult | null) =>
  `hermes-tui: ${snap.level} memory (${formatBytes(snap.heapUsed)}) — auto heap dump → ${dump?.heapPath ?? '(failed)'}\n`

setupGracefulExit({
  cleanups: [() => gw.kill()],
  onError: (scope, err) => {
    const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err)

    process.stderr.write(`hermes-tui ${scope}: ${message.slice(0, 2000)}\n`)
  },
  onSignal: signal => process.stderr.write(`hermes-tui: received ${signal}\n`)
})

const stopMemoryMonitor = startMemoryMonitor({
  onCritical: (snap, dump) => {
    process.stderr.write(dumpNotice(snap, dump))
    process.stderr.write('hermes-tui: exiting to avoid OOM; restart to recover\n')
    process.exit(137)
  },
  onHigh: (snap, dump) => process.stderr.write(dumpNotice(snap, dump))
})

if (process.env.HERMES_HEAPDUMP_ON_START === '1') {
  void performHeapDump('manual')
}

process.on('beforeExit', () => stopMemoryMonitor())

const [{ render }, { App }, { logFrameEvent }, { trackFrame }] = await Promise.all([
  import('@hermes/ink'),
  import('./app.js'),
  import('./lib/perfPane.js'),
  import('./lib/fpsStore.js')
])

// Compose onFrame from the two opt-in consumers (HERMES_DEV_PERF and
// HERMES_TUI_FPS).  Each is undefined when its env flag is off; we only
// attach onFrame at all when at least one is on, so ink skips the
// handler entirely in the default disabled case.
type InkFrameEvent = { durationMs: number }
type OnFrame = (event: InkFrameEvent) => void

const onFrame: OnFrame | undefined =
  logFrameEvent || trackFrame
    ? (event: InkFrameEvent) => {
        logFrameEvent?.(event as Parameters<NonNullable<typeof logFrameEvent>>[0])
        trackFrame?.(event.durationMs)
      }
    : undefined

render(<App gw={gw} />, { exitOnCtrlC: false, onFrame })
