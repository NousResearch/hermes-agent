#!/usr/bin/env -S node --max-old-space-size=8192 --expose-gc
// Must be first import. If the user explicitly opts into truecolor, this
// nudges chalk / supports-color before either package is initialized.
import './lib/forceTruecolor.js'

import type { FrameEvent } from '@hermes/ink'

import { TERMUX_TUI_MODE } from './config/env.js'
import { GatewayClient } from './gatewayClient.js'
import { setupGracefulExit } from './lib/gracefulExit.js'
import { formatBytes, type HeapDumpResult, performHeapDump } from './lib/memory.js'
import { type MemorySnapshot, startMemoryMonitor } from './lib/memoryMonitor.js'
import { openExternalUrl } from './lib/openExternalUrl.js'
import { recordParentLifecycle } from './lib/parentLog.js'
import { clampStdoutDimensions } from './lib/terminalDimensions.js'
import { resetTerminalModes } from './lib/terminalModes.js'

if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY')
  process.exit(0)
}

// Some hosts (notably WSL) report bogus window sizes such as 131072x1. Clamp
// `process.stdout.columns`/`rows` at the source so the Ink renderer, its
// resize handler, and every component read see sane values. Must run before
// `ink.render` constructs the renderer.
clampStdoutDimensions()

// Start from a clean slate. If a previous TUI crashed or was kill -9'd, the
// terminal tab can still have mouse/focus/paste modes enabled.
resetTerminalModes()

// Desktop terminals benefit from a clean startup slate because the TUI usually
// runs in AlternateScreen. On Termux we keep prior output intact so users can
// review/copy earlier assistant replies after reopening the app.
if (TERMUX_TUI_MODE) {
  process.stdout.write('\n')
} else {
  process.stdout.write('\x1b[2J\x1b[H\x1b[3J')
}

const gw = new GatewayClient()

gw.start()

const dumpNotice = (snap: MemorySnapshot, dump: HeapDumpResult | null) =>
  `hermes-tui: ${snap.level} memory (${formatBytes(snap.heapUsed)}) — auto heap dump → ${dump?.heapPath ?? '(failed)'}\n`

setupGracefulExit({
  cleanups: [
    () => {
      resetTerminalModes()

      return gw.kill('graceful-exit-cleanup')
    }
  ],
  onError: (scope, err) => {
    const message = err instanceof Error ? `${err.name}: ${err.message}\n${err.stack ?? ''}` : String(err)

    recordParentLifecycle(`${scope}: ${message.split('\n')[0]?.slice(0, 400) ?? ''}`)
    process.stderr.write(`hermes-tui lifecycle ${scope}: ${message.slice(0, 2000)}\n`)
  },
  onSignal: signal => {
    // The next line in the crash log is the child's `=== SIGTERM received ===`
    // (gw.kill forwards SIGTERM regardless of which signal hit us) — this is
    // what tells SIGHUP (terminal/SSH dropped) apart from a real SIGTERM.
    recordParentLifecycle(`graceful-exit received signal=${signal} → killing gateway`)
    resetTerminalModes()
    process.stderr.write(`hermes-tui lifecycle: received ${signal}\n`)
  }
})

const stopMemoryMonitor = startMemoryMonitor({
  onCritical: (snap, dump) => {
    // process.exit(137) closes the child's stdin → the gateway logs a clean
    // EOF, NOT SIGTERM. Recording it here is the only way a crash report can
    // attribute a death to Node OOM rather than a signal-driven kill.
    recordParentLifecycle(`memory-critical process.exit(137) heap=${formatBytes(snap.heapUsed)} rss=${formatBytes(snap.rss)} dump=${dump?.heapPath ?? 'failed'}`)
    resetTerminalModes()
    process.stderr.write(`hermes-tui lifecycle: memory critical exit heap=${formatBytes(snap.heapUsed)} rss=${formatBytes(snap.rss)}\n`)
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

const [ink, { App }, { logFrameEvent }, { trackFrame }] = await Promise.all([
  import('@hermes/ink'),
  import('./app.js'),
  import('./lib/perfPane.js'),
  import('./lib/fpsStore.js')
])

// Both consumers are undefined when their env flags are off; only attach
// onFrame when at least one is on so ink skips timing in the default case.
const onFrame =
  logFrameEvent || trackFrame
    ? (event: FrameEvent) => {
        logFrameEvent?.(event)
        trackFrame?.(event.durationMs)
      }
    : undefined

ink.render(<App gw={gw} />, {
  exitOnCtrlC: false,
  onFrame,
  // Open URLs in the user's default browser when a link cell is clicked.
  // The TUI's mouse tracking captures click events before Terminal.app's
  // own URL detection can fire, so without this hook clicks on `<Link>`
  // do nothing in any terminal where mouseTracking is on.
  onHyperlinkClick: url => {
    openExternalUrl(url)
  }
})
