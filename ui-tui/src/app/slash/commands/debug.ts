import { formatBytes, performHeapDump } from '../../../lib/memory.js'
import { translate } from '../../../i18n/index.js'
import type { SlashCommand } from '../types.js'

export const debugCommands: SlashCommand[] = [
  {
    help: 'write a V8 heap snapshot + memory diagnostics (see HERMES_HEAPDUMP_DIR)',
    name: 'heapdump',
    run: (_arg, ctx) => {
      const { heapUsed, rss } = process.memoryUsage()
      const locale = ctx.ui.locale

      ctx.transcript.sys(translate(locale, 'debug.writingHeapDump', { heap: formatBytes(heapUsed), rss: formatBytes(rss) }))

      void performHeapDump('manual').then(r => {
        if (ctx.stale()) {
          return
        }

        if (!r.success) {
          return ctx.transcript.sys(translate(locale, 'debug.heapdumpFailed', { error: r.error ?? 'unknown error' }))
        }

        ctx.transcript.sys(translate(locale, 'debug.heapdumpPath', { path: r.heapPath ?? '' }))
        ctx.transcript.sys(translate(locale, 'debug.diagPath', { path: r.diagPath ?? '' }))
      })
    }
  },

  {
    help: 'print live V8 heap + rss numbers',
    name: 'mem',
    run: (_arg, ctx) => {
      const { arrayBuffers, external, heapTotal, heapUsed, rss } = process.memoryUsage()
      const locale = ctx.ui.locale

      ctx.transcript.panel(translate(locale, 'section.memory'), [
        {
          rows: [
            [translate(locale, 'debug.heapUsed'), formatBytes(heapUsed)],
            [translate(locale, 'debug.heapTotal'), formatBytes(heapTotal)],
            [translate(locale, 'debug.external'), formatBytes(external)],
            [translate(locale, 'debug.arrayBuffers'), formatBytes(arrayBuffers)],
            [translate(locale, 'debug.rss'), formatBytes(rss)],
            [translate(locale, 'debug.uptime'), `${process.uptime().toFixed(0)}s`]
          ]
        }
      ])
    }
  }
]
