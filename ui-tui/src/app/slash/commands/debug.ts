// Importing the apps barrel registers the reference apps before launch.
import '../../../sdk/apps/index.js'

import { terminalBackgroundHex } from '@hermes/ink'

import { formatBytes, performHeapDump } from '../../../lib/memory.js'
import { launchWidget } from '../../../sdk/host.js'
import { detectLightMode } from '../../../theme.js'
import { getUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

/** Slash command → SDK widget-app launch. The app owns parsing (init),
 *  keybindings (reduce), and placement (render); refusals print usage. */
const widgetCommand = (name: string, help: string): SlashCommand => ({
  help,
  name,
  run: (arg, ctx) => {
    const err = launchWidget(name, arg)

    if (err) {
      ctx.transcript.sys(err)
    }
  }
})

export const debugCommands: SlashCommand[] = [
  widgetCommand('grid-test', 'open an interactive widget-grid demo overlay'),
  widgetCommand('dialog-test', 'open a sample dialog overlay with a faked backdrop'),

  {
    help: 'write a V8 heap snapshot + memory diagnostics (see HERMES_HEAPDUMP_DIR)',
    name: 'heapdump',
    run: (_arg, ctx) => {
      const { heapUsed, rss } = process.memoryUsage()

      ctx.transcript.sys(`writing heap dump (heap ${formatBytes(heapUsed)} · rss ${formatBytes(rss)})…`)

      void performHeapDump('manual').then(r => {
        if (ctx.stale()) {
          return
        }

        if (!r.success) {
          return ctx.transcript.sys(`heapdump failed: ${r.error ?? 'unknown error'}`)
        }

        ctx.transcript.sys(`heapdump: ${r.heapPath}`)
        ctx.transcript.sys(`diagnostics: ${r.diagPath}`)
      })
    }
  },

  {
    help: 'print live theme diagnostics (background probe, light mode, palette)',
    name: 'theme-info',
    run: (_arg, ctx) => {
      const { theme } = getUiState()

      ctx.transcript.panel('Theme', [
        {
          rows: [
            ['OSC-11 background', terminalBackgroundHex() ?? '(no reply)'],
            ['HERMES_TUI_BACKGROUND', process.env.HERMES_TUI_BACKGROUND ?? '(unset)'],
            ['HERMES_TUI_THEME', process.env.HERMES_TUI_THEME ?? '(unset)'],
            ['COLORFGBG', process.env.COLORFGBG ?? '(unset)'],
            ['TERM_PROGRAM', process.env.TERM_PROGRAM ?? '(unset)'],
            ['detected mode', detectLightMode() ? 'light' : 'dark'],
            ['text', theme.color.text],
            ['completionBg', theme.color.completionBg],
            ['selectionBg', theme.color.selectionBg],
            ['statusBg', theme.color.statusBg]
          ]
        }
      ])
    }
  },

  {
    help: 'print live V8 heap + rss numbers',
    name: 'mem',
    run: (_arg, ctx) => {
      const { arrayBuffers, external, heapTotal, heapUsed, rss } = process.memoryUsage()

      ctx.transcript.panel('Memory', [
        {
          rows: [
            ['heap used', formatBytes(heapUsed)],
            ['heap total', formatBytes(heapTotal)],
            ['external', formatBytes(external)],
            ['array buffers', formatBytes(arrayBuffers)],
            ['rss', formatBytes(rss)],
            ['uptime', `${process.uptime().toFixed(0)}s`]
          ]
        }
      ])
    }
  }
]
