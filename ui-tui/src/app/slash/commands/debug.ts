import { terminalBackgroundHex } from '@hermes/ink'

import { formatBytes, performHeapDump } from '../../../lib/memory.js'
import { detectLightMode } from '../../../theme.js'
import type { DialogState } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { getUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

const GRID_TEST_USAGE = 'usage: /grid-test [cols]x[rows]  ·  /grid-test [cols] [rows]  ·  /grid-test streams'
const GRID_TEST_MAX_SIZE = 12

const DIALOG_TEST_ZONES = new Set<DialogState['zone']>([
  'bottom',
  'bottom-left',
  'bottom-right',
  'center',
  'left',
  'right',
  'top',
  'top-left',
  'top-right'
])

const DIALOG_TEST_USAGE = `usage: /dialog-test [zone]   zones: ${[...DIALOG_TEST_ZONES].join(', ')}`

const clampGridSize = (value: number, fallback: number) => {
  if (!Number.isFinite(value)) {
    return fallback
  }

  return Math.max(1, Math.min(GRID_TEST_MAX_SIZE, Math.trunc(value)))
}

const parseGridTestSize = (arg: string) => {
  const trimmed = arg.trim()

  if (!trimmed) {
    return { cols: 4, rows: 3 }
  }

  const grid = trimmed.match(/^(\d+)\s*x\s*(\d+)$/i)

  if (grid) {
    return { cols: clampGridSize(Number(grid[1]), 4), rows: clampGridSize(Number(grid[2]), 3) }
  }

  const [cols, rows, ...rest] = trimmed.split(/\s+/)

  if (rest.length || !cols || !rows || Number.isNaN(Number(cols)) || Number.isNaN(Number(rows))) {
    return null
  }

  return { cols: clampGridSize(Number(cols), 4), rows: clampGridSize(Number(rows), 3) }
}

export const debugCommands: SlashCommand[] = [
  {
    help: 'open an interactive widget-grid demo overlay',
    name: 'grid-test',
    run: (arg, ctx) => {
      const streams = arg.trim().toLowerCase() === 'streams'
      const size = streams ? { cols: 4, rows: 3 } : parseGridTestSize(arg)

      if (!size) {
        return ctx.transcript.sys(GRID_TEST_USAGE)
      }

      patchOverlayState({
        gridTest: {
          activeCol: 0,
          activeRow: 0,
          areas: false,
          cols: size.cols,
          gap: null,
          nested: false,
          paddingX: null,
          rows: size.rows,
          streamFocus: 0,
          streamMain: 0,
          streams,
          zoomed: false
        }
      })
    }
  },

  {
    help: 'open a sample dialog overlay with a faked backdrop',
    name: 'dialog-test',
    run: (arg, ctx) => {
      const trimmed = arg.trim().toLowerCase()
      const zone = (trimmed || 'center') as DialogState['zone']

      if (!DIALOG_TEST_ZONES.has(zone)) {
        return ctx.transcript.sys(DIALOG_TEST_USAGE)
      }

      patchOverlayState({
        dialog: {
          body: [
            'This is a viewport-level overlay with a backdrop.',
            '',
            `Zone: ${zone}`,
            'Try: /dialog-test top-right · bottom · left · ...'
          ].join('\n'),
          hint: 'Esc/q/Enter close · Ctrl+C close',
          title: 'Dialog primitive',
          zone
        }
      })
    }
  },

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
