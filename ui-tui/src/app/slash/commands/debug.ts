// Importing the apps barrel registers the reference apps before launch.
import '../../../sdk/apps/index.js'

import { terminalBackgroundHex } from '@hermes/ink'

import { t } from '../../../i18n/runtime.js'
import { formatBytes, performHeapDump } from '../../../lib/memory.js'
import { launchWidget } from '../../../sdk/host.js'
import { listWidgetApps } from '../../../sdk/registry.js'
import { loadUserWidgets } from '../../../sdk/userWidgets.js'
import { detectLightMode } from '../../../theme.js'
import { getUiState } from '../../uiStore.js'
import type { SlashCommand } from '../types.js'

/** The registry IS the catalog: every registered widget app becomes a slash
 *  command carrying the app's own help/usage — nothing hardcoded per app.
 *  The app owns parsing (init), keybindings (reduce), placement (render). */
export const widgetAppCommands: SlashCommand[] = listWidgetApps().map(app => ({
  help: app.help,
  name: app.id,
  run: (arg, ctx) => {
    const err = launchWidget(app.id, arg)

    if (err) {
      ctx.transcript.sys(err)
    }
  }
}))

export const debugCommands: SlashCommand[] = [
  ...widgetAppCommands,

  {
    help: 'rescan $HERMES_HOME/tui-widgets and (re)register user widget apps',
    name: 'widgets-reload',
    run: (_arg, ctx) => {
      void loadUserWidgets().then(({ errors, loaded }) => {
        const parts = [
          loaded.length
            ? t('slashCmd.debug.widgetsReload.loaded', loaded.join(', '))
            : t('slashCmd.debug.widgetsReload.none'),
          ...errors.map(e => `${e.file}: ${e.message}`)
        ]

        ctx.transcript.sys(t('slashCmd.debug.widgetsReload.summary', parts.join(' · ')))
      })
    }
  },

  {
    help: 'write a V8 heap snapshot + memory diagnostics (see HERMES_HEAPDUMP_DIR)',
    name: 'heapdump',
    run: (_arg, ctx) => {
      const { heapUsed, rss } = process.memoryUsage()

      ctx.transcript.sys(t('slashCmd.debug.heapdump.writing', formatBytes(heapUsed), formatBytes(rss)))

      void performHeapDump('manual').then(r => {
        if (ctx.stale()) {
          return
        }

        if (!r.success) {
          return ctx.transcript.sys(
            t('slashCmd.debug.heapdump.failed', r.error ?? t('slashCmd.debug.heapdump.unknownError'))
          )
        }

        ctx.transcript.sys(t('slashCmd.debug.heapdump.heapPath', r.heapPath))
        ctx.transcript.sys(t('slashCmd.debug.heapdump.diagPath', r.diagPath))
      })
    }
  },

  {
    help: 'print live theme diagnostics (background probe, light mode, palette)',
    name: 'theme-info',
    run: (_arg, ctx) => {
      const { theme } = getUiState()

      const unset = t('slashCmd.debug.themeInfo.unset')

      ctx.transcript.panel(t('slashCmd.debug.themeInfo.panelTitle'), [
        {
          rows: [
            [
              t('slashCmd.debug.themeInfo.osc11Background'),
              terminalBackgroundHex() ?? t('slashCmd.debug.themeInfo.noReply')
            ],
            ['HERMES_TUI_BACKGROUND', process.env.HERMES_TUI_BACKGROUND ?? unset],
            ['HERMES_TUI_THEME', process.env.HERMES_TUI_THEME ?? unset],
            ['COLORFGBG', process.env.COLORFGBG ?? unset],
            ['TERM_PROGRAM', process.env.TERM_PROGRAM ?? unset],
            [
              t('slashCmd.debug.themeInfo.detectedMode'),
              detectLightMode() ? t('slashCmd.debug.themeInfo.light') : t('slashCmd.debug.themeInfo.dark')
            ],
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

      ctx.transcript.panel(t('slashCmd.debug.mem.panelTitle'), [
        {
          rows: [
            [t('slashCmd.debug.mem.heapUsed'), formatBytes(heapUsed)],
            [t('slashCmd.debug.mem.heapTotal'), formatBytes(heapTotal)],
            [t('slashCmd.debug.mem.external'), formatBytes(external)],
            [t('slashCmd.debug.mem.arrayBuffers'), formatBytes(arrayBuffers)],
            [t('slashCmd.debug.mem.rss'), formatBytes(rss)],
            [t('slashCmd.debug.mem.uptime'), t('slashCmd.debug.mem.seconds', process.uptime().toFixed(0))]
          ]
        }
      ])
    }
  }
]
