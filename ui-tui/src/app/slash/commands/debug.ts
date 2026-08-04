// Importing the apps barrel registers the reference apps before launch.
import '../../../sdk/apps/index.js'

import { terminalBackgroundHex } from '@hermes/ink'

import { translate } from '../../../i18n/index.js'
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
        const locale = ctx.ui.locale

        const parts = [
          loaded.length
            ? translate(locale, 'widget.reload.loaded', { names: loaded.join(', ') })
            : translate(locale, 'widget.reload.none'),
          ...errors.map(e => `${e.file}: ${e.message}`)
        ]

        ctx.transcript.sys(translate(locale, 'widget.reload.result', { result: parts.join(' · ') }))
      })
    }
  },

  {
    name: 'heapdump',
    run: (_arg, ctx) => {
      const { heapUsed, rss } = process.memoryUsage()
      const locale = ctx.ui.locale

      ctx.transcript.sys(
        translate(locale, 'debug.writingHeapDump', { heap: formatBytes(heapUsed), rss: formatBytes(rss) })
      )

      void performHeapDump('manual').then(r => {
        if (ctx.stale()) {
          return
        }

        if (!r.success) {
          return ctx.transcript.sys(
            translate(locale, 'debug.heapdumpFailed', {
              error: r.error ?? translate(locale, 'common.unknownError')
            })
          )
        }

        ctx.transcript.sys(translate(locale, 'debug.heapdumpPath', { path: r.heapPath ?? '' }))
        ctx.transcript.sys(translate(locale, 'debug.diagPath', { path: r.diagPath ?? '' }))
      })
    }
  },

  {
    help: 'print live theme diagnostics (background probe, light mode, palette)',
    name: 'theme-info',
    run: (_arg, ctx) => {
      const { theme } = getUiState()
      const locale = ctx.ui.locale

      ctx.transcript.panel(translate(locale, 'debug.theme.title'), [
        {
          rows: [
            [translate(locale, 'debug.theme.oscBackground'), terminalBackgroundHex() ?? translate(locale, 'common.noReply')],
            ['HERMES_TUI_BACKGROUND', process.env.HERMES_TUI_BACKGROUND ?? translate(locale, 'common.unset')],
            ['HERMES_TUI_THEME', process.env.HERMES_TUI_THEME ?? translate(locale, 'common.unset')],
            ['COLORFGBG', process.env.COLORFGBG ?? translate(locale, 'common.unset')],
            ['TERM_PROGRAM', process.env.TERM_PROGRAM ?? translate(locale, 'common.unset')],
            [
              translate(locale, 'debug.theme.detectedMode'),
              translate(locale, detectLightMode() ? 'theme.light' : 'theme.dark')
            ],
            [translate(locale, 'debug.theme.text'), theme.color.text],
            [translate(locale, 'debug.theme.completionBackground'), theme.color.completionBg],
            [translate(locale, 'debug.theme.selectionBackground'), theme.color.selectionBg],
            [translate(locale, 'debug.theme.statusBackground'), theme.color.statusBg]
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
