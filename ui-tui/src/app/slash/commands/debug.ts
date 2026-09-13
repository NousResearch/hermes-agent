// Importing the apps barrel registers the reference apps before launch.
import '../../../sdk/apps/index.js'

import { terminalBackgroundHex } from '@hermes/ink'

import { formatBytes, performHeapDump } from '../../../lib/memory.js'
import { launchWidget } from '../../../sdk/host.js'
import { getWidgetApp, listWidgetApps } from '../../../sdk/registry.js'
import {
  listWidgetSources,
  loadUserWidgets,
  loadWidgetPath,
  reloadWidgetFile,
  requestWidgetRefresh,
  unloadWidgetApp
} from '../../../sdk/userWidgets.js'
import { detectLightMode } from '../../../theme.js'
import { getOverlayState } from '../../overlayStore.js'
import { getUiState } from '../../uiStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

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

const WIDGETS_USAGE = 'Usage: /widgets (list|reload|load|unload|update) [target]'

const formatLoadResult = (
  label: string,
  r: {
    added: string[]
    errors: { file: string; message: string }[]
    loaded: string[]
    removed: string[]
  }
): string => {
  const parts = [
    r.loaded.length ? `loaded: ${r.loaded.join(', ')}` : 'no user widgets found',
    r.added.length ? `added: ${r.added.join(', ')}` : '',
    r.removed.length ? `removed: ${r.removed.join(', ')}` : '',
    ...r.errors.map(e => `${e.file}: ${e.message}`)
  ].filter(Boolean)

  return `${label} — ${parts.join(' · ')}`
}

const runWidgetsList = (ctx: SlashRunCtx): void => {
  const sources = listWidgetSources()
  const overlay = getOverlayState()

  const open = new Set([...overlay.ambient.map(a => a.appId), ...(overlay.widget ? [overlay.widget.appId] : [])])

  if (!sources.length) {
    ctx.transcript.sys('widgets (0): none registered')

    return
  }

  const lines = sources.map(s => {
    const src = s.file ?? 'built-in'
    const state = open.has(s.id) ? 'open' : 'loaded'
    const help = getWidgetApp(s.id)?.help

    return `  ${s.id}  [${state}]  ${src}${help ? `  - ${help}` : ''}`
  })

  ctx.transcript.sys(`widgets (${sources.length}):\n${lines.join('\n')}`)
}

const runWidgetsReload = async (target: string, ctx: SlashRunCtx): Promise<void> => {
  if (!target) {
    const r = await loadUserWidgets()

    if (!ctx.stale()) {
      ctx.transcript.sys(formatLoadResult('widgets reloaded', r))
    }

    return
  }

  const r = await reloadWidgetFile(target)

  if (!ctx.stale()) {
    ctx.transcript.sys(formatLoadResult(`widgets reload ${target}`, r))
  }
}

const runWidgetsLoad = async (target: string, ctx: SlashRunCtx): Promise<void> => {
  if (!target) {
    ctx.transcript.sys('usage: /widgets load <path-to.mjs>')

    return
  }

  const r = await loadWidgetPath(target)

  if (!ctx.stale()) {
    ctx.transcript.sys(formatLoadResult('widgets load', r))
  }
}

const runWidgetsUnload = (target: string, ctx: SlashRunCtx): void => {
  if (!target) {
    ctx.transcript.sys('usage: /widgets unload <id>')

    return
  }

  const r = unloadWidgetApp(target)

  ctx.transcript.sys(r.ok ? `widgets: unloaded ${target}` : `widgets: ${r.reason}`)
}

const runWidgetsUpdate = (target: string, ctx: SlashRunCtx): void => {
  if (target && !getWidgetApp(target)) {
    ctx.transcript.sys(`widgets: unknown widget app: ${target}`)

    return
  }

  const listeners = requestWidgetRefresh(target || undefined)
  const overlay = getOverlayState()

  const active = [...overlay.ambient.map(a => a.appId), ...(overlay.widget ? [overlay.widget.appId] : [])]

  const scope = target || 'all'
  const activeNote = active.length ? ` · docked: ${active.join(', ')}` : ''

  ctx.transcript.sys(
    `widgets: update ${scope} — signaled ${listeners} listener${listeners === 1 ? '' : 's'}${activeNote}`
  )
}

const WIDGET_SUBCOMMANDS: Record<string, (target: string, ctx: SlashRunCtx) => void | Promise<void>> = {
  list: (_target, ctx) => runWidgetsList(ctx),
  ls: (_target, ctx) => runWidgetsList(ctx),
  load: (target, ctx) => void runWidgetsLoad(target, ctx),
  reload: (target, ctx) => void runWidgetsReload(target, ctx),
  rl: (target, ctx) => void runWidgetsReload(target, ctx),
  rm: (target, ctx) => runWidgetsUnload(target, ctx),
  unload: (target, ctx) => runWidgetsUnload(target, ctx),
  up: (target, ctx) => runWidgetsUpdate(target, ctx),
  update: (target, ctx) => runWidgetsUpdate(target, ctx)
}

const runWidgetsFamily = (arg: string, ctx: SlashRunCtx): void => {
  const [rawSub, ...rest] = (arg ?? '').trim().split(/\s+/).filter(Boolean)
  const sub = rawSub?.toLowerCase()

  if (!sub) {
    ctx.transcript.sys(WIDGETS_USAGE)

    return
  }

  const handler = WIDGET_SUBCOMMANDS[sub]

  if (!handler) {
    ctx.transcript.sys(WIDGETS_USAGE)

    return
  }

  void handler(rest.join(' ').trim(), ctx)
}

export const debugCommands: SlashCommand[] = [
  ...widgetAppCommands,

  {
    help: 'list / reload / load / unload / update widget apps',
    name: 'widgets',
    usage: WIDGETS_USAGE,
    run: (arg, ctx) => runWidgetsFamily(arg, ctx)
  },

  // Backward-compat alias for the old flat command.
  {
    help: 'rescan $HERMES_HOME/tui-widgets and (re)register user widget apps',
    name: 'widgets-reload',
    run: (_arg, ctx) => void runWidgetsReload('', ctx)
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
