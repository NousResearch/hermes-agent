import { watch } from 'fs'
import { readdir } from 'fs/promises'
import { homedir } from 'os'
import { dirname, join } from 'path'
import { pathToFileURL } from 'url'

import { Box, Text } from '@hermes/ink'
import * as React from 'react'

import { Accordion } from '../components/accordion.js'
import { Shimmer, ShimmerRows, useShimmerPhase } from '../components/loaders.js'
import { Dialog, Overlay } from '../components/overlay.js'
import { GridAreas, WidgetGrid } from '../components/widgetGrid.js'
import { gauge, hbars, sparkline, sparkRows } from '../lib/charts.js'
import { recordParentLifecycle } from '../lib/parentLog.js'

import { openWidget, updateWidget } from './host.js'
import { defineWidgetApp, listWidgetApps, removeWidgetApp } from './registry.js'
import { isCtrl } from './types.js'

/**
 * User widget apps — Hermes authors its own TUI widgets, mirroring the
 * Python plugin contract: drop `<name>.mjs` into `$HERMES_HOME/tui-widgets/`,
 * default-export `register(sdk)`, and the app surfaces in `/` completions
 * and dispatch automatically (the registry is the catalog). Plain ESM so the
 * production bundle can import it — no bundler, no JSX; `sdk.h` is
 * React.createElement.
 *
 * Trust model matches `~/.hermes/plugins/`: files under HERMES_HOME execute
 * with the TUI's privileges. Load errors log and skip — a broken widget
 * never takes the TUI down.
 */

/** Everything a user widget may touch, passed INTO its register() — user
 *  files have no resolvable import path to the bundle. */
export const widgetSdk = {
  Accordion,
  Box,
  Dialog,
  GridAreas,
  Overlay,
  React,
  Shimmer,
  ShimmerRows,
  Text,
  WidgetGrid,
  defineWidgetApp,
  gauge,
  h: React.createElement,
  hbars,
  isCtrl,
  openWidget,
  sparkRows,
  sparkline,
  updateWidget,
  useShimmerPhase
} as const

export type WidgetSdk = typeof widgetSdk

const widgetsDir = () => join(process.env.HERMES_HOME?.trim() || join(homedir(), '.hermes'), 'tui-widgets')

export interface UserWidgetLoadResult {
  /** App ids newly registered by this scan. */
  added: string[]
  errors: { file: string; message: string }[]
  loaded: string[]
  /** App ids unregistered because their file disappeared. */
  removed: string[]
}

/** Which app ids each user file registered — the delete-sync source of
 *  truth (file gone on the next scan ⇒ its apps unregister). */
const fileApps = new Map<string, string[]>()

const listeners = new Set<(result: UserWidgetLoadResult) => void>()

/** Subscribe to scan results — the app layer announces loads in the
 *  transcript so a hot-loaded widget is VISIBLY live (silent success is
 *  indistinguishable from failure). */
export function onUserWidgets(listener: (result: UserWidgetLoadResult) => void): () => void {
  listeners.add(listener)

  return () => listeners.delete(listener)
}

/** Scan + import + register, diffing the registry per file. Cache-busted so
 *  edits reload without restarting the TUI (last-writer-wins shadows stale
 *  definitions). Files that vanished unregister their apps. */
export async function loadUserWidgets(dir = widgetsDir()): Promise<UserWidgetLoadResult> {
  const result: UserWidgetLoadResult = { added: [], errors: [], loaded: [], removed: [] }

  let files: string[] = []

  try {
    files = (await readdir(dir)).filter(f => f.endsWith('.mjs')).sort()
  } catch {
    // No directory: fall through so previously-loaded files still delete-sync.
  }

  for (const [file, ids] of fileApps) {
    if (!files.includes(file)) {
      fileApps.delete(file)

      for (const id of ids) {
        if (removeWidgetApp(id)) {
          result.removed.push(id)
        }
      }
    }
  }

  for (const file of files) {
    const before = new Set(listWidgetApps().map(app => app.id))

    try {
      const mod = (await import(`${pathToFileURL(join(dir, file)).href}?t=${Date.now()}`)) as {
        default?: (sdk: WidgetSdk) => void
      }

      if (typeof mod.default !== 'function') {
        throw new Error('default export must be register(sdk)')
      }

      mod.default(widgetSdk)
      result.loaded.push(file)

      const ids = listWidgetApps()
        .map(app => app.id)
        .filter(id => !before.has(id))

      // Re-registrations of existing ids keep their prior file attribution.
      if (ids.length) {
        fileApps.set(file, ids)
        result.added.push(...ids)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)

      result.errors.push({ file, message })
      recordParentLifecycle(`user widget ${file} failed to load: ${message}`)
    }
  }

  if (result.added.length) {
    recordParentLifecycle(`user widgets registered: ${result.added.join(', ')}`)
  }

  for (const listener of listeners) {
    listener(result)
  }

  return result
}

let watching = false

/** Generative-UI hot loading: watch the widgets directory and re-scan on
 *  every change, so a widget Hermes writes appears within ~a second — no
 *  `/widgets-reload`, no restart (GUI parity). Debounced (editors and
 *  write_file emit bursts); polls until the directory exists so the very
 *  first widget ever written also hot-loads. */
export function watchUserWidgets(dir = widgetsDir()): void {
  if (watching) {
    return
  }

  watching = true

  let timer: NodeJS.Timeout | undefined

  const attach = () => {
    try {
      const watcher = watch(dir, () => {
        clearTimeout(timer)
        timer = setTimeout(() => void loadUserWidgets(dir), 300)
        timer.unref?.()
      })

      watcher.unref?.()

      return true
    } catch {
      return false // directory doesn't exist yet
    }
  }

  if (!attach()) {
    // Event-driven first-creation: watch the PARENT for the widgets dir to
    // appear, attach + scan the instant it does. The very first widget a
    // user (or Hermes) ever writes must hot-load too — a 10s poll here read
    // as "requires a restart" in live use.
    try {
      const parent = watch(dirname(dir), () => {
        if (attach()) {
          parent.close()
          void loadUserWidgets(dir)
        }
      })

      parent.unref?.()
    } catch {
      const poll = setInterval(() => {
        if (attach()) {
          clearInterval(poll)
          void loadUserWidgets(dir)
        }
      }, 2_000)

      poll.unref?.()
    }
  }
}
