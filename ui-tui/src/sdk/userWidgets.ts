import { watch } from 'fs'
import { access, readdir, readFile, rm, writeFile } from 'fs/promises'
import { homedir, tmpdir } from 'os'
import { basename, dirname, isAbsolute, join, resolve } from 'path'
import { pathToFileURL } from 'url'

import { Box, Text } from '@hermes/ink'
import * as React from 'react'

import { Accordion } from '../components/accordion.js'
import { Shimmer, ShimmerRows, useShimmerPhase } from '../components/loaders.js'
import { Dialog, Overlay } from '../components/overlay.js'
import { GridAreas, WidgetGrid } from '../components/widgetGrid.js'
import { gauge, hbars, sparkline, sparkRows } from '../lib/charts.js'
import { recordParentLifecycle } from '../lib/parentLog.js'

import { closeWidget, openWidget, updateWidget } from './host.js'
import { defineWidgetApp, getWidgetApp, listWidgetApps, removeWidgetApp } from './registry.js'
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

const widgetsDir = () => join(process.env.HERMES_HOME?.trim() || join(homedir(), '.hermes'), 'tui-widgets')

export interface UserWidgetLoadResult {
  /** App ids newly registered by this scan. */
  added: string[]
  errors: { file: string; message: string }[]
  loaded: string[]
  /** App ids unregistered because their file disappeared. */
  removed: string[]
}

export interface WidgetSource {
  file: null | string
  id: string
}

/** Which app ids each user file registered — the delete-sync source of
 *  truth (file gone on the next scan ⇒ its apps unregister). Absolute-path
 *  keys are external loads (via `/widgets load`) and skip delete-sync. */
const fileApps = new Map<string, string[]>()

const listeners = new Set<(result: UserWidgetLoadResult) => void>()

/** Manual refresh bus for `/widgets update`. Widgets subscribe in register()
 *  or inside a component effect. `id` is null when every widget should refresh. */
type WidgetRefreshListener = (id: null | string) => void
const refreshListeners = new Set<WidgetRefreshListener>()

export function onWidgetRefresh(listener: WidgetRefreshListener): () => void {
  refreshListeners.add(listener)

  return () => {
    refreshListeners.delete(listener)
  }
}

/** Notify refresh subscribers. Returns the listener count at fire time. */
export function requestWidgetRefresh(id?: string): number {
  const target = id?.trim() ? id.trim() : null

  for (const listener of refreshListeners) {
    listener(target)
  }

  return refreshListeners.size
}

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
  onWidgetRefresh,
  openWidget,
  requestWidgetRefresh,
  sparkRows,
  sparkline,
  updateWidget,
  useShimmerPhase
} as const

export type WidgetSdk = typeof widgetSdk

/** Subscribe to scan results — the app layer announces loads in the
 *  transcript so a hot-loaded widget is VISIBLY live (silent success is
 *  indistinguishable from failure). */
export function onUserWidgets(listener: (result: UserWidgetLoadResult) => void): () => void {
  listeners.add(listener)

  return () => listeners.delete(listener)
}

const emptyResult = (): UserWidgetLoadResult => ({ added: [], errors: [], loaded: [], removed: [] })

const emitLoadResult = (result: UserWidgetLoadResult): UserWidgetLoadResult => {
  if (result.added.length) {
    recordParentLifecycle(`user widgets registered: ${result.added.join(', ')}`)
  }

  for (const listener of listeners) {
    listener(result)
  }

  return result
}

/** Import one on-disk .mjs via a unique temp path so ESM loaders (Node query
 *  bust and Vitest/vite-node) always re-execute after edits. */
async function importRegister(absPath: string, fileKey: string, result: UserWidgetLoadResult): Promise<void> {
  const before = new Set(listWidgetApps().map(app => app.id))
  const previous = fileApps.get(fileKey) ?? []

  const tmp = join(
    tmpdir(),
    `hermes-widget-${Date.now()}-${Math.random().toString(36).slice(2, 10)}.mjs`
  )

  try {
    const src = await readFile(absPath, 'utf8')

    await writeFile(tmp, src)

    const mod = (await import(pathToFileURL(tmp).href)) as {
      default?: (sdk: WidgetSdk) => void
    }

    if (typeof mod.default !== 'function') {
      throw new Error('default export must be register(sdk)')
    }

    mod.default(widgetSdk)
    result.loaded.push(fileKey)

    const added = listWidgetApps()
      .map(app => app.id)
      .filter(id => !before.has(id))

    // First registration of this file, or new ids from a multi-app file.
    if (added.length) {
      fileApps.set(fileKey, [...new Set([...previous, ...added])])
      result.added.push(...added)

      return
    }

    // Re-import of existing ids (hot reload): keep prior attribution.
    if (previous.length) {
      fileApps.set(fileKey, previous)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    result.errors.push({ file: fileKey, message })
    recordParentLifecycle(`user widget ${fileKey} failed to load: ${message}`)
  } finally {
    await rm(tmp, { force: true }).catch(() => {})
  }
}

/** Resolve a user target (app id, basename, or basename.mjs) to a fileApps key. */
export function findWidgetFile(target: string): null | string {
  const needle = target.trim()

  if (!needle) {
    return null
  }

  if (fileApps.has(needle)) {
    return needle
  }

  if (!needle.endsWith('.mjs') && fileApps.has(`${needle}.mjs`)) {
    return `${needle}.mjs`
  }

  const base = basename(needle)

  for (const [file, ids] of fileApps) {
    if (ids.includes(needle)) {
      return file
    }

    if (file === base || basename(file) === base) {
      return file
    }

    if (basename(file, '.mjs') === needle || basename(file, '.mjs') === base.replace(/\.mjs$/, '')) {
      return file
    }
  }

  return null
}

/** Catalog entry: every registered app plus its user-file source (if any). */
export function listWidgetSources(): WidgetSource[] {
  const byId = new Map<string, string>()

  for (const [file, ids] of fileApps) {
    for (const id of ids) {
      byId.set(id, file)
    }
  }

  return listWidgetApps().map(app => ({ file: byId.get(app.id) ?? null, id: app.id }))
}

/** Hot-reload one user widget file (or the file that owns `target` id). */
export async function reloadWidgetFile(target: string, dir = widgetsDir()): Promise<UserWidgetLoadResult> {
  const result = emptyResult()
  const key = findWidgetFile(target)
  let abs: null | string = null
  let fileKey: null | string = key

  if (key && isAbsolute(key)) {
    abs = key
  } else if (key) {
    abs = join(dir, key)
  } else {
    // Allow reload by on-disk basename before it is attributed (first load).
    const candidate = target.endsWith('.mjs') ? target : `${target}.mjs`
    const path = isAbsolute(target) ? resolve(target) : join(dir, basename(candidate))

    try {
      await access(path)
      abs = path
      fileKey = isAbsolute(target) ? path : basename(path)
    } catch {
      result.errors.push({ file: target, message: `no widget file found for '${target}'` })

      return emitLoadResult(result)
    }
  }

  if (!abs || !fileKey) {
    result.errors.push({ file: target, message: `no widget file found for '${target}'` })

    return emitLoadResult(result)
  }

  // Drop prior ids owned solely by this file so a rename does not leave ghosts.
  const previous = fileApps.get(fileKey) ?? []

  for (const id of previous) {
    const claimedElsewhere = [...fileApps.entries()].some(([k, ids]) => k !== fileKey && ids.includes(id))

    if (!claimedElsewhere) {
      closeWidget(id)
      removeWidgetApp(id)
      result.removed.push(id)
    }
  }

  fileApps.delete(fileKey)
  await importRegister(abs, fileKey, result)

  return emitLoadResult(result)
}

/** Load a .mjs widget from an arbitrary path (dev copy, backup, template). */
export async function loadWidgetPath(path: string): Promise<UserWidgetLoadResult> {
  const result = emptyResult()
  const abs = resolve(path.trim())

  try {
    await access(abs)
  } catch {
    result.errors.push({ file: abs, message: 'file not found' })

    return emitLoadResult(result)
  }

  if (!abs.endsWith('.mjs')) {
    result.errors.push({ file: abs, message: 'widget files must end in .mjs' })

    return emitLoadResult(result)
  }

  await importRegister(abs, abs, result)

  return emitLoadResult(result)
}

/** Unregister one app, dismiss it from the dock/modal, and drop file attribution. */
export function unloadWidgetApp(id: string): { ok: boolean; reason?: string } {
  const appId = id.trim()

  if (!appId) {
    return { ok: false, reason: 'missing app id' }
  }

  if (!getWidgetApp(appId)) {
    return { ok: false, reason: `unknown widget app: ${appId}` }
  }

  closeWidget(appId)
  removeWidgetApp(appId)

  for (const [file, ids] of [...fileApps.entries()]) {
    if (!ids.includes(appId)) {
      continue
    }

    const next = ids.filter(x => x !== appId)

    if (next.length) {
      fileApps.set(file, next)
    } else {
      fileApps.delete(file)
    }
  }

  return { ok: true }
}

/** Scan + import + register, diffing the registry per file. Cache-busted so
 *  edits reload without restarting the TUI (last-writer-wins shadows stale
 *  definitions). Files that vanished unregister their apps. */
export async function loadUserWidgets(dir = widgetsDir()): Promise<UserWidgetLoadResult> {
  const result = emptyResult()

  let files: string[] = []

  try {
    files = (await readdir(dir)).filter(f => f.endsWith('.mjs')).sort()
  } catch {
    // No directory: fall through so previously-loaded files still delete-sync.
  }

  for (const [file, ids] of fileApps) {
    // External loads use absolute keys and persist until `/widgets unload`.
    if (isAbsolute(file)) {
      continue
    }

    if (!files.includes(file)) {
      fileApps.delete(file)

      for (const id of ids) {
        closeWidget(id)

        if (removeWidgetApp(id)) {
          result.removed.push(id)
        }
      }
    }
  }

  for (const file of files) {
    await importRegister(join(dir, file), file, result)
  }

  return emitLoadResult(result)
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
