import { realpathSync } from 'node:fs'
import { normalize } from 'node:path'

import { atom } from 'nanostores'

import type { ComposerToken } from './interfaces.js'

export interface SessionTabDraft {
  input: string
  inputBuf: string[]
  tokens: ComposerToken[]
}

export interface SessionTab {
  id: string
  storedId: string
  cwd: string
  title: string
  aliases: string[]
  running: boolean
  blocked?: boolean
  draft: SessionTabDraft
}

export interface SessionTabView {
  id: string
  storedId?: string
  cwd: string
  title?: string
  running?: boolean
  blocked?: boolean
  requestedId?: string
}

export type SessionTabAction =
  | { kind: 'new' }
  | { kind: 'close'; id?: string }
  | { kind: 'select'; id: string }
  | { kind: 'index'; index: number }
  | { kind: 'cycle'; direction: 1 | -1 }

/** Open views in THIS terminal process, not every session on the gateway.
 * History remains durable in the owner; closing a tab never deletes history. */
export const $sessionTabs = atom<SessionTab[]>([])

export const emptyTabDraft = (): SessionTabDraft => ({ input: '', inputBuf: [], tokens: [] })
export const hasTabDraft = (draft: SessionTabDraft) =>
  Boolean(draft.input || draft.inputBuf.length || draft.tokens.length)

export const tabDirectory = (cwd?: string | null) => {
  if (!cwd) {
    return ''
  }

  try {
    return realpathSync(cwd)
  } catch {
    return normalize(cwd).replace(/[/\\]+$/, '') || '/'
  }
}

export const tabsInDirectory = (tabs: readonly SessionTab[], cwd?: string | null) =>
  tabs.filter(tab => Boolean(tabDirectory(cwd)) && tabDirectory(tab.cwd) === tabDirectory(cwd))

export const findSessionTab = (id: string | null | undefined) =>
  id ? $sessionTabs.get().find(tab => tab.id === id || tab.storedId === id || tab.aliases.includes(id)) : undefined

export function openSessionTab(view: SessionTabView) {
  const aliases = [...new Set([view.id, view.storedId, view.requestedId].filter((id): id is string => Boolean(id)))]
  const previous = $sessionTabs.get()

  const existing = previous.find(tab =>
    aliases.some(id => tab.id === id || tab.storedId === id || tab.aliases.includes(id))
  )

  const tab: SessionTab = {
    id: view.id,
    storedId: view.storedId || view.id,
    cwd: view.cwd,
    title: view.title || existing?.title || '',
    aliases: [...new Set([...(existing?.aliases || []), ...aliases])],
    running: view.running ?? existing?.running ?? false,
    blocked: view.blocked ?? existing?.blocked ?? false,
    draft: existing?.draft || emptyTabDraft()
  }

  if (!existing) {
    $sessionTabs.set([...previous, tab])
  } else if (JSON.stringify({ ...existing, draft: undefined }) !== JSON.stringify({ ...tab, draft: undefined })) {
    $sessionTabs.set(previous.map(item => (item === existing ? tab : item)))
  }

  return tab
}

export function saveSessionTabDraft(id: string | null | undefined, draft: SessionTabDraft) {
  const tab = findSessionTab(id)

  if (!tab) {
    return
  }

  $sessionTabs.set(
    $sessionTabs.get().map(item =>
      item === tab
        ? {
            ...item,
            draft: {
              input: draft.input,
              inputBuf: [...draft.inputBuf],
              tokens: draft.tokens.map(token => ({ ...token }))
            }
          }
        : item
    )
  )
}

export function removeSessionTab(id: string) {
  $sessionTabs.set($sessionTabs.get().filter(tab => tab.id !== id))
}

export function refreshSessionTabs(
  rows: readonly { id: string; title?: string; running?: boolean; status?: string }[]
) {
  let changed = false

  const tabs = $sessionTabs.get().map(tab => {
    const row = rows.find(item => item.id === tab.id || item.id === tab.storedId || tab.aliases.includes(item.id))

    if (!row) {
      return tab
    }

    const title = row.title?.trim() || tab.title
    const running = row.running ?? ['working', 'streaming', 'starting', 'waiting'].includes(row.status || '')
    const blocked = row.status === 'waiting' || row.status === 'blocked'

    if (title === tab.title && running === tab.running && blocked === Boolean(tab.blocked)) {
      return tab
    }

    changed = true

    return { ...tab, title, running, blocked }
  })

  if (changed) {
    $sessionTabs.set(tabs)
  }
}

export const resetSessionTabs = () => $sessionTabs.set([])
