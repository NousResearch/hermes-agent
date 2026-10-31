import { looksLikeSlashCommand, parseSlashCommand } from '@hermes/shared/slash'

import { readableSessionTitle } from '../../../domain/sessionTabs.js'
import { $sessionTabs, findSessionTab, type SessionTabAction, tabsInDirectory } from '../../sessionTabsStore.js'
import { sessionCwd } from '../../uiStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

export const TAB_COMMAND_NAMES = new Set([
  'new',
  'tabnew',
  'tab',
  'tabs',
  'tabnext',
  'next',
  'tabprev',
  'prev',
  'previous',
  'tabclose'
])

const act = (ctx: SlashRunCtx, action: SessionTabAction) => {
  if (!ctx.session.tabAction) {
    ctx.transcript.sys('Tab controls are not available in this terminal.')

    return
  }

  ctx.session.tabAction(action)
}

export function normalizeTabCommandInput(input: string): string {
  const trimmed = input.trimStart()

  return looksLikeSlashCommand(trimmed) && TAB_COMMAND_NAMES.has(parseSlashCommand(trimmed).name) ? trimmed : input
}

export const tabCommands: SlashCommand[] = [
  {
    name: 'new',
    aliases: ['tabnew'],
    help: 'open a new tab without closing or interrupting the current chat',
    run: (arg, ctx) => {
      ctx.ran?.()
      ctx.session.newLiveSession(undefined, arg.trim() || undefined)
    }
  },
  {
    name: 'tabnext',
    aliases: ['next'],
    help: 'switch to the next session tab',
    run: (_arg, ctx) => act(ctx, { kind: 'cycle', direction: 1 })
  },
  {
    name: 'tabprev',
    aliases: ['prev', 'previous'],
    help: 'switch to the previous session tab',
    run: (_arg, ctx) => act(ctx, { kind: 'cycle', direction: -1 })
  },
  {
    name: 'tabclose',
    help: 'close this tab view without deleting its history; unsent drafts are protected',
    run: (_arg, ctx) => act(ctx, { kind: 'close' })
  },
  {
    name: 'tab',
    aliases: ['tabs'],
    help: 'list or select tabs: /tab <number|next|prev|new|close|title|id>',
    run: (arg, ctx) => {
      const value = arg.trim()
      const tabs = tabsInDirectory($sessionTabs.get(), sessionCwd())

      if (!value || value === 'list') {
        const active = findSessionTab(ctx.sid)
        ctx.transcript.panel('Session tabs', [
          {
            rows: tabs.map((tab, index) => [
              String(index + 1),
              `${tab.id === active?.id ? '* ' : ''}${readableSessionTitle(tab.title) || '(untitled)'}${tab.running ? ' · running' : ''}`
            ])
          }
        ])

        return
      }

      if (value === 'new') {
        return act(ctx, { kind: 'new' })
      }

      if (value === 'close') {
        return act(ctx, { kind: 'close' })
      }

      if (value === 'next') {
        return act(ctx, { kind: 'cycle', direction: 1 })
      }

      if (value === 'prev' || value === 'previous') {
        return act(ctx, { kind: 'cycle', direction: -1 })
      }

      if (/^[0-9]+$/.test(value)) {
        const index = Number(value) - 1

        if (Number.isSafeInteger(index) && tabs[index]) {
          return act(ctx, { kind: 'index', index })
        }

        ctx.transcript.sys(`No tab ${value} in this directory. Use /tabs to list open tabs.`)

        return
      }

      const exact = tabs.filter(tab => [tab.id, tab.storedId, ...tab.aliases].includes(value))

      const matches = exact.length
        ? exact
        : tabs.filter(tab => readableSessionTitle(tab.title).toLowerCase().includes(value.toLowerCase()))

      if (matches.length === 1) {
        return act(ctx, { kind: 'select', id: matches[0]!.id })
      }

      ctx.transcript.sys(
        matches.length
          ? 'More than one tab matches. Use /tab <number>.'
          : 'No matching tab in this directory. Use /tabs to list open tabs.'
      )
    }
  }
]
