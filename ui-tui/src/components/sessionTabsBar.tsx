import { Box, NoSelect, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { $isBlocked } from '../app/overlayStore.js'
import { $sessionTabs, type SessionTabAction, tabsInDirectory } from '../app/sessionTabsStore.js'
import { readableSessionTitle } from '../domain/sessionTabs.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

import { listRowStyle } from './overlayPrimitives.js'

export function sessionTabWindow(total: number, active: number, columns: number) {
  const width = Math.max(8, Math.min(30, Math.floor((Math.max(18, columns) - 9) / Math.min(Math.max(1, total), 5))))
  const count = Math.min(total, Math.max(1, Math.floor((Math.max(18, columns) - 9) / width)))
  const offset = Math.max(0, Math.min(Math.max(0, active) - count + 1, total - count))

  return { width, count, offset }
}

/** A process-local strip of opened views; history is opened from /sessions. */
export function SessionTabsBar({
  columns,
  cwd,
  currentId,
  onAction,
  t
}: {
  columns: number
  cwd?: string
  currentId: string | null
  onAction?: (action: SessionTabAction) => void
  t: Theme
}) {
  const tabs = tabsInDirectory(useStore($sessionTabs), cwd)
  const blocked = useStore($isBlocked)
  const disabled = blocked || !currentId
  const T = useT().pickers.session.tabs

  if (!onAction) {
    return null
  }

  const current = tabs.findIndex(tab => tab.id === currentId || tab.aliases.includes(currentId || ''))
  const { width, count, offset } = sessionTabWindow(tabs.length, current, columns)

  const click = (action: SessionTabAction) => (event: { stopImmediatePropagation?: () => void }) => {
    event.stopImmediatePropagation?.()

    if (!disabled) {
      onAction(action)
    }
  }

  return (
    <NoSelect flexDirection="column" flexShrink={0}>
      <Box flexDirection="row" height={1} paddingX={1} width={columns}>
        <Box flexShrink={0} onClick={offset > 0 ? click({ kind: 'index', index: offset - 1 }) : undefined} width={2}>
          <Text color={t.color.muted}>{offset > 0 ? '‹ ' : '  '}</Text>
        </Box>
        {tabs.slice(offset, offset + count).map((tab, index) => {
          const active = index + offset === current
          const style = listRowStyle(t, active)
          const title = readableSessionTitle(tab.title) || T.newChat

          return (
            <Box
              backgroundColor={active ? style.backgroundColor : undefined}
              flexDirection="row"
              flexShrink={0}
              key={tab.id}
              onClick={click({ kind: 'select', id: tab.id })}
              width={width}
            >
              <Box flexShrink={0} width={Math.max(1, width - 3)}>
                <Text bold={active} color={active ? style.color : t.color.muted} wrap="truncate-end">
                  {index + offset + 1} {tab.running ? '● ' : ''}
                  {title}
                </Text>
              </Box>
              <Box flexShrink={0} onClick={click({ kind: 'close', id: tab.id })} width={3}>
                <Text color={active ? style.color : t.color.muted}> × </Text>
              </Box>
            </Box>
          )
        })}
        <Box
          flexShrink={0}
          onClick={offset + count < tabs.length ? click({ kind: 'index', index: offset + count }) : undefined}
          width={2}
        >
          <Text color={t.color.muted}>{offset + count < tabs.length ? ' ›' : '  '}</Text>
        </Box>
        <Box flexGrow={1} />
        <Box flexShrink={0} onClick={click({ kind: 'new' })} width={3}>
          <Text bold color={disabled ? t.color.muted : t.color.accent}>
            {' '}
            +{' '}
          </Text>
        </Box>
      </Box>
      <Text color={t.color.border}>{'─'.repeat(Math.max(0, columns))}</Text>
    </NoSelect>
  )
}
