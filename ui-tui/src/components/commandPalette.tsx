import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useMemo, useState } from 'react'

import { SLASH_COMMANDS } from '../app/slash/registry.js'
import { buildCommandPaletteItems, filterCommandPaletteItems } from '../lib/commandPalette.js'
import type { Theme } from '../theme.js'
import type { SlashCatalog } from '../types.js'

const VISIBLE = 12
const MIN_WIDTH = 58
const MAX_WIDTH = 118

const pageOffset = (count: number, sel: number) => Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), count - VISIBLE))

const keyName = (key: { backspace?: boolean; delete?: boolean }) => key.backspace || key.delete

export function CommandPalette({ catalog, initialQuery = '', onClose, onSelect, t }: CommandPaletteProps) {
  const [query, setQuery] = useState(initialQuery)
  const [sel, setSel] = useState(0)
  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  const allItems = useMemo(() => buildCommandPaletteItems(SLASH_COMMANDS, catalog), [catalog])
  const items = useMemo(() => filterCommandPaletteItems(allItems, query), [allItems, query])
  const active = items[Math.min(sel, Math.max(0, items.length - 1))]
  const off = pageOffset(items.length, Math.min(sel, Math.max(0, items.length - 1)))

  useInput((ch, key) => {
    if (key.escape || (key.ctrl && ch.toLowerCase() === 'c')) {
      return onClose()
    }

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)

      return
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(v => v + 1)

      return
    }

    if (key.return && active) {
      onSelect(active.usage === active.command ? `${active.command} ` : `${active.usage} `)

      return
    }

    if (keyName(key)) {
      setQuery(v => v.slice(0, -1))
      setSel(0)

      return
    }

    if (ch && !key.ctrl && !key.meta && ch >= ' ' && ch !== '\x7f') {
      setQuery(v => v + ch)
      setSel(0)
    }
  })

  return (
    <Box flexDirection="column" paddingX={1} paddingY={1} width={width}>
      <Box justifyContent="space-between" marginBottom={1}>
        <Text bold color={t.color.gold}>☤ Command Palette</Text>
        <Text color={t.color.dim}>{items.length}/{allItems.length}</Text>
      </Box>

      <Text color={t.color.dim}>
        query: <Text color={t.color.cornsilk}>{query || 'type to filter…'}</Text>
      </Text>

      <Box flexDirection="column" marginTop={1}>
        {!items.length ? (
          <Text color={t.color.dim}>no matching commands</Text>
        ) : (
          items.slice(off, off + VISIBLE).map((item, vi) => {
            const i = off + vi
            const selected = i === sel
            const aliasText = item.aliases.length ? ` · ${item.aliases.map(a => `/${a}`).join(', ')}` : ''

            return (
              <Box key={item.command} width="100%">
                <Box width={3}>
                  <Text bold={selected} color={selected ? t.color.amber : t.color.dim} inverse={selected}>
                    {selected ? '▸ ' : '  '}
                  </Text>
                </Box>

                <Box width={3}>
                  <Text color={item.lane === 'core' ? t.color.gold : t.color.dim} inverse={selected}>
                    {item.rune}{' '}
                  </Text>
                </Box>

                <Box width={23}>
                  <Text bold={selected} color={selected ? t.color.amber : t.color.label} inverse={selected} wrap="truncate-end">
                    {item.command}
                  </Text>
                </Box>

                <Text color={selected ? t.color.cornsilk : t.color.dim} inverse={selected} wrap="truncate-end">
                  {item.description || item.usage}{aliasText}
                </Text>
              </Box>
            )
          })
        )}
      </Box>

      <Box marginTop={1}>
        <Text color={t.color.dim}>↑/↓ select · type filter · Backspace edit · Enter stage command · Esc close</Text>
      </Box>
    </Box>
  )
}

interface CommandPaletteProps {
  catalog: null | SlashCatalog
  initialQuery?: string
  onClose: () => void
  onSelect: (command: string) => void
  t: Theme
}
