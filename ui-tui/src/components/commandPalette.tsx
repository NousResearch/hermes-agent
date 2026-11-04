import { Box, stringWidth, Text, useInput } from '@hermes/ink'
import { fuzzyScore } from '@hermes/shared/fuzzy'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { $slashCatalog } from '../app/slashCatalog.js'
import type { Theme } from '../theme.js'
import type { SlashCatalog } from '../types.js'

import { clampOverlayWidth, listRowStyle } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const MAX_ROWS = 12

export interface CommandPaletteEntry {
  aliases?: string[]
  category: string
  command: string
  description: string
}

/** Flatten the categorized gateway catalog while retaining uncategorized skill rows. */
export function commandPaletteEntries(catalog: null | SlashCatalog): CommandPaletteEntry[] {
  if (!catalog) {
    return []
  }

  const categoryByCommand = new Map<string, string>()

  for (const category of catalog.categories) {
    for (const [command] of category.pairs) {
      categoryByCommand.set(command, category.name)
    }
  }

  return catalog.pairs.map(([command, description]) => {
    const aliases = Object.entries(catalog.canon)
      .filter(([alias, canonical]) => canonical === command && alias !== command.toLowerCase())
      .map(([alias]) => alias)

    return {
      ...(aliases.length ? { aliases } : {}),
      category: categoryByCommand.get(command) ?? 'Skills & extensions',
      command,
      description
    }
  })
}

/**
 * Rank names before descriptions, then use the shared scorer's start,
 * boundary, contiguity and gap signals. Equal scores keep catalog order.
 */
export function rankCommandPaletteEntries(entries: CommandPaletteEntry[], query: string): CommandPaletteEntry[] {
  const normalized = query.trim().replace(/^\/+/, '')

  if (!normalized) {
    return entries
  }

  return entries
    .map((entry, index) => {
      const name = [entry.command, ...(entry.aliases ?? [])]
        .map(value => fuzzyScore(value.replace(/^\//, ''), normalized))
        .filter((match): match is NonNullable<typeof match> => match !== null)
        .sort((a, b) => b.score - a.score)[0]
      const description = fuzzyScore(entry.description, normalized)
      const category = fuzzyScore(entry.category, normalized)
      const score = name
        ? name.score + 100
        : description
          ? description.score + 20
          : category
            ? category.score
            : Number.NEGATIVE_INFINITY

      return { entry, index, score }
    })
    .filter(row => row.score !== Number.NEGATIVE_INFINITY)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(row => row.entry)
}

export function CommandPalette({
  maxWidth,
  onCancel,
  onSelect,
  t
}: {
  maxWidth?: number
  onCancel: () => void
  onSelect: (command: string) => void
  t: Theme
}) {
  const catalog = useStore($slashCatalog)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)
  const entries = useMemo(() => commandPaletteEntries(catalog), [catalog])
  const ranked = useMemo(() => rankCommandPaletteEntries(entries, query), [entries, query])
  const active = Math.min(selected, Math.max(0, ranked.length - 1))
  const start = Math.max(0, Math.min(active - Math.floor(MAX_ROWS / 2), ranked.length - MAX_ROWS))
  const visible = ranked.slice(start, start + MAX_ROWS)
  const width = clampOverlayWidth(76, maxWidth, 32)
  const commandWidth = Math.min(28, Math.max(12, ...visible.map(row => stringWidth(row.command) + 2)))
  const selectCurrent = () => {
    const row = ranked[active]

    if (row) {
      onSelect(row.command)
    }
  }

  useEffect(() => setSelected(0), [query])

  useInput((ch, key) => {
    if (key.escape || (key.ctrl && ch.toLowerCase() === 'p')) {
      return onCancel()
    }

    if (key.upArrow && ranked.length) {
      return setSelected(value => (value - 1 + ranked.length) % ranked.length)
    }

    if (key.downArrow && ranked.length) {
      return setSelected(value => (value + 1) % ranked.length)
    }
  })

  return (
    <Box flexDirection="column" paddingX={1} paddingY={1} width={width}>
      <Box marginBottom={1}>
        <Text bold color={t.color.primary}>
          Command palette
        </Text>
        <Text color={t.color.muted}> · Ctrl+P</Text>
      </Box>

      <Box marginBottom={1}>
        <Text color={t.color.accent}>› </Text>
        <TextInput
          accentColor={t.color.accent}
          color={t.color.text}
          columns={Math.max(8, width - 6)}
          ignoreVerticalArrows
          onChange={setQuery}
          onSubmit={selectCurrent}
          placeholder="Search commands"
          placeholderColor={t.color.muted}
          value={query}
        />
      </Box>

      {visible.length ? (
        visible.map((row, index) => {
          const rowIndex = start + index
          const selectedRow = rowIndex === active
          const style = listRowStyle(t, selectedRow)

          return (
            <Box backgroundColor={style.backgroundColor} key={row.command} width="100%">
              <Box flexShrink={0} width={commandWidth}>
                <Text bold={selectedRow} color={selectedRow ? (style.color ?? t.color.label) : t.color.label}>
                  {selectedRow ? '▸ ' : '  '}
                  {row.command}
                </Text>
              </Box>
              <Box flexGrow={1}>
                <Text color={selectedRow ? (style.color ?? t.color.text) : t.color.statusFg} wrap="truncate-end">
                  {row.description}
                </Text>
              </Box>
              <Box flexShrink={0} marginLeft={1} width={18}>
                <Text color={selectedRow ? (style.color ?? t.color.muted) : t.color.muted} wrap="truncate-end">
                  {row.category}
                </Text>
              </Box>
            </Box>
          )
        })
      ) : (
        <Text color={t.color.muted}>No matching commands</Text>
      )}

      <Box marginTop={1}>
        <Text color={t.color.muted}>↑↓ navigate · Enter run · Esc close</Text>
      </Box>
    </Box>
  )
}
