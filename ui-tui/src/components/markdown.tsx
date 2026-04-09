import { Box, Text } from 'ink'
import type { ReactNode } from 'react'

import type { Theme } from '../theme.js'

/** OSC 8 hyperlink — wrap-ansi / Ink keep the link active across soft line wraps. */
const osc8 = (url: string) => '\x1b]8;;' + url + '\x1b\\'
const OSC8_END = '\x1b]8;;\x1b\\'

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: ReactNode[] = []
  const re = /(\[(.+?)\]\((https?:\/\/[^\s)]+)\)|\*\*(.+?)\*\*|`([^`]+)`|\*(.+?)\*|(https?:\/\/[^\s]+))/g

  let last = 0

  for (const m of text.matchAll(re)) {
    const i = m.index ?? 0

    if (i > last) {
      parts.push(<Text key={parts.length}>{text.slice(last, i)}</Text>)
    }

    if (m[2] && m[3]) {
      parts.push(
        <Text key={parts.length}>
          {osc8(m[3])}
          <Text color={t.color.amber} underline>
            {m[2]}
          </Text>
          {OSC8_END}
        </Text>
      )
    } else if (m[4]) {
      parts.push(
        <Text bold key={parts.length}>
          {m[4]}
        </Text>
      )
    } else if (m[5]) {
      parts.push(
        <Text color={t.color.amber} dimColor key={parts.length}>
          {m[5]}
        </Text>
      )
    } else if (m[6]) {
      parts.push(
        <Text italic key={parts.length}>
          {m[6]}
        </Text>
      )
    } else if (m[7]) {
      const u = m[7]
      parts.push(
        <Text key={parts.length}>
          {osc8(u)}
          <Text color={t.color.amber} underline>
            {u}
          </Text>
          {OSC8_END}
        </Text>
      )
    }

    last = i + m[0].length
  }

  if (last < text.length) {
    parts.push(<Text key={parts.length}>{text.slice(last)}</Text>)
  }

  return <Text>{parts.length ? parts : <Text>{text}</Text>}</Text>
}

export function Md({ compact, t, text }: { compact?: boolean; t: Theme; text: string }) {
  const lines = text.split('\n')
  const nodes: ReactNode[] = []
  let i = 0
  let prevKind: 'blank' | 'code' | 'heading' | 'list' | 'paragraph' | 'quote' | 'table' | null = null

  const gap = () => {
    if (nodes.length && prevKind !== 'blank') {
      nodes.push(<Text key={`gap-${nodes.length}`}> </Text>)
      prevKind = 'blank'
    }
  }

  const start = (kind: Exclude<typeof prevKind, null | 'blank'>) => {
    if (prevKind && prevKind !== 'blank' && prevKind !== kind) {
      gap()
    }

    prevKind = kind
  }

  while (i < lines.length) {
    const line = lines[i]!
    const key = nodes.length

    if (compact && !line.trim()) {
      i++

      continue
    }

    if (!line.trim()) {
      gap()
      i++

      continue
    }

    if (line.startsWith('```')) {
      start('code')
      const lang = line.slice(3).trim()
      const block: string[] = []

      for (i++; i < lines.length && !lines[i]!.startsWith('```'); i++) {
        block.push(lines[i]!)
      }

      i++
      const isDiff = lang === 'diff'

      nodes.push(
        <Box flexDirection="column" key={key} paddingLeft={2}>
          {lang && !isDiff && <Text color={t.color.dim}>{'─ ' + lang}</Text>}
          {block.map((l, j) => {
            const add = isDiff && l.startsWith('+')
            const del = isDiff && l.startsWith('-')
            const hunk = isDiff && l.startsWith('@@')

            return (
              <Text
                backgroundColor={add ? t.color.diffAdded : del ? t.color.diffRemoved : undefined}
                color={add ? t.color.diffAddedWord : del ? t.color.diffRemovedWord : hunk ? t.color.dim : undefined}
                dimColor={isDiff && !add && !del && !hunk && l.startsWith(' ')}
                key={j}
              >
                {l}
              </Text>
            )
          })}
        </Box>
      )

      continue
    }

    const heading = line.match(/^#{1,3}\s+(.*)/)

    if (heading) {
      start('heading')
      nodes.push(
        <Text bold color={t.color.amber} key={key}>
          {heading[1]}
        </Text>
      )
      i++

      continue
    }

    const bullet = line.match(/^\s*[-*]\s(.*)/)

    if (bullet) {
      start('list')
      nodes.push(
        <Text key={key}>
          <Text color={t.color.dim}> • </Text>
          <MdInline t={t} text={bullet[1]!} />
        </Text>
      )
      i++

      continue
    }

    const numbered = line.match(/^\s*(\d+)\.\s(.*)/)

    if (numbered) {
      start('list')
      nodes.push(
        <Text key={key}>
          <Text color={t.color.dim}> {numbered[1]}. </Text>
          <MdInline t={t} text={numbered[2]!} />
        </Text>
      )
      i++

      continue
    }

    if (line.match(/^>\s?/)) {
      start('quote')
      const quoteLines: string[] = []

      while (i < lines.length && lines[i]!.match(/^>\s?/)) {
        quoteLines.push(lines[i]!.replace(/^>\s?/, ''))
        i++
      }

      nodes.push(
        <Box flexDirection="column" key={key}>
          {quoteLines.map((ql, qi) => (
            <Text color={t.color.dim} key={qi}>
              {'  │ '}
              <MdInline t={t} text={ql} />
            </Text>
          ))}
        </Box>
      )

      continue
    }

    if (line.includes('|') && line.trim().startsWith('|')) {
      start('table')
      const tableRows: string[][] = []

      while (i < lines.length && lines[i]!.trim().startsWith('|')) {
        const row = lines[i]!.trim()

        if (!/^[|\s:-]+$/.test(row)) {
          tableRows.push(
            row
              .split('|')
              .filter(Boolean)
              .map(c => c.trim())
          )
        }

        i++
      }

      if (tableRows.length) {
        const widths = tableRows[0]!.map((_, ci) => Math.max(...tableRows.map(r => (r[ci] ?? '').length)))

        nodes.push(
          <Box flexDirection="column" key={key} paddingLeft={2}>
            {tableRows.map((row, ri) => (
              <Text color={ri === 0 ? t.color.amber : undefined} key={ri}>
                {row.map((cell, ci) => cell.padEnd(widths[ci] ?? 0)).join('  ')}
              </Text>
            ))}
          </Box>
        )
      }

      continue
    }

    start('paragraph')
    nodes.push(<MdInline key={key} t={t} text={line} />)

    i++
  }

  return <Box flexDirection="column">{nodes}</Box>
}
