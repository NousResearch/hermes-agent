import { Box, Link, Text } from '@hermes/ink'
import { memo, type ReactNode, useMemo } from 'react'

import { highlightLine, isHighlightable } from '../lib/syntax.js'
import type { Theme } from '../theme.js'

const FENCE_RE = /^\s*(`{3,}|~{3,})(.*)$/
const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+\s*)?$/
const FOOTNOTE_RE = /^\[\^([^\]]+)\]:\s*(.*)$/
const DEF_RE = /^\s*:\s+(.+)$/
const TABLE_DIVIDER_CELL_RE = /^:?-{3,}:?$/
const MD_URL_RE = '((?:[^\\s()]|\\([^\\s()]*\\))+?)'

const INLINE_RE = new RegExp(
  `(!\\[(.*?)\\]\\(${MD_URL_RE}\\)|\\[(.+?)\\]\\(${MD_URL_RE}\\)|<((?:https?:\\/\\/|mailto:)[^>\\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,})>|~~(.+?)~~|\`([^\\\`]+)\`|\\*\\*(.+?)\\*\\*|__(.+?)__|\\*(.+?)\\*|_(.+?)_|==(.+?)==|\\[\\^([^\\]]+)\\]|\\^([^^\\s][^^]*?)\\^|~([^~\\s][^~]*?)~|(https?:\\/\\/[^\\s<]+))`,
  'g'
)

type Fence = {
  char: '`' | '~'
  lang: string
  len: number
}

const renderLink = (key: number, t: Theme, label: string, url: string) => (
  <Link key={key} url={url}>
    <Text color={t.color.amber} underline>
      {label}
    </Text>
  </Link>
)

const trimBareUrl = (value: string) => {
  const trimmed = value.replace(/[),.;:!?]+$/g, '')

  return {
    tail: value.slice(trimmed.length),
    url: trimmed
  }
}

const renderAutolink = (key: number, t: Theme, raw: string) => {
  const url = raw.startsWith('mailto:') ? raw : raw.includes('@') && !raw.startsWith('http') ? `mailto:${raw}` : raw

  return (
    <Link key={key} url={url}>
      <Text color={t.color.amber} underline>
        {raw.replace(/^mailto:/, '')}
      </Text>
    </Link>
  )
}

const indentDepth = (indent: string) => Math.floor(indent.replace(/\t/g, '  ').length / 2)

const parseFence = (line: string): Fence | null => {
  const m = line.match(FENCE_RE)

  if (!m) {
    return null
  }

  return {
    char: m[1]![0] as '`' | '~',
    lang: m[2]!.trim().toLowerCase(),
    len: m[1]!.length
  }
}

const isFenceClose = (line: string, fence: Fence) => {
  const end = line.match(/^\s*(`{3,}|~{3,})\s*$/)

  return Boolean(end && end[1]![0] === fence.char && end[1]!.length >= fence.len)
}

const isMarkdownFence = (lang: string) => ['md', 'markdown'].includes(lang)

const splitTableRow = (row: string) =>
  row
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(cell => cell.trim())

const isTableDivider = (row: string) => {
  const cells = splitTableRow(row)

  return cells.length > 1 && cells.every(cell => TABLE_DIVIDER_CELL_RE.test(cell))
}

const stripInlineMarkup = (value: string) =>
  value
    .replace(/!\[(.*?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '[image: $1] $2')
    .replace(/\[(.+?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '$1')
    .replace(/<((?:https?:\/\/|mailto:)[^>\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/_(.+?)_/g, '$1')
    .replace(/==(.+?)==/g, '$1')
    .replace(/\[\^([^\]]+)\]/g, '[$1]')
    .replace(/\^([^^\s][^^]*?)\^/g, '^$1')
    .replace(/~([^~\s][^~]*?)~/g, '_$1')

const renderTable = (key: number, rows: string[][], t: Theme) => {
  const widths = rows[0]!.map((_, ci) => Math.max(...rows.map(r => stripInlineMarkup(r[ci] ?? '').length)))

  return (
    <Box flexDirection="column" key={key} paddingLeft={2}>
      {rows.map((row, ri) => (
        <Box key={ri}>
          {widths.map((width, ci) => {
            const cell = row[ci] ?? ''
            const pad = ' '.repeat(Math.max(0, width - stripInlineMarkup(cell).length))

            return (
              <Text color={ri === 0 ? t.color.amber : undefined} key={ci}>
                <MdInline t={t} text={cell} />
                {pad}
                {ci < widths.length - 1 ? '  ' : ''}
              </Text>
            )
          })}
        </Box>
      ))}
    </Box>
  )
}

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: ReactNode[] = []

  let last = 0

  for (const m of text.matchAll(INLINE_RE)) {
    const i = m.index ?? 0

    if (i > last) {
      parts.push(<Text key={parts.length}>{text.slice(last, i)}</Text>)
    }

    if (m[2] && m[3]) {
      parts.push(
        <Text color={t.color.dim} key={parts.length}>
          [image: {m[2]}] {m[3]}
        </Text>
      )
    } else if (m[4] && m[5]) {
      parts.push(renderLink(parts.length, t, m[4], m[5]))
    } else if (m[6]) {
      parts.push(renderAutolink(parts.length, t, m[6]))
    } else if (m[7]) {
      parts.push(
        <Text key={parts.length} strikethrough>
          {m[7]}
        </Text>
      )
    } else if (m[8]) {
      parts.push(
        <Text color={t.color.amber} dimColor key={parts.length}>
          {m[8]}
        </Text>
      )
    } else if (m[9] || m[10]) {
      parts.push(
        <Text bold key={parts.length}>
          {m[9] ?? m[10]}
        </Text>
      )
    } else if (m[11] || m[12]) {
      parts.push(
        <Text italic key={parts.length}>
          {m[11] ?? m[12]}
        </Text>
      )
    } else if (m[13]) {
      parts.push(
        <Text backgroundColor={t.color.diffAdded} color={t.color.diffAddedWord} key={parts.length}>
          {m[13]}
        </Text>
      )
    } else if (m[14]) {
      parts.push(
        <Text color={t.color.dim} key={parts.length}>
          [{m[14]}]
        </Text>
      )
    } else if (m[15]) {
      parts.push(
        <Text color={t.color.dim} key={parts.length}>
          ^{m[15]}
        </Text>
      )
    } else if (m[16]) {
      parts.push(
        <Text color={t.color.dim} key={parts.length}>
          _{m[16]}
        </Text>
      )
    } else if (m[17]) {
      const { tail, url } = trimBareUrl(m[17])

      parts.push(renderAutolink(parts.length, t, url))

      if (tail) {
        parts.push(<Text key={parts.length}>{tail}</Text>)
      }
    }

    last = i + m[0].length
  }

  if (last < text.length) {
    parts.push(<Text key={parts.length}>{text.slice(last)}</Text>)
  }

  return <Text>{parts.length ? parts : <Text>{text}</Text>}</Text>
}

interface MdProps {
  compact?: boolean
  t: Theme
  text: string
}

function MdImpl({ compact, t, text }: MdProps) {
  const nodes = useMemo(() => {
    const lines = text.split('\n')
    const nodes: ReactNode[] = []
    let i = 0

    let prevKind: 'blank' | 'code' | 'heading' | 'list' | 'paragraph' | 'quote' | 'rule' | 'table' | null = null

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

      const fence = parseFence(line)

      if (fence) {
        const block: string[] = []
        const lang = fence.lang

        for (i++; i < lines.length && !isFenceClose(lines[i]!, fence); i++) {
          block.push(lines[i]!)
        }

        if (i < lines.length) {
          i++
        }

        if (isMarkdownFence(lang)) {
          start('paragraph')
          nodes.push(<Md compact={compact} key={key} t={t} text={block.join('\n')} />)

          continue
        }

        start('code')

        const isDiff = lang === 'diff'
        const highlighted = !isDiff && isHighlightable(lang)

        nodes.push(
          <Box flexDirection="column" key={key} paddingLeft={2}>
            {lang && !isDiff && <Text color={t.color.dim}>{'─ ' + lang}</Text>}
            {block.map((l, j) => {
              if (highlighted) {
                return (
                  <Text key={j}>
                    {highlightLine(l, lang, t).map(([color, text], k) =>
                      color ? (
                        <Text color={color} key={k}>
                          {text}
                        </Text>
                      ) : (
                        <Text key={k}>{text}</Text>
                      )
                    )}
                  </Text>
                )
              }

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

      if (line.trim().startsWith('$$')) {
        start('code')

        const block: string[] = []

        for (i++; i < lines.length; i++) {
          if (lines[i]!.trim().startsWith('$$')) {
            i++

            break
          }

          block.push(lines[i]!)
        }

        nodes.push(
          <Box flexDirection="column" key={key} paddingLeft={2}>
            <Text color={t.color.dim}>─ math</Text>
            {block.map((l, j) => (
              <Text color={t.color.amber} key={j}>
                {l}
              </Text>
            ))}
          </Box>
        )

        continue
      }

      const heading = line.match(HEADING_RE)

      if (heading) {
        start('heading')
        nodes.push(
          <Text bold color={t.color.amber} key={key}>
            {heading[2]}
          </Text>
        )
        i++

        continue
      }

      if (i + 1 < lines.length && line.trim()) {
        const setext = lines[i + 1]!.match(/^\s{0,3}(=+|-+)\s*$/)

        if (setext) {
          start('heading')
          nodes.push(
            <Text bold color={t.color.amber} key={key}>
              {line.trim()}
            </Text>
          )
          i += 2

          continue
        }
      }

      if (HR_RE.test(line)) {
        start('rule')
        nodes.push(
          <Text color={t.color.dim} key={key}>
            {'─'.repeat(36)}
          </Text>
        )
        i++

        continue
      }

      const footnote = line.match(FOOTNOTE_RE)

      if (footnote) {
        start('list')
        nodes.push(
          <Text color={t.color.dim} key={key}>
            [{footnote[1]}] <MdInline t={t} text={footnote[2] ?? ''} />
          </Text>
        )
        i++

        while (i < lines.length && /^\s{2,}\S/.test(lines[i]!)) {
          nodes.push(
            <Box key={`${key}-cont-${i}`} paddingLeft={2}>
              <Text color={t.color.dim}>
                <MdInline t={t} text={lines[i]!.trim()} />
              </Text>
            </Box>
          )
          i++
        }

        continue
      }

      if (i + 1 < lines.length && DEF_RE.test(lines[i + 1]!)) {
        start('list')
        nodes.push(
          <Text bold key={key}>
            {line.trim()}
          </Text>
        )
        i++

        while (i < lines.length) {
          const def = lines[i]!.match(DEF_RE)

          if (!def) {
            break
          }

          nodes.push(
            <Text key={`${key}-def-${i}`}>
              <Text color={t.color.dim}> · </Text>
              <MdInline t={t} text={def[1]!} />
            </Text>
          )
          i++
        }

        continue
      }

      const bullet = line.match(/^(\s*)[-+*]\s+(.*)$/)

      if (bullet) {
        start('list')
        const depth = indentDepth(bullet[1]!)
        const task = bullet[2]!.match(/^\[( |x|X)\]\s+(.*)$/)
        const marker = task ? (task[1]!.toLowerCase() === 'x' ? '☑' : '☐') : '•'
        const body = task ? task[2]! : bullet[2]!

        nodes.push(
          <Text key={key}>
            <Text color={t.color.dim}>
              {' '.repeat(depth * 2)}
              {marker}{' '}
            </Text>
            <MdInline t={t} text={body} />
          </Text>
        )
        i++

        continue
      }

      const numbered = line.match(/^(\s*)(\d+)[.)]\s+(.*)$/)

      if (numbered) {
        start('list')
        const depth = indentDepth(numbered[1]!)

        nodes.push(
          <Text key={key}>
            <Text color={t.color.dim}>
              {' '.repeat(depth * 2)}
              {numbered[2]}.{' '}
            </Text>
            <MdInline t={t} text={numbered[3]!} />
          </Text>
        )
        i++

        continue
      }

      if (/^\s*(?:>\s*)+/.test(line)) {
        start('quote')
        const quoteLines: Array<{ depth: number; text: string }> = []

        while (i < lines.length && /^\s*(?:>\s*)+/.test(lines[i]!)) {
          const raw = lines[i]!
          const prefix = raw.match(/^\s*(?:>\s*)+/)?.[0] ?? ''

          quoteLines.push({
            depth: (prefix.match(/>/g) ?? []).length,
            text: raw.slice(prefix.length)
          })
          i++
        }

        nodes.push(
          <Box flexDirection="column" key={key}>
            {quoteLines.map((ql, qi) => (
              <Text color={t.color.dim} key={qi}>
                {' '.repeat(Math.max(0, ql.depth - 1) * 2)}
                {'│ '}
                <MdInline t={t} text={ql.text} />
              </Text>
            ))}
          </Box>
        )

        continue
      }

      if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1]!)) {
        start('table')
        const tableRows: string[][] = []

        tableRows.push(splitTableRow(line))
        i += 2

        while (i < lines.length && lines[i]!.includes('|') && lines[i]!.trim()) {
          tableRows.push(splitTableRow(lines[i]!))
          i++
        }

        nodes.push(renderTable(key, tableRows, t))

        continue
      }

      if (/^<details\b/i.test(line) || /^<\/details>/i.test(line)) {
        i++

        continue
      }

      const summary = line.match(/^<summary>(.*?)<\/summary>$/i)

      if (summary) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.dim} key={key}>
            ▶ {summary[1]}
          </Text>
        )
        i++

        continue
      }

      if (/^<\/?[^>]+>$/.test(line.trim())) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.dim} key={key}>
            {line.trim()}
          </Text>
        )
        i++

        continue
      }

      if (line.includes('|') && line.trim().startsWith('|')) {
        start('table')
        const tableRows: string[][] = []

        while (i < lines.length && lines[i]!.trim().startsWith('|')) {
          const row = lines[i]!.trim()

          if (!/^[|\s:-]+$/.test(row)) {
            tableRows.push(splitTableRow(row))
          }

          i++
        }

        if (tableRows.length) {
          nodes.push(renderTable(key, tableRows, t))
        }

        continue
      }

      start('paragraph')
      nodes.push(<MdInline key={key} t={t} text={line} />)

      i++
    }

    return nodes
  }, [compact, t, text])

  return <Box flexDirection="column">{nodes}</Box>
}

export const Md = memo(MdImpl)
