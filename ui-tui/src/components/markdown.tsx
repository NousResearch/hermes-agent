import { Box, Link, Text } from '@hermes/ink'
import { Fragment, memo, type ReactNode, useMemo } from 'react'

import { ensureEmojiPresentation } from '../lib/emoji.js'
import { highlightLine, isHighlightable } from '../lib/syntax.js'
import type { Theme } from '../theme.js'

const FENCE_RE = /^\s*(`{3,}|~{3,})(.*)$/
const FENCE_CLOSE_RE = /^\s*(`{3,}|~{3,})\s*$/
const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+\s*)?$/
const SETEXT_RE = /^\s{0,3}(=+|-+)\s*$/
const FOOTNOTE_RE = /^\[\^([^\]]+)\]:\s*(.*)$/
const DEF_RE = /^\s*:\s+(.+)$/
const BULLET_RE = /^(\s*)[-+*]\s+(.*)$/
const TASK_RE = /^\[( |x|X)\]\s+(.*)$/
const NUMBERED_RE = /^(\s*)(\d+)[.)]\s+(.*)$/
const QUOTE_RE = /^\s*(?:>\s*)+/
const TABLE_DIVIDER_CELL_RE = /^:?-{3,}:?$/
const MD_URL_RE = '((?:[^\\s()]|\\([^\\s()]*\\))+?)'

export const MEDIA_LINE_RE = /^\s*[`"']?MEDIA:\s*(\S+?)[`"']?\s*$/
export const AUDIO_DIRECTIVE_RE = /^\s*\[\[audio_as_voice\]\]\s*$/

// Inline markdown tokens, in priority order. The outer regex picks the
// leftmost match at each position, preferring earlier alternatives on tie —
// so `**` must come before `*`, `__` before `_`, etc. Each pattern owns its
// own capture groups; MdInline dispatches on which group matched.
//
// Subscript (`~x~`) is restricted to short alphanumeric runs so prose like
// `thing ~! more ~?` from Kimi / Qwen / GLM (kaomoji-style decorators)
// doesn't pair up the first `~` with the next one on the line and swallow
// the text between them as a dim `_`-prefixed span.
export const INLINE_RE = new RegExp(
  [
    `!\\[(.*?)\\]\\(${MD_URL_RE}\\)`, // 1,2  image
    `\\[(.+?)\\]\\(${MD_URL_RE}\\)`, // 3,4  link
    `<((?:https?:\\/\\/|mailto:)[^>\\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,})>`, // 5   autolink
    `~~(.+?)~~`, // 6    strike
    `\`([^\\\`]+)\``, // 7    code
    `\\*\\*(.+?)\\*\\*`, // 8    bold *
    `(?<!\\w)__(.+?)__(?!\\w)`, // 9    bold _
    `\\*(.+?)\\*`, // 10   italic *
    `(?<!\\w)_(.+?)_(?!\\w)`, // 11   italic _
    `==(.+?)==`, // 12   highlight
    `\\[\\^([^\\]]+)\\]`, // 13   footnote ref
    `\\^([^^\\s][^^]*?)\\^`, // 14   superscript
    `~([A-Za-z0-9]{1,8})~`, // 15   subscript
    `https?:\\/\\/[^\\s<]+` //  16   bare URL
  ].join('|'),
  'g'
)

const indentDepth = (s: string) => Math.floor(s.replace(/\t/g, '  ').length / 2)

const splitRow = (row: string) =>
  row
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(c => c.trim())

const isTableDivider = (row: string) => {
  const cells = splitRow(row)

  return cells.length > 1 && cells.every(c => TABLE_DIVIDER_CELL_RE.test(c))
}

const autolinkUrl = (raw: string) =>
  raw.startsWith('mailto:') || raw.startsWith('http') || !raw.includes('@') ? raw : `mailto:${raw}`

const renderAutolink = (k: number, t: Theme, raw: string) => (
  <Link key={k} url={autolinkUrl(raw)}>
    <Text color={t.color.accent} underline>
      {raw.replace(/^mailto:/, '')}
    </Text>
  </Link>
)

export const stripInlineMarkup = (v: string) =>
  v
    .replace(/!\[(.*?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '[image: $1] $2')
    .replace(/\[(.+?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '$1')
    .replace(/<((?:https?:\/\/|mailto:)[^>\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/(?<!\w)__(.+?)__(?!\w)/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/(?<!\w)_(.+?)_(?!\w)/g, '$1')
    .replace(/==(.+?)==/g, '$1')
    .replace(/\[\^([^\]]+)\]/g, '[$1]')
    .replace(/\^([^^\s][^^]*?)\^/g, '^$1')
    .replace(/~([A-Za-z0-9]{1,8})~/g, '_$1')

const renderTable = (k: number, rows: string[][], t: Theme) => {
  const widths = rows[0]!.map((_, ci) => Math.max(...rows.map(r => stripInlineMarkup(r[ci] ?? '').length)))

  // Thin divider under the header.  Without it tables look like prose
  // with extra spacing because the header is just amber-coloured text
  // (#15534).  We avoid full borders on purpose — column widths come
  // from `stripInlineMarkup(...).length` (UTF-16 code units, not
  // display width), so a real outline often misaligns on emoji and
  // East-Asian wide characters; one dim solid rule (`─`) under row 0
  // plus tab-style column gaps reads cleanly on every terminal we
  // tested.
  const sep = widths.map(w => '─'.repeat(Math.max(1, w))).join('  ')

  return (
    <Box flexDirection="column" key={k} paddingLeft={2}>
      {rows.map((row, ri) => (
        <Fragment key={ri}>
          <Box>
            {widths.map((w, ci) => (
              <Text bold={ri === 0} color={ri === 0 ? t.color.accent : undefined} key={ci}>
                <MdInline t={t} text={row[ci] ?? ''} />
                {' '.repeat(Math.max(0, w - stripInlineMarkup(row[ci] ?? '').length))}
                {ci < widths.length - 1 ? '  ' : ''}
              </Text>
            ))}
          </Box>
          {ri === 0 && rows.length > 1 ? (
            <Text color={t.color.muted} dimColor>
              {sep}
            </Text>
          ) : null}
        </Fragment>
      ))}
    </Box>
  )
}

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: ReactNode[] = []

  let last = 0

  for (const m of text.matchAll(INLINE_RE)) {
    const i = m.index ?? 0
    const k = parts.length

    if (i > last) {
      parts.push(<Text key={k}>{text.slice(last, i)}</Text>)
    }

    if (m[1] && m[2]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          [image: {m[1]}] {m[2]}
        </Text>
      )
    } else if (m[3] && m[4]) {
      parts.push(
        <Link key={parts.length} url={m[4]}>
          <Text color={t.color.accent} underline>
            {m[3]}
          </Text>
        </Link>
      )
    } else if (m[5]) {
      parts.push(renderAutolink(parts.length, t, m[5]))
    } else if (m[6]) {
      parts.push(
        <Text key={parts.length} strikethrough>
          {m[6]}
        </Text>
      )
    } else if (m[7]) {
      parts.push(
        <Text color={t.color.accent} dimColor key={parts.length}>
          {m[7]}
        </Text>
      )
    } else if (m[8] ?? m[9]) {
      parts.push(
        <Text bold key={parts.length}>
          {m[8] ?? m[9]}
        </Text>
      )
    } else if (m[10] ?? m[11]) {
      parts.push(
        <Text italic key={parts.length}>
          {m[10] ?? m[11]}
        </Text>
      )
    } else if (m[12]) {
      parts.push(
        <Text backgroundColor={t.color.diffAdded} color={t.color.diffAddedWord} key={parts.length}>
          {m[12]}
        </Text>
      )
    } else if (m[13]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          [{m[13]}]
        </Text>
      )
    } else if (m[14]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          ^{m[14]}
        </Text>
      )
    } else if (m[15]) {
      parts.push(
        <Text color={t.color.muted} key={parts.length}>
          _{m[15]}
        </Text>
      )
    } else if (m[16]) {
      // Bare URL — trim trailing prose punctuation into a sibling text node
      // so `see https://x.com/, which…` keeps the comma outside the link.
      const url = m[16].replace(/[),.;:!?]+$/g, '')

      parts.push(renderAutolink(parts.length, t, url))

      if (url.length < m[16].length) {
        parts.push(<Text key={parts.length}>{m[16].slice(url.length)}</Text>)
      }
    }

    last = i + m[0].length
  }

  if (last < text.length) {
    parts.push(<Text key={parts.length}>{text.slice(last)}</Text>)
  }

  return <Text>{parts.length ? parts : <Text>{text}</Text>}</Text>
}

// Cross-instance parsed-children cache: useMemo's per-instance cache dies
// on remount, so virtualization re-parses every row that scrolls back into
// view. Theme-keyed WeakMap drops stale palettes; inner Map is LRU-bounded.
const MD_CACHE_LIMIT = 512
const mdCache = new WeakMap<Theme, Map<string, ReactNode[]>>()

const cacheBucket = (t: Theme) => {
  const b = mdCache.get(t)

  if (b) {
    return b
  }

  const fresh = new Map<string, ReactNode[]>()
  mdCache.set(t, fresh)

  return fresh
}

const cacheGet = (b: Map<string, ReactNode[]>, key: string) => {
  const v = b.get(key)

  if (v) {
    b.delete(key)
    b.set(key, v)
  }

  return v
}

const cacheSet = (b: Map<string, ReactNode[]>, key: string, v: ReactNode[]) => {
  b.set(key, v)

  if (b.size > MD_CACHE_LIMIT) {
    b.delete(b.keys().next().value!)
  }
}

function MdImpl({ compact, t, text }: MdProps) {
  const nodes = useMemo(() => {
    const bucket = cacheBucket(t)
    const cacheKey = `${compact ? '1' : '0'}|${text}`
    const cached = cacheGet(bucket, cacheKey)

    if (cached) {
      return cached
    }

    const lines = ensureEmojiPresentation(text).split('\n')
    const nodes: ReactNode[] = []

    let prevKind: Kind = null
    let i = 0

    const gap = () => {
      if (nodes.length && prevKind !== 'blank') {
        nodes.push(<Text key={`gap-${nodes.length}`}> </Text>)
        prevKind = 'blank'
      }
    }

    const start = (kind: Exclude<Kind, null | 'blank'>) => {
      if (prevKind && prevKind !== 'blank' && prevKind !== kind) {
        gap()
      }

      prevKind = kind
    }

    while (i < lines.length) {
      const line = lines[i]!
      const key = nodes.length

      if (!line.trim()) {
        if (!compact) {
          gap()
        }

        i++

        continue
      }

      if (AUDIO_DIRECTIVE_RE.test(line)) {
        i++

        continue
      }

      const media = line.match(MEDIA_LINE_RE)?.[1]

      if (media) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.muted} key={key}>
            {'▸ '}

            <Link url={/^(?:\/|[a-z]:[\\/])/i.test(media) ? `file://${media}` : media}>
              <Text color={t.color.accent} underline>
                {media}
              </Text>
            </Link>
          </Text>
        )
        i++

        continue
      }

      const fence = line.match(FENCE_RE)

      if (fence) {
        const char = fence[1]![0] as '`' | '~'
        const len = fence[1]!.length
        const lang = fence[2]!.trim().toLowerCase()
        const block: string[] = []

        for (i++; i < lines.length; i++) {
          const close = lines[i]!.match(FENCE_CLOSE_RE)?.[1]

          if (close && close[0] === char && close.length >= len) {
            break
          }

          block.push(lines[i]!)
        }

        if (i < lines.length) {
          i++
        }

        if (['md', 'markdown'].includes(lang)) {
          start('paragraph')
          nodes.push(<Md compact={compact} key={key} t={t} text={block.join('\n')} />)

          continue
        }

        start('code')

        const isDiff = lang === 'diff'
        const highlighted = !isDiff && isHighlightable(lang)

        nodes.push(
          <Box flexDirection="column" key={key} paddingLeft={2}>
            {lang && !isDiff && <Text color={t.color.muted}>{'─ ' + lang}</Text>}

            {block.map((l, j) => {
              if (highlighted) {
                return (
                  <Text key={j}>
                    {highlightLine(l, lang, t).map(([color, text], kk) =>
                      color ? (
                        <Text color={color} key={kk}>
                          {text}
                        </Text>
                      ) : (
                        <Text key={kk}>{text}</Text>
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
                  color={add ? t.color.diffAddedWord : del ? t.color.diffRemovedWord : hunk ? t.color.muted : undefined}
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
            <Text color={t.color.muted}>─ math</Text>

            {block.map((l, j) => (
              <Text color={t.color.accent} key={j}>
                {l}
              </Text>
            ))}
          </Box>
        )

        continue
      }

      const heading = line.match(HEADING_RE)?.[2]

      if (heading) {
        start('heading')
        nodes.push(
          <Text bold color={t.color.accent} key={key}>
            {heading}
          </Text>
        )
        i++

        continue
      }

      if (i + 1 < lines.length && SETEXT_RE.test(lines[i + 1]!)) {
        start('heading')
        nodes.push(
          <Text bold color={t.color.accent} key={key}>
            {line.trim()}
          </Text>
        )
        i += 2

        continue
      }

      if (HR_RE.test(line)) {
        start('rule')
        nodes.push(
          <Text color={t.color.muted} key={key}>
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
          <Text color={t.color.muted} key={key}>
            [{footnote[1]}] <MdInline t={t} text={footnote[2] ?? ''} />
          </Text>
        )
        i++

        while (i < lines.length && /^\s{2,}\S/.test(lines[i]!)) {
          nodes.push(
            <Box key={`${key}-cont-${i}`} paddingLeft={2}>
              <Text color={t.color.muted}>
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
          const def = lines[i]!.match(DEF_RE)?.[1]

          if (!def) {
            break
          }

          nodes.push(
            <Text key={`${key}-def-${i}`}>
              <Text color={t.color.muted}> · </Text>
              <MdInline t={t} text={def} />
            </Text>
          )
          i++
        }

        continue
      }

      const bullet = line.match(BULLET_RE)

      if (bullet) {
        start('list')

        const task = bullet[2]!.match(TASK_RE)
        const marker = task ? (task[1]!.toLowerCase() === 'x' ? '☑' : '☐') : '•'

        nodes.push(
          <Text key={key}>
            <Text color={t.color.muted}>
              {' '.repeat(indentDepth(bullet[1]!) * 2)}
              {marker}{' '}
            </Text>

            <MdInline t={t} text={task ? task[2]! : bullet[2]!} />
          </Text>
        )
        i++

        continue
      }

      const numbered = line.match(NUMBERED_RE)

      if (numbered) {
        start('list')
        nodes.push(
          <Text key={key}>
            <Text color={t.color.muted}>
              {' '.repeat(indentDepth(numbered[1]!) * 2)}
              {numbered[2]}.{' '}
            </Text>

            <MdInline t={t} text={numbered[3]!} />
          </Text>
        )
        i++

        continue
      }

      if (QUOTE_RE.test(line)) {
        start('quote')

        const quoteLines: Array<{ depth: number; text: string }> = []

        while (i < lines.length && QUOTE_RE.test(lines[i]!)) {
          const prefix = lines[i]!.match(QUOTE_RE)?.[0] ?? ''

          quoteLines.push({ depth: (prefix.match(/>/g) ?? []).length, text: lines[i]!.slice(prefix.length) })
          i++
        }

        nodes.push(
          <Box flexDirection="column" key={key}>
            {quoteLines.map((ql, qi) => (
              <Text color={t.color.muted} key={qi}>
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

        const rows: string[][] = [splitRow(line)]

        for (i += 2; i < lines.length && lines[i]!.includes('|') && lines[i]!.trim(); i++) {
          rows.push(splitRow(lines[i]!))
        }

        nodes.push(renderTable(key, rows, t))

        continue
      }

      if (/^<\/?details\b/i.test(line)) {
        i++

        continue
      }

      const summary = line.match(/^<summary>(.*?)<\/summary>$/i)?.[1]

      if (summary) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.muted} key={key}>
            ▶ {summary}
          </Text>
        )
        i++

        continue
      }

      if (/^<\/?[^>]+>$/.test(line.trim())) {
        start('paragraph')
        nodes.push(
          <Text color={t.color.muted} key={key}>
            {line.trim()}
          </Text>
        )
        i++

        continue
      }

      if (line.includes('|') && line.trim().startsWith('|')) {
        start('table')

        const rows: string[][] = []

        while (i < lines.length && lines[i]!.trim().startsWith('|')) {
          const row = lines[i]!.trim()

          if (!/^[|\s:-]+$/.test(row)) {
            rows.push(splitRow(row))
          }

          i++
        }

        if (rows.length) {
          nodes.push(renderTable(key, rows, t))
        }

        continue
      }

      start('paragraph')
      nodes.push(<MdInline key={key} t={t} text={line} />)
      i++
    }

    cacheSet(bucket, cacheKey, nodes)

    return nodes
  }, [compact, t, text])

  return <Box flexDirection="column">{nodes}</Box>
}

export const Md = memo(MdImpl)

type Kind = 'blank' | 'code' | 'heading' | 'list' | 'paragraph' | 'quote' | 'rule' | 'table' | null

interface MdProps {
  compact?: boolean
  t: Theme
  text: string
}
