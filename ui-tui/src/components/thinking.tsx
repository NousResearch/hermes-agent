import { Box, Text } from '@hermes/ink'
import { memo, type ReactNode, useEffect, useMemo, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import {
  compactPreview,
  estimateTokensRough,
  fmtK,
  formatToolCall,
  parseToolTrailResultLine,
  pick,
  THINKING_COT_MAX,
  thinkingPreview,
  toolTrailLabel
} from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool, ActivityItem, DetailsMode, SubagentProgress, ThinkingMode } from '../types.js'

const THINK: BrailleSpinnerName[] = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse']
const TOOL: BrailleSpinnerName[] = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle']

const fmtElapsed = (ms: number) => {
  const sec = Math.max(0, ms) / 1000

  return sec < 10 ? `${sec.toFixed(1)}s` : `${Math.round(sec)}s`
}

// ── Primitives ───────────────────────────────────────────────────────

export function Spinner({ color, variant = 'think' }: { color: string; variant?: 'think' | 'tool' }) {
  const [spin] = useState(() => {
    const raw = spinners[pick(variant === 'tool' ? TOOL : THINK)]

    return { ...raw, frames: raw.frames.map(f => [...f][0] ?? '⠀') }
  })

  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % spin.frames.length), spin.interval)

    return () => clearInterval(id)
  }, [spin])

  return <Text color={color}>{spin.frames[frame]}</Text>
}

interface DetailRow {
  color: string
  content: ReactNode
  dimColor?: boolean
  key: string
}

function Detail({ color, content, dimColor }: DetailRow) {
  return (
    <Text color={color} dimColor={dimColor} wrap="wrap-trim">
      <Text dimColor>└ </Text>
      {content}
    </Text>
  )
}

function StreamCursor({
  color,
  dimColor,
  streaming = false,
  visible = false
}: {
  color: string
  dimColor?: boolean
  streaming?: boolean
  visible?: boolean
}) {
  const [on, setOn] = useState(true)

  useEffect(() => {
    if (!visible || !streaming) {
      setOn(true)

      return
    }

    const id = setInterval(() => setOn(v => !v), 420)

    return () => clearInterval(id)
  }, [streaming, visible])

  return visible ? (
    <Text color={color} dimColor={dimColor}>
      {streaming && on ? '▍' : ' '}
    </Text>
  ) : null
}

function Chevron({
  count,
  onClick,
  open,
  suffix,
  t,
  title,
  tone = 'dim'
}: {
  count?: number
  onClick: (deep?: boolean) => void
  open: boolean
  suffix?: string
  t: Theme
  title: string
  tone?: 'dim' | 'error' | 'warn'
}) {
  const color = tone === 'error' ? t.color.error : tone === 'warn' ? t.color.warn : t.color.dim

  return (
    <Box onClick={(e: { ctrlKey?: boolean; shiftKey?: boolean }) => onClick(!!e?.shiftKey || !!e?.ctrlKey)}>
      <Text color={color} dimColor={tone === 'dim'}>
        <Text color={t.color.amber}>{open ? '▾ ' : '▸ '}</Text>
        {title}
        {typeof count === 'number' ? ` (${count})` : ''}
        {suffix ? (
          <Text color={t.color.statusFg} dimColor>
            {'  '}
            {suffix}
          </Text>
        ) : null}
      </Text>
    </Box>
  )
}

function SubagentAccordion({ expanded, item, t }: { expanded: boolean; item: SubagentProgress; t: Theme }) {
  const [open, setOpen] = useState(expanded)
  const [deep, setDeep] = useState(expanded)
  const [openThinking, setOpenThinking] = useState(expanded)
  const [openTools, setOpenTools] = useState(expanded)
  const [openNotes, setOpenNotes] = useState(expanded)

  useEffect(() => {
    if (!expanded) {
      return
    }

    setOpen(true)
    setDeep(true)
    setOpenThinking(true)
    setOpenTools(true)
    setOpenNotes(true)
  }, [expanded])

  const expandAll = () => {
    setOpen(true)
    setDeep(true)
    setOpenThinking(true)
    setOpenTools(true)
    setOpenNotes(true)
  }

  const statusTone: 'dim' | 'error' | 'warn' =
    item.status === 'failed' ? 'error' : item.status === 'interrupted' ? 'warn' : 'dim'

  const prefix = item.taskCount > 1 ? `[${item.index + 1}/${item.taskCount}] ` : ''
  const goalLabel = item.goal || `Subagent ${item.index + 1}`
  const title = `${prefix}${open ? goalLabel : compactPreview(goalLabel, 60)}`
  const summary = compactPreview((item.summary || '').replace(/\s+/g, ' ').trim(), 72)

  const suffix =
    item.status === 'running'
      ? 'running'
      : `${item.status}${item.durationSeconds ? ` · ${fmtElapsed(item.durationSeconds * 1000)}` : ''}`

  const thinkingText = item.thinking.join('\n')
  const hasThinking = Boolean(thinkingText)
  const hasTools = item.tools.length > 0
  const noteRows = [...(summary ? [summary] : []), ...item.notes]
  const hasNotes = noteRows.length > 0
  const showChildren = expanded || deep

  return (
    <Box flexDirection="column" paddingLeft={1}>
      <Chevron
        onClick={shift => shift ? expandAll() : setOpen(v => { if (!v) setDeep(false); return !v })}
        open={open}
        suffix={suffix}
        t={t}
        title={title}
        tone={statusTone}
      />

      {open && (
        <Box flexDirection="column" paddingLeft={2}>
          {hasThinking && (
            <>
              <Chevron
                count={item.thinking.length}
                onClick={shift => { if (shift) expandAll(); else setOpenThinking(v => !v) }}
                open={showChildren || openThinking}
                t={t}
                title="Thinking"
              />

              {(showChildren || openThinking) && (
                <Thinking
                  active={item.status === 'running'}
                  mode="full"
                  reasoning={thinkingText}
                  streaming={item.status === 'running'}
                  t={t}
                />
              )}
            </>
          )}

          {hasTools && (
            <>
              <Chevron
                count={item.tools.length}
                onClick={shift => { if (shift) expandAll(); else setOpenTools(v => !v) }}
                open={showChildren || openTools}
                t={t}
                title="Tool calls"
              />

              {(showChildren || openTools) && (
                <Box flexDirection="column">
                  {item.tools.map((line, index) => (
                    <Text color={t.color.cornsilk} key={`${item.id}-tool-${index}`} wrap="wrap-trim">
                      <Text color={t.color.amber}>● </Text>
                      {line}
                    </Text>
                  ))}
                </Box>
              )}
            </>
          )}

          {hasNotes && (
            <>
              <Chevron
                count={noteRows.length}
                onClick={shift => { if (shift) expandAll(); else setOpenNotes(v => !v) }}
                open={showChildren || openNotes}
                t={t}
                title="Progress"
                tone={statusTone}
              />

              {(showChildren || openNotes) && (
                <Box flexDirection="column">
                  {noteRows.map((line, index) => (
                    <Text
                      color={statusTone === 'error' ? t.color.error : t.color.dim}
                      dimColor
                      key={`${item.id}-note-${index}`}
                    >
                      <Text dimColor>{index === noteRows.length - 1 ? '└ ' : '├ '}</Text>
                      {line}
                    </Text>
                  ))}
                </Box>
              )}
            </>
          )}
        </Box>
      )}
    </Box>
  )
}

// ── Thinking ─────────────────────────────────────────────────────────

export const Thinking = memo(function Thinking({
  active = false,
  mode = 'truncated',
  reasoning,
  streaming = false,
  t
}: {
  active?: boolean
  mode?: ThinkingMode
  reasoning: string
  streaming?: boolean
  t: Theme
}) {
  const preview = useMemo(() => thinkingPreview(reasoning, mode, THINKING_COT_MAX), [mode, reasoning])
  const lines = useMemo(() => preview.split('\n').map(line => line.replace(/\t/g, '  ')), [preview])

  return (
    <Box flexDirection="column">
      {preview ? (
        mode === 'full' ? (
          <Box flexDirection="row">
            <Text color={t.color.dim} dimColor>
              └{' '}
            </Text>
            <Box flexDirection="column" flexGrow={1}>
              {lines.map((line, index) => (
                <Text color={t.color.dim} dimColor key={index} wrap="wrap-trim">
                  {line || ' '}
                  {index === lines.length - 1 ? (
                    <StreamCursor color={t.color.dim} dimColor streaming={streaming} visible={active} />
                  ) : null}
                </Text>
              ))}
            </Box>
          </Box>
        ) : (
          <Text color={t.color.dim} dimColor wrap="truncate-end">
            <Text dimColor>└ </Text>
            {preview}
            <StreamCursor color={t.color.dim} dimColor streaming={streaming} visible={active} />
          </Text>
        )
      ) : active ? (
        <Text color={t.color.dim} dimColor>
          <Text dimColor>└ </Text>
          <StreamCursor color={t.color.dim} dimColor streaming={streaming} visible={active} />
        </Text>
      ) : null}
    </Box>
  )
})

// ── ToolTrail ────────────────────────────────────────────────────────

interface Group {
  color: string
  content: ReactNode
  details: DetailRow[]
  key: string
}

export const ToolTrail = memo(function ToolTrail({
  busy = false,
  detailsMode = 'collapsed',
  reasoningActive = false,
  reasoning = '',
  reasoningTokens,
  reasoningStreaming = false,
  subagents = [],
  t,
  tools = [],
  toolTokens,
  trail = [],
  activity = []
}: {
  busy?: boolean
  detailsMode?: DetailsMode
  reasoningActive?: boolean
  reasoning?: string
  reasoningTokens?: number
  reasoningStreaming?: boolean
  subagents?: SubagentProgress[]
  t: Theme
  tools?: ActiveTool[]
  toolTokens?: number
  trail?: string[]
  activity?: ActivityItem[]
}) {
  const [now, setNow] = useState(() => Date.now())
  const [openThinking, setOpenThinking] = useState(false)
  const [openTools, setOpenTools] = useState(false)
  const [openSubagents, setOpenSubagents] = useState(false)
  const [deepSubagents, setDeepSubagents] = useState(false)
  const [openMeta, setOpenMeta] = useState(false)

  useEffect(() => {
    if (!tools.length || (detailsMode === 'collapsed' && !openTools)) {
      return
    }

    const id = setInterval(() => setNow(Date.now()), 500)

    return () => clearInterval(id)
  }, [detailsMode, openTools, tools.length])

  useEffect(() => {
    if (detailsMode === 'expanded') {
      setOpenThinking(true)
      setOpenTools(true)
      setOpenSubagents(true)
      setOpenMeta(true)
    }

    if (detailsMode === 'hidden') {
      setOpenThinking(false)
      setOpenTools(false)
      setOpenSubagents(false)
      setOpenMeta(false)
    }
  }, [detailsMode])

  const cot = useMemo(() => thinkingPreview(reasoning, 'full', THINKING_COT_MAX), [reasoning])

  if (!busy && !trail.length && !tools.length && !subagents.length && !activity.length && !cot && !reasoningActive) {
    return null
  }

  // ── Build groups + meta ────────────────────────────────────────

  const groups: Group[] = []
  const meta: DetailRow[] = []
  const pushDetail = (row: DetailRow) => (groups.at(-1)?.details ?? meta).push(row)

  for (const [i, line] of trail.entries()) {
    const parsed = parseToolTrailResultLine(line)

    if (parsed) {
      groups.push({
        color: parsed.mark === '✗' ? t.color.error : t.color.cornsilk,
        content: parsed.detail ? parsed.call : `${parsed.call} ${parsed.mark}`,
        details: [],
        key: `tr-${i}`
      })

      if (parsed.detail) {
        pushDetail({
          color: parsed.mark === '✗' ? t.color.error : t.color.dim,
          content: parsed.detail,
          dimColor: parsed.mark !== '✗',
          key: `tr-${i}-d`
        })
      }

      continue
    }

    if (line.startsWith('drafting ')) {
      groups.push({
        color: t.color.cornsilk,
        content: toolTrailLabel(line.slice(9).replace(/…$/, '').trim()),
        details: [{ color: t.color.dim, content: 'drafting...', dimColor: true, key: `tr-${i}-d` }],
        key: `tr-${i}`
      })

      continue
    }

    if (line === 'analyzing tool output…') {
      pushDetail({
        color: t.color.dim,
        dimColor: true,
        key: `tr-${i}`,
        content: groups.length ? (
          <>
            <Spinner color={t.color.amber} variant="think" /> {line}
          </>
        ) : (
          line
        )
      })

      continue
    }

    meta.push({ color: t.color.dim, content: line, dimColor: true, key: `tr-${i}` })
  }

  for (const tool of tools) {
    groups.push({
      color: t.color.cornsilk,
      key: tool.id,
      details: [],
      content: (
        <>
          <Spinner color={t.color.amber} variant="tool" /> {formatToolCall(tool.name, tool.context || '')}
          {tool.startedAt ? ` (${fmtElapsed(now - tool.startedAt)})` : ''}
        </>
      )
    })
  }

  for (const item of activity.slice(-4)) {
    const glyph = item.tone === 'error' ? '✗' : item.tone === 'warn' ? '!' : '·'
    const color = item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim
    meta.push({ color, content: `${glyph} ${item.text}`, dimColor: item.tone === 'info', key: `a-${item.id}` })
  }

  // ── Derived ────────────────────────────────────────────────────

  const hasTools = groups.length > 0
  const hasSubagents = subagents.length > 0
  const hasMeta = meta.length > 0
  const hasThinking = !!cot || reasoningActive || (busy && !hasTools)
  const thinkingLive = reasoningActive || reasoningStreaming

  const tokenCount = reasoningTokens !== undefined ? reasoningTokens : reasoning ? estimateTokensRough(reasoning) : 0

  const toolTokenCount = toolTokens ?? 0
  const totalTokenCount = tokenCount + toolTokenCount
  const thinkingTokensLabel = tokenCount > 0 ? `~${fmtK(tokenCount)} tokens` : null

  const toolTokensLabel = toolTokens !== undefined && toolTokens > 0 ? `~${fmtK(toolTokens)} tokens` : undefined

  const totalTokensLabel = tokenCount > 0 && toolTokenCount > 0 ? `~${fmtK(totalTokenCount)} total` : null

  // ── Hidden: errors/warnings only ──────────────────────────────

  if (detailsMode === 'hidden') {
    const alerts = activity.filter(i => i.tone !== 'info').slice(-2)

    return alerts.length ? (
      <Box flexDirection="column">
        {alerts.map(i => (
          <Text color={i.tone === 'error' ? t.color.error : t.color.warn} key={`ha-${i.id}`}>
            {i.tone === 'error' ? '✗' : '!'} {i.text}
          </Text>
        ))}
      </Box>
    ) : null
  }

  // ── Shared render fragments ────────────────────────────────────

  const thinkingBlock = hasThinking ? (
    busy ? (
      <Thinking active={reasoningActive} mode="full" reasoning={reasoning} streaming={reasoningStreaming} t={t} />
    ) : cot ? (
      <Detail color={t.color.dim} content={cot} dimColor key="cot" />
    ) : (
      <Detail
        color={t.color.dim}
        content={<StreamCursor color={t.color.dim} dimColor streaming={reasoningStreaming} visible={reasoningActive} />}
        dimColor
        key="cot"
      />
    )
  ) : null

  const toolBlock = hasTools
    ? groups.map(g => (
        <Box flexDirection="column" key={g.key}>
          <Text color={g.color}>
            <Text color={t.color.amber}>● </Text>
            {g.content}
          </Text>
          {g.details.map(d => (
            <Detail {...d} key={d.key} />
          ))}
        </Box>
      ))
    : null

  const subagentBlock = hasSubagents
    ? subagents.map(item => (
        <SubagentAccordion expanded={detailsMode === 'expanded' || deepSubagents} item={item} key={item.id} t={t} />
      ))
    : null

  const metaBlock = hasMeta
    ? meta.map((row, i) => (
        <Text color={row.color} dimColor={row.dimColor} key={row.key}>
          <Text dimColor>{i === meta.length - 1 ? '└ ' : '├ '}</Text>
          {row.content}
        </Text>
      ))
    : null

  const totalBlock = totalTokensLabel ? (
    <Text color={t.color.statusFg} dimColor>
      <Text color={t.color.amber}>Σ </Text>
      {totalTokensLabel}
    </Text>
  ) : null

  // ── Expanded: flat, no accordions ──────────────────────────────

  if (detailsMode === 'expanded') {
    return (
      <Box flexDirection="column">
        {thinkingBlock}
        {toolBlock}
        {subagentBlock}
        {metaBlock}
        {totalBlock}
      </Box>
    )
  }

  // ── Collapsed: clickable accordions ────────────────────────────

  const expandAll = () => {
    setOpenThinking(true)
    setOpenTools(true)
    setOpenSubagents(true)
    setDeepSubagents(true)
    setOpenMeta(true)
  }

  const metaTone: 'dim' | 'error' | 'warn' = activity.some(i => i.tone === 'error')
    ? 'error'
    : activity.some(i => i.tone === 'warn')
      ? 'warn'
      : 'dim'

  return (
    <Box flexDirection="column">
      {hasThinking && (
        <>
          <Box onClick={(e: { ctrlKey?: boolean; shiftKey?: boolean }) => (e?.shiftKey || e?.ctrlKey) ? expandAll() : setOpenThinking(v => !v)}>
            <Text color={t.color.dim} dimColor={!thinkingLive}>
              <Text color={t.color.amber}>{openThinking ? '▾ ' : '▸ '}</Text>
              <Text bold={thinkingLive} color={thinkingLive ? t.color.cornsilk : t.color.dim} dimColor={!thinkingLive}>
                Thinking
              </Text>
              {thinkingTokensLabel ? (
                <Text color={t.color.statusFg} dimColor>
                  {'  '}
                  {thinkingTokensLabel}
                </Text>
              ) : null}
            </Text>
          </Box>
          {openThinking && thinkingBlock}
        </>
      )}

      {hasTools && (
        <>
          <Chevron
            count={groups.length}
            onClick={shift => shift ? expandAll() : setOpenTools(v => !v)}
            open={openTools}
            suffix={toolTokensLabel}
            t={t}
            title="Tool calls"
          />
          {openTools && toolBlock}
        </>
      )}

      {hasSubagents && (
        <>
          <Chevron
            count={subagents.length}
            onClick={shift => {
              if (shift) {
                expandAll()
                setDeepSubagents(true)
              } else {
                setOpenSubagents(v => !v)
                setDeepSubagents(false)
              }
            }}
            open={openSubagents}
            t={t}
            title="Subagents"
          />
          {openSubagents && subagentBlock}
        </>
      )}

      {hasMeta && (
        <>
          <Chevron
            count={meta.length}
            onClick={shift => shift ? expandAll() : setOpenMeta(v => !v)}
            open={openMeta}
            t={t}
            title="Activity"
            tone={metaTone}
          />
          {openMeta && metaBlock}
        </>
      )}

      {totalBlock}
    </Box>
  )
})
