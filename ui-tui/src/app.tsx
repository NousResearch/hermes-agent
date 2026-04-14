import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  AlternateScreen,
  Box,
  NoSelect,
  ScrollBox,
  type ScrollBoxHandle,
  Text,
  useApp,
  useHasSelection,
  useInput,
  useSelection,
  useStdout
} from '@hermes/ink'
import { type RefObject, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'

import { Banner, Panel, SessionPanel } from './components/branding.js'
import { MaskedPrompt } from './components/maskedPrompt.js'
import { MessageLine } from './components/messageLine.js'
import { ModelPicker } from './components/modelPicker.js'
import { ApprovalPrompt, ClarifyPrompt } from './components/prompts.js'
import { QueuedMessages } from './components/queuedMessages.js'
import { SessionPicker } from './components/sessionPicker.js'
import { type PasteEvent, TextInput } from './components/textInput.js'
import { ToolTrail } from './components/thinking.js'
import { HOTKEYS, INTERPOLATION_RE, PLACEHOLDERS, ZERO } from './constants.js'
import { type GatewayClient, type GatewayEvent } from './gatewayClient.js'
import { useCompletion } from './hooks/useCompletion.js'
import { useInputHistory } from './hooks/useInputHistory.js'
import { useQueue } from './hooks/useQueue.js'
import { useVirtualHistory } from './hooks/useVirtualHistory.js'
import { writeOsc52Clipboard } from './lib/osc52.js'
import { asRpcResult, rpcErrorMessage } from './lib/rpc.js'
import {
  buildToolTrailLine,
  fmtK,
  hasInterpolation,
  isToolTrailResultLine,
  isTransientTrailLine,
  pasteTokenLabel,
  pick,
  sameToolTrailGroup,
  stripTrailingPasteNewlines,
  toolTrailLabel,
  userDisplay
} from './lib/text.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'
import type {
  ActiveTool,
  ActivityItem,
  ApprovalReq,
  ClarifyReq,
  DetailsMode,
  Msg,
  PanelSection,
  SecretReq,
  SessionInfo,
  SlashCatalog,
  SudoReq,
  Usage
} from './types.js'

// ── Constants ────────────────────────────────────────────────────────

const PLACEHOLDER = pick(PLACEHOLDERS)
const STARTUP_RESUME_ID = (process.env.HERMES_TUI_RESUME ?? '').trim()

const LARGE_PASTE = { chars: 8000, lines: 80 }
const MAX_HISTORY = 800
const REASONING_PULSE_MS = 700
const STREAM_BATCH_MS = 16
const WHEEL_SCROLL_STEP = 3
const MOUSE_TRACKING = !/^(1|true|yes|on)$/.test((process.env.HERMES_TUI_DISABLE_MOUSE ?? '').trim().toLowerCase())
const PASTE_SNIPPET_RE = /\[\[[^\n]*?\]\]/g

const DETAILS_MODES: DetailsMode[] = ['hidden', 'collapsed', 'expanded']

const parseDetailsMode = (v: unknown): DetailsMode | null => {
  const s = typeof v === 'string' ? v.trim().toLowerCase() : ''

  return DETAILS_MODES.includes(s as DetailsMode) ? (s as DetailsMode) : null
}

const resolveDetailsMode = (d: any): DetailsMode =>
  parseDetailsMode(d?.details_mode) ??
  { full: 'expanded' as const, collapsed: 'collapsed' as const, truncated: 'collapsed' as const }[
    String(d?.thinking_mode ?? '')
      .trim()
      .toLowerCase()
  ] ??
  'collapsed'

const nextDetailsMode = (m: DetailsMode): DetailsMode =>
  DETAILS_MODES[(DETAILS_MODES.indexOf(m) + 1) % DETAILS_MODES.length]!

// ── Pure helpers ─────────────────────────────────────────────────────

type PasteSnippet = { label: string; text: string }

const introMsg = (info: SessionInfo): Msg => ({ role: 'system', text: '', kind: 'intro', info })

const shortCwd = (cwd: string, max = 28) => {
  const p = process.env.HOME && cwd.startsWith(process.env.HOME) ? `~${cwd.slice(process.env.HOME.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}

const imageTokenMeta = (info: { height?: number; token_estimate?: number; width?: number } | null | undefined) => {
  const dims = info?.width && info?.height ? `${info.width}x${info.height}` : ''

  const tok =
    typeof info?.token_estimate === 'number' && info.token_estimate > 0 ? `~${fmtK(info.token_estimate)} tok` : ''

  return [dims, tok].filter(Boolean).join(' · ')
}

const looksLikeSlashCommand = (text: string) => {
  if (!text.startsWith('/')) {
    return false
  }

  const first = text.split(/\s+/, 1)[0] || ''

  return !first.slice(1).includes('/')
}

const toTranscriptMessages = (rows: unknown): Msg[] => {
  if (!Array.isArray(rows)) {
    return []
  }

  const result: Msg[] = []
  let pendingTools: string[] = []

  for (const row of rows) {
    if (!row || typeof row !== 'object') {
      continue
    }

    const role = (row as any).role
    const text = (row as any).text

    if (role === 'tool') {
      const name = (row as any).name ?? 'tool'
      const ctx = (row as any).context ?? ''
      pendingTools.push(buildToolTrailLine(name, ctx))

      continue
    }

    if (typeof text !== 'string' || !text.trim()) {
      continue
    }

    if (role === 'assistant') {
      const msg: Msg = { role, text }

      if (pendingTools.length) {
        msg.tools = pendingTools
        pendingTools = []
      }

      result.push(msg)

      continue
    }

    if (role === 'user' || role === 'system') {
      pendingTools = []
      result.push({ role, text })
    }
  }

  return result
}

// ── StatusRule ────────────────────────────────────────────────────────

function ctxBarColor(pct: number | undefined, t: Theme) {
  if (pct == null) {
    return t.color.dim
  }

  if (pct >= 95) {
    return t.color.statusCritical
  }

  if (pct > 80) {
    return t.color.statusBad
  }

  if (pct >= 50) {
    return t.color.statusWarn
  }

  return t.color.statusGood
}

function ctxBar(pct: number | undefined, w = 10) {
  const p = Math.max(0, Math.min(100, pct ?? 0))
  const filled = Math.round((p / 100) * w)

  return '█'.repeat(filled) + '░'.repeat(w - filled)
}

function fmtDuration(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000))
  const hours = Math.floor(total / 3600)
  const mins = Math.floor((total % 3600) / 60)
  const secs = total % 60

  if (hours > 0) {
    return `${hours}h ${mins}m`
  }

  if (mins > 0) {
    return `${mins}m ${secs}s`
  }

  return `${secs}s`
}

function StatusRule({
  cwdLabel,
  cols,
  status,
  statusColor,
  model,
  usage,
  bgCount,
  durationLabel,
  voiceLabel,
  t
}: {
  cwdLabel: string
  cols: number
  status: string
  statusColor: string
  model: string
  usage: Usage
  bgCount: number
  durationLabel?: string
  voiceLabel?: string
  t: Theme
}) {
  const pct = usage.context_percent
  const barColor = ctxBarColor(pct, t)

  const ctxLabel = usage.context_max
    ? `${fmtK(usage.context_used ?? 0)}/${fmtK(usage.context_max)}`
    : usage.total > 0
      ? `${fmtK(usage.total)} tok`
      : ''

  const pctLabel = pct != null ? `${pct}%` : ''
  const bar = usage.context_max ? ctxBar(pct) : ''

  const leftWidth = Math.max(12, cols - cwdLabel.length - 3)

  return (
    <Box>
      <Box flexShrink={1} width={leftWidth}>
        <Text color={t.color.bronze} wrap="truncate-end">
          {'─ '}
          <Text color={statusColor}>{status}</Text>
          <Text color={t.color.dim}> │ {model}</Text>
          {ctxLabel ? <Text color={t.color.dim}> │ {ctxLabel}</Text> : null}
          {bar ? (
            <Text color={t.color.dim}>
              {' │ '}
              <Text color={barColor}>[{bar}]</Text> <Text color={barColor}>{pctLabel}</Text>
            </Text>
          ) : null}
          {durationLabel ? <Text color={t.color.dim}> │ {durationLabel}</Text> : null}
          {voiceLabel ? <Text color={t.color.dim}> │ {voiceLabel}</Text> : null}
          {bgCount > 0 ? <Text color={t.color.dim}> │ {bgCount} bg</Text> : null}
        </Text>
      </Box>
      <Text color={t.color.bronze}> ─ </Text>
      <Text color={t.color.label}>{cwdLabel}</Text>
    </Box>
  )
}

// ── PromptBox ────────────────────────────────────────────────────────

function PromptBox({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <Box borderColor={color} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
      {children}
    </Box>
  )
}

const upperBound = (arr: ArrayLike<number>, target: number) => {
  let lo = 0
  let hi = arr.length

  while (lo < hi) {
    const mid = (lo + hi) >> 1

    if (arr[mid]! <= target) {
      lo = mid + 1
    } else {
      hi = mid
    }
  }

  return lo
}

function StickyPromptTracker({
  messages,
  offsets,
  scrollRef,
  onChange
}: {
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  scrollRef: RefObject<ScrollBoxHandle | null>
  onChange: (text: string) => void
}) {
  useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => {
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      const top = Math.max(0, s.getScrollTop() + s.getPendingDelta())

      return s.isSticky() ? -1 - top : top
    },
    () => NaN
  )

  const s = scrollRef.current
  const top = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))

  let text = ''

  if (!(s?.isSticky() ?? true) && messages.length) {
    const first = Math.max(0, Math.min(messages.length - 1, upperBound(offsets, top) - 1))

    if (!(messages[first]?.role === 'user' && (offsets[first] ?? 0) + 1 >= top)) {
      for (let i = first - 1; i >= 0; i--) {
        if (messages[i]?.role !== 'user') {
          continue
        }

        if ((offsets[i] ?? 0) + 1 >= top) {
          continue
        }

        text = userDisplay(messages[i]!.text.trim()).replace(/\s+/g, ' ').trim()

        break
      }
    }
  }

  useEffect(() => onChange(text), [onChange, text])

  return null
}

function TranscriptScrollbar({ scrollRef, t }: { scrollRef: RefObject<ScrollBoxHandle | null>; t: Theme }) {
  useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => {
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      return `${s.getScrollTop() + s.getPendingDelta()}:${s.getViewportHeight()}:${s.getScrollHeight()}`
    },
    () => ''
  )

  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<number | null>(null)

  const s = scrollRef.current
  const vp = Math.max(0, s?.getViewportHeight() ?? 0)

  if (!vp) {
    return <Box width={1} />
  }

  const total = Math.max(vp, s?.getScrollHeight() ?? vp)
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const pos = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0

  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }
    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => setGrab(null)}
      width={1}
    >
      {Array.from({ length: vp }, (_, i) => {
        const active = i >= thumbTop && i < thumbTop + thumb

        const color = active
          ? grab !== null
            ? t.color.gold
            : hover
              ? t.color.amber
              : t.color.bronze
          : hover
            ? t.color.bronze
            : t.color.dim

        return (
          <Text color={color} dimColor={!active && !hover} key={i}>
            {scrollable ? (active ? '┃' : '│') : ' '}
          </Text>
        )
      })}
    </Box>
  )
}

// ── App ──────────────────────────────────────────────────────────────

export function App({ gw }: { gw: GatewayClient }) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const [cols, setCols] = useState(stdout?.columns ?? 80)

  useEffect(() => {
    if (!stdout) {
      return
    }

    const sync = () => setCols(stdout.columns ?? 80)
    stdout.on('resize', sync)

    // Enable bracketed paste so image-only clipboard paste reaches the app
    if (stdout.isTTY) {
      stdout.write('\x1b[?2004h')
    }

    return () => {
      stdout.off('resize', sync)

      if (stdout.isTTY) {
        stdout.write('\x1b[?2004l')
      }
    }
  }, [stdout])

  // ── State ────────────────────────────────────────────────────────

  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [messages, setMessages] = useState<Msg[]>([])
  const [historyItems, setHistoryItems] = useState<Msg[]>([])
  const [status, setStatus] = useState('summoning hermes…')
  const [sid, setSid] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(DEFAULT_THEME)
  const [info, setInfo] = useState<SessionInfo | null>(null)
  const [activity, setActivity] = useState<ActivityItem[]>([])
  const [tools, setTools] = useState<ActiveTool[]>([])
  const [busy, setBusy] = useState(false)
  const [compact, setCompact] = useState(false)
  const [usage, setUsage] = useState<Usage>(ZERO)
  const [clarify, setClarify] = useState<ClarifyReq | null>(null)
  const [approval, setApproval] = useState<ApprovalReq | null>(null)
  const [sudo, setSudo] = useState<SudoReq | null>(null)
  const [secret, setSecret] = useState<SecretReq | null>(null)
  const [modelPicker, setModelPicker] = useState(false)
  const [picker, setPicker] = useState(false)
  const [reasoning, setReasoning] = useState('')
  const [reasoningActive, setReasoningActive] = useState(false)
  const [reasoningStreaming, setReasoningStreaming] = useState(false)
  const [statusBar, setStatusBar] = useState(true)
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [stickyPrompt, setStickyPrompt] = useState('')
  const [pasteSnips, setPasteSnips] = useState<PasteSnippet[]>([])
  const [streaming, setStreaming] = useState('')
  const [turnTrail, setTurnTrail] = useState<string[]>([])
  const [bgTasks, setBgTasks] = useState<Set<string>>(new Set())
  const [catalog, setCatalog] = useState<SlashCatalog | null>(null)
  const [pager, setPager] = useState<{ lines: string[]; offset: number; title?: string } | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [voiceRecording, setVoiceRecording] = useState(false)
  const [voiceProcessing, setVoiceProcessing] = useState(false)
  const [sessionStartedAt, setSessionStartedAt] = useState(() => Date.now())
  const [bellOnComplete, setBellOnComplete] = useState(false)
  const [clockNow, setClockNow] = useState(() => Date.now())
  const [detailsMode, setDetailsMode] = useState<DetailsMode>('collapsed')

  // ── Refs ─────────────────────────────────────────────────────────

  const activityIdRef = useRef(0)
  const toolCompleteRibbonRef = useRef<{ label: string; line: string } | null>(null)
  const buf = useRef('')
  const interruptedRef = useRef(false)
  const reasoningRef = useRef('')
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const lastEmptyAt = useRef(0)
  const lastStatusNoteRef = useRef('')
  const protocolWarnedRef = useRef(false)
  const colsRef = useRef(cols)
  const turnToolsRef = useRef<string[]>([])
  const persistedToolLabelsRef = useRef<Set<string>>(new Set())
  const streamTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reasoningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reasoningStreamingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const busyRef = useRef(busy)
  const sidRef = useRef<string | null>(sid)
  const scrollRef = useRef<ScrollBoxHandle | null>(null)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  const configMtimeRef = useRef(0)
  colsRef.current = cols
  busyRef.current = busy
  sidRef.current = sid
  reasoningRef.current = reasoning

  // ── Hooks ────────────────────────────────────────────────────────

  const hasSelection = useHasSelection()
  const selection = useSelection()

  const { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue } =
    useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, blocked(), gw)

  const pulseReasoningStreaming = useCallback(() => {
    if (reasoningStreamingTimerRef.current) {
      clearTimeout(reasoningStreamingTimerRef.current)
    }

    setReasoningActive(true)
    setReasoningStreaming(true)
    reasoningStreamingTimerRef.current = setTimeout(() => {
      reasoningStreamingTimerRef.current = null
      setReasoningStreaming(false)
    }, REASONING_PULSE_MS)
  }, [])

  const scheduleStreaming = useCallback(() => {
    if (streamTimerRef.current) {
      return
    }

    streamTimerRef.current = setTimeout(() => {
      streamTimerRef.current = null
      setStreaming(buf.current.trimStart())
    }, STREAM_BATCH_MS)
  }, [])

  const scheduleReasoning = useCallback(() => {
    if (reasoningTimerRef.current) {
      return
    }

    reasoningTimerRef.current = setTimeout(() => {
      reasoningTimerRef.current = null
      setReasoning(reasoningRef.current)
    }, STREAM_BATCH_MS)
  }, [])

  const endReasoningPhase = useCallback(() => {
    if (reasoningStreamingTimerRef.current) {
      clearTimeout(reasoningStreamingTimerRef.current)
      reasoningStreamingTimerRef.current = null
    }

    setReasoningStreaming(false)
    setReasoningActive(false)
  }, [])

  useEffect(
    () => () => {
      if (streamTimerRef.current) {
        clearTimeout(streamTimerRef.current)
      }

      if (reasoningTimerRef.current) {
        clearTimeout(reasoningTimerRef.current)
      }

      if (reasoningStreamingTimerRef.current) {
        clearTimeout(reasoningStreamingTimerRef.current)
      }
    },
    []
  )

  function blocked() {
    return !!(clarify || approval || modelPicker || picker || secret || sudo || pager)
  }

  const empty = !messages.length
  const isBlocked = blocked()

  const virtualRows = useMemo(
    () =>
      historyItems.map((msg, index) => ({
        index,
        key: `${index}:${msg.role}:${msg.kind ?? ''}:${msg.text.slice(0, 40)}`,
        msg
      })),
    [historyItems]
  )

  const virtualHistory = useVirtualHistory(scrollRef, virtualRows)

  const scrollWithSelection = useCallback(
    (delta: number) => {
      const s = scrollRef.current

      const sel = selection.getState() as {
        anchor?: { row: number }
        focus?: { row: number }
        isDragging?: boolean
      } | null

      if (!s || !sel?.anchor || !sel.focus) {
        s?.scrollBy(delta)

        return
      }

      const top = s.getViewportTop()
      const bottom = top + s.getViewportHeight() - 1

      if (sel.anchor.row < top || sel.anchor.row > bottom) {
        s.scrollBy(delta)

        return
      }

      if (!sel.isDragging && (sel.focus.row < top || sel.focus.row > bottom)) {
        s.scrollBy(delta)

        return
      }

      const max = Math.max(0, s.getScrollHeight() - s.getViewportHeight())
      const cur = s.getScrollTop() + s.getPendingDelta()
      const actual = Math.max(0, Math.min(max, cur + delta)) - cur

      if (actual === 0) {
        return
      }

      if (actual > 0) {
        selection.captureScrolledRows(top, top + actual - 1, 'above')
        sel.isDragging ? selection.shiftAnchor(-actual, top, bottom) : selection.shiftSelection(-actual, top, bottom)
      } else {
        const amount = -actual
        selection.captureScrolledRows(bottom - amount + 1, bottom, 'below')
        sel.isDragging ? selection.shiftAnchor(amount, top, bottom) : selection.shiftSelection(amount, top, bottom)
      }

      s.scrollBy(delta)
    },
    [selection]
  )

  // ── Resize RPC ───────────────────────────────────────────────────

  useEffect(() => {
    if (!sid || !stdout) {
      return
    }

    const onResize = () => rpc('terminal.resize', { session_id: sid, cols: stdout.columns ?? 80 })
    stdout.on('resize', onResize)

    return () => {
      stdout.off('resize', onResize)
    }
  }, [sid, stdout]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const id = setInterval(() => setClockNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [])

  // ── Core actions ─────────────────────────────────────────────────

  const appendMessage = useCallback((msg: Msg) => {
    const cap = (items: Msg[]) =>
      items.length <= MAX_HISTORY
        ? items
        : items[0]?.kind === 'intro'
          ? [items[0]!, ...items.slice(-(MAX_HISTORY - 1))]
          : items.slice(-MAX_HISTORY)

    setMessages(prev => cap([...prev, msg]))
    setHistoryItems(prev => cap([...prev, msg]))
  }, [])

  const sys = useCallback((text: string) => appendMessage({ role: 'system' as const, text }), [appendMessage])

  const page = useCallback((text: string, title?: string) => {
    const lines = text.split('\n')
    setPager({ lines, offset: 0, title })
  }, [])

  const panel = useCallback(
    (title: string, sections: PanelSection[]) => {
      appendMessage({ role: 'system', text: '', kind: 'panel', panelData: { title, sections } })
    },
    [appendMessage]
  )

  const maybeWarn = useCallback(
    (value: any) => {
      if (value?.warning) {
        sys(`warning: ${value.warning}`)
      }
    },
    [sys]
  )

  const pushActivity = useCallback((text: string, tone: ActivityItem['tone'] = 'info', replaceLabel?: string) => {
    setActivity(prev => {
      const base = replaceLabel ? prev.filter(a => !sameToolTrailGroup(replaceLabel, a.text)) : prev

      if (base.at(-1)?.text === text && base.at(-1)?.tone === tone) {
        return base
      }

      activityIdRef.current++

      return [...base, { id: activityIdRef.current, text, tone }].slice(-8)
    })
  }, [])

  const setTrail = (next: string[]) => {
    turnToolsRef.current = next

    return next
  }

  const pruneTransient = useCallback(() => {
    setTurnTrail(prev => {
      const next = prev.filter(l => !isTransientTrailLine(l))

      return next.length === prev.length ? prev : setTrail(next)
    })
  }, [])

  const pushTrail = useCallback((line: string) => {
    setTurnTrail(prev =>
      prev.at(-1) === line ? prev : setTrail([...prev.filter(l => !isTransientTrailLine(l)), line].slice(-8))
    )
  }, [])

  const rpc = useCallback(
    async (method: string, params: Record<string, unknown> = {}) => {
      try {
        const result = asRpcResult(await gw.request(method, params))

        if (result) {
          return result
        }

        sys(`error: invalid response: ${method}`)
      } catch (e) {
        sys(`error: ${rpcErrorMessage(e)}`)
      }

      return null
    },
    [gw, sys]
  )

  const answerClarify = useCallback(
    (answer: string) => {
      if (!clarify) {
        return
      }

      const label = toolTrailLabel('clarify')

      setTrail(turnToolsRef.current.filter(l => !sameToolTrailGroup(label, l)))
      setTurnTrail(turnToolsRef.current)

      rpc('clarify.respond', { answer, request_id: clarify.requestId }).then(r => {
        if (!r) {
          return
        }

        if (answer) {
          persistedToolLabelsRef.current.add(label)
          appendMessage({
            role: 'system',
            text: '',
            kind: 'trail',
            tools: [buildToolTrailLine('clarify', clarify.question)]
          })
          appendMessage({ role: 'user', text: answer })
          setStatus('running…')
        } else {
          sys('prompt cancelled')
        }

        setClarify(null)
      })
    },
    [appendMessage, clarify, rpc, sys]
  )

  useEffect(() => {
    if (!sid) {
      return
    }

    rpc('voice.toggle', { action: 'status' }).then((r: any) => setVoiceEnabled(!!r?.enabled))
    rpc('config.get', { key: 'mtime' }).then((r: any) => {
      configMtimeRef.current = Number(r?.mtime ?? 0)
    })
    rpc('config.get', { key: 'full' }).then((r: any) => {
      const display = r?.config?.display ?? {}
      setBellOnComplete(!!display?.bell_on_complete)
      setCompact(!!display?.tui_compact)
      setStatusBar(display?.tui_statusbar !== false)
      setDetailsMode(resolveDetailsMode(display))
    })
  }, [rpc, sid])

  useEffect(() => {
    if (!sid) {
      return
    }

    const id = setInterval(() => {
      rpc('config.get', { key: 'mtime' }).then((r: any) => {
        const next = Number(r?.mtime ?? 0)

        if (configMtimeRef.current && next && next !== configMtimeRef.current) {
          configMtimeRef.current = next
          rpc('reload.mcp', { session_id: sid }).then(r => {
            if (!r) {
              return
            }

            pushActivity('MCP reloaded after config change')
          })
          rpc('config.get', { key: 'full' }).then((cfg: any) => {
            const display = cfg?.config?.display ?? {}
            setBellOnComplete(!!display?.bell_on_complete)
            setCompact(!!display?.tui_compact)
            setStatusBar(display?.tui_statusbar !== false)
            setDetailsMode(resolveDetailsMode(display))
          })
        } else if (!configMtimeRef.current && next) {
          configMtimeRef.current = next
        }
      })
    }, 5000)

    return () => clearInterval(id)
  }, [pushActivity, rpc, sid])

  const idle = () => {
    endReasoningPhase()
    setTools([])
    setTurnTrail([])
    setBusy(false)
    setClarify(null)
    setApproval(null)
    setSudo(null)
    setSecret(null)

    if (streamTimerRef.current) {
      clearTimeout(streamTimerRef.current)
      streamTimerRef.current = null
    }

    setStreaming('')
    buf.current = ''
  }

  const clearReasoning = () => {
    if (reasoningTimerRef.current) {
      clearTimeout(reasoningTimerRef.current)
      reasoningTimerRef.current = null
    }

    reasoningRef.current = ''
    setReasoning('')
  }

  const die = () => {
    gw.kill()
    exit()
  }

  const clearIn = () => {
    setInput('')
    setInputBuf([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }

  const resetSession = () => {
    idle()
    clearReasoning()
    setVoiceRecording(false)
    setVoiceProcessing(false)
    setSid(null as any) // will be set by caller
    setInfo(null)
    setHistoryItems([])
    setMessages([])
    setStickyPrompt('')
    setPasteSnips([])
    setActivity([])
    setBgTasks(new Set())
    setUsage(ZERO)
    turnToolsRef.current = []
    lastStatusNoteRef.current = ''
    protocolWarnedRef.current = false
  }

  const resetVisibleHistory = (info: SessionInfo | null = null) => {
    idle()
    clearReasoning()
    setMessages([])
    setHistoryItems(info ? [introMsg(info)] : [])
    setInfo(info)
    setUsage(info?.usage ? { ...ZERO, ...info.usage } : ZERO)
    setStickyPrompt('')
    setPasteSnips([])
    setActivity([])
    setLastUserMsg('')
    turnToolsRef.current = []
    persistedToolLabelsRef.current.clear()
  }

  const trimLastExchange = (items: Msg[]) => {
    const q = [...items]

    while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
      q.pop()
    }

    if (q.at(-1)?.role === 'user') {
      q.pop()
    }

    return q
  }

  const guardBusySessionSwitch = useCallback(
    (what = 'switch sessions') => {
      if (!busyRef.current) {
        return false
      }

      sys(`interrupt the current turn before trying to ${what}`)

      return true
    },
    [sys]
  )

  const closeSession = useCallback(
    (targetSid?: string | null) => {
      if (!targetSid) {
        return Promise.resolve(null)
      }

      return rpc('session.close', { session_id: targetSid })
    },
    [rpc]
  )

  // ── Session management ───────────────────────────────────────────

  const newSession = useCallback(
    async (msg?: string) => {
      await closeSession(sidRef.current)

      return rpc('session.create', { cols: colsRef.current }).then((r: any) => {
        if (!r) {
          setStatus('ready')

          return
        }

        resetSession()
        setSid(r.session_id)
        setSessionStartedAt(Date.now())
        setStatus('ready')

        if (r.info) {
          setInfo(r.info)

          if (r.info.usage) {
            setUsage(prev => ({ ...prev, ...r.info.usage }))
          }

          setHistoryItems([introMsg(r.info)])
        } else {
          setInfo(null)
        }

        if (r.info?.credential_warning) {
          sys(`warning: ${r.info.credential_warning}`)
        }

        if (msg) {
          sys(msg)
        }
      })
    },
    [closeSession, rpc, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      setPicker(false)
      setStatus('resuming…')
      closeSession(sidRef.current === id ? null : sidRef.current).then(() =>
        gw
          .request('session.resume', { cols: colsRef.current, session_id: id })
          .then((raw: any) => {
            const r = asRpcResult(raw)

            if (!r) {
              sys('error: invalid response: session.resume')
              setStatus('ready')

              return
            }

            resetSession()
            setSid(r.session_id)
            setSessionStartedAt(Date.now())
            setInfo(r.info ?? null)
            const resumed = toTranscriptMessages(r.messages)

            if (r.info?.usage) {
              setUsage(prev => ({ ...prev, ...r.info.usage }))
            }

            setMessages(resumed)
            setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
            setStatus('ready')
          })
          .catch((e: Error) => {
            sys(`error: ${e.message}`)
            setStatus('ready')
          })
      )
    },
    [closeSession, gw, sys]
  )

  // ── Paste pipeline ───────────────────────────────────────────────

  const paste = useCallback(
    (quiet = false) =>
      rpc('clipboard.paste', { session_id: sid }).then((r: any) => {
        if (!r) {
          return
        }

        if (r.attached) {
          const meta = imageTokenMeta(r)
          sys(`📎 Image #${r.count} attached from clipboard${meta ? ` · ${meta}` : ''}`)

          return
        }

        quiet || sys(r.message || 'No image found in clipboard')
      }),
    [rpc, sid, sys]
  )

  const handleTextPaste = useCallback(
    ({ bracketed, cursor, hotkey, text, value }: PasteEvent) => {
      if (hotkey) {
        void paste(false)

        return null
      }

      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        if (bracketed) {
          void paste(true)
        }

        return null
      }

      const lineCount = cleanedText.split('\n').length

      if (cleanedText.length < LARGE_PASTE.chars && lineCount < LARGE_PASTE.lines) {
        return {
          cursor: cursor + cleanedText.length,
          value: value.slice(0, cursor) + cleanedText + value.slice(cursor)
        }
      }

      const label = pasteTokenLabel(cleanedText, lineCount)
      const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
      const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
      const insert = `${lead}${label}${tail}`

      setPasteSnips(prev => [...prev, { label, text: cleanedText }].slice(-32))

      return {
        cursor: cursor + insert.length,
        value: value.slice(0, cursor) + insert + value.slice(cursor)
      }
    },
    [paste]
  )

  // ── Send ─────────────────────────────────────────────────────────

  const send = (text: string) => {
    const expandPasteSnips = (value: string) => {
      const byLabel = new Map<string, string[]>()

      for (const item of pasteSnips) {
        const list = byLabel.get(item.label)
        list ? list.push(item.text) : byLabel.set(item.label, [item.text])
      }

      return value.replace(PASTE_SNIPPET_RE, token => byLabel.get(token)?.shift() ?? token)
    }

    const startSubmit = (displayText: string, submitText: string) => {
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current)
        statusTimerRef.current = null
      }

      setLastUserMsg(text)
      appendMessage({ role: 'user', text: displayText })
      setBusy(true)
      setStatus('running…')
      buf.current = ''
      interruptedRef.current = false

      gw.request('prompt.submit', { session_id: sid, text: submitText }).catch((e: Error) => {
        sys(`error: ${e.message}`)
        setStatus('ready')
        setBusy(false)
      })
    }

    gw.request('input.detect_drop', { session_id: sid, text })
      .then((r: any) => {
        if (r?.matched) {
          if (r.is_image) {
            const meta = imageTokenMeta(r)
            pushActivity(`attached image: ${r.name}${meta ? ` · ${meta}` : ''}`)
          } else {
            pushActivity(`detected file: ${r.name}`)
          }

          startSubmit(r.text || text, expandPasteSnips(r.text || text))

          return
        }

        startSubmit(text, expandPasteSnips(text))
      })
      .catch(() => startSubmit(text, expandPasteSnips(text)))
  }

  const shellExec = (cmd: string) => {
    appendMessage({ role: 'user', text: `!${cmd}` })
    setBusy(true)
    setStatus('running…')

    gw.request('shell.exec', { command: cmd })
      .then((raw: any) => {
        const r = asRpcResult(raw)

        if (!r) {
          sys('error: invalid response: shell.exec')

          return
        }

        const out = [r.stdout, r.stderr].filter(Boolean).join('\n').trim()

        if (out) {
          sys(out)
        }

        if (r.code !== 0 || !out) {
          sys(`exit ${r.code}`)
        }
      })
      .catch((e: Error) => sys(`error: ${e.message}`))
      .finally(() => {
        setStatus('ready')
        setBusy(false)
      })
  }

  const openEditor = () => {
    const editor = process.env.EDITOR || process.env.VISUAL || 'vi'
    const file = join(mkdtempSync(join(tmpdir(), 'hermes-')), 'prompt.md')

    writeFileSync(file, [...inputBuf, input].join('\n'))
    process.stdout.write('\x1b[?1049l')
    const { status: code } = spawnSync(editor, [file], { stdio: 'inherit' })
    process.stdout.write('\x1b[?1049h\x1b[2J\x1b[H')

    if (code === 0) {
      const text = readFileSync(file, 'utf8').trimEnd()

      if (text) {
        setInput('')
        setInputBuf([])
        submit(text)
      }
    }

    try {
      unlinkSync(file)
    } catch {
      /* noop */
    }
  }

  const interpolate = (text: string, then: (result: string) => void) => {
    setStatus('interpolating…')
    const matches = [...text.matchAll(new RegExp(INTERPOLATION_RE.source, 'g'))]

    Promise.all(
      matches.map(m =>
        gw
          .request('shell.exec', { command: m[1]! })
          .then((raw: any) => {
            const r = asRpcResult(raw)

            return [r?.stdout, r?.stderr].filter(Boolean).join('\n').trim()
          })
          .catch(() => '(error)')
      )
    ).then(results => {
      let out = text

      for (let i = matches.length - 1; i >= 0; i--) {
        out = out.slice(0, matches[i]!.index!) + results[i] + out.slice(matches[i]!.index! + matches[i]![0].length)
      }

      then(out)
    })
  }

  const sendQueued = (text: string) => {
    if (text.startsWith('!')) {
      shellExec(text.slice(1).trim())

      return
    }

    if (hasInterpolation(text)) {
      setBusy(true)
      interpolate(text, send)

      return
    }

    send(text)
  }

  // ── Dispatch ─────────────────────────────────────────────────────

  const dispatchSubmission = useCallback(
    (full: string) => {
      if (!full.trim()) {
        return
      }

      if (!sid) {
        sys('session not ready yet')

        return
      }

      const clearInput = () => {
        setInputBuf([])
        setInput('')
        setHistoryIdx(null)
        historyDraftRef.current = ''
      }

      if (looksLikeSlashCommand(full)) {
        appendMessage({ role: 'system', text: full, kind: 'slash' })
        pushHistory(full)
        slashRef.current(full)
        clearInput()

        return
      }

      if (full.startsWith('!')) {
        clearInput()
        shellExec(full.slice(1).trim())

        return
      }

      clearInput()

      const editIdx = queueEditRef.current

      if (editIdx !== null) {
        replaceQ(editIdx, full)
        const picked = queueRef.current.splice(editIdx, 1)[0]
        syncQueue()
        setQueueEdit(null)

        if (picked && busy && sid) {
          queueRef.current.unshift(picked)
          syncQueue()

          return
        }

        if (picked && sid) {
          sendQueued(picked)
        }

        return
      }

      pushHistory(full)

      if (busy) {
        enqueue(full)

        return
      }

      if (hasInterpolation(full)) {
        setBusy(true)
        interpolate(full, send)

        return
      }

      send(full)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [appendMessage, busy, enqueue, gw, pushHistory, sid]
  )

  // ── Input handling ───────────────────────────────────────────────

  const ctrl = (key: { ctrl: boolean }, ch: string, target: string) => key.ctrl && ch.toLowerCase() === target

  const pagerPageSize = Math.max(5, (stdout?.rows ?? 24) - 6)

  useInput((ch, key) => {
    if (isBlocked) {
      if (pager) {
        if (key.return || ch === ' ') {
          const next = pager.offset + pagerPageSize

          if (next >= pager.lines.length) {
            setPager(null)
          } else {
            setPager({ ...pager, offset: next })
          }
        } else if (key.escape || ctrl(key, ch, 'c') || ch === 'q') {
          setPager(null)
        }

        return
      }

      if (ctrl(key, ch, 'c')) {
        if (clarify) {
          answerClarify('')
        } else if (approval) {
          rpc('approval.respond', { choice: 'deny', session_id: sid }).then(r => {
            if (!r) {
              return
            }

            setApproval(null)
            sys('denied')
          })
        } else if (sudo) {
          rpc('sudo.respond', { request_id: sudo.requestId, password: '' }).then(r => {
            if (!r) {
              return
            }

            setSudo(null)
            sys('sudo cancelled')
          })
        } else if (secret) {
          rpc('secret.respond', { request_id: secret.requestId, value: '' }).then(r => {
            if (!r) {
              return
            }

            setSecret(null)
            sys('secret entry cancelled')
          })
        } else if (modelPicker) {
          setModelPicker(false)
        } else if (picker) {
          setPicker(false)
        }
      } else if (key.escape && picker) {
        setPicker(false)
      }

      return
    }

    if (completions.length && input && historyIdx === null && (key.upArrow || key.downArrow)) {
      setCompIdx(i => (key.upArrow ? (i - 1 + completions.length) % completions.length : (i + 1) % completions.length))

      return
    }

    if (key.wheelUp) {
      scrollWithSelection(-WHEEL_SCROLL_STEP)

      return
    }

    if (key.wheelDown) {
      scrollWithSelection(WHEEL_SCROLL_STEP)

      return
    }

    if (key.pageUp || key.pageDown) {
      const viewport = scrollRef.current?.getViewportHeight() ?? Math.max(6, (stdout?.rows ?? 24) - 8)
      const step = Math.max(4, viewport - 2)
      scrollWithSelection(key.pageUp ? -step : step)

      return
    }

    if (key.tab && completions.length) {
      const row = completions[compIdx]

      if (row?.text) {
        const text = input.startsWith('/') && row.text.startsWith('/') && compReplace > 0 ? row.text.slice(1) : row.text
        setInput(input.slice(0, compReplace) + text)
      }

      return
    }

    if (key.upArrow && !inputBuf.length) {
      if (queueRef.current.length) {
        const idx = queueEditIdx === null ? 0 : (queueEditIdx + 1) % queueRef.current.length
        setQueueEdit(idx)
        setHistoryIdx(null)
        setInput(queueRef.current[idx] ?? '')
      } else if (historyRef.current.length) {
        const idx = historyIdx === null ? historyRef.current.length - 1 : Math.max(0, historyIdx - 1)

        if (historyIdx === null) {
          historyDraftRef.current = input
        }

        setHistoryIdx(idx)
        setQueueEdit(null)
        setInput(historyRef.current[idx] ?? '')
      }

      return
    }

    if (key.downArrow && !inputBuf.length) {
      if (queueRef.current.length) {
        const idx =
          queueEditIdx === null
            ? queueRef.current.length - 1
            : (queueEditIdx - 1 + queueRef.current.length) % queueRef.current.length

        setQueueEdit(idx)
        setHistoryIdx(null)
        setInput(queueRef.current[idx] ?? '')
      } else if (historyIdx !== null) {
        const next = historyIdx + 1

        if (next >= historyRef.current.length) {
          setHistoryIdx(null)
          setInput(historyDraftRef.current)
        } else {
          setHistoryIdx(next)
          setInput(historyRef.current[next] ?? '')
        }
      }

      return
    }

    if (ctrl(key, ch, 'c')) {
      if (hasSelection) {
        const copied = selection.copySelection()

        if (copied) {
          sys('copied selection')
        }
      } else if (busy && sid) {
        interruptedRef.current = true
        gw.request('session.interrupt', { session_id: sid }).catch(() => {})
        const partial = (streaming || buf.current).trimStart()
        partial ? appendMessage({ role: 'assistant', text: partial + '\n\n*[interrupted]*' }) : sys('interrupted')

        idle()
        clearReasoning()
        setActivity([])
        turnToolsRef.current = []
        setStatus('interrupted')

        if (statusTimerRef.current) {
          clearTimeout(statusTimerRef.current)
        }

        statusTimerRef.current = setTimeout(() => {
          statusTimerRef.current = null
          setStatus('ready')
        }, 1500)
      } else if (input || inputBuf.length) {
        clearIn()
      } else {
        return die()
      }

      return
    }

    if (ctrl(key, ch, 'd')) {
      return die()
    }

    if (ctrl(key, ch, 'l')) {
      if (guardBusySessionSwitch()) {
        return
      }

      setStatus('forging session…')
      newSession()

      return
    }

    if (ctrl(key, ch, 'b')) {
      if (voiceRecording) {
        setVoiceRecording(false)
        setVoiceProcessing(true)
        rpc('voice.record', { action: 'stop' })
          .then((r: any) => {
            if (!r) {
              return
            }

            const transcript = String(r?.text || '').trim()

            if (transcript) {
              setInput(prev => (prev ? `${prev}${/\s$/.test(prev) ? '' : ' '}${transcript}` : transcript))
            } else {
              sys('voice: no speech detected')
            }
          })
          .catch((e: Error) => sys(`voice error: ${e.message}`))
          .finally(() => {
            setVoiceProcessing(false)
            setStatus('ready')
          })
      } else {
        rpc('voice.record', { action: 'start' })
          .then(r => {
            if (!r) {
              return
            }

            setVoiceRecording(true)
            setStatus('recording…')
          })
          .catch((e: Error) => sys(`voice error: ${e.message}`))
      }

      return
    }

    if (ctrl(key, ch, 'g')) {
      return openEditor()
    }
  })

  // ── Gateway events ───────────────────────────────────────────────

  const onEvent = useCallback(
    (ev: GatewayEvent) => {
      if (ev.session_id && sidRef.current && ev.session_id !== sidRef.current && !ev.type.startsWith('gateway.')) {
        return
      }

      const p = ev.payload as any

      switch (ev.type) {
        case 'gateway.ready':
          if (p?.skin) {
            setTheme(
              fromSkin(p.skin.colors ?? {}, p.skin.branding ?? {}, p.skin.banner_logo ?? '', p.skin.banner_hero ?? '')
            )
          }

          rpc('commands.catalog', {})
            .then((r: any) => {
              if (!r?.pairs) {
                return
              }

              setCatalog({
                canon: (r.canon ?? {}) as Record<string, string>,
                categories: (r.categories ?? []) as SlashCatalog['categories'],
                pairs: r.pairs as [string, string][],
                skillCount: (r.skill_count ?? 0) as number,
                sub: (r.sub ?? {}) as Record<string, string[]>
              })

              if (r.warning) {
                pushActivity(String(r.warning), 'warn')
              }
            })
            .catch((e: unknown) => pushActivity(`command catalog unavailable: ${rpcErrorMessage(e)}`, 'warn'))

          if (STARTUP_RESUME_ID) {
            setStatus('resuming…')
            gw.request('session.resume', { cols: colsRef.current, session_id: STARTUP_RESUME_ID })
              .then((raw: any) => {
                const r = asRpcResult(raw)

                if (!r) {
                  throw new Error('invalid response: session.resume')
                }

                resetSession()
                setSid(r.session_id)
                setInfo(r.info ?? null)
                const resumed = toTranscriptMessages(r.messages)

                if (r.info?.usage) {
                  setUsage(prev => ({ ...prev, ...r.info.usage }))
                }

                setMessages(resumed)
                setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
                setStatus('ready')
              })
              .catch((e: unknown) => {
                sys(`resume failed: ${rpcErrorMessage(e)}`)
                setStatus('forging session…')
                newSession('started a new session')
              })
          } else {
            setStatus('forging session…')
            newSession()
          }

          break

        case 'skin.changed':
          if (p) {
            setTheme(fromSkin(p.colors ?? {}, p.branding ?? {}, p.banner_logo ?? '', p.banner_hero ?? ''))
          }

          break

        case 'session.info':
          setInfo(p as SessionInfo)

          if (p?.usage) {
            setUsage(prev => ({ ...prev, ...p.usage }))
          }

          break

        case 'thinking.delta':
          if (p && Object.prototype.hasOwnProperty.call(p, 'text')) {
            setStatus(p.text ? String(p.text) : busyRef.current ? 'running…' : 'ready')
          }

          break

        case 'message.start':
          setBusy(true)
          endReasoningPhase()
          clearReasoning()
          setActivity([])
          setTurnTrail([])
          turnToolsRef.current = []
          persistedToolLabelsRef.current.clear()

          break

        case 'status.update':
          if (p?.text) {
            setStatus(p.text)

            if (p.kind && p.kind !== 'status') {
              if (lastStatusNoteRef.current !== p.text) {
                lastStatusNoteRef.current = p.text
                pushActivity(
                  p.text,
                  p.kind === 'error' ? 'error' : p.kind === 'warn' || p.kind === 'approval' ? 'warn' : 'info'
                )
              }

              if (statusTimerRef.current) {
                clearTimeout(statusTimerRef.current)
              }

              statusTimerRef.current = setTimeout(() => {
                statusTimerRef.current = null
                setStatus(busyRef.current ? 'running…' : 'ready')
              }, 4000)
            }
          }

          break

        case 'gateway.stderr':
          if (p?.line) {
            const line = String(p.line).slice(0, 120)
            const tone = /\b(error|traceback|exception|failed|spawn)\b/i.test(line) ? 'error' : 'warn'
            pushActivity(line, tone)
          }

          break

        case 'gateway.start_timeout':
          setStatus('gateway startup timeout')
          pushActivity(
            `gateway startup timed out${p?.python || p?.cwd ? ` · ${String(p?.python || '')} ${String(p?.cwd || '')}`.trim() : ''} · /logs to inspect`,
            'error'
          )

          break

        case 'gateway.protocol_error':
          setStatus('protocol warning')

          if (statusTimerRef.current) {
            clearTimeout(statusTimerRef.current)
          }

          statusTimerRef.current = setTimeout(() => {
            statusTimerRef.current = null
            setStatus(busyRef.current ? 'running…' : 'ready')
          }, 4000)

          if (!protocolWarnedRef.current) {
            protocolWarnedRef.current = true
            pushActivity('protocol noise detected · /logs to inspect', 'warn')
          }

          if (p?.preview) {
            pushActivity(`protocol noise: ${String(p.preview).slice(0, 120)}`, 'warn')
          }

          break

        case 'reasoning.delta':
          if (p?.text) {
            reasoningRef.current += p.text
            scheduleReasoning()
            pulseReasoningStreaming()
          }

          break

        case 'tool.progress':
          if (p?.preview) {
            setTools(prev => {
              const idx = prev.findIndex(t => t.name === p.name)

              return idx >= 0
                ? [...prev.slice(0, idx), { ...prev[idx]!, context: p.preview as string }, ...prev.slice(idx + 1)]
                : prev
            })
          }

          break

        case 'tool.generating':
          if (p?.name) {
            pushTrail(`drafting ${p.name}…`)
          }

          break

        case 'tool.start':
          pruneTransient()
          endReasoningPhase()
          setTools(prev => [
            ...prev,
            { id: p.tool_id, name: p.name, context: (p.context as string) || '', startedAt: Date.now() }
          ])

          break
        case 'tool.complete': {
          toolCompleteRibbonRef.current = null
          setTools(prev => {
            const done = prev.find(t => t.id === p.tool_id)
            const name = done?.name ?? p.name
            const label = toolTrailLabel(name)

            const line = buildToolTrailLine(
              name,
              done?.context || '',
              !!p.error,
              (p.error as string) || (p.summary as string) || ''
            )

            toolCompleteRibbonRef.current = { label, line }
            const remaining = prev.filter(t => t.id !== p.tool_id)
            const next = [...turnToolsRef.current.filter(s => !sameToolTrailGroup(label, s)), line]

            if (!remaining.length) {
              next.push('analyzing tool output…')
            }

            const pruned = next.slice(-8)
            turnToolsRef.current = pruned
            setTurnTrail(pruned)

            return remaining
          })

          if (p?.inline_diff) {
            sys(p.inline_diff as string)
          }

          break
        }

        case 'clarify.request':
          setClarify({ choices: p.choices, question: p.question, requestId: p.request_id })
          setStatus('waiting for input…')

          break

        case 'approval.request':
          setApproval({ command: p.command, description: p.description })
          setStatus('approval needed')

          break

        case 'sudo.request':
          setSudo({ requestId: p.request_id })
          setStatus('sudo password needed')

          break

        case 'secret.request':
          setSecret({ requestId: p.request_id, prompt: p.prompt, envVar: p.env_var })
          setStatus('secret input needed')

          break

        case 'background.complete':
          setBgTasks(prev => {
            const next = new Set(prev)
            next.delete(p.task_id)

            return next
          })
          sys(`[bg ${p.task_id}] ${p.text}`)

          break

        case 'btw.complete':
          setBgTasks(prev => {
            const next = new Set(prev)
            next.delete('btw:x')

            return next
          })
          sys(`[btw] ${p.text}`)

          break

        case 'message.delta':
          pruneTransient()
          endReasoningPhase()

          if (p?.text && !interruptedRef.current) {
            buf.current = p.rendered ?? buf.current + p.text
            scheduleStreaming()
          }

          break
        case 'message.complete': {
          const wasInterrupted = interruptedRef.current
          const savedReasoning = reasoningRef.current.trim()
          const persisted = persistedToolLabelsRef.current

          const savedTools = turnToolsRef.current.filter(
            l => isToolTrailResultLine(l) && ![...persisted].some(p => sameToolTrailGroup(p, l))
          )

          const finalText = (p?.rendered ?? p?.text ?? buf.current).trimStart()

          idle()
          clearReasoning()
          setStreaming('')

          if (!wasInterrupted) {
            appendMessage({
              role: 'assistant',
              text: finalText,
              thinking: savedReasoning || undefined,
              tools: savedTools.length ? savedTools : undefined
            })

            if (bellOnComplete && stdout?.isTTY) {
              stdout.write('\x07')
            }
          }

          turnToolsRef.current = []
          persistedToolLabelsRef.current.clear()
          setActivity([])

          buf.current = ''
          setStatus('ready')

          if (p?.usage) {
            setUsage(p.usage)
          }

          if (queueEditRef.current !== null) {
            break
          }

          const next = dequeue()

          if (next) {
            sendQueued(next)
          }

          break
        }

        case 'error':
          idle()
          clearReasoning()
          turnToolsRef.current = []
          persistedToolLabelsRef.current.clear()

          if (statusTimerRef.current) {
            clearTimeout(statusTimerRef.current)
            statusTimerRef.current = null
          }

          pushActivity(String(p?.message || 'unknown error'), 'error')
          sys(`error: ${p?.message}`)
          setStatus('ready')

          break
      }
    },
    [
      appendMessage,
      bellOnComplete,
      clearReasoning,
      dequeue,
      endReasoningPhase,
      gw,
      newSession,
      pruneTransient,
      pulseReasoningStreaming,
      pushActivity,
      pushTrail,
      rpc,
      scheduleReasoning,
      scheduleStreaming,
      sendQueued,
      sys,
      stdout
    ]
  )

  onEventRef.current = onEvent

  useEffect(() => {
    const handler = (ev: GatewayEvent) => onEventRef.current(ev)

    const exitHandler = () => {
      setStatus('gateway exited')
      setSid(null)
      setBusy(false)
      pushActivity('gateway exited · /logs to inspect', 'error')
      sys('error: gateway exited')
    }

    gw.on('event', handler)
    gw.on('exit', exitHandler)
    gw.drain()

    return () => {
      gw.off('event', handler)
      gw.off('exit', exitHandler)
      gw.kill()
    }
  }, [gw, pushActivity, sys])

  // ── Slash commands ───────────────────────────────────────────────

  const slash = useCallback(
    (cmd: string): boolean => {
      const [rawName, ...rest] = cmd.slice(1).split(/\s+/)
      const name = rawName.toLowerCase()
      const arg = rest.join(' ')

      switch (name) {
        case 'help': {
          const sections: PanelSection[] = (catalog?.categories ?? []).map(({ name: catName, pairs }) => ({
            title: catName,
            rows: pairs
          }))

          if (catalog?.skillCount) {
            sections.push({ text: `${catalog.skillCount} skill commands available — /skills to browse` })
          }

          sections.push({
            title: 'TUI',
            rows: [['/details [hidden|collapsed|expanded|cycle]', 'set agent detail visibility mode']]
          })

          sections.push({ title: 'Hotkeys', rows: HOTKEYS })

          panel('Commands', sections)

          return true
        }

        case 'quit':

        case 'exit':

        case 'q':
          die()

          return true

        case 'clear':
          if (guardBusySessionSwitch('switch sessions')) {
            return true
          }

          setStatus('forging session…')
          newSession()

          return true

        case 'new':
          if (guardBusySessionSwitch('switch sessions')) {
            return true
          }

          setStatus('forging session…')
          newSession('new session started')

          return true

        case 'resume':
          if (guardBusySessionSwitch('switch sessions')) {
            return true
          }

          if (arg) {
            resumeById(arg)
          } else {
            setPicker(true)
          }

          return true

        case 'compact':
          if (arg && !['on', 'off', 'toggle'].includes(arg.trim().toLowerCase())) {
            sys('usage: /compact [on|off|toggle]')

            return true
          }

          {
            const mode = arg.trim().toLowerCase()
            setCompact(current => {
              const next = mode === 'on' ? true : mode === 'off' ? false : !current
              rpc('config.set', { key: 'compact', value: next ? 'on' : 'off' }).catch(() => {})
              queueMicrotask(() => sys(`compact ${next ? 'on' : 'off'}`))

              return next
            })
          }

          return true

        case 'details':

        case 'detail':
          if (!arg) {
            rpc('config.get', { key: 'details_mode' })
              .then((r: any) => {
                const mode = parseDetailsMode(r?.value) ?? detailsMode
                setDetailsMode(mode)
                sys(`details: ${mode}`)
              })
              .catch(() => sys(`details: ${detailsMode}`))

            return true
          }

          {
            const mode = arg.trim().toLowerCase()

            if (!['hidden', 'collapsed', 'expanded', 'cycle', 'toggle'].includes(mode)) {
              sys('usage: /details [hidden|collapsed|expanded|cycle]')

              return true
            }

            const next = mode === 'cycle' || mode === 'toggle' ? nextDetailsMode(detailsMode) : (mode as DetailsMode)
            setDetailsMode(next)
            rpc('config.set', { key: 'details_mode', value: next }).catch(() => {})
            sys(`details: ${next}`)
          }

          return true
        case 'copy': {
          if (!arg && hasSelection) {
            const copied = selection.copySelection()

            if (copied) {
              sys('copied selection')

              return true
            }
          }

          const all = messages.filter(m => m.role === 'assistant')

          if (arg && Number.isNaN(parseInt(arg, 10))) {
            sys('usage: /copy [number]')

            return true
          }

          const target = all[arg ? Math.min(parseInt(arg, 10), all.length) - 1 : all.length - 1]

          if (!target) {
            sys('nothing to copy')

            return true
          }

          writeOsc52Clipboard(target.text)
          sys('sent OSC52 copy sequence (terminal support required)')

          return true
        }

        case 'paste':
          if (!arg) {
            paste()

            return true
          }

          sys('usage: /paste')

          return true
        case 'logs': {
          const logText = gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))
          logText ? page(logText, 'Logs') : sys('no gateway logs')

          return true
        }

        case 'statusbar':

        case 'sb':
          if (arg && !['on', 'off', 'toggle'].includes(arg.trim().toLowerCase())) {
            sys('usage: /statusbar [on|off|toggle]')

            return true
          }

          setStatusBar(current => {
            const mode = arg.trim().toLowerCase()
            const next = mode === 'on' ? true : mode === 'off' ? false : !current
            rpc('config.set', { key: 'statusbar', value: next ? 'on' : 'off' }).catch(() => {})
            queueMicrotask(() => sys(`status bar ${next ? 'on' : 'off'}`))

            return next
          })

          return true

        case 'queue':
          if (!arg) {
            sys(`${queueRef.current.length} queued message(s)`)

            return true
          }

          enqueue(arg)
          sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)

          return true

        case 'undo':
          if (!sid) {
            sys('nothing to undo')

            return true
          }

          rpc('session.undo', { session_id: sid }).then((r: any) => {
            if (!r) {
              return
            }

            if (r.removed > 0) {
              setMessages(prev => trimLastExchange(prev))
              setHistoryItems(prev => trimLastExchange(prev))
              sys(`undid ${r.removed} messages`)
            } else {
              sys('nothing to undo')
            }
          })

          return true

        case 'retry':
          if (!lastUserMsg) {
            sys('nothing to retry')

            return true
          }

          if (sid) {
            rpc('session.undo', { session_id: sid }).then((r: any) => {
              if (!r) {
                return
              }

              if (r.removed <= 0) {
                sys('nothing to retry')

                return
              }

              setMessages(prev => trimLastExchange(prev))
              setHistoryItems(prev => trimLastExchange(prev))
              send(lastUserMsg)
            })

            return true
          }

          send(lastUserMsg)

          return true

        case 'background':

        case 'bg':
          if (!arg) {
            sys('/background <prompt>')

            return true
          }

          rpc('prompt.background', { session_id: sid, text: arg }).then((r: any) => {
            if (!r?.task_id) {
              return
            }

            setBgTasks(prev => new Set(prev).add(r.task_id))
            sys(`bg ${r.task_id} started`)
          })

          return true

        case 'btw':
          if (!arg) {
            sys('/btw <question>')

            return true
          }

          rpc('prompt.btw', { session_id: sid, text: arg }).then(r => {
            if (!r) {
              return
            }

            setBgTasks(prev => new Set(prev).add('btw:x'))
            sys('btw running…')
          })

          return true

        case 'model':
          if (guardBusySessionSwitch('change models')) {
            return true
          }

          if (!arg) {
            setModelPicker(true)
          } else {
            rpc('config.set', { session_id: sid, key: 'model', value: arg.trim() }).then((r: any) => {
              if (!r) {
                return
              }

              if (!r.value) {
                sys('error: invalid response: model switch')

                return
              }

              sys(`model → ${r.value}`)
              maybeWarn(r)
              setInfo(prev => (prev ? { ...prev, model: r.value } : { model: r.value, skills: {}, tools: {} }))
            })
          }

          return true

        case 'image':
          rpc('image.attach', { session_id: sid, path: arg }).then((r: any) => {
            if (!r) {
              return
            }

            const meta = imageTokenMeta(r)
            sys(`attached image: ${r.name}${meta ? ` · ${meta}` : ''}`)

            if (r?.remainder) {
              setInput(r.remainder)
            }
          })

          return true

        case 'provider':
          gw.request('slash.exec', { command: 'provider', session_id: sid })
            .then((r: any) => {
              page(
                r?.warning ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}` : r?.output || '(no output)',
                'Provider'
              )
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

          return true

        case 'skin':
          if (arg) {
            rpc('config.set', { key: 'skin', value: arg }).then((r: any) => {
              if (!r?.value) {
                return
              }

              sys(`skin → ${r.value}`)
            })
          } else {
            rpc('config.get', { key: 'skin' }).then((r: any) => {
              if (!r) {
                return
              }

              sys(`skin: ${r.value || 'default'}`)
            })
          }

          return true

        case 'yolo':
          rpc('config.set', { session_id: sid, key: 'yolo' }).then((r: any) => {
            if (!r) {
              return
            }

            sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)
          })

          return true

        case 'reasoning':
          if (!arg) {
            rpc('config.get', { key: 'reasoning' }).then((r: any) => {
              if (!r?.value) {
                return
              }

              sys(`reasoning: ${r.value} · display ${r.display || 'hide'}`)
            })
          } else {
            rpc('config.set', { session_id: sid, key: 'reasoning', value: arg }).then((r: any) => {
              if (!r?.value) {
                return
              }

              sys(`reasoning: ${r.value}`)
            })
          }

          return true

        case 'verbose':
          rpc('config.set', { session_id: sid, key: 'verbose', value: arg || 'cycle' }).then((r: any) => {
            if (!r?.value) {
              return
            }

            sys(`verbose: ${r.value}`)
          })

          return true

        case 'personality':
          if (arg) {
            rpc('config.set', { session_id: sid, key: 'personality', value: arg }).then((r: any) => {
              if (!r) {
                return
              }

              if (r.history_reset) {
                resetVisibleHistory(r.info ?? null)
              }

              sys(`personality: ${r.value || 'default'}${r.history_reset ? ' · transcript cleared' : ''}`)
              maybeWarn(r)
            })
          } else {
            gw.request('slash.exec', { command: 'personality', session_id: sid })
              .then((r: any) => {
                panel('Personality', [
                  {
                    text: r?.warning
                      ? `warning: ${r.warning}\n\n${r?.output || '(no output)'}`
                      : r?.output || '(no output)'
                  }
                ])
              })
              .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
          }

          return true

        case 'compress':
          rpc('session.compress', { session_id: sid, ...(arg ? { focus_topic: arg } : {}) }).then((r: any) => {
            if (!r) {
              return
            }

            if (Array.isArray(r.messages)) {
              const resumed = toTranscriptMessages(r.messages)
              setMessages(resumed)
              setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
            }

            if (r.info) {
              setInfo(r.info)
            }

            if (r.usage) {
              setUsage(prev => ({ ...prev, ...r.usage }))
            }

            if ((r.removed ?? 0) <= 0) {
              sys('nothing to compress')

              return
            }

            sys(`compressed ${r.removed} messages${r.usage?.total ? ' · ' + fmtK(r.usage.total) + ' tok' : ''}`)
          })

          return true

        case 'stop':
          rpc('process.stop', {}).then((r: any) => {
            if (!r) {
              return
            }

            sys(`killed ${r.killed ?? 0} registered process(es)`)
          })

          return true

        case 'branch':

        case 'fork':
          {
            const prevSid = sid
            rpc('session.branch', { session_id: sid, name: arg }).then((r: any) => {
              if (r?.session_id) {
                void closeSession(prevSid)
                setSid(r.session_id)
                setSessionStartedAt(Date.now())
                setHistoryItems([])
                setMessages([])
                sys(`branched → ${r.title}`)
              }
            })
          }

          return true

        case 'reload-mcp':

        case 'reload_mcp':
          rpc('reload.mcp', { session_id: sid }).then(r => {
            if (!r) {
              return
            }

            sys('MCP reloaded')
          })

          return true

        case 'title':
          rpc('session.title', { session_id: sid, ...(arg ? { title: arg } : {}) }).then((r: any) => {
            if (!r) {
              return
            }

            sys(`title: ${r.title || '(none)'}`)
          })

          return true

        case 'usage':
          rpc('session.usage', { session_id: sid }).then((r: any) => {
            if (r) {
              setUsage({ input: r.input ?? 0, output: r.output ?? 0, total: r.total ?? 0, calls: r.calls ?? 0 })
            }

            if (!r?.calls) {
              sys('no API calls yet')

              return
            }

            const f = (v: number) => (v ?? 0).toLocaleString()

            const cost =
              r.cost_usd != null ? `${r.cost_status === 'estimated' ? '~' : ''}$${r.cost_usd.toFixed(4)}` : null

            const rows: [string, string][] = [
              ['Model', r.model ?? ''],
              ['Input tokens', f(r.input)],
              ['Cache read tokens', f(r.cache_read)],
              ['Cache write tokens', f(r.cache_write)],
              ['Output tokens', f(r.output)],
              ['Total tokens', f(r.total)],
              ['API calls', f(r.calls)]
            ]

            if (cost) {
              rows.push(['Cost', cost])
            }

            const sections: PanelSection[] = [{ rows }]

            if (r.context_max) {
              sections.push({ text: `Context: ${f(r.context_used)} / ${f(r.context_max)} (${r.context_percent}%)` })
            }

            if (r.compressions) {
              sections.push({ text: `Compressions: ${r.compressions}` })
            }

            panel('Usage', sections)
          })

          return true

        case 'save':
          rpc('session.save', { session_id: sid }).then((r: any) => {
            if (!r?.file) {
              return
            }

            sys(`saved: ${r.file}`)
          })

          return true

        case 'history':
          rpc('session.history', { session_id: sid }).then((r: any) => {
            if (typeof r?.count !== 'number') {
              return
            }

            sys(`${r.count} messages`)
          })

          return true

        case 'profile':
          rpc('config.get', { key: 'profile' }).then((r: any) => {
            if (!r) {
              return
            }

            const text = r.display || r.home || '(unknown profile)'
            const lines = text.split('\n').filter(Boolean)

            if (lines.length <= 2) {
              panel('Profile', [{ text }])
            } else {
              page(text, 'Profile')
            }
          })

          return true

        case 'voice':
          rpc('voice.toggle', { action: arg === 'on' || arg === 'off' ? arg : 'status' }).then((r: any) => {
            if (!r) {
              return
            }

            setVoiceEnabled(!!r?.enabled)
            sys(`voice: ${r.enabled ? 'on' : 'off'}`)
          })

          return true

        case 'insights':
          rpc('insights.get', { days: parseInt(arg) || 30 }).then((r: any) => {
            if (!r) {
              return
            }

            panel('Insights', [
              {
                rows: [
                  ['Period', `${r.days} days`],
                  ['Sessions', `${r.sessions}`],
                  ['Messages', `${r.messages}`]
                ]
              }
            ])
          })

          return true
        case 'rollback': {
          const [sub, ...rArgs] = (arg || 'list').split(/\s+/)

          if (!sub || sub === 'list') {
            rpc('rollback.list', { session_id: sid }).then((r: any) => {
              if (!r) {
                return
              }

              if (!r.checkpoints?.length) {
                return sys('no checkpoints')
              }

              panel('Checkpoints', [
                {
                  rows: r.checkpoints.map(
                    (c: any, i: number) => [`${i + 1} ${c.hash?.slice(0, 8)}`, c.message] as [string, string]
                  )
                }
              ])
            })
          } else {
            const hash = sub === 'restore' || sub === 'diff' ? rArgs[0] : sub

            const filePath =
              sub === 'restore' || sub === 'diff' ? rArgs.slice(1).join(' ').trim() : rArgs.join(' ').trim()

            rpc(sub === 'diff' ? 'rollback.diff' : 'rollback.restore', {
              session_id: sid,
              hash,
              ...(sub === 'diff' || !filePath ? {} : { file_path: filePath })
            }).then((r: any) => {
              if (!r) {
                return
              }

              sys(r.rendered || r.diff || r.message || 'done')
            })
          }

          return true
        }

        case 'browser': {
          const [act, ...bArgs] = (arg || 'status').split(/\s+/)
          rpc('browser.manage', { action: act, ...(bArgs[0] ? { url: bArgs[0] } : {}) }).then((r: any) => {
            if (!r) {
              return
            }

            sys(r.connected ? `browser: ${r.url}` : 'browser: disconnected')
          })

          return true
        }

        case 'plugins':
          rpc('plugins.list', {}).then((r: any) => {
            if (!r) {
              return
            }

            if (!r.plugins?.length) {
              return sys('no plugins')
            }

            panel('Plugins', [
              {
                items: r.plugins.map((p: any) => `${p.name} v${p.version}${p.enabled ? '' : ' (disabled)'}`)
              }
            ])
          })

          return true
        case 'skills': {
          const [sub, ...sArgs] = (arg || '').split(/\s+/).filter(Boolean)

          if (!sub || sub === 'list') {
            rpc('skills.manage', { action: 'list' }).then((r: any) => {
              if (!r) {
                return
              }

              const sk = r.skills as Record<string, string[]> | undefined

              if (!sk || !Object.keys(sk).length) {
                return sys('no skills installed')
              }

              panel(
                'Installed Skills',
                Object.entries(sk).map(([cat, names]) => ({
                  title: cat,
                  items: names as string[]
                }))
              )
            })

            return true
          }

          if (sub === 'browse') {
            const pg = parseInt(sArgs[0] ?? '1', 10) || 1
            rpc('skills.manage', { action: 'browse', page: pg }).then((r: any) => {
              if (!r) {
                return
              }

              if (!r.items?.length) {
                return sys('no skills found in the hub')
              }

              const sections: PanelSection[] = [
                {
                  rows: r.items.map(
                    (s: any) =>
                      [s.name ?? '', (s.description ?? '').slice(0, 60) + (s.description?.length > 60 ? '…' : '')] as [
                        string,
                        string
                      ]
                  )
                }
              ]

              if (r.page < r.total_pages) {
                sections.push({ text: `/skills browse ${r.page + 1} → next page` })
              }

              if (r.page > 1) {
                sections.push({ text: `/skills browse ${r.page - 1} → prev page` })
              }

              panel(`Skills Hub (page ${r.page}/${r.total_pages}, ${r.total} total)`, sections)
            })

            return true
          }

          gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then((r: any) => {
              sys(
                r?.warning
                  ? `warning: ${r.warning}\n${r?.output || '/skills: no output'}`
                  : r?.output || '/skills: no output'
              )
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

          return true
        }

        case 'agents':

        case 'tasks':
          rpc('agents.list', {})
            .then((r: any) => {
              if (!r) {
                return
              }

              const procs = r.processes ?? []
              const running = procs.filter((p: any) => p.status === 'running')
              const finished = procs.filter((p: any) => p.status !== 'running')
              const sections: PanelSection[] = []

              if (running.length) {
                sections.push({
                  title: `Running (${running.length})`,
                  rows: running.map((p: any) => [p.session_id.slice(0, 8), p.command])
                })
              }

              if (finished.length) {
                sections.push({
                  title: `Finished (${finished.length})`,
                  rows: finished.map((p: any) => [p.session_id.slice(0, 8), p.command])
                })
              }

              if (!sections.length) {
                sections.push({ text: 'No active processes' })
              }

              panel('Agents', sections)
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

          return true

        case 'cron':
          if (!arg || arg === 'list') {
            rpc('cron.manage', { action: 'list' })
              .then((r: any) => {
                if (!r) {
                  return
                }

                const jobs = r.jobs ?? []

                if (!jobs.length) {
                  return sys('no scheduled jobs')
                }

                panel('Cron', [
                  {
                    rows: jobs.map(
                      (j: any) =>
                        [j.name || j.job_id?.slice(0, 12), `${j.schedule} · ${j.state ?? 'active'}`] as [string, string]
                    )
                  }
                ])
              })
              .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
          } else {
            gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
              .then((r: any) => {
                sys(r?.warning ? `warning: ${r.warning}\n${r?.output || '(no output)'}` : r?.output || '(no output)')
              })
              .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
          }

          return true

        case 'config':
          rpc('config.show', {})
            .then((r: any) => {
              if (!r) {
                return
              }

              panel(
                'Config',
                (r.sections ?? []).map((s: any) => ({
                  title: s.title,
                  rows: s.rows
                }))
              )
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

          return true

        case 'tools':
          rpc('tools.list', { session_id: sid })
            .then((r: any) => {
              if (!r) {
                return
              }

              if (!r.toolsets?.length) {
                return sys('no tools')
              }

              panel(
                'Tools',
                r.toolsets.map((ts: any) => ({
                  title: `${ts.enabled ? '*' : ' '} ${ts.name} [${ts.tool_count} tools]`,
                  items: ts.tools
                }))
              )
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

          return true

        case 'toolsets':
          rpc('toolsets.list', { session_id: sid })
            .then((r: any) => {
              if (!r) {
                return
              }

              if (!r.toolsets?.length) {
                return sys('no toolsets')
              }

              panel('Toolsets', [
                {
                  rows: r.toolsets.map(
                    (ts: any) =>
                      [`${ts.enabled ? '(*)' : '   '} ${ts.name}`, `[${ts.tool_count}] ${ts.description}`] as [
                        string,
                        string
                      ]
                  )
                }
              ])
            })
            .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))

          return true

        default:
          gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then((r: any) => {
              sys(
                r?.warning
                  ? `warning: ${r.warning}\n${r?.output || `/${name}: no output`}`
                  : r?.output || `/${name}: no output`
              )
            })
            .catch(() => {
              gw.request('command.dispatch', { name: name ?? '', arg, session_id: sid })
                .then((raw: any) => {
                  const d = asRpcResult(raw)

                  if (!d?.type) {
                    sys('error: invalid response: command.dispatch')

                    return
                  }

                  if (d.type === 'exec') {
                    sys(d.output || '(no output)')
                  } else if (d.type === 'alias') {
                    slash(`/${d.target}${arg ? ' ' + arg : ''}`)
                  } else if (d.type === 'plugin') {
                    sys(d.output || '(no output)')
                  } else if (d.type === 'skill') {
                    sys(`⚡ loading skill: ${d.name}`)

                    if (typeof d.message === 'string' && d.message.trim()) {
                      send(d.message)
                    } else {
                      sys(`/${name}: skill payload missing message`)
                    }
                  }
                })
                .catch((e: unknown) => sys(`error: ${rpcErrorMessage(e)}`))
            })

          return true
      }
    },
    [
      catalog,
      compact,
      detailsMode,
      guardBusySessionSwitch,
      gw,
      hasSelection,
      lastUserMsg,
      maybeWarn,
      messages,
      newSession,
      page,
      panel,
      pushActivity,
      rpc,
      resetVisibleHistory,
      selection,
      send,
      sid,
      statusBar,
      sys
    ]
  )

  slashRef.current = slash

  // ── Submit ───────────────────────────────────────────────────────

  const submit = useCallback(
    (value: string) => {
      if (!value.trim() && !inputBuf.length) {
        const now = Date.now()
        const dbl = now - lastEmptyAt.current < 450
        lastEmptyAt.current = now

        if (dbl && busy && sid) {
          interruptedRef.current = true
          gw.request('session.interrupt', { session_id: sid }).catch(() => {})
          const partial = (streaming || buf.current).trimStart()

          if (partial) {
            appendMessage({ role: 'assistant', text: partial + '\n\n*[interrupted]*' })
          } else {
            sys('interrupted')
          }

          idle()
          clearReasoning()
          setActivity([])
          turnToolsRef.current = []
          setStatus('interrupted')

          if (statusTimerRef.current) {
            clearTimeout(statusTimerRef.current)
          }

          statusTimerRef.current = setTimeout(() => {
            statusTimerRef.current = null
            setStatus('ready')
          }, 1500)

          return
        }

        if (dbl && queueRef.current.length) {
          const next = dequeue()

          if (next && sid) {
            setQueueEdit(null)
            dispatchSubmission(next)
          }
        }

        return
      }

      lastEmptyAt.current = 0

      if (value.endsWith('\\')) {
        setInputBuf(prev => [...prev, value.slice(0, -1)])
        setInput('')

        return
      }

      dispatchSubmission([...inputBuf, value].join('\n'))
    },
    [dequeue, dispatchSubmission, inputBuf, sid]
  )

  // ── Derived ──────────────────────────────────────────────────────

  const statusColor =
    status === 'ready'
      ? theme.color.ok
      : status.startsWith('error')
        ? theme.color.error
        : status === 'interrupted'
          ? theme.color.warn
          : theme.color.dim

  const durationLabel = sid ? fmtDuration(clockNow - sessionStartedAt) : ''
  const voiceLabel = voiceRecording ? 'REC' : voiceProcessing ? 'STT' : `voice ${voiceEnabled ? 'on' : 'off'}`
  const cwdLabel = shortCwd(info?.cwd || process.env.HERMES_CWD || process.cwd())
  const showStreamingArea = Boolean(streaming)
  const visibleHistory = virtualRows.slice(virtualHistory.start, virtualHistory.end)
  const showStickyPrompt = !!stickyPrompt

  const hasReasoning = Boolean(reasoning.trim())

  const showProgressArea =
    detailsMode === 'hidden'
      ? activity.some(i => i.tone !== 'info')
      : Boolean(busy || tools.length || turnTrail.length || hasReasoning || activity.length)

  // ── Render ───────────────────────────────────────────────────────

  return (
    <AlternateScreen mouseTracking={MOUSE_TRACKING}>
      <Box flexDirection="column" flexGrow={1}>
        <Box flexDirection="row" flexGrow={1}>
          <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1} ref={scrollRef} stickyScroll>
            <Box flexDirection="column" paddingX={1}>
              {virtualHistory.topSpacer > 0 ? <Box height={virtualHistory.topSpacer} /> : null}

              {visibleHistory.map(row => (
                <Box flexDirection="column" key={row.key} ref={virtualHistory.measureRef(row.key)}>
                  {row.msg.kind === 'intro' && row.msg.info ? (
                    <Box flexDirection="column" paddingTop={1}>
                      <Banner t={theme} />
                      <SessionPanel info={row.msg.info} sid={sid} t={theme} />
                    </Box>
                  ) : row.msg.kind === 'panel' && row.msg.panelData ? (
                    <Panel sections={row.msg.panelData.sections} t={theme} title={row.msg.panelData.title} />
                  ) : (
                    <MessageLine cols={cols} compact={compact} detailsMode={detailsMode} msg={row.msg} t={theme} />
                  )}
                </Box>
              ))}

              {virtualHistory.bottomSpacer > 0 ? <Box height={virtualHistory.bottomSpacer} /> : null}

              {showProgressArea && (
                <ToolTrail
                  activity={activity}
                  busy={busy && !streaming}
                  detailsMode={detailsMode}
                  reasoning={reasoning}
                  reasoningActive={reasoningActive}
                  reasoningStreaming={reasoningStreaming}
                  t={theme}
                  tools={tools}
                  trail={turnTrail}
                />
              )}

              {showStreamingArea && (
                <MessageLine
                  cols={cols}
                  compact={compact}
                  detailsMode={detailsMode}
                  isStreaming
                  msg={{ role: 'assistant', text: streaming }}
                  t={theme}
                />
              )}
            </Box>
          </ScrollBox>

          <NoSelect flexShrink={0} marginLeft={1}>
            <TranscriptScrollbar scrollRef={scrollRef} t={theme} />
          </NoSelect>

          <StickyPromptTracker
            messages={historyItems}
            offsets={virtualHistory.offsets}
            onChange={setStickyPrompt}
            scrollRef={scrollRef}
          />
        </Box>

        <NoSelect flexDirection="column" flexShrink={0} fromLeftEdge paddingX={1}>
          {clarify && (
            <PromptBox color={theme.color.bronze}>
              <ClarifyPrompt
                cols={cols}
                onAnswer={answerClarify}
                onCancel={() => answerClarify('')}
                req={clarify}
                t={theme}
              />
            </PromptBox>
          )}

          {approval && (
            <PromptBox color={theme.color.bronze}>
              <ApprovalPrompt
                onChoice={choice => {
                  rpc('approval.respond', { choice, session_id: sid }).then(r => {
                    if (!r) {
                      return
                    }

                    setApproval(null)
                    sys(choice === 'deny' ? 'denied' : `approved (${choice})`)
                    setStatus('running…')
                  })
                }}
                req={approval}
                t={theme}
              />
            </PromptBox>
          )}

          {sudo && (
            <PromptBox color={theme.color.bronze}>
              <MaskedPrompt
                cols={cols}
                icon="🔐"
                label="sudo password required"
                onSubmit={pw => {
                  rpc('sudo.respond', { request_id: sudo.requestId, password: pw }).then(r => {
                    if (!r) {
                      return
                    }

                    setSudo(null)
                    setStatus('running…')
                  })
                }}
                t={theme}
              />
            </PromptBox>
          )}

          {secret && (
            <PromptBox color={theme.color.bronze}>
              <MaskedPrompt
                cols={cols}
                icon="🔑"
                label={secret.prompt}
                onSubmit={val => {
                  rpc('secret.respond', { request_id: secret.requestId, value: val }).then(r => {
                    if (!r) {
                      return
                    }

                    setSecret(null)
                    setStatus('running…')
                  })
                }}
                sub={`for ${secret.envVar}`}
                t={theme}
              />
            </PromptBox>
          )}

          {picker && (
            <PromptBox color={theme.color.bronze}>
              <SessionPicker gw={gw} onCancel={() => setPicker(false)} onSelect={resumeById} t={theme} />
            </PromptBox>
          )}

          {modelPicker && (
            <PromptBox color={theme.color.bronze}>
              <ModelPicker
                gw={gw}
                onCancel={() => setModelPicker(false)}
                onSelect={value => {
                  setModelPicker(false)
                  slash(`/model ${value}`)
                }}
                sessionId={sid}
                t={theme}
              />
            </PromptBox>
          )}

          <QueuedMessages cols={cols} queued={queuedDisplay} queueEditIdx={queueEditIdx} t={theme} />

          {bgTasks.size > 0 && (
            <Text color={theme.color.dim} dimColor>
              {bgTasks.size} background {bgTasks.size === 1 ? 'task' : 'tasks'} running
            </Text>
          )}

          {showStickyPrompt ? (
            <Text color={theme.color.dim} dimColor wrap="truncate-end">
              <Text color={theme.color.label}>↳ </Text>
              {stickyPrompt}
            </Text>
          ) : (
            <Text> </Text>
          )}

          {statusBar && (
            <StatusRule
              bgCount={bgTasks.size}
              cols={cols}
              cwdLabel={cwdLabel}
              durationLabel={durationLabel}
              model={info?.model?.split('/').pop() ?? ''}
              status={status}
              statusColor={statusColor}
              t={theme}
              usage={usage}
              voiceLabel={voiceLabel}
            />
          )}

          {pager && (
            <Box borderColor={theme.color.bronze} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
              {pager.title && (
                <Box justifyContent="center" marginBottom={1}>
                  <Text bold color={theme.color.gold}>
                    {pager.title}
                  </Text>
                </Box>
              )}

              {pager.lines.slice(pager.offset, pager.offset + pagerPageSize).map((line, i) => (
                <Text key={i}>{line}</Text>
              ))}

              <Box marginTop={1}>
                <Text color={theme.color.dim}>
                  {pager.offset + pagerPageSize < pager.lines.length
                    ? `Enter/Space for more · q to close (${Math.min(pager.offset + pagerPageSize, pager.lines.length)}/${pager.lines.length})`
                    : `end · q to close (${pager.lines.length} lines)`}
                </Text>
              </Box>
            </Box>
          )}

          {!isBlocked && (
            <Box flexDirection="column" marginBottom={1}>
              {inputBuf.map((line, i) => (
                <Box key={i}>
                  <Box width={3}>
                    <Text color={theme.color.dim}>{i === 0 ? `${theme.brand.prompt} ` : '  '}</Text>
                  </Box>

                  <Text color={theme.color.cornsilk}>{line || ' '}</Text>
                </Box>
              ))}

              <Box>
                <Box width={3}>
                  <Text bold color={theme.color.gold}>
                    {inputBuf.length ? '  ' : `${theme.brand.prompt} `}
                  </Text>
                </Box>

                <TextInput
                  columns={Math.max(20, cols - 3)}
                  onChange={setInput}
                  onPaste={handleTextPaste}
                  onSubmit={submit}
                  placeholder={empty ? PLACEHOLDER : busy ? 'Ctrl+C to interrupt…' : ''}
                  value={input}
                />
              </Box>
            </Box>
          )}

          {!!completions.length && (
            <Box borderColor={theme.color.bronze} borderStyle="single" flexDirection="column" paddingX={1}>
              {completions.slice(Math.max(0, compIdx - 8), compIdx + 8).map((item, i) => {
                const active = Math.max(0, compIdx - 8) + i === compIdx

                return (
                  <Text key={item.text}>
                    <Text bold={active} color={active ? theme.color.amber : theme.color.cornsilk}>
                      {item.display}
                    </Text>
                    {item.meta ? <Text color={theme.color.dim}> {item.meta}</Text> : null}
                  </Text>
                )
              })}
            </Box>
          )}

          {!empty && !sid && <Text color={theme.color.dim}>⚕ {status}</Text>}
        </NoSelect>
      </Box>
    </AlternateScreen>
  )
}
