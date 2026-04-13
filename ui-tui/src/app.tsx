import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Box, Text, useApp, useInput, useStdout } from '@hermes/ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Banner, Panel, SessionPanel } from './components/branding.js'
import { MaskedPrompt } from './components/maskedPrompt.js'
import { MessageLine } from './components/messageLine.js'
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
import { writeOsc52Clipboard } from './lib/osc52.js'
import {
  buildToolTrailLine,
  compactPreview,
  fmtK,
  hasInterpolation,
  isToolTrailResultLine,
  isTransientTrailLine,
  pick,
  sameToolTrailGroup,
  stripTrailingPasteNewlines,
  toolTrailLabel
} from './lib/text.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'
import type {
  ActiveTool,
  ActivityItem,
  ApprovalReq,
  ClarifyReq,
  Msg,
  PanelSection,
  PasteMode,
  PendingPaste,
  SecretReq,
  SessionInfo,
  SlashCatalog,
  SudoReq,
  ThinkingMode,
  Usage
} from './types.js'

// ── Constants ────────────────────────────────────────────────────────

const PLACEHOLDER = pick(PLACEHOLDERS)
const PASTE_TOKEN_RE = /\[\[paste:(\d+)\]\]/g
const STARTUP_RESUME_ID = (process.env.HERMES_TUI_RESUME ?? '').trim()

const LARGE_PASTE = { chars: 8000, lines: 80 }
const EXCERPT = { chars: 1200, lines: 14 }
const MAX_HISTORY = 800
const REASONING_PULSE_MS = 700

const SECRET_PATTERNS = [
  /AKIA[0-9A-Z]{16}/g,
  /AIza[0-9A-Za-z-_]{30,}/g,
  /gh[pousr]_[A-Za-z0-9]{20,}/g,
  /sk-[A-Za-z0-9]{20,}/g,
  /sk-ant-[A-Za-z0-9-]{20,}/g,
  /xox[baprs]-[A-Za-z0-9-]{10,}/g,
  /\b(?:api[_-]?key|token|secret)\b\s*[:=]\s*["']?[A-Za-z0-9_-]{12,}/gi
]

// ── Pure helpers ─────────────────────────────────────────────────────

const introMsg = (info: SessionInfo): Msg => ({ role: 'system', text: '', kind: 'intro', info })

const classifyPaste = (text: string): PendingPaste['kind'] => {
  if (/error|warn|traceback|exception|stack|debug|\[\d{2}:\d{2}:\d{2}\]/i.test(text)) {
    return 'log'
  }

  if (
    /```|function\s+\w+|class\s+\w+|import\s+.+from|const\s+\w+\s*=|def\s+\w+\(|<\w+/.test(text) ||
    text.split('\n').filter(l => /[{}()[\];<>]/.test(l)).length >= 3
  ) {
    return 'code'
  }

  return 'text'
}

const redactSecrets = (text: string) => {
  let redactions = 0

  const cleaned = SECRET_PATTERNS.reduce(
    (t, pat) =>
      t.replace(pat, val => {
        redactions++

        return val.includes(':') || val.includes('=')
          ? `${val.split(/[:=]/)[0]}: [REDACTED_SECRET]`
          : '[REDACTED_SECRET]'
      }),
    text
  )

  return { redactions, text: cleaned }
}

const pasteToken = (id: number) => `[[paste:${id}]]`

const stripTokens = (text: string, re: RegExp) =>
  text
    .replace(re, '')
    .replace(/\s{2,}/g, ' ')
    .trim()

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

  const segs = [
    status,
    model,
    ctxLabel,
    bar ? `[${bar}]` : '',
    pctLabel,
    durationLabel || '',
    voiceLabel || '',
    bgCount > 0 ? `${bgCount} bg` : ''
  ].filter(Boolean)

  const inner = segs.join(' │ ')
  const pad = Math.max(0, cols - inner.length - 5)

  return (
    <Text color={t.color.bronze}>
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
      {' ' + '─'.repeat(pad)}
    </Text>
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
  const [picker, setPicker] = useState(false)
  const [reasoning, setReasoning] = useState('')
  const [reasoningActive, setReasoningActive] = useState(false)
  const [reasoningStreaming, setReasoningStreaming] = useState(false)
  const [statusBar, setStatusBar] = useState(true)
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [pastes, setPastes] = useState<PendingPaste[]>([])
  const [pasteReview, setPasteReview] = useState<{ largeIds: number[]; text: string } | null>(null)
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
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>('truncated')

  // ── Refs ─────────────────────────────────────────────────────────

  const activityIdRef = useRef(0)
  const toolCompleteRibbonRef = useRef<{ label: string; line: string } | null>(null)
  const buf = useRef('')
  const inflightPasteIdsRef = useRef<number[]>([])
  const interruptedRef = useRef(false)
  const reasoningRef = useRef('')
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const lastEmptyAt = useRef(0)
  const lastStatusNoteRef = useRef('')
  const protocolWarnedRef = useRef(false)
  const pasteCounterRef = useRef(0)
  const colsRef = useRef(cols)
  const turnToolsRef = useRef<string[]>([])
  const persistedToolLabelsRef = useRef<Set<string>>(new Set())
  const reasoningStreamingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const busyRef = useRef(busy)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  const configMtimeRef = useRef(0)
  colsRef.current = cols
  busyRef.current = busy
  reasoningRef.current = reasoning

  // ── Hooks ────────────────────────────────────────────────────────

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
      if (reasoningStreamingTimerRef.current) {
        clearTimeout(reasoningStreamingTimerRef.current)
      }
    },
    []
  )

  function blocked() {
    return !!(clarify || approval || pasteReview || picker || secret || sudo || pager)
  }

  const empty = !messages.length
  const isBlocked = blocked()
  const hasAnyThinking = Boolean(reasoning.trim() || historyItems.some(m => m.thinking?.trim()))

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
    (method: string, params: Record<string, unknown> = {}) =>
      gw.request(method, params).catch((e: Error) => {
        sys(`error: ${e.message}`)
      }),
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

      gw.request('clarify.respond', { answer, request_id: clarify.requestId }).catch(() => {})

      if (answer) {
        persistedToolLabelsRef.current.add(label)
        appendMessage({
          role: 'system',
          text: '',
          kind: 'trail',
          tools: [buildToolTrailLine('clarify', clarify.question)]
        })
        appendMessage({ role: 'user', text: answer })
      } else {
        sys('prompt cancelled')
      }

      setClarify(null)
      setStatus('running…')
    },
    [appendMessage, clarify, gw, sys]
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
      setBellOnComplete(!!r?.config?.display?.bell_on_complete)
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
          rpc('reload.mcp', { session_id: sid }).then(() => pushActivity('MCP reloaded after config change'))
          rpc('config.get', { key: 'full' }).then((cfg: any) => {
            setBellOnComplete(!!cfg?.config?.display?.bell_on_complete)
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
    setPasteReview(null)
    setSudo(null)
    setSecret(null)
    setStreaming('')
    buf.current = ''
  }

  const die = () => {
    gw.kill()
    exit()
  }

  const clearIn = () => {
    setInput('')
    setInputBuf([])
    setPasteReview(null)
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }

  const resetSession = () => {
    idle()
    setReasoning('')
    setVoiceRecording(false)
    setVoiceProcessing(false)
    setSid(null as any) // will be set by caller
    setHistoryItems([])
    setMessages([])
    setPastes([])
    setActivity([])
    setBgTasks(new Set())
    setUsage(ZERO)
    turnToolsRef.current = []
    lastStatusNoteRef.current = ''
    protocolWarnedRef.current = false
  }

  // ── Session management ───────────────────────────────────────────

  const newSession = useCallback(
    (msg?: string) =>
      rpc('session.create', { cols: colsRef.current }).then((r: any) => {
        if (!r) {
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

        if (msg) {
          sys(msg)
        }
      }),
    [rpc, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      setPicker(false)
      setStatus('resuming…')
      gw.request('session.resume', { cols: colsRef.current, session_id: id })
        .then((r: any) => {
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
    },
    [gw, sys]
  )

  // ── Paste pipeline ───────────────────────────────────────────────

  const listPasteIds = useCallback((text: string) => {
    const ids = new Set<number>()

    for (const m of text.matchAll(PASTE_TOKEN_RE)) {
      const id = parseInt(m[1] ?? '-1', 10)

      if (id > 0) {
        ids.add(id)
      }
    }

    return [...ids]
  }, [])

  const resolvePasteTokens = useCallback(
    (text: string) => {
      const byId = new Map(pastes.map(p => [p.id, p]))
      const missingIds = new Set<number>()
      const usedIds = new Set<number>()
      let redactions = 0

      const resolved = text.replace(PASTE_TOKEN_RE, (_m, rawId: string) => {
        const id = parseInt(rawId, 10)
        const paste = byId.get(id)

        if (!paste) {
          missingIds.add(id)

          return `[missing paste:${id}]`
        }

        usedIds.add(id)
        const cleaned = redactSecrets(paste.text)
        redactions += cleaned.redactions

        if (paste.mode === 'inline') {
          return cleaned.text
        }

        const lang = paste.kind === 'code' ? 'text' : ''
        const lines = cleaned.text.split('\n')

        if (paste.mode === 'excerpt') {
          let excerpt = lines.slice(0, EXCERPT.lines).join('\n')

          if (excerpt.length > EXCERPT.chars) {
            excerpt = excerpt.slice(0, EXCERPT.chars).trimEnd() + '…'
          }

          const truncated = lines.length > EXCERPT.lines || cleaned.text.length > excerpt.length
          const tail = truncated ? `\n…[paste #${id} truncated]` : ''

          return `[paste #${id} excerpt]\n\`\`\`${lang}\n${excerpt}${tail}\n\`\`\``
        }

        return `[paste #${id} attached · ${paste.lineCount} lines]\n\`\`\`${lang}\n${cleaned.text}\n\`\`\``
      })

      return { missingIds: [...missingIds], redactions, text: resolved, usedIds: [...usedIds] }
    },
    [pastes]
  )

  const paste = useCallback(
    (quiet = false) =>
      rpc('clipboard.paste', { session_id: sid }).then((r: any) =>
        r?.attached
          ? sys(`📎 Image #${r.count} attached from clipboard`)
          : quiet || sys(r?.message || 'No image found in clipboard')
      ),
    [rpc, sid, sys]
  )

  const handleTextPaste = useCallback(
    ({ bracketed, cursor, hotkey, text, value }: PasteEvent) => {
      if (hotkey) {
        void paste(false)

        return null
      }

      if (bracketed) {
        void paste(true)
      }

      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText) {
        return null
      }

      const lineCount = cleanedText.split('\n').length

      if (cleanedText.length < LARGE_PASTE.chars && lineCount < LARGE_PASTE.lines) {
        return {
          cursor: cursor + cleanedText.length,
          value: value.slice(0, cursor) + cleanedText + value.slice(cursor)
        }
      }

      pasteCounterRef.current++
      const id = pasteCounterRef.current
      const mode: PasteMode = 'attach'
      const token = pasteToken(id)
      const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
      const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
      const insert = `${lead}${token}${tail}`

      setPastes(prev =>
        [
          ...prev,
          {
            charCount: cleanedText.length,
            createdAt: Date.now(),
            id,
            kind: classifyPaste(cleanedText),
            lineCount,
            mode,
            text: cleanedText
          }
        ].slice(-24)
      )

      pushActivity(`captured ${lineCount}L paste as ${token} (${mode})`)

      return { cursor: cursor + insert.length, value: value.slice(0, cursor) + insert + value.slice(cursor) }
    },
    [paste, pushActivity]
  )

  // ── Send ─────────────────────────────────────────────────────────

  const send = (text: string) => {
    const payload = resolvePasteTokens(text)

    if (payload.missingIds.length) {
      pushActivity(`missing paste token(s): ${payload.missingIds.join(', ')}`, 'warn')

      return
    }

    if (payload.redactions > 0) {
      pushActivity(`redacted ${payload.redactions} secret-like value(s)`, 'warn')
    }

    const startSubmit = (displayText: string, submitText: string) => {
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current)
        statusTimerRef.current = null
      }

      inflightPasteIdsRef.current = payload.usedIds
      setLastUserMsg(text)
      appendMessage({ role: 'user', text: displayText })
      setBusy(true)
      setStatus('running…')
      buf.current = ''
      interruptedRef.current = false

      gw.request('prompt.submit', { session_id: sid, text: submitText }).catch((e: Error) => {
        inflightPasteIdsRef.current = []
        sys(`error: ${e.message}`)
        setStatus('ready')
        setBusy(false)
      })
    }

    gw.request('input.detect_drop', { session_id: sid, text: payload.text })
      .then((r: any) => {
        if (r?.matched) {
          if (r.is_image) {
            pushActivity(`attached image: ${r.name}`)
          } else {
            pushActivity(`detected file: ${r.name}`)
          }

          startSubmit(r.text || text, r.text || payload.text)

          return
        }

        startSubmit(text, payload.text)
      })
      .catch(() => startSubmit(text, payload.text))
  }

  const shellExec = (cmd: string) => {
    appendMessage({ role: 'user', text: `!${cmd}` })
    setBusy(true)
    setStatus('running…')

    gw.request('shell.exec', { command: cmd })
      .then((r: any) => {
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
          .then((r: any) => [r.stdout, r.stderr].filter(Boolean).join('\n').trim())
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
    (full: string, allowLarge = false) => {
      if (!full.trim() || !sid) {
        return
      }

      const clearInput = () => {
        setInputBuf([])
        setInput('')
        setHistoryIdx(null)
        historyDraftRef.current = ''
      }

      if (full.startsWith('/')) {
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

      const { missingIds } = resolvePasteTokens(full)

      if (missingIds.length) {
        pushActivity(`missing paste token(s): ${missingIds.join(', ')}`, 'warn')

        return
      }

      const largeIds = listPasteIds(full).filter(id => {
        const p = pastes.find(x => x.id === id)

        return !!p && (p.charCount >= LARGE_PASTE.chars || p.lineCount >= LARGE_PASTE.lines)
      })

      if (!allowLarge && largeIds.length) {
        setPasteReview({ largeIds, text: full })
        setStatus(`review large paste (${largeIds.length})`)

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
    [appendMessage, busy, enqueue, gw, listPasteIds, pastes, pushHistory, resolvePasteTokens, sid]
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

      if (pasteReview) {
        if (key.return) {
          setPasteReview(null)
          dispatchSubmission(pasteReview.text, true)
        } else if (key.escape || ctrl(key, ch, 'c')) {
          setPasteReview(null)
          setStatus('ready')
        }

        return
      }

      if (ctrl(key, ch, 'c')) {
        if (clarify) {
          answerClarify('')
        } else if (approval) {
          gw.request('approval.respond', { choice: 'deny', session_id: sid }).catch(() => {})
          setApproval(null)
          sys('denied')
        } else if (sudo) {
          gw.request('sudo.respond', { request_id: sudo.requestId, password: '' }).catch(() => {})
          setSudo(null)
          sys('sudo cancelled')
        } else if (secret) {
          gw.request('secret.respond', { request_id: secret.requestId, value: '' }).catch(() => {})
          setSecret(null)
          sys('secret entry cancelled')
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

    if (!inputBuf.length && key.tab && completions.length) {
      const row = completions[compIdx]

      if (row) {
        setInput(input.slice(0, compReplace) + row.text)
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
      if (busy && sid) {
        interruptedRef.current = true
        gw.request('session.interrupt', { session_id: sid }).catch(() => {})
        const partial = (streaming || buf.current).trimStart()
        partial ? appendMessage({ role: 'assistant', text: partial + '\n\n*[interrupted]*' }) : sys('interrupted')

        idle()
        setReasoning('')
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
      setStatus('forging session…')
      newSession()

      return
    }

    if (ctrl(key, ch, 't')) {
      if (hasAnyThinking) {
        setThinkingMode(mode => (mode === 'collapsed' ? 'truncated' : mode === 'truncated' ? 'full' : 'collapsed'))
      } else {
        sys('no thinking available')
      }

      return
    }

    if (ctrl(key, ch, 'b')) {
      if (voiceRecording) {
        setVoiceRecording(false)
        setVoiceProcessing(true)
        rpc('voice.record', { action: 'stop' })
          .then((r: any) => {
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
          .then(() => {
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
            })
            .catch(() => {})

          if (STARTUP_RESUME_ID) {
            setStatus('resuming…')
            gw.request('session.resume', { cols: colsRef.current, session_id: STARTUP_RESUME_ID })
              .then((r: any) => {
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
              .catch(() => {
                setStatus('forging session…')
                newSession('resume failed, started a new session')
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
          break

        case 'message.start':
          setBusy(true)
          endReasoningPhase()
          setReasoning('')
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

        case 'gateway.protocol_error':
          setStatus('protocol warning')

          if (!protocolWarnedRef.current) {
            protocolWarnedRef.current = true
            pushActivity('protocol noise detected · /logs to inspect', 'warn')
          }

          break

        case 'reasoning.delta':
          if (p?.text) {
            setReasoning(prev => prev + p.text)
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
            setStreaming(buf.current.trimStart())
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
          setReasoning('')
          setStreaming('')

          if (inflightPasteIdsRef.current.length) {
            setPastes(prev => prev.filter(paste => !inflightPasteIdsRef.current.includes(paste.id)))
            inflightPasteIdsRef.current = []
          }

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
          inflightPasteIdsRef.current = []
          sys(`error: ${p?.message}`)
          idle()
          setReasoning('')
          setActivity([])
          turnToolsRef.current = []
          persistedToolLabelsRef.current.clear()
          setStatus('ready')

          break
      }
    },
    [
      appendMessage,
      bellOnComplete,
      dequeue,
      endReasoningPhase,
      newSession,
      pruneTransient,
      pulseReasoningStreaming,
      pushActivity,
      pushTrail,
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
      exit()
    }

    gw.on('event', handler)
    gw.on('exit', exitHandler)

    return () => {
      gw.off('event', handler)
      gw.off('exit', exitHandler)
    }
  }, [gw, exit])

  // ── Slash commands ───────────────────────────────────────────────

  const slash = useCallback(
    (cmd: string): boolean => {
      const [name, ...rest] = cmd.slice(1).split(/\s+/)
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
          setStatus('forging session…')
          newSession()

          return true

        case 'new':
          setStatus('forging session…')
          newSession('new session started')

          return true

        case 'resume':
          if (arg) {
            resumeById(arg)
          } else {
            setPicker(true)
          }

          return true

        case 'compact':
          setCompact(c => (arg ? true : !c))
          sys(arg ? `compact on, focus: ${arg}` : `compact ${compact ? 'off' : 'on'}`)

          return true
        case 'copy': {
          const all = messages.filter(m => m.role === 'assistant')
          const target = all[arg ? Math.min(parseInt(arg), all.length) - 1 : all.length - 1]

          if (!target) {
            sys('nothing to copy')

            return true
          }

          writeOsc52Clipboard(target.text)
          sys('copied to clipboard')

          return true
        }

        case 'paste':
          if (!arg) {
            paste()

            return true
          }

          if (arg === 'list') {
            if (!pastes.length) {
              sys('no text pastes')
            } else {
              panel('Paste Shelf', [
                {
                  rows: pastes.map(
                    p =>
                      [
                        `#${p.id} ${p.mode}`,
                        `${p.lineCount}L · ${p.kind} · ${compactPreview(p.text, 60) || '(empty)'}`
                      ] as [string, string]
                  )
                }
              ])
            }

            return true
          }

          if (arg === 'clear') {
            setPastes([])
            setInput(v => stripTokens(v, PASTE_TOKEN_RE))
            setInputBuf(prev => prev.map(l => stripTokens(l, PASTE_TOKEN_RE)).filter(Boolean))
            pushActivity('cleared paste shelf')

            return true
          }

          if (arg.startsWith('drop ')) {
            const id = parseInt(arg.split(/\s+/)[1] ?? '-1', 10)

            if (!id || !pastes.some(p => p.id === id)) {
              sys('usage: /paste drop <id>')

              return true
            }

            const re = new RegExp(`\\s*\\[\\[paste:${id}\\]\\]\\s*`, 'g')
            setPastes(prev => prev.filter(p => p.id !== id))
            setInput(v => stripTokens(v, re))
            setInputBuf(prev => prev.map(l => stripTokens(l, re)).filter(Boolean))
            pushActivity(`dropped paste #${id}`)

            return true
          }

          if (arg.startsWith('mode ')) {
            const [, rawId, rawMode] = arg.split(/\s+/)
            const id = parseInt(rawId ?? '-1', 10)
            const mode = rawMode as PasteMode

            if (!id || !['attach', 'excerpt', 'inline'].includes(mode) || !pastes.some(p => p.id === id)) {
              sys('usage: /paste mode <id> <attach|excerpt|inline>')

              return true
            }

            setPastes(prev => prev.map(p => (p.id === id ? { ...p, mode } : p)))
            pushActivity(`paste #${id} mode → ${mode}`)

            return true
          }

          sys('usage: /paste [list|mode <id> <attach|excerpt|inline>|drop <id>|clear]')

          return true
        case 'logs': {
          const logText = gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))
          logText ? page(logText, 'Logs') : sys('no gateway logs')

          return true
        }

        case 'statusbar':

        case 'sb':
          setStatusBar(v => !v)
          sys(`status bar ${statusBar ? 'off' : 'on'}`)

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
            return true
          }

          rpc('session.undo', { session_id: sid }).then((r: any) => {
            if (r.removed > 0) {
              setMessages(prev => {
                const q = [...prev]

                while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
                  q.pop()
                }

                if (q.at(-1)?.role === 'user') {
                  q.pop()
                }

                return q
              })
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
            gw.request('session.undo', { session_id: sid }).catch(() => {})
          }

          setMessages(prev => {
            const q = [...prev]

            while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
              q.pop()
            }

            return q
          })
          send(lastUserMsg)

          return true

        case 'background':

        case 'bg':
          if (!arg) {
            sys('/background <prompt>')

            return true
          }

          rpc('prompt.background', { session_id: sid, text: arg }).then((r: any) => {
            setBgTasks(prev => new Set(prev).add(r.task_id))
            sys(`bg ${r.task_id} started`)
          })

          return true

        case 'btw':
          if (!arg) {
            sys('/btw <question>')

            return true
          }

          rpc('prompt.btw', { session_id: sid, text: arg }).then(() => {
            setBgTasks(prev => new Set(prev).add('btw:x'))
            sys('btw running…')
          })

          return true

        case 'model':
          if (!arg) {
            rpc('config.get', { key: 'provider' }).then((r: any) =>
              panel('Model', [
                {
                  rows: [
                    ['Model', r.model],
                    ['Provider', r.provider]
                  ]
                }
              ])
            )
          } else {
            rpc('config.set', { session_id: sid, key: 'model', value: arg.replace('--global', '').trim() }).then(
              (r: any) => {
                sys(`model → ${r.value}`)
                setInfo(prev => (prev ? { ...prev, model: r.value } : prev))
              }
            )
          }

          return true

        case 'image':
          rpc('image.attach', { session_id: sid, path: arg }).then((r: any) => {
            if (!r) {
              return
            }

            sys(`attached image: ${r.name}`)

            if (r?.remainder) {
              setInput(r.remainder)
            }
          })

          return true

        case 'provider':
          gw.request('slash.exec', { command: 'provider', session_id: sid })
            .then((r: any) => page(r?.output || '(no output)', 'Provider'))
            .catch(() => sys('provider command failed'))

          return true

        case 'skin':
          if (arg) {
            rpc('config.set', { key: 'skin', value: arg }).then((r: any) => sys(`skin → ${r.value}`))
          } else {
            rpc('config.get', { key: 'skin' }).then((r: any) => sys(`skin: ${r.value || 'default'}`))
          }

          return true

        case 'yolo':
          rpc('config.set', { session_id: sid, key: 'yolo' }).then((r: any) =>
            sys(`yolo ${r.value === '1' ? 'on' : 'off'}`)
          )

          return true

        case 'reasoning':
          rpc('config.set', { session_id: sid, key: 'reasoning', value: arg || 'medium' }).then((r: any) =>
            sys(`reasoning: ${r.value}`)
          )

          return true

        case 'verbose':
          rpc('config.set', { session_id: sid, key: 'verbose', value: arg || 'cycle' }).then((r: any) =>
            sys(`verbose: ${r.value}`)
          )

          return true

        case 'personality':
          if (arg) {
            rpc('config.set', { key: 'personality', value: arg }).then((r: any) =>
              sys(`personality: ${r.value || 'default'}`)
            )
          } else {
            gw.request('slash.exec', { command: 'personality', session_id: sid })
              .then((r: any) => panel('Personality', [{ text: r?.output || '(no output)' }]))
              .catch(() => sys('personality command failed'))
          }

          return true

        case 'compress':
          rpc('session.compress', { session_id: sid }).then((r: any) =>
            sys(`compressed${r.usage?.total ? ' · ' + fmtK(r.usage.total) + ' tok' : ''}`)
          )

          return true

        case 'stop':
          rpc('process.stop', {}).then((r: any) => sys(`killed ${r.killed ?? 0} process(es)`))

          return true

        case 'branch':

        case 'fork':
          rpc('session.branch', { session_id: sid, name: arg }).then((r: any) => {
            if (r?.session_id) {
              setSid(r.session_id)
              setSessionStartedAt(Date.now())
              setHistoryItems([])
              setMessages([])
              sys(`branched → ${r.title}`)
            }
          })

          return true

        case 'reload-mcp':

        case 'reload_mcp':
          rpc('reload.mcp', { session_id: sid }).then(() => sys('MCP reloaded'))

          return true

        case 'title':
          rpc('session.title', { session_id: sid, ...(arg ? { title: arg } : {}) }).then((r: any) =>
            sys(`title: ${r.title || '(none)'}`)
          )

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
          rpc('session.save', { session_id: sid }).then((r: any) => sys(`saved: ${r.file}`))

          return true

        case 'history':
          rpc('session.history', { session_id: sid }).then((r: any) => sys(`${r.count} messages`))

          return true

        case 'profile':
          rpc('config.get', { key: 'profile' }).then((r: any) => {
            const text = r.display || r.home
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
            sys(`voice${arg === 'on' || arg === 'off' ? '' : ':'} ${r.enabled ? 'on' : 'off'}`)
          })

          return true

        case 'insights':
          rpc('insights.get', { days: parseInt(arg) || 30 }).then((r: any) =>
            panel('Insights', [
              {
                rows: [
                  ['Period', `${r.days} days`],
                  ['Sessions', `${r.sessions}`],
                  ['Messages', `${r.messages}`]
                ]
              }
            ])
          )

          return true
        case 'rollback': {
          const [sub, ...rArgs] = (arg || 'list').split(/\s+/)

          if (!sub || sub === 'list') {
            rpc('rollback.list', { session_id: sid }).then((r: any) => {
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
            }).then((r: any) => sys(r.rendered || r.diff || r.message || 'done'))
          }

          return true
        }

        case 'browser': {
          const [act, ...bArgs] = (arg || 'status').split(/\s+/)
          rpc('browser.manage', { action: act, ...(bArgs[0] ? { url: bArgs[0] } : {}) }).then((r: any) =>
            sys(r.connected ? `browser: ${r.url}` : 'browser: disconnected')
          )

          return true
        }

        case 'plugins':
          rpc('plugins.list', {}).then((r: any) => {
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
            .then((r: any) => sys(r?.output || '/skills: no output'))
            .catch(() => sys(`skills: ${sub} failed`))

          return true
        }

        case 'agents':

        case 'tasks':
          rpc('agents.list', {})
            .then((r: any) => {
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
            .catch(() => sys('agents command failed'))

          return true

        case 'cron':
          if (!arg || arg === 'list') {
            rpc('cron.manage', { action: 'list' })
              .then((r: any) => {
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
              .catch(() => sys('cron command failed'))
          } else {
            gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
              .then((r: any) => sys(r?.output || '(no output)'))
              .catch(() => sys('cron command failed'))
          }

          return true

        case 'config':
          rpc('config.show', {})
            .then((r: any) => {
              panel(
                'Config',
                (r.sections ?? []).map((s: any) => ({
                  title: s.title,
                  rows: s.rows
                }))
              )
            })
            .catch(() => sys('config command failed'))

          return true

        case 'tools':
          rpc('tools.list', { session_id: sid })
            .then((r: any) => {
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
            .catch(() => sys('tools command failed'))

          return true

        case 'toolsets':
          rpc('toolsets.list', { session_id: sid })
            .then((r: any) => {
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
            .catch(() => sys('toolsets command failed'))

          return true

        default:
          gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then((r: any) => sys(r?.output || `/${name}: no output`))
            .catch(() => {
              gw.request('command.dispatch', { name: name ?? '', arg, session_id: sid })
                .then((d: any) => {
                  if (d.type === 'exec') {
                    sys(d.output || '(no output)')
                  } else if (d.type === 'alias') {
                    slash(`/${d.target}${arg ? ' ' + arg : ''}`)
                  } else if (d.type === 'plugin') {
                    sys(d.output || '(no output)')
                  } else if (d.type === 'skill') {
                    sys(`⚡ loading skill: ${d.name}`)
                    send(d.message)
                  }
                })
                .catch(() => sys(`unknown command: /${name}`))
            })

          return true
      }
    },
    [
      catalog,
      compact,
      gw,
      lastUserMsg,
      messages,
      newSession,
      page,
      panel,
      pastes,
      pushActivity,
      rpc,
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
          setReasoning('')
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
            dispatchSubmission(next, true)
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

  const hasReasoning = Boolean(reasoning.trim())

  const showProgressArea = Boolean(busy || tools.length || turnTrail.length || hasReasoning)

  const showStreamingArea = Boolean(streaming)

  // ── Render ───────────────────────────────────────────────────────

  return (
    <Box flexDirection="column">
      {historyItems.map((m, i) => (
        <Box flexDirection="column" key={i} paddingX={1}>
          {m.kind === 'intro' && m.info ? (
            <Box flexDirection="column" paddingTop={1}>
              <Banner t={theme} />
              <SessionPanel info={m.info} sid={sid} t={theme} />
            </Box>
          ) : m.kind === 'panel' && m.panelData ? (
            <Panel sections={m.panelData.sections} t={theme} title={m.panelData.title} />
          ) : (
            <MessageLine cols={cols} compact={compact} msg={m} t={theme} thinkingMode={thinkingMode} />
          )}
        </Box>
      ))}

      <Box flexDirection="column" paddingX={1}>
        {showProgressArea && (
          <Box marginBottom={showStreamingArea ? 1 : 0}>
            <ToolTrail
              activity={busy ? activity : []}
              busy={busy && !streaming}
              reasoning={reasoning}
              reasoningActive={reasoningActive}
              reasoningStreaming={reasoningStreaming}
              t={theme}
              thinkingMode={hasReasoning ? thinkingMode : 'truncated'}
              tools={tools}
              trail={turnTrail}
            />
          </Box>
        )}

        {showStreamingArea && (
          <Box>
            <MessageLine cols={cols} compact={compact} msg={{ role: 'assistant', text: streaming }} t={theme} />
          </Box>
        )}

        {pasteReview && (
          <PromptBox color={theme.color.warn}>
            <Text bold color={theme.color.warn}>
              Review large paste before send
            </Text>
            <Text color={theme.color.dim}>pastes: {pasteReview.largeIds.map(id => `#${id}`).join(', ')}</Text>
            <Text color={theme.color.dim}>Enter to send · Esc/Ctrl+C to cancel</Text>
          </PromptBox>
        )}

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
                gw.request('approval.respond', { choice, session_id: sid }).catch(() => {})
                setApproval(null)
                sys(choice === 'deny' ? 'denied' : `approved (${choice})`)
                setStatus('running…')
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
                gw.request('sudo.respond', { request_id: sudo.requestId, password: pw }).catch(() => {})
                setSudo(null)
                setStatus('running…')
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
                gw.request('secret.respond', { request_id: secret.requestId, value: val }).catch(() => {})
                setSecret(null)
                setStatus('running…')
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

        <QueuedMessages cols={cols} queued={queuedDisplay} queueEditIdx={queueEditIdx} t={theme} />

        {bgTasks.size > 0 && (
          <Text color={theme.color.dim} dimColor>
            {bgTasks.size} background {bgTasks.size === 1 ? 'task' : 'tasks'} running · /stop to cancel
          </Text>
        )}

        <Text> </Text>

        {statusBar && (
          <StatusRule
            bgCount={bgTasks.size}
            cols={cols}
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
          <Box flexDirection="column">
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
      </Box>
    </Box>
  )
}
