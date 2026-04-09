import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Box, Text, useApp, useInput, useStdout } from 'ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Banner, SessionPanel } from './components/branding.js'
import { MaskedPrompt } from './components/maskedPrompt.js'
import { MessageLine } from './components/messageLine.js'
import { ApprovalPrompt, ClarifyPrompt } from './components/prompts.js'
import { QueuedMessages } from './components/queuedMessages.js'
import { SessionPicker } from './components/sessionPicker.js'
import { type PasteEvent, TextInput } from './components/textInput.js'
import { Thinking, ToolTrail } from './components/thinking.js'
import { HOTKEYS, INTERPOLATION_RE, PLACEHOLDERS, TOOL_VERBS, ZERO } from './constants.js'
import { type GatewayClient, type GatewayEvent } from './gatewayClient.js'
import { useCompletion } from './hooks/useCompletion.js'
import { useInputHistory } from './hooks/useInputHistory.js'
import { useQueue } from './hooks/useQueue.js'
import { writeOsc52Clipboard } from './lib/osc52.js'
import { compactPreview, fmtK, hasInterpolation, isToolTrailResultLine, pick, sameToolTrailGroup } from './lib/text.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'
import type {
  ActiveTool,
  ActivityItem,
  ApprovalReq,
  ClarifyReq,
  Msg,
  PasteMode,
  PendingPaste,
  SecretReq,
  SessionInfo,
  SlashCatalog,
  SudoReq,
  Usage
} from './types.js'

// ── Constants ────────────────────────────────────────────────────────

const PLACEHOLDER = pick(PLACEHOLDERS)
const PASTE_TOKEN_RE = /\[\[paste:(\d+)\]\]/g
const STARTUP_RESUME_ID = (process.env.HERMES_TUI_RESUME ?? '').trim()

const LARGE_PASTE = { chars: 8000, lines: 80 }
const EXCERPT = { chars: 1200, lines: 14 }
const MAX_HISTORY = 800

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

  return rows.flatMap(row => {
    if (!row || typeof row !== 'object') {
      return []
    }

    const role = (row as any).role
    const text = (row as any).text

    if (
      (role !== 'assistant' && role !== 'system' && role !== 'tool' && role !== 'user') ||
      typeof text !== 'string' ||
      !text.trim()
    ) {
      return []
    }

    return [{ role, text }]
  })
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

function StatusRule({
  cols,
  status,
  statusColor,
  model,
  usage,
  bgCount,
  t
}: {
  cols: number
  status: string
  statusColor: string
  model: string
  usage: Usage
  bgCount: number
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

  const segs = [status, model, ctxLabel, bar ? `[${bar}]` : '', pctLabel, bgCount > 0 ? `${bgCount} bg` : ''].filter(
    Boolean
  )

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
  const [thinking, setThinking] = useState(false)
  const [turnKey, setTurnKey] = useState(0)
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
  const [statusBar, setStatusBar] = useState(true)
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [pastes, setPastes] = useState<PendingPaste[]>([])
  const [pasteReview, setPasteReview] = useState<{ largeIds: number[]; text: string } | null>(null)
  const [streaming, setStreaming] = useState('')
  const [turnTrail, setTurnTrail] = useState<string[]>([])
  const [bgTasks, setBgTasks] = useState<Set<string>>(new Set())
  const [catalog, setCatalog] = useState<SlashCatalog | null>(null)

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
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const busyRef = useRef(busy)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  colsRef.current = cols
  busyRef.current = busy
  reasoningRef.current = reasoning

  // ── Hooks ────────────────────────────────────────────────────────

  const { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue } =
    useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, blocked(), gw)

  function blocked() {
    return !!(clarify || approval || pasteReview || picker || secret || sudo)
  }

  const empty = !messages.length
  const isBlocked = blocked()

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

  const pushTrail = useCallback((line: string) => {
    setTurnTrail(prev => {
      if (prev.at(-1) === line) {
        return prev
      }

      const next = [...prev, line].slice(-8)
      turnToolsRef.current = next

      return next
    })
  }, [])

  const rpc = useCallback(
    (method: string, params: Record<string, unknown> = {}) =>
      gw.request(method, params).catch((e: Error) => {
        sys(`error: ${e.message}`)
      }),
    [gw, sys]
  )

  const idle = () => {
    setThinking(false)
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

      if (!text) {
        return null
      }

      const lineCount = text.split('\n').length

      if (text.length < LARGE_PASTE.chars && lineCount < LARGE_PASTE.lines) {
        return { cursor: cursor + text.length, value: value.slice(0, cursor) + text + value.slice(cursor) }
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
            charCount: text.length,
            createdAt: Date.now(),
            id,
            kind: classifyPaste(text),
            lineCount,
            mode,
            text
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

    if (statusTimerRef.current) {
      clearTimeout(statusTimerRef.current)
      statusTimerRef.current = null
    }

    inflightPasteIdsRef.current = payload.usedIds
    setLastUserMsg(text)
    appendMessage({ role: 'user', text })
    setBusy(true)
    setStatus('running…')
    buf.current = ''
    interruptedRef.current = false

    gw.request('prompt.submit', { session_id: sid, text: payload.text }).catch((e: Error) => {
      inflightPasteIdsRef.current = []
      sys(`error: ${e.message}`)
      setStatus('ready')
      setBusy(false)
    })
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
          gw.request('session.interrupt', { session_id: sid }).catch(() => {})
          setStatus('interrupting…')

          return
        }

        if (picked && sid) {
          send(picked)
        }

        return
      }

      pushHistory(full)

      if (busy) {
        if (hasInterpolation(full)) {
          interpolate(full, enqueue)

          return
        }

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

  useInput((ch, key) => {
    if (isBlocked) {
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
        if (approval) {
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

    if (completions.length && input && (key.upArrow || key.downArrow)) {
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
                pairs: r.pairs as [string, string][],
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

        case 'session.info':
          setInfo(p as SessionInfo)

          if (p?.usage) {
            setUsage(prev => ({ ...prev, ...p.usage }))
          }

          break

        case 'thinking.delta':
          break

        case 'message.start':
          setThinking(true)
          setTurnKey(k => k + 1)
          setBusy(true)
          setReasoning('')
          setActivity([])
          setTurnTrail([])
          turnToolsRef.current = []

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
          setTools(prev => [...prev, { id: p.tool_id, name: p.name, context: (p.context as string) || '' }])

          break
        case 'tool.complete': {
          const mark = p.error ? '✗' : '✓'

          toolCompleteRibbonRef.current = null
          setTools(prev => {
            const done = prev.find(t => t.id === p.tool_id)
            const label = TOOL_VERBS[done?.name ?? p.name] ?? done?.name ?? p.name
            const ctx = (p.error as string) || done?.context || ''
            const line = `${label}${ctx ? ': ' + compactPreview(ctx, 72) : ''} ${mark}`

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
          if (p?.text && !interruptedRef.current) {
            buf.current += p.rendered ?? p.text
            setStreaming(buf.current.trimStart())
          }

          break
        case 'message.complete': {
          const wasInterrupted = interruptedRef.current
          const savedReasoning = reasoningRef.current.trim()
          const savedTools = turnToolsRef.current.filter(isToolTrailResultLine)
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
          }

          turnToolsRef.current = []
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
            send(next)
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
          setStatus('ready')

          break
      }
    },
    [appendMessage, dequeue, newSession, pushActivity, pushTrail, send, sys]
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
          const rows = catalog?.pairs ?? []
          const cap = 52

          sys(
            [
              '  Commands:',
              ...rows.slice(0, cap).map(([c, d]) => `    ${c.padEnd(16)} ${d}`),
              rows.length > cap ? `    … ${rows.length - cap} more` : '',
              '',
              '  Hotkeys:',
              ...HOTKEYS.map(([k, d]) => `    ${k.padEnd(14)} ${d}`)
            ]
              .filter(Boolean)
              .join('\n')
          )

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
          setPicker(true)

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
            sys(
              pastes.length
                ? pastes
                    .map(
                      p =>
                        `#${p.id} ${p.mode} · ${p.lineCount}L · ${p.kind} · ${compactPreview(p.text, 60) || '(empty)'}`
                    )
                    .join('\n')
                : 'no text pastes'
            )

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

        case 'logs':
          sys(gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20))) || 'no gateway logs')

          return true

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
            rpc('config.get', { key: 'provider' }).then((r: any) => sys(`${r.model} (${r.provider})`))
          } else {
            rpc('config.set', { key: 'model', value: arg.replace('--global', '').trim() }).then((r: any) => {
              sys(`model → ${r.value}`)
              setInfo(prev => (prev ? { ...prev, model: r.value } : prev))
            })
          }

          return true

        case 'yolo':
          rpc('config.set', { key: 'yolo' }).then((r: any) => sys(`yolo ${r.value === '1' ? 'on' : 'off'}`))

          return true

        case 'reasoning':
          rpc('config.set', { key: 'reasoning', value: arg || 'medium' }).then((r: any) => sys(`reasoning: ${r.value}`))

          return true

        case 'verbose':
          rpc('config.set', { key: 'verbose', value: arg || 'cycle' }).then((r: any) => sys(`verbose: ${r.value}`))

          return true

        case 'personality':
          if (arg) {
            rpc('config.set', { key: 'personality', value: arg }).then((r: any) =>
              sys(`personality: ${r.value || 'default'}`)
            )
          } else {
            gw.request('slash.exec', { command: 'personality', session_id: sid })
              .then((r: any) => sys(r?.output || '(no output)'))
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
            const ln = (k: string, v: string) => `  ${k.padEnd(26)}${v.padStart(10)}`
            const hr = `  ${'─'.repeat(36)}`

            const cost =
              r.cost_usd != null ? `${r.cost_status === 'estimated' ? '~' : ''}$${r.cost_usd.toFixed(4)}` : null

            sys(
              [
                hr,
                ln('Model:', r.model ?? ''),
                ln('Input tokens:', f(r.input)),
                ln('Cache read tokens:', f(r.cache_read)),
                ln('Cache write tokens:', f(r.cache_write)),
                ln('Output tokens:', f(r.output)),
                ln('Total tokens:', f(r.total)),
                ln('API calls:', f(r.calls)),
                cost && ln('Cost:', cost),
                hr,
                r.context_max && `  Context: ${f(r.context_used)} / ${f(r.context_max)} (${r.context_percent}%)`,
                r.compressions && `  Compressions: ${r.compressions}`
              ]
                .filter(Boolean)
                .join('\n')
            )
          })

          return true

        case 'save':
          rpc('session.save', { session_id: sid }).then((r: any) => sys(`saved: ${r.file}`))

          return true

        case 'history':
          rpc('session.history', { session_id: sid }).then((r: any) => sys(`${r.count} messages`))

          return true

        case 'profile':
          rpc('config.get', { key: 'profile' }).then((r: any) => sys(r.display || r.home))

          return true

        case 'voice':
          rpc('voice.toggle', { action: arg === 'on' || arg === 'off' ? arg : 'status' }).then((r: any) =>
            sys(`voice${arg === 'on' || arg === 'off' ? '' : ':'} ${r.enabled ? 'on' : 'off'}`)
          )

          return true

        case 'insights':
          rpc('insights.get', { days: parseInt(arg) || 30 }).then((r: any) =>
            sys(`${r.days}d: ${r.sessions} sessions, ${r.messages} messages`)
          )

          return true
        case 'rollback': {
          const [sub, ...rArgs] = (arg || 'list').split(/\s+/)

          if (!sub || sub === 'list') {
            rpc('rollback.list', { session_id: sid }).then((r: any) => {
              if (!r.checkpoints?.length) {
                return sys('no checkpoints')
              }

              sys(r.checkpoints.map((c: any, i: number) => `  ${i} ${c.hash?.slice(0, 8)} ${c.message}`).join('\n'))
            })
          } else {
            const hash = sub === 'restore' || sub === 'diff' ? rArgs[0] : sub
            rpc(sub === 'diff' ? 'rollback.diff' : 'rollback.restore', { session_id: sid, hash }).then((r: any) =>
              sys(r.rendered || r.diff || r.message || 'done')
            )
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

            sys(r.plugins.map((p: any) => `  ${p.name} v${p.version}${p.enabled ? '' : ' (disabled)'}`).join('\n'))
          })

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
    [catalog, compact, gw, lastUserMsg, messages, newSession, pastes, pushActivity, rpc, send, sid, statusBar, sys]
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
          ) : (
            <MessageLine cols={cols} compact={compact} msg={m} t={theme} />
          )}
        </Box>
      ))}

      <Box flexDirection="column" paddingX={1}>
        <ToolTrail
          activity={busy ? activity : []}
          animateCot={busy && !streaming}
          t={theme}
          tools={tools}
          trail={turnTrail}
        />

        {busy && !tools.length && !streaming && <Thinking key={turnKey} reasoning={reasoning} t={theme} />}

        {streaming && (
          <MessageLine cols={cols} compact={compact} msg={{ role: 'assistant', text: streaming }} t={theme} />
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
              onAnswer={answer => {
                gw.request('clarify.respond', { answer, request_id: clarify.requestId }).catch(() => {})
                appendMessage({ role: 'user', text: answer })
                setClarify(null)
              }}
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
            <SessionPicker
              gw={gw}
              onCancel={() => setPicker(false)}
              onSelect={id => {
                setPicker(false)
                setStatus('resuming…')
                gw.request('session.resume', { cols: colsRef.current, session_id: id })
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
                  .catch((e: Error) => {
                    sys(`error: ${e.message}`)
                    setStatus('ready')
                  })
              }}
              t={theme}
            />
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
            model={info?.model?.split('/').pop() ?? ''}
            status={status}
            statusColor={statusColor}
            t={theme}
            usage={usage}
          />
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
