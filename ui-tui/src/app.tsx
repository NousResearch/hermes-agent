import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { Box, Static, Text, useApp, useInput, useStdout } from 'ink'
import { useCallback, useEffect, useRef, useState } from 'react'

import { ActivityLane } from './components/activityLane.js'
import { Banner, SessionPanel } from './components/branding.js'
import { MaskedPrompt } from './components/maskedPrompt.js'
import { MessageLine } from './components/messageLine.js'
import { ApprovalPrompt, ClarifyPrompt } from './components/prompts.js'
import { QueuedMessages } from './components/queuedMessages.js'
import { SessionPicker } from './components/sessionPicker.js'
import { TextInput } from './components/textInput.js'
import { Thinking } from './components/thinking.js'
import { HOTKEYS, INTERPOLATION_RE, PLACEHOLDERS, TOOL_VERBS, ZERO } from './constants.js'
import { type GatewayClient, type GatewayEvent } from './gatewayClient.js'
import { useCompletion } from './hooks/useCompletion.js'
import { useInputHistory } from './hooks/useInputHistory.js'
import { useQueue } from './hooks/useQueue.js'
import { writeOsc52Clipboard } from './lib/osc52.js'
import { compactPreview, fmtK, hasInterpolation, pick } from './lib/text.js'
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

const PLACEHOLDER = pick(PLACEHOLDERS)
const PASTE_TOKEN_RE = /\[\[paste:(\d+)\]\]/g
const EXCERPT_CHAR_LIMIT = 1200
const EXCERPT_LINE_LIMIT = 14
const LARGE_PASTE_CHAR_LIMIT = 8000
const LARGE_PASTE_LINE_LIMIT = 80
const SMALL_PASTE_CHAR_LIMIT = 400
const SMALL_PASTE_LINE_LIMIT = 4

const SECRET_PATTERNS = [
  /AKIA[0-9A-Z]{16}/g,
  /AIza[0-9A-Za-z-_]{30,}/g,
  /gh[pousr]_[A-Za-z0-9]{20,}/g,
  /sk-[A-Za-z0-9]{20,}/g,
  /sk-ant-[A-Za-z0-9-]{20,}/g,
  /xox[baprs]-[A-Za-z0-9-]{10,}/g,
  /\b(?:api[_-]?key|token|secret)\b\s*[:=]\s*["']?[A-Za-z0-9_-]{12,}/gi
]

const introMsg = (info: SessionInfo): Msg => ({
  role: 'system',
  text: '',
  kind: 'intro',
  info
})

const classifyPaste = (text: string): PendingPaste['kind'] => {
  const t = text.toLowerCase()
  const lines = text.split('\n')

  if (/error|warn|traceback|exception|stack|debug|\[\d{2}:\d{2}:\d{2}\]/.test(t)) {
    return 'log'
  }

  if (
    /```|function\s+\w+|class\s+\w+|import\s+.+from|const\s+\w+\s*=|def\s+\w+\(|<\w+/.test(text) ||
    lines.filter(line => /[{}()[\];<>]/.test(line)).length >= 3
  ) {
    return 'code'
  }

  return 'text'
}

const redactSecrets = (text: string) => {
  let output = text
  let redactions = 0

  for (const pattern of SECRET_PATTERNS) {
    output = output.replace(pattern, value => {
      redactions += 1

      if (value.includes(':') || value.includes('=')) {
        const [head] = value.split(/[:=]/)
        return `${head}: [REDACTED_SECRET]`
      }

      return '[REDACTED_SECRET]'
    })
  }

  return { redactions, text: output }
}

const tokenForPaste = (id: number) => `[[paste:${id}]]`

function StatusRule({
  cols,
  color,
  dimColor,
  statusColor,
  parts
}: {
  cols: number
  color: string
  dimColor: string
  statusColor: string
  parts: (string | false | undefined | null)[]
}) {
  const label = parts.filter(Boolean).join(' · ')
  const lead = String(parts[0] ?? '')
  const fill = Math.max(0, cols - label.length - 5)

  return (
    <Text color={color}>
      {'─ '}
      <Text color={dimColor}>
        <Text color={statusColor}>{parts[0]}</Text>
        {label.slice(lead.length)}
      </Text>
      {' ' + '─'.repeat(fill)}
    </Text>
  )
}

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

    return () => {
      stdout.off('resize', sync)
    }
  }, [stdout])

  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [messages, setMessages] = useState<Msg[]>([])
  const [historyItems, setHistoryItems] = useState<Msg[]>([])
  const [status, setStatus] = useState('summoning hermes…')
  const [sid, setSid] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(DEFAULT_THEME)
  const [info, setInfo] = useState<SessionInfo | null>(null)
  const [introCollapsed, setIntroCollapsed] = useState(false)
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
  const [thinkingText, setThinkingText] = useState('')
  const [statusBar, setStatusBar] = useState(true)
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [pastes, setPastes] = useState<PendingPaste[]>([])
  const [pasteReview, setPasteReview] = useState<{ largeIds: number[]; text: string } | null>(null)
  const [streaming, setStreaming] = useState('')
  const [catalog, setCatalog] = useState<SlashCatalog | null>(null)

  const activityIdRef = useRef(0)
  const buf = useRef('')
  const inflightPasteIdsRef = useRef<number[]>([])
  const interruptedRef = useRef(false)
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const lastEmptyAt = useRef(0)
  const lastStatusNoteRef = useRef('')
  const protocolWarnedRef = useRef(false)
  const pasteCounterRef = useRef(0)

  const { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue } =
    useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()

  const empty = !messages.length
  const blocked = !!(clarify || approval || pasteReview || picker || secret || sudo)

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

  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, blocked, gw)

  const appendMessage = useCallback((msg: Msg) => {
    setMessages(prev => [...prev, msg])
    setHistoryItems(prev => [...prev, msg])
  }, [])

  const appendHistory = useCallback((msg: Msg) => {
    setHistoryItems(prev => [...prev, msg])
  }, [])

  const sys = useCallback((text: string) => appendMessage({ role: 'system' as const, text }), [appendMessage])

  const pushActivity = useCallback((text: string, tone: ActivityItem['tone'] = 'info') => {
    setActivity(prev => {
      if (prev.at(-1)?.text === text && prev.at(-1)?.tone === tone) {
        return prev
      }

      activityIdRef.current += 1
      return [...prev, { id: activityIdRef.current, text, tone }].slice(-8)
    })
  }, [])

  const colsRef = useRef(cols)
  colsRef.current = cols

  const rpc = useCallback(
    (method: string, params: Record<string, unknown> = {}) =>
      gw.request(method, params).catch((e: Error) => {
        sys(`error: ${e.message}`)
      }),
    [gw, sys]
  )

  const newSession = useCallback(
    (msg?: string) =>
      rpc('session.create', { cols: colsRef.current }).then((r: any) => {
        if (!r) {
          return
        }

        setSid(r.session_id)
        setHistoryItems([])
        setMessages([])
        setPastes([])
        setActivity([])
        setUsage(ZERO)
        setStatus('ready')
        setIntroCollapsed(false)
        lastStatusNoteRef.current = ''
        protocolWarnedRef.current = false

        if (r.info) {
          setInfo(r.info)
          appendHistory(introMsg(r.info))
        } else {
          setInfo(null)
        }

        if (msg) {
          sys(msg)
        }
      }),
    [appendHistory, rpc, sys]
  )

  const idle = () => {
    setThinking(false)
    setTools([])
    setActivity([])
    setBusy(false)
    setClarify(null)
    setApproval(null)
    setPasteReview(null)
    setSudo(null)
    setSecret(null)
    setReasoning('')
    setThinkingText('')
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

      const resolved = text.replace(PASTE_TOKEN_RE, (_match, rawId: string) => {
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
          const clippedLines = lines.slice(0, EXCERPT_LINE_LIMIT)
          let excerpt = clippedLines.join('\n')

          if (excerpt.length > EXCERPT_CHAR_LIMIT) {
            excerpt = excerpt.slice(0, EXCERPT_CHAR_LIMIT).trimEnd() + '…'
          }

          const isTruncated = lines.length > EXCERPT_LINE_LIMIT || cleaned.text.length > excerpt.length
          const truncLabel = isTruncated ? `\n…[paste #${id} truncated]` : ''

          return `[paste #${id} excerpt]\n\`\`\`${lang}\n${excerpt}${truncLabel}\n\`\`\``
        }

        return `[paste #${id} attached · ${paste.lineCount} lines]\n\`\`\`${lang}\n${cleaned.text}\n\`\`\``
      })

      return {
        missingIds: [...missingIds],
        redactions,
        text: resolved,
        usedIds: [...usedIds]
      }
    },
    [pastes]
  )

  const handleTextPaste = useCallback(
    ({ cursor, text, value }: { cursor: number; text: string; value: string }) => {
      pasteCounterRef.current += 1
      const id = pasteCounterRef.current
      const charCount = text.length
      const lineCount = text.split('\n').length
      const mode: PasteMode =
        lineCount > SMALL_PASTE_LINE_LIMIT || charCount > SMALL_PASTE_CHAR_LIMIT ? 'attach' : 'excerpt'
      const token = tokenForPaste(id)
      const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
      const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
      const insert = `${lead}${token}${tail}`

      setPastes(prev =>
        [
          ...prev,
          {
            charCount,
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

      return {
        cursor: cursor + insert.length,
        value: value.slice(0, cursor) + insert + value.slice(cursor)
      }
    },
    [pushActivity]
  )

  const send = (text: string) => {
    const payload = resolvePasteTokens(text)

    if (payload.missingIds.length) {
      setStatus('missing paste token')
      pushActivity(`missing paste token(s): ${payload.missingIds.join(', ')}`, 'warn')
      return
    }

    if (payload.redactions > 0) {
      pushActivity(`redacted ${payload.redactions} secret-like value(s)`, 'warn')
    }

    inflightPasteIdsRef.current = payload.usedIds
    setLastUserMsg(text)
    setIntroCollapsed(true)
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

  const paste = () =>
    rpc('clipboard.paste', { session_id: sid }).then((r: any) =>
      pushActivity(r.attached ? `image #${r.count} attached` : r.message || 'no image in clipboard')
    )

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
      matches.map(match =>
        gw
          .request('shell.exec', { command: match[1]! })
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

      // Slash commands and shell — route immediately, no paste logic needed
      if (full.startsWith('/') && slashRef.current(full)) {
        clearInput()
        return
      }

      if (full.startsWith('!')) {
        clearInput()
        shellExec(full.slice(1).trim())
        return
      }

      // Paste token validation for non-command text
      const missing = resolvePasteTokens(full).missingIds

      if (missing.length) {
        setStatus('missing paste token')
        pushActivity(`missing paste token(s): ${missing.join(', ')}`, 'warn')
        return
      }

      const largeIds = listPasteIds(full).filter(id => {
        const paste = pastes.find(p => p.id === id)

        return !!paste && (paste.charCount >= LARGE_PASTE_CHAR_LIMIT || paste.lineCount >= LARGE_PASTE_LINE_LIMIT)
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
    [busy, enqueue, gw, listPasteIds, pastes, resolvePasteTokens, sid]
  )

  useInput((ch, key) => {
    if (blocked) {
      if (pasteReview) {
        if (key.return) {
          const text = pasteReview.text
          setPasteReview(null)
          dispatchSubmission(text, true)
        } else if (key.escape || (key.ctrl && ch === 'c')) {
          setPasteReview(null)
          setStatus('ready')
        }

        return
      }

      if (key.ctrl && ch === 'c') {
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
        const len = queueRef.current.length
        const idx = queueEditIdx === null ? 0 : (queueEditIdx + 1) % len
        setQueueEdit(idx)
        setHistoryIdx(null)
        setInput(queueRef.current[idx] ?? '')
      } else if (historyRef.current.length) {
        const hist = historyRef.current
        const idx = historyIdx === null ? hist.length - 1 : Math.max(0, historyIdx - 1)

        if (historyIdx === null) {
          historyDraftRef.current = input
        }

        setHistoryIdx(idx)
        setQueueEdit(null)
        setInput(hist[idx] ?? '')
      }

      return
    }

    if (key.downArrow && !inputBuf.length) {
      if (queueRef.current.length) {
        const len = queueRef.current.length
        const idx = queueEditIdx === null ? len - 1 : (queueEditIdx - 1 + len) % len
        setQueueEdit(idx)
        setHistoryIdx(null)
        setInput(queueRef.current[idx] ?? '')
      } else if (historyIdx !== null) {
        const hist = historyRef.current
        const next = historyIdx + 1

        if (next >= hist.length) {
          setHistoryIdx(null)
          setInput(historyDraftRef.current)
        } else {
          setHistoryIdx(next)
          setInput(hist[next] ?? '')
        }
      }

      return
    }

    if (key.ctrl && ch === 'c') {
      if (busy && sid) {
        interruptedRef.current = true
        gw.request('session.interrupt', { session_id: sid }).catch(() => {})

        const partial = (streaming || buf.current).trimStart()
        if (partial) {
          appendMessage({ role: 'assistant' as const, text: partial + '\n\n*[interrupted]*' })
        } else {
          sys('interrupted')
        }

        idle()
        setStatus('interrupted')
        setTimeout(() => setStatus('ready'), 1500)
      } else if (input || inputBuf.length) {
        clearIn()
      } else {
        die()
      }

      return
    }

    if (key.ctrl && ch === 'd') {
      die()
    }

    if (key.ctrl && ch === 'l') {
      setStatus('forging session…')
      newSession()

      return
    }

    if (key.ctrl && ch === 'v') {
      paste()

      return
    }

    if (key.ctrl && ch === 'g') {
      return openEditor()
    }

    if (key.escape) {
      clearIn()
    }
  })

  const onEvent = useCallback(
    (ev: GatewayEvent) => {
      const p = ev.payload as any

      switch (ev.type) {
        case 'gateway.ready':
          if (p?.skin) {
            setTheme(fromSkin(p.skin.colors ?? {}, p.skin.branding ?? {}, p.skin.banner_logo ?? '', p.skin.banner_hero ?? ''))
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

          setStatus('forging session…')
          newSession()

          break

        case 'session.info':
          setInfo(p as SessionInfo)

          break

        case 'thinking.delta':
          if (p?.text) {
            setThinkingText(prev => prev + p.text)
          }

          break

        case 'message.start':
          setThinking(true)
          setTurnKey(k => k + 1)
          setBusy(true)
          setReasoning('')
          setThinkingText('')

          break

        case 'status.update':
          if (p?.text) {
            setStatus(p.text)

            if (p.kind && p.kind !== 'status' && lastStatusNoteRef.current !== p.text) {
              lastStatusNoteRef.current = p.text
              pushActivity(
                p.text,
                p.kind === 'error' ? 'error' : p.kind === 'warn' || p.kind === 'approval' ? 'warn' : 'info'
              )
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

              if (idx >= 0) {
                return [...prev.slice(0, idx), { ...prev[idx]!, context: p.preview as string }, ...prev.slice(idx + 1)]
              }

              return prev
            })
          }

          break
        case 'tool.start': {
          const ctx = (p.context as string) || ''
          setTools(prev => [...prev, { id: p.tool_id, name: p.name, context: ctx }])

          break
        }

        case 'tool.complete': {
          const mark = p.error ? '✗' : '✓'
          setTools(prev => {
            const done = prev.find(t => t.id === p.tool_id)
            const label = TOOL_VERBS[done?.name ?? p.name] ?? done?.name ?? p.name
            const ctx = (p.error as string) || done?.context || ''
            pushActivity(`${label}${ctx ? ': ' + ctx : ''} ${mark}`, p.error ? 'error' : 'info')

            return prev.filter(t => t.id !== p.tool_id)
          })
        }

          break

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
          sys(`[bg ${p.task_id}] ${p.text}`)

          break

        case 'btw.complete':
          sys(`[btw] ${p.text}`)

          break

        case 'message.delta':
          if (!p?.text || interruptedRef.current) {
            break
          }

          buf.current += p.rendered ?? p.text
          setStreaming(buf.current.trimStart())

          break
        case 'message.complete': {
          const wasInterrupted = interruptedRef.current
          idle()
          setStreaming('')

          if (inflightPasteIdsRef.current.length) {
            setPastes(prev => prev.filter(paste => !inflightPasteIdsRef.current.includes(paste.id)))
            inflightPasteIdsRef.current = []
          }

          if (!wasInterrupted) {
            appendMessage({ role: 'assistant' as const, text: (p?.rendered ?? p?.text ?? buf.current).trimStart() })
          }

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
          setStatus('ready')

          break
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [appendMessage, dequeue, newSession, pushActivity, send, sys]
  )

  const onExit = useCallback(() => {
    setStatus('gateway exited')
    exit()
  }, [exit])

  useEffect(() => {
    gw.on('event', onEvent)
    gw.on('exit', onExit)

    return () => {
      gw.off('event', onEvent)
      gw.off('exit', onExit)
    }
  }, [gw, onEvent, onExit])

  const slash = useCallback(
    (cmd: string): boolean => {
      const [name, ...rest] = cmd.slice(1).split(/\s+/)
      const arg = rest.join(' ')

      switch (name) {
        case 'help': {
          const rows = catalog?.pairs ?? []
          const cap = 52
          const lines = rows.slice(0, cap).map(([c, d]) => `    ${c.padEnd(16)} ${d}`)
          sys(
            [
              '  Commands:',
              ...lines,
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

        case 'compact':
          setCompact(c => (arg ? true : !c))
          sys(arg ? `compact on, focus: ${arg}` : `compact ${compact ? 'off' : 'on'}`)

          return true

        case 'resume':
          setPicker(true)

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
              sys(
                pastes
                  .map(
                    p =>
                      `#${p.id} ${p.mode} · ${p.lineCount}L · ${p.kind} · ${compactPreview(p.text, 60) || '(empty)'}`
                  )
                  .join('\n')
              )
            }

            return true
          }

          if (arg === 'clear') {
            setPastes([])
            setInput(v => v.replace(PASTE_TOKEN_RE, '').replace(/\s{2,}/g, ' ').trim())
            setInputBuf(prev =>
              prev
                .map(line => line.replace(PASTE_TOKEN_RE, '').replace(/\s{2,}/g, ' ').trim())
                .filter(Boolean)
            )
            pushActivity('cleared paste shelf')

            return true
          }

          if (arg.startsWith('drop ')) {
            const id = parseInt(arg.split(/\s+/)[1] ?? '-1', 10)

            if (!id || !pastes.some(p => p.id === id)) {
              sys('usage: /paste drop <id>')
              return true
            }

            setPastes(prev => prev.filter(p => p.id !== id))
            setInput(v => v.replace(new RegExp(`\\s*\\[\\[paste:${id}\\]\\]\\s*`, 'g'), ' ').replace(/\s{2,}/g, ' ').trim())
            setInputBuf(prev =>
              prev
                .map(line =>
                  line.replace(new RegExp(`\\s*\\[\\[paste:${id}\\]\\]\\s*`, 'g'), ' ').replace(/\s{2,}/g, ' ').trim()
                )
                .filter(Boolean)
            )
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
          const limit = Math.min(80, Math.max(1, parseInt(arg, 10) || 20))
          sys(gw.getLogTail(limit) || 'no gateway logs')

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
          rpc('prompt.background', { session_id: sid, text: arg }).then((r: any) => sys(`bg ${r.task_id} started`))
          return true

        case 'btw':
          if (!arg) {
            sys('/btw <question>')
            return true
          }
          rpc('prompt.btw', { session_id: sid, text: arg }).then(() => sys('btw running…'))
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
          rpc('config.set', { key: 'personality', value: arg }).then((r: any) =>
            sys(`personality: ${r.value || 'default'}`)
          )
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
            if (r) { setUsage({ input: r.input ?? 0, output: r.output ?? 0, total: r.total ?? 0, calls: r.calls ?? 0 }) }
            sys(`${fmtK(r?.input ?? 0)} in · ${fmtK(r?.output ?? 0)} out · ${fmtK(r?.total ?? 0)} total · ${r?.calls ?? 0} calls`)
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
          if (arg === 'on' || arg === 'off') {
            rpc('voice.toggle', { action: arg }).then((r: any) => sys(`voice ${r.enabled ? 'on' : 'off'}`))
          } else {
            rpc('voice.toggle', { action: 'status' }).then((r: any) => sys(`voice: ${r.enabled ? 'on' : 'off'}`))
          }
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
              if (!r.checkpoints?.length) { return sys('no checkpoints') }
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
          const action = arg || 'status'
          const [act, ...bArgs] = action.split(/\s+/)
          rpc('browser.manage', { action: act, ...(bArgs[0] ? { url: bArgs[0] } : {}) }).then((r: any) =>
            sys(r.connected ? `browser: ${r.url}` : 'browser: disconnected')
          )
          return true
        }

        case 'plugins':
          rpc('plugins.list', {}).then((r: any) => {
            if (!r.plugins?.length) { return sys('no plugins') }
            sys(r.plugins.map((p: any) => `  ${p.name} v${p.version}${p.enabled ? '' : ' (disabled)'}`).join('\n'))
          })
          return true

        default:
          gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then((r: any) => {
              if (r?.output) {
                sys(r.output)
              } else {
                sys(`/${name}: no output`)
              }
            })
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [catalog, compact, gw, lastUserMsg, messages, newSession, pastes, pushActivity, rpc, send, sid, statusBar, sys]
  )

  slashRef.current = slash

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
            appendMessage({ role: 'assistant' as const, text: partial + '\n\n*[interrupted]*' })
          } else {
            sys('interrupted')
          }

          idle()
          setStatus('interrupted')
          setTimeout(() => setStatus('ready'), 1500)

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

      const full = [...inputBuf, value].join('\n')
      dispatchSubmission(full)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dequeue, dispatchSubmission, inputBuf, sid]
  )

  const statusColor =
    status === 'ready'
      ? theme.color.ok
      : status.startsWith('error')
        ? theme.color.error
        : status === 'interrupted'
          ? theme.color.warn
          : theme.color.dim

  return (
    <Box flexDirection="column">
      <Static items={historyItems}>
        {(m, i) => (
          <Box flexDirection="column" key={i} paddingX={1}>
            {m.kind === 'intro' && m.info ? (
              <Box flexDirection="column" paddingTop={1}>
                {introCollapsed ? (
                  <Text color={theme.color.dim}>
                    {theme.brand.icon} {theme.brand.name} · {m.info.model.split('/').pop()}
                  </Text>
                ) : (
                  <>
                    <Banner t={theme} />
                    <SessionPanel info={m.info} sid={sid} t={theme} />
                  </>
                )}
              </Box>
            ) : (
              <MessageLine cols={cols} compact={compact} msg={m} t={theme} />
            )}
          </Box>
        )}
      </Static>

      <Box flexDirection="column" paddingX={1}>
        {streaming && (
          <Box flexDirection="column">
            <MessageLine cols={cols} compact={compact} msg={{ role: 'assistant', text: streaming }} t={theme} />
          </Box>
        )}

        {(thinking || tools.length > 0) && (!streaming || tools.length > 0) && <Thinking key={turnKey} reasoning={reasoning} t={theme} tools={tools} />}
        <ActivityLane items={activity} t={theme} />

        {pasteReview && (
          <Box borderColor={theme.color.warn} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
            <Text bold color={theme.color.warn}>
              Review large paste before send
            </Text>
            <Text color={theme.color.dim}>pastes: {pasteReview.largeIds.map(id => `#${id}`).join(', ')}</Text>
            <Text color={theme.color.dim}>Enter to send · Esc/Ctrl+C to cancel</Text>
          </Box>
        )}

        {clarify && (
          <Box borderColor={theme.color.bronze} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
            <ClarifyPrompt
              onAnswer={answer => {
                gw.request('clarify.respond', { answer, request_id: clarify.requestId }).catch(() => {})
                appendMessage({ role: 'user', text: answer })
                setClarify(null)
              }}
              req={clarify}
              t={theme}
            />
          </Box>
        )}

        {approval && (
          <Box borderColor={theme.color.bronze} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
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
          </Box>
        )}

        {sudo && (
          <Box borderColor={theme.color.bronze} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
            <MaskedPrompt
              icon="🔐"
              label="sudo password required"
              onSubmit={password => {
                gw.request('sudo.respond', { request_id: sudo.requestId, password }).catch(() => {})
                setSudo(null)
                setStatus('running…')
              }}
              t={theme}
            />
          </Box>
        )}

        {secret && (
          <Box borderColor={theme.color.bronze} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
            <MaskedPrompt
              icon="🔑"
              label={secret.prompt}
              onSubmit={value => {
                gw.request('secret.respond', { request_id: secret.requestId, value }).catch(() => {})
                setSecret(null)
                setStatus('running…')
              }}
              sub={`for ${secret.envVar}`}
              t={theme}
            />
          </Box>
        )}

        {picker && (
          <Box borderColor={theme.color.bronze} borderStyle="round" flexDirection="column" marginTop={1} paddingX={1}>
            <SessionPicker
              gw={gw}
              onCancel={() => setPicker(false)}
              onSelect={id => {
                setPicker(false)
                setStatus('resuming…')
                gw.request('session.resume', { cols, session_id: id })
                  .then((r: any) => {
                    setSid(r.session_id)
                  setHistoryItems([])
                    setMessages([])
                    setInfo(r.info ?? null)
                    setPastes([])
                    setActivity([])
                    setIntroCollapsed(false)

                    if (r.info) {
                      appendHistory(introMsg(r.info))
                    }

                    setUsage(ZERO)
                    lastStatusNoteRef.current = ''
                    protocolWarnedRef.current = false
                    sys(`resumed session (${r.message_count} messages)`)
                    setStatus('ready')
                  })
                  .catch((e: Error) => {
                    sys(`error: ${e.message}`)
                    setStatus('ready')
                  })
              }}
              t={theme}
            />
          </Box>
        )}

        <QueuedMessages cols={cols} queued={queuedDisplay} queueEditIdx={queueEditIdx} t={theme} />

        <Text> </Text>

        {statusBar && (
          <StatusRule
            color={theme.color.bronze}
            cols={cols}
            dimColor={theme.color.dim}
            parts={[status, sid, info?.model?.split('/').pop(), usage.total > 0 && `${fmtK(usage.total)} tok`]}
            statusColor={statusColor}
          />
        )}

        {!blocked && (
          <Box>
            <Box width={3}>
              <Text bold color={theme.color.gold}>
                {inputBuf.length ? '… ' : `${theme.brand.prompt} `}
              </Text>
            </Box>

            <TextInput
              onChange={setInput}
              onPaste={handleTextPaste}
              onSubmit={submit}
              placeholder={
                empty
                  ? PLACEHOLDER
                  : busy
                    ? 'Ctrl+C to interrupt…'
                    : inputBuf.length
                      ? 'continue (or Enter to send)'
                      : ''
              }
              value={input}
            />
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
