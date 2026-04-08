import { spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

import { Box, Static, Text, useApp, useInput, useStdout } from 'ink'
import { useCallback, useEffect, useRef, useState } from 'react'

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
import { fmtK, hasInterpolation, pick } from './lib/text.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'
import type {
  ActiveTool,
  ApprovalReq,
  ClarifyReq,
  Msg,
  SecretReq,
  SessionInfo,
  SlashCatalog,
  SudoReq,
  Usage
} from './types.js'

const PLACEHOLDER = pick(PLACEHOLDERS)
const PASTE_REF_RE = /\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]/g

const introMsg = (info: SessionInfo): Msg => ({
  role: 'system',
  text: '',
  kind: 'intro',
  info
})

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
  const [thinking, setThinking] = useState(false)
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
  const [streaming, setStreaming] = useState('')
  const [catalog, setCatalog] = useState<SlashCatalog | null>(null)

  const buf = useRef('')
  const interruptedRef = useRef(false)
  const lastEmptyAt = useRef(0)
  const lastStatusNoteRef = useRef('')
  const protocolWarnedRef = useRef(false)
  const pasteCounterRef = useRef(0)

  const { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue } =
    useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()

  const empty = !messages.length
  const blocked = !!(clarify || approval || sudo || secret || picker)

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
        setMessages([])
        setUsage(ZERO)
        setStatus('ready')
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
    setBusy(false)
    setClarify(null)
    setApproval(null)
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
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }

  const expandPastes = (text: string) =>
    text.replace(PASTE_REF_RE, (m, path) => {
      try {
        return readFileSync(path, 'utf8')
      } catch {
        return m
      }
    })

  const collapsePaste = (text: string) => {
    pasteCounterRef.current += 1
    const lineCount = text.split('\n').length
    const pasteDir = join(process.env.HERMES_HOME ?? join(homedir(), '.hermes'), 'pastes')
    mkdirSync(pasteDir, { recursive: true })

    const pasteFile = join(
      pasteDir,
      `paste_${pasteCounterRef.current}_${new Date().toTimeString().slice(0, 8).replace(/:/g, '')}.txt`
    )

    writeFileSync(pasteFile, text, 'utf8')

    return `[Pasted text #${pasteCounterRef.current}: ${lineCount} lines → ${pasteFile}]`
  }

  const send = (text: string) => {
    setLastUserMsg(text)
    appendMessage({ role: 'user', text })
    setBusy(true)
    buf.current = ''
    interruptedRef.current = false
    gw.request('prompt.submit', { session_id: sid, text: expandPastes(text) }).catch((e: Error) => {
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
      sys(r.attached ? `📎 image #${r.count} attached` : r.message || 'no image in clipboard')
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

  useInput((ch, key) => {
    if (blocked) {
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

        if (buf.current.trim()) {
          appendMessage({ role: 'assistant' as const, text: buf.current.trimStart() })
        }

        idle()
        setStatus('interrupted')
        sys('interrupted by user')
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
          setBusy(true)
          setReasoning('')
          setThinkingText('')

          break

        case 'status.update':
          if (p?.text) {
            setStatus(p.text)

            if (p.kind && p.kind !== 'status' && lastStatusNoteRef.current !== p.text) {
              lastStatusNoteRef.current = p.text
              sys(p.text)
            }
          }

          break

        case 'gateway.protocol_error':
          setStatus('protocol warning')

          if (!protocolWarnedRef.current) {
            protocolWarnedRef.current = true
            sys('protocol noise detected · /logs to inspect')
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
            appendMessage({ role: 'tool', text: `${label}${ctx ? ': ' + ctx : ''} ${mark}` })

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
          idle()
          setStreaming('')
          appendMessage({ role: 'assistant' as const, text: (p?.rendered ?? p?.text ?? buf.current).trimStart() })
          buf.current = ''
          setStatus('ready')

          if (p?.usage) {
            setUsage(p.usage)
          }

          if (p?.status === 'interrupted') {
            sys('response interrupted')
          }

          if (queueEditRef.current !== null) {
            break
          }

          const next = dequeue()

          if (next) {
            setLastUserMsg(next)
            appendMessage({ role: 'user' as const, text: next })
            setBusy(true)
            buf.current = ''
            gw.request('prompt.submit', { session_id: ev.session_id, text: next }).catch((e: Error) => {
              sys(`error: ${e.message}`)
              setStatus('ready')
              setBusy(false)
            })
          }

          break
        }

        case 'error':
          sys(`error: ${p?.message}`)
          idle()
          setStatus('ready')

          break
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [appendMessage, gw, sys, newSession]
  )

  useEffect(() => {
    gw.on('event', onEvent)
    gw.on('exit', () => {
      setStatus('gateway exited')
      exit()
    })

    return () => {
      gw.off('event', onEvent)
    }
  }, [exit, gw, onEvent])

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
          paste()

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
    [catalog, compact, gw, info, lastUserMsg, messages, newSession, rpc, send, sid, status, sys, usage, statusBar]
  )

  const submit = useCallback(
    (value: string) => {
      if (!value.trim() && !inputBuf.length) {
        const now = Date.now()
        const dbl = now - lastEmptyAt.current < 450
        lastEmptyAt.current = now

        if (dbl && queueRef.current.length) {
          if (busy && sid) {
            gw.request('session.interrupt', { session_id: sid }).catch(() => {})
            setStatus('interrupting…')

            return
          }

          const next = dequeue()

          if (next && sid) {
            setQueueEdit(null)
            send(next)
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
      setInputBuf([])
      setInput('')
      setHistoryIdx(null)
      historyDraftRef.current = ''

      if (!full.trim() || !sid) {
        return
      }

      const editIdx = queueEditRef.current

      if (editIdx !== null && !full.startsWith('/') && !full.startsWith('!')) {
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

          return
        }

        return
      }

      if (editIdx !== null) {
        setQueueEdit(null)
      }

      pushHistory(full)

      if (busy && !full.startsWith('/') && !full.startsWith('!')) {
        if (hasInterpolation(full)) {
          interpolate(full, enqueue)

          return
        }

        enqueue(full)

        return
      }

      if (full.startsWith('!')) {
        shellExec(full.slice(1).trim())

        return
      }

      if (full.startsWith('/') && slash(full)) {
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
    [busy, gw, inputBuf, sid, slash, sys]
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
                <Banner t={theme} />
                <SessionPanel info={m.info} sid={sid} t={theme} />
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

        {(thinking || tools.length > 0) && !streaming && <Thinking reasoning={reasoning} t={theme} tools={tools} />}

        {clarify && (
          <ClarifyPrompt
            onAnswer={answer => {
              gw.request('clarify.respond', { answer, request_id: clarify.requestId }).catch(() => {})
              appendMessage({ role: 'user', text: answer })
              setClarify(null)
            }}
            req={clarify}
            t={theme}
          />
        )}

        {approval && (
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
        )}

        {sudo && (
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
        )}

        {secret && (
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
        )}

        {picker && (
          <SessionPicker
            gw={gw}
            onCancel={() => setPicker(false)}
            onSelect={id => {
              setPicker(false)
              setStatus('resuming…')
              gw.request('session.resume', { session_id: id, cols })
                .then((r: any) => {
                  setSid(r.session_id)
                  setMessages([])
                  setInfo(r.info ?? null)

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
        )}

        <QueuedMessages cols={cols} queued={queuedDisplay} queueEditIdx={queueEditIdx} t={theme} />

        <Text> </Text>

        <StatusRule
          color={theme.color.bronze}
          cols={cols}
          dimColor={theme.color.dim}
          parts={[status, sid, info?.model?.split('/').pop(), usage.total > 0 && `${fmtK(usage.total)} tok`]}
          statusColor={statusColor}
        />

        {!blocked && (
          <Box>
            <Box width={3}>
              <Text bold color={theme.color.gold}>
                {inputBuf.length ? '… ' : `${theme.brand.prompt} `}
              </Text>
            </Box>

            <TextInput
              onChange={setInput}
              onLargePaste={collapsePaste}
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
