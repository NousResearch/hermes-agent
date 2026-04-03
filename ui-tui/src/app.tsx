import { Box, Text, useApp, useInput, useStdout } from 'ink'
import TextInput from 'ink-text-input'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { AltScreen } from './altScreen.js'
import { Banner, SessionPanel } from './components/branding.js'
import { CommandPalette } from './components/commandPalette.js'
import { MessageLine } from './components/messageLine.js'
import { ApprovalPrompt, ClarifyPrompt } from './components/prompts.js'
import { QueuedMessages } from './components/queuedMessages.js'
import { Thinking } from './components/thinking.js'
import { COMMANDS, HOTKEYS, INTERPOLATION_RE, MAX_CTX, PLACEHOLDERS, TOOL_VERBS, ZERO } from './constants.js'
import type { GatewayClient } from './gatewayClient.js'
import { type GatewayEvent } from './gatewayClient.js'
import { upsert } from './lib/messages.js'
import { estimateRows, flat, fmtK, hasInterpolation, pick, userDisplay } from './lib/text.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'
import type { ActiveTool, ApprovalReq, ClarifyReq, Msg, SessionInfo, Usage } from './types.js'

const PLACEHOLDER = pick(PLACEHOLDERS)

export function App({ gw }: { gw: GatewayClient }) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const cols = stdout?.columns ?? 80
  const rows = stdout?.rows ?? 24

  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [messages, setMessages] = useState<Msg[]>([])
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
  const [reasoning, setReasoning] = useState('')
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)
  const [historyIdx, setHistoryIdx] = useState<number | null>(null)
  const [scrollOffset, setScrollOffset] = useState(0)
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])

  const buf = useRef('')
  const stickyRef = useRef(true)
  const queueRef = useRef<string[]>([])
  const historyRef = useRef<string[]>([])
  const historyDraftRef = useRef('')
  const queueEditRef = useRef<number | null>(null)
  const lastEmptyAt = useRef(0)

  const empty = !messages.length
  const blocked = !!(clarify || approval)

  const syncQueue = () => setQueuedDisplay([...queueRef.current])

  const setQueueEdit = (idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }

  const enqueue = (text: string) => {
    queueRef.current.push(text)
    syncQueue()
  }

  const dequeue = () => {
    const [head, ...rest] = queueRef.current
    queueRef.current = rest
    syncQueue()

    return head
  }

  const replaceQ = (i: number, text: string) => {
    queueRef.current[i] = text
    syncQueue()
  }

  const pushHistory = (text: string) => {
    const trimmed = text.trim()

    if (trimmed && historyRef.current.at(-1) !== trimmed) {
      historyRef.current.push(trimmed)
    }
  }

  useEffect(() => {
    if (stickyRef.current) {
      setScrollOffset(0)
    }
  }, [messages.length])

  const msgBudget = Math.max(3, rows - 2 - (empty ? 0 : 2) - (thinking ? 2 : 0) - 2)

  const viewport = useMemo(() => {
    if (!messages.length) {
      return { above: 0, end: 0, start: 0 }
    }

    const end = Math.max(0, messages.length - scrollOffset)
    const width = Math.max(20, cols - 5)

    let budget = msgBudget
    let start = end

    for (let i = end - 1; i >= 0 && budget > 0; i--) {
      const msg = messages[i]!
      const margin = msg.role === 'user' && i > 0 && messages[i - 1]?.role !== 'user' ? 1 : 0
      budget -= margin + estimateRows(msg.role === 'user' ? userDisplay(msg.text) : msg.text, width)

      if (budget >= 0) {
        start = i
      }
    }

    if (start === end && end > 0) {
      start = end - 1
    }

    return { above: start, end, start }
  }, [cols, messages, msgBudget, scrollOffset])

  const sys = useCallback((text: string) => setMessages(prev => [...prev, { role: 'system' as const, text }]), [])

  const idle = () => {
    setThinking(false)
    setTools([])
    setBusy(false)
    setClarify(null)
    setApproval(null)
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

  const scrollBot = () => {
    setScrollOffset(0)
    stickyRef.current = true
  }

  const scrollUp = (n: number) => {
    setScrollOffset(prev => Math.min(Math.max(0, messages.length - 1), prev + n))
    stickyRef.current = false
  }

  const scrollDown = (n: number) => {
    setScrollOffset(prev => {
      const v = Math.max(0, prev - n)

      if (!v) {
        stickyRef.current = true
      }

      return v
    })
  }

  const send = (text: string) => {
    setLastUserMsg(text)
    setMessages(prev => [...prev, { role: 'user', text }])
    scrollBot()
    setStatus('thinking…')
    setBusy(true)
    buf.current = ''
    gw.request('prompt.submit', { session_id: sid, text }).catch((e: Error) => {
      sys(`error: ${e.message}`)
      setStatus('ready')
      setBusy(false)
    })
  }

  const shellExec = (cmd: string) => {
    setMessages(prev => [...prev, { role: 'user', text: `!${cmd}` }])
    setBusy(true)
    setStatus('running…')
    gw.request('shell.exec', { command: cmd })
      .then((r: any) => {
        const out = [r.stdout, r.stderr].filter(Boolean).join('\n').trim()
        sys(out || `exit ${r.code}`)

        if (r.code !== 0 && out) {
          sys(`exit ${r.code}`)
        }
      })
      .catch((e: Error) => sys(`error: ${e.message}`))
      .finally(() => {
        setStatus('ready')
        setBusy(false)
      })
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
      if (key.ctrl && ch === 'c' && approval) {
        gw.request('approval.respond', { choice: 'deny', session_id: sid }).catch(() => {})
        setApproval(null)
        sys('denied')
      }

      return
    }

    if (key.pageUp) {
      scrollUp(5)

      return
    }

    if (key.pageDown) {
      scrollDown(5)

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
        gw.request('session.interrupt', { session_id: sid }).catch(() => {})
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
      setMessages([])
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
            setTheme(fromSkin(p.skin.colors ?? {}, p.skin.branding ?? {}))
          }

          setStatus('forging session…')
          gw.request('session.create')
            .then((r: any) => {
              setSid(r.session_id)
              setStatus('ready')
            })
            .catch((e: Error) => setStatus(`error: ${e.message}`))

          break

        case 'session.info':
          setInfo(p as SessionInfo)

          break

        case 'thinking.delta':
          break

        case 'message.start':
          setThinking(true)
          setBusy(true)
          setReasoning('')
          setStatus('thinking…')

          break

        case 'status.update':
          if (p?.text) {
            setStatus(p.text)
          }

          break

        case 'reasoning.delta':
          if (p?.text) {
            setReasoning(prev => prev + p.text)
          }

          break

        case 'tool.generating':
          if (p?.name) {
            setStatus(`preparing ${p.name}…`)
          }

          break

        case 'tool.progress':
          if (p?.preview) {
            setMessages(prev =>
              prev.at(-1)?.role === 'tool'
                ? [...prev.slice(0, -1), { role: 'tool' as const, text: `${p.name}: ${p.preview}` }]
                : [...prev, { role: 'tool' as const, text: `${p.name}: ${p.preview}` }]
            )
          }

          break

        case 'tool.start':
          setTools(prev => [...prev, { id: p.tool_id, name: p.name }])
          setStatus(`running ${p.name}…`)
          setMessages(prev => [...prev, { role: 'tool', text: `${TOOL_VERBS[p.name] ?? p.name}…` }])

          break

        case 'tool.complete':
          setTools(prev => prev.filter(t => t.id !== p.tool_id))

          break

        case 'clarify.request':
          setClarify({ choices: p.choices, question: p.question, requestId: p.request_id })
          setStatus('waiting for input…')

          break

        case 'approval.request':
          setApproval({ command: p.command, description: p.description })
          setStatus('approval needed')

          break

        case 'message.delta':
          if (!p?.text) {
            break
          }

          buf.current += p.text
          setThinking(false)
          setTools([])
          setReasoning('')
          setMessages(prev => upsert(prev, 'assistant', buf.current.trimStart()))

          break
        case 'message.complete': {
          idle()
          setMessages(prev => upsert(prev, 'assistant', (p?.text ?? buf.current).trimStart()))
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
            setMessages(prev => [...prev, { role: 'user' as const, text: next }])
            setStatus('thinking…')
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
    [gw, sys]
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
        case 'help':
          sys(
            [
              '  Commands:',
              ...COMMANDS.map(([c, d]) => `    ${c.padEnd(12)} ${d}`),
              '',
              '  Hotkeys:',
              ...HOTKEYS.map(([k, d]) => `    ${k.padEnd(12)} ${d}`)
            ].join('\n')
          )

          return true

        case 'clear':
          setMessages([])

          return true

        case 'quit': // falls through

        case 'exit':
          die()

          return true

        case 'new':
          setStatus('forging session…')
          gw.request('session.create')
            .then((r: any) => {
              setSid(r.session_id)
              setMessages([])
              setUsage(ZERO)
              setStatus('ready')
              sys('new session started')
            })
            .catch((e: Error) => setStatus(`error: ${e.message}`))

          return true

        case 'undo':
          if (!sid) {
            return true
          }

          gw.request('session.undo', { session_id: sid })
            .then((r: any) => {
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
            .catch((e: Error) => sys(`error: ${e.message}`))

          return true

        case 'retry':
          if (!lastUserMsg) {
            sys('nothing to retry')

            return true
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

        case 'compact':
          setCompact(c => (arg ? true : !c))
          sys(arg ? `compact on, focus: ${arg}` : `compact ${compact ? 'off' : 'on'}`)

          return true

        case 'compress':
          if (!sid) {
            return true
          }

          gw.request('session.compress', { session_id: sid })
            .then((r: any) => {
              sys('context compressed')

              if (r.usage) {
                setUsage(r.usage)
              }
            })
            .catch((e: Error) => sys(`error: ${e.message}`))

          return true

        case 'cost': // falls through

        case 'usage':
          sys(
            `in: ${fmtK(usage.input)}  out: ${fmtK(usage.output)}  total: ${fmtK(usage.total)}  calls: ${usage.calls}`
          )

          return true
        case 'copy': {
          const all = messages.filter(m => m.role === 'assistant')
          const target = all[arg ? Math.min(parseInt(arg), all.length) - 1 : all.length - 1]

          if (!target) {
            sys('nothing to copy')

            return true
          }

          process.stdout.write(`\x1b]52;c;${Buffer.from(target.text).toString('base64')}\x07`)
          sys('copied to clipboard')

          return true
        }

        case 'context': {
          const pct = Math.min(100, Math.round((usage.total / MAX_CTX) * 100))
          const bar = Math.round((pct / 100) * 30)
          const icon = pct < 50 ? '✓' : pct < 80 ? '⚠' : '✗'
          sys(
            `context: ${fmtK(usage.total)} / ${fmtK(MAX_CTX)} (${pct}%)\n[${'█'.repeat(bar)}${'░'.repeat(30 - bar)}] ${icon}`
          )

          return true
        }

        case 'config':
          sys(
            `model: ${info?.model ?? '?'}  session: ${sid ?? 'none'}  compact: ${compact}\ntools: ${flat(info?.tools ?? {}).length}  skills: ${flat(info?.skills ?? {}).length}`
          )

          return true

        case 'status':
          sys(
            `session: ${sid ?? 'none'}  status: ${status}  tokens: ${fmtK(usage.input)}↑ ${fmtK(usage.output)}↓ (${usage.calls} calls)`
          )

          return true

        case 'skills':
          if (!info?.skills || !Object.keys(info.skills).length) {
            sys('no skills loaded')

            return true
          }

          sys(
            Object.entries(info.skills)
              .map(([k, vs]) => `${k}: ${vs.join(', ')}`)
              .join('\n')
          )

          return true

        case 'model':
          if (!arg) {
            sys('usage: /model <name>')

            return true
          }

          gw.request('config.set', { key: 'model', value: arg })
            .then(() => sys(`model → ${arg}`))
            .catch((e: Error) => sys(`error: ${e.message}`))

          return true

        case 'skin':
          if (!arg) {
            sys('usage: /skin <name>')

            return true
          }

          gw.request('config.set', { key: 'skin', value: arg })
            .then(() => sys(`skin → ${arg} (restart to apply)`))
            .catch((e: Error) => sys(`error: ${e.message}`))

          return true

        default:
          return false
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [compact, gw, info, lastUserMsg, messages, sid, status, sys, usage]
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
        setQueueEdit(null)

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
    <AltScreen>
      <Box flexDirection="column" flexGrow={1} padding={1}>
        {empty ? (
          <>
            <Banner t={theme} />
            {info && <SessionPanel info={info} t={theme} />}
            {!sid ? (
              <Text color={theme.color.dim}>⚕ {status}</Text>
            ) : (
              <Text color={theme.color.dim}>
                type <Text color={theme.color.amber}>/</Text> for commands
                {' · '}
                <Text color={theme.color.amber}>!</Text> for shell
                {' · '}
                <Text color={theme.color.amber}>Ctrl+C</Text> to interrupt
              </Text>
            )}
          </>
        ) : (
          <Box marginBottom={1}>
            <Text bold color={theme.color.gold}>
              {theme.brand.icon}{' '}
            </Text>
            <Text bold color={theme.color.amber}>
              {theme.brand.name}
            </Text>
            <Text color={theme.color.dim}>
              {info?.model ? ` · ${info.model.split('/').pop()}` : ''}
              {' · '}
              <Text color={statusColor}>{status}</Text>
              {busy && ' · Ctrl+C to stop'}
            </Text>
            {usage.total > 0 && (
              <Text color={theme.color.dim}>
                {' · '}
                {fmtK(usage.input)}↑ {fmtK(usage.output)}↓ ({usage.calls} calls)
              </Text>
            )}
          </Box>
        )}

        <Box flexDirection="column" flexGrow={1} overflow="hidden">
          {viewport.above > 0 && (
            <Text color={theme.color.dim} dimColor>
              ↑ {viewport.above} above · PgUp/PgDn to scroll
            </Text>
          )}

          {messages.slice(viewport.start, viewport.end).map((m, i) => {
            const ri = viewport.start + i

            return (
              <Box
                flexDirection="column"
                key={ri}
                marginTop={m.role === 'user' && ri > 0 && messages[ri - 1]!.role !== 'user' ? 1 : 0}
              >
                <MessageLine compact={compact} msg={m} t={theme} />
              </Box>
            )
          })}

          {scrollOffset > 0 && (
            <Text color={theme.color.dim} dimColor>
              ↓ {scrollOffset} below · PgDn or Enter to return
            </Text>
          )}

          {thinking && <Thinking reasoning={reasoning} t={theme} tools={tools} />}
        </Box>

        {clarify && (
          <ClarifyPrompt
            onAnswer={answer => {
              gw.request('clarify.respond', { answer, request_id: clarify.requestId }).catch(() => {})
              setMessages(prev => [...prev, { role: 'user', text: answer }])
              setClarify(null)
              setStatus('thinking…')
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

        {!blocked && input.startsWith('/') && <CommandPalette filter={input} t={theme} />}

        <QueuedMessages cols={cols} queued={queuedDisplay} queueEditIdx={queueEditIdx} t={theme} />

        <Text color={theme.color.bronze}>{'─'.repeat(cols - 2)}</Text>

        {!blocked && (
          <Box>
            <Box width={3}>
              <Text bold color={theme.color.gold}>
                {inputBuf.length ? '… ' : `${theme.brand.prompt} `}
              </Text>
            </Box>
            <TextInput
              onChange={setInput}
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
      </Box>
    </AltScreen>
  )
}
