import { Box, Text, useApp, useInput, useStdout } from 'ink'
import TextInput from 'ink-text-input'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { AltScreen } from './altScreen.js'
import { Banner, SessionPanel } from './components/branding.js'
import { CommandPalette } from './components/commandPalette.js'
import { MaskedPrompt } from './components/maskedPrompt.js'
import { MessageLine } from './components/messageLine.js'
import { ApprovalPrompt, ClarifyPrompt } from './components/prompts.js'
import { QueuedMessages } from './components/queuedMessages.js'
import { SessionPicker } from './components/sessionPicker.js'
import { Thinking } from './components/thinking.js'
import { HOTKEYS, INTERPOLATION_RE, MAX_CTX, PLACEHOLDERS, TOOL_VERBS, ZERO } from './constants.js'
import { type GatewayClient, type GatewayEvent } from './gatewayClient.js'
import * as inputHistory from './lib/history.js'
import { upsert } from './lib/messages.js'
import { writeOsc52Clipboard } from './lib/osc52.js'
import { paletteForLine, tabAdvance } from './lib/slash.js'
import { estimateRows, flat, fmtK, hasInterpolation, pick, userDisplay } from './lib/text.js'
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
  const [sudo, setSudo] = useState<SudoReq | null>(null)
  const [secret, setSecret] = useState<SecretReq | null>(null)
  const [picker, setPicker] = useState(false)
  const [reasoning, setReasoning] = useState('')
  const [thinkingText, setThinkingText] = useState('')
  const [statusBar, setStatusBar] = useState(true)
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)
  const [historyIdx, setHistoryIdx] = useState<number | null>(null)
  const [scrollOffset, setScrollOffset] = useState(0)
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const [catalog, setCatalog] = useState<SlashCatalog | null>(null)

  const buf = useRef('')
  const stickyRef = useRef(true)
  const queueRef = useRef<string[]>([])
  const historyRef = useRef<string[]>(inputHistory.load())
  const historyDraftRef = useRef('')
  const queueEditRef = useRef<number | null>(null)
  const lastEmptyAt = useRef(0)

  const empty = !messages.length
  const blocked = !!(clarify || approval || sudo || secret || picker)

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
      inputHistory.append(trimmed)
    }
  }

  useEffect(() => {
    if (stickyRef.current) {
      setScrollOffset(0)
    }
  }, [messages.length])

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

    if (start > 0 && messages[start - 1]?.role === 'user') {
      start--
    }

    return { above: start, end, start }
  }, [cols, messages, msgBudget, scrollOffset])

  const sys = useCallback((text: string) => setMessages(prev => [...prev, { role: 'system' as const, text }]), [])

  const rpc = (method: string, params: Record<string, unknown> = {}) =>
    gw.request(method, params).catch((e: Error) => {
      sys(`error: ${e.message}`)
    })

  const newSession = (msg?: string) =>
    rpc('session.create', { cols }).then((r: any) => {
      if (!r) {
        return
      }

      setSid(r.session_id)
      setMessages([])
      setUsage(ZERO)
      setStatus('ready')

      if (msg) {
        sys(msg)
      }
    })

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

  const paste = () =>
    rpc('clipboard.paste', { session_id: sid }).then((r: any) =>
      sys(r.attached ? `📎 image #${r.count} attached` : r.message || 'no image in clipboard')
    )

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

    if (!inputBuf.length && key.tab && input.startsWith('/')) {
      const next = tabAdvance(input, catalog)

      if (next) {
        setInput(next)
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

    if (key.ctrl && ch === 'v') {
      return paste()
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
          if (!p?.text) {
            break
          }

          buf.current += p.rendered ?? p.text
          setThinking(false)
          setTools([])
          setReasoning('')
          setMessages(prev => upsert(prev, 'assistant', buf.current.trimStart()))

          break
        case 'message.complete': {
          idle()
          setMessages(prev => upsert(prev, 'assistant', (p?.rendered ?? p?.text ?? buf.current).trimStart()))
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
    [gw, sys, newSession]
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

        case 'clear':
          setStatus('forging session…')
          newSession()

          return true

        case 'quit': // falls through

        case 'exit':
          die()

          return true

        case 'new':
          setStatus('forging session…')
          newSession('new session started')

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

        case 'compact':
          setCompact(c => (arg ? true : !c))
          sys(arg ? `compact on, focus: ${arg}` : `compact ${compact ? 'off' : 'on'}`)

          return true

        case 'compress':
          if (!sid) {
            return true
          }

          rpc('session.compress', { session_id: sid }).then((r: any) => {
            sys('context compressed')

            if (r.usage) {
              setUsage(r.usage)
            }
          })

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

          writeOsc52Clipboard(target.text)
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

        case 'resume':
          setPicker(true)

          return true

        case 'history':
          if (!sid) {
            setPicker(true)

            return true
          }

          rpc('session.history', { session_id: sid }).then((r: any) =>
            sys(`session ${sid}: ${r.count} messages in context`)
          )

          return true

        case 'title':
          if (!sid) {
            return true
          }

          if (!arg) {
            rpc('session.title', { session_id: sid }).then((r: any) =>
              sys(`title: ${r.title || '(none)'}  session: ${r.session_key}`)
            )

            return true
          }

          rpc('session.title', { session_id: sid, title: arg }).then(() => sys(`title → ${arg}`))

          return true

        case 'tools':
          if (!info?.tools || !Object.keys(info.tools).length) {
            sys('no tools loaded')

            return true
          }

          sys(
            Object.entries(info.tools)
              .map(([k, vs]) => `${k} (${vs.length}): ${vs.join(', ')}`)
              .join('\n')
          )

          return true

        case 'skills':
          if (!arg || arg === 'list') {
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
          }

          if (arg.startsWith('search ')) {
            rpc('skills.manage', { action: 'search', query: arg.slice(7).trim() }).then((r: any) => {
              if (!r.results?.length) {
                sys('no results')

                return
              }

              sys(r.results.map((s: any) => `  ${s.name}: ${s.description}`).join('\n'))
            })

            return true
          }

          if (arg.startsWith('install ')) {
            rpc('skills.manage', { action: 'install', query: arg.slice(8).trim() }).then((r: any) =>
              sys(r.installed ? `installed ${r.name}` : 'install failed')
            )

            return true
          }

          if (arg === 'browse' || arg.startsWith('browse ')) {
            rpc('skills.manage', { action: 'browse', query: arg.slice(6).trim() }).then((r: any) => {
              if (!r.results?.length) {
                sys('no skills available')

                return
              }

              sys(r.results.map((s: any) => `  ${s.name}: ${s.description}`).join('\n'))
            })

            return true
          }

          if (arg.startsWith('inspect ')) {
            rpc('skills.manage', { action: 'inspect', query: arg.slice(8).trim() }).then((r: any) =>
              sys(JSON.stringify(r.info, null, 2))
            )

            return true
          }

          sys('usage: /skills [list|search <q>|install <name>|browse|inspect <name>]')

          return true

        case 'verbose':
          rpc('config.set', { key: 'verbose', value: arg || 'cycle' }).then((r: any) => sys(`verbose → ${r.value}`))

          return true

        case 'yolo':
          rpc('config.set', { key: 'yolo', value: '' }).then((r: any) =>
            sys(`yolo → ${r.value === '1' ? 'on' : 'off'}`)
          )

          return true

        case 'reasoning':
          if (!arg) {
            sys('usage: /reasoning <none|low|medium|high|xhigh|show|hide>')

            return true
          }

          rpc('config.set', { key: 'reasoning', value: arg }).then((r: any) => sys(`reasoning → ${r.value}`))

          return true

        case 'stop':
          rpc('process.stop').then((r: any) => sys(`killed ${r.killed} process(es)`))

          return true

        case 'profile':
          gw.request('config.get', { key: 'profile' })
            .then((r: any) => sys(`profile: ${r.display}`))
            .catch(() => sys(`profile: ${process.env.HERMES_HOME ?? '~/.hermes'}`))

          return true

        case 'save':
          if (!sid) {
            return true
          }

          rpc('session.save', { session_id: sid }).then((r: any) => sys(`saved to ${r.file}`))

          return true

        case 'provider':
          rpc('config.get', { key: 'provider' }).then((r: any) => {
            const lines = [`model: ${r.model}  provider: ${r.provider}`]

            if (r.providers?.length) {
              lines.push(`available: ${r.providers.join(', ')}`)
            }

            sys(lines.join('\n'))
          })

          return true

        case 'prompt':
          if (!arg) {
            rpc('config.get', { key: 'prompt' }).then((r: any) => sys(`custom prompt: ${r.prompt || '(none set)'}`))

            return true
          }

          rpc('config.set', { key: 'prompt', value: arg }).then((r: any) =>
            sys(r.value ? `prompt set (${r.value.length} chars)` : 'prompt cleared')
          )

          return true

        case 'personality':
          if (!arg) {
            sys('usage: /personality <name>  (concise, creative, analytical, friendly, none)')

            return true
          }

          rpc('config.set', { key: 'personality', value: arg }).then((r: any) =>
            sys(`personality → ${r.value || 'default'}`)
          )

          return true

        case 'plan':
          send(arg ? `/plan ${arg}` : 'Create a detailed plan for the current task.')

          return true

        case 'background':

        case 'bg':
          if (!arg) {
            sys('usage: /background <prompt>')

            return true
          }

          rpc('prompt.background', { session_id: sid, text: arg }).then((r: any) =>
            sys(`background task ${r.task_id} started`)
          )

          return true

        case 'btw':
          if (!arg) {
            sys('usage: /btw <question>')

            return true
          }

          rpc('prompt.btw', { session_id: sid, text: arg }).then(() => sys('btw running…'))

          return true

        case 'queue':
          if (!arg) {
            sys(`${queueRef.current.length} queued message(s)`)

            return true
          }

          enqueue(arg)
          sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)

          return true

        case 'rollback':
          if (!sid) {
            return true
          }

          if (!arg) {
            rpc('rollback.list', { session_id: sid }).then((r: any) => {
              if (!r.enabled) {
                sys('checkpoints not enabled — use hermes --checkpoints')

                return
              }

              if (!r.checkpoints?.length) {
                sys('no checkpoints')

                return
              }

              sys(
                r.checkpoints
                  .map((c: any, i: number) => `  ${i + 1}. ${c.message || c.hash.slice(0, 8)} (${c.timestamp})`)
                  .join('\n')
              )
            })

            return true
          }

          if (arg.startsWith('diff ')) {
            const ref = arg.slice(5).trim()
            rpc('rollback.list', { session_id: sid }).then((r: any) => {
              const hash = /^\d+$/.test(ref) ? r.checkpoints?.[parseInt(ref) - 1]?.hash : ref

              if (!hash) {
                sys(`checkpoint ${ref} not found`)

                return
              }

              rpc('rollback.diff', { session_id: sid, hash }).then((d: any) =>
                sys(d.rendered || d.stat || d.diff || 'no changes')
              )
            })

            return true
          }

          {
            const parts = arg.trim().split(/\s+/)
            const ref = parts[0]!
            const file = parts[1]
            rpc('rollback.list', { session_id: sid }).then((r: any) => {
              const hash = /^\d+$/.test(ref) ? r.checkpoints?.[parseInt(ref) - 1]?.hash : ref

              if (!hash) {
                sys(`checkpoint ${ref} not found`)

                return
              }

              rpc('rollback.restore', { session_id: sid, hash, ...(file ? { file } : {}) }).then((d: any) =>
                sys(d.success ? `restored${file ? ` ${file}` : ''}` : `failed: ${d.error || 'unknown'}`)
              )
            })
          }

          return true

        case 'insights':
          rpc('insights.get', { days: arg ? parseInt(arg) : 30 }).then((r: any) =>
            sys(`last ${r.days}d: ${r.sessions} sessions, ${r.messages} messages`)
          )

          return true

        case 'toolsets':
          if (!info?.tools) {
            sys('no toolsets loaded')

            return true
          }

          sys(
            Object.entries(info.tools)
              .map(([k, vs]) => `${k}: ${vs.length} tools`)
              .join('\n')
          )

          return true

        case 'paste':
          paste()

          return true

        case 'reload-mcp':

        case 'reload_mcp':
          rpc('reload.mcp', { session_id: sid }).then(() => sys('MCP servers reloaded'))

          return true

        case 'browser':
          if (!arg || arg === 'status') {
            rpc('browser.manage', { action: 'status' }).then((r: any) =>
              sys(r.connected ? `browser: connected (${r.url})` : 'browser: not connected')
            )
          } else if (arg === 'connect' || arg.startsWith('connect ')) {
            const url = arg.split(/\s+/)[1]
            rpc('browser.manage', { action: 'connect', ...(url ? { url } : {}) }).then((r: any) =>
              sys(`browser connected: ${r.url}`)
            )
          } else if (arg === 'disconnect') {
            rpc('browser.manage', { action: 'disconnect' }).then(() => sys('browser disconnected'))
          } else {
            sys('usage: /browser [connect|disconnect|status]')
          }

          return true

        case 'platforms':

        case 'gateway':
          sys('gateway status is not available in TUI mode')

          return true

        case 'statusbar':

        case 'sb':
          setStatusBar(v => !v)
          sys(`status bar ${statusBar ? 'off' : 'on'}`)

          return true

        case 'voice':
          if (!arg || arg === 'status') {
            rpc('voice.toggle', { action: 'status' }).then((r: any) => sys(`voice: ${r.enabled ? 'on' : 'off'}`))
          } else if (arg === 'on' || arg === 'off') {
            rpc('voice.toggle', { action: arg }).then((r: any) => sys(`voice → ${r.enabled ? 'on' : 'off'}`))
          } else if (arg === 'record') {
            rpc('voice.record', { action: 'start' }).then(() => sys('recording… (use /voice stop to transcribe)'))
          } else if (arg === 'stop') {
            rpc('voice.record', { action: 'stop' }).then((r: any) => {
              if (r.text) {
                send(r.text)
              } else {
                sys('no speech detected')
              }
            })
          } else if (arg === 'tts') {
            const last = messages.filter(m => m.role === 'assistant').at(-1)

            if (last) {
              rpc('voice.tts', { text: last.text }).then(() => sys('speaking…'))
            } else {
              sys('no response to speak')
            }
          } else {
            sys('usage: /voice [on|off|status|record|stop|tts]')
          }

          return true

        case 'plugins':
          rpc('plugins.list').then((r: any) => {
            if (!r.plugins?.length) {
              sys('no plugins installed')

              return
            }

            sys(r.plugins.map((p: any) => `  ${p.name} v${p.version} ${p.enabled ? '✓' : '✗'}`).join('\n'))
          })

          return true

        case 'cron':
          if (!arg || arg === 'list') {
            rpc('cron.manage', { action: 'list' }).then((r: any) => {
              const jobs = r.jobs || r.schedules || []

              if (!jobs.length) {
                sys('no cron jobs')

                return
              }

              sys(jobs.map((j: any) => `  ${j.name}: ${j.schedule} ${j.paused ? '(paused)' : ''}`).join('\n'))
            })
          } else {
            const parts = arg.split(/\s+/)
            const sub = parts[0]!

            if (sub === 'add' || sub === 'create') {
              const name = parts[1] || ''
              const schedule = parts[2] || ''
              const prompt = parts.slice(3).join(' ')
              rpc('cron.manage', { action: 'add', name, schedule, prompt }).then((r: any) =>
                sys(r.message || r.status || 'created')
              )
            } else {
              rpc('cron.manage', { action: sub, name: parts[1] || '' }).then((r: any) =>
                sys(r.message || r.status || JSON.stringify(r))
              )
            }
          }

          return true

        case 'update':
        case 'hermes': {
          const argv = name === 'update' ? ['update'] : arg.split(/\s+/).filter(Boolean)

          if (!argv.length) {
            sys('usage: /hermes <args…>  (e.g. sessions list, chat -q "hi")')

            return true
          }

          if (name === 'update') {
            setBusy(true)
            setStatus('updating…')
          }

          rpc('cli.exec', { argv, timeout: name === 'update' ? 600 : 240 })
            .then((r: any) => {
              if (r.blocked) {
                return sys(r.hint ?? 'blocked')
              }

              sys(r.output ?? '(no output)')

              if (r.code !== 0) {
                sys(`exit ${r.code}`)
              }
            })
            .catch((e: Error) => sys(`error: ${e.message}`))
            .finally(() => {
              if (name === 'update') {
                setStatus('ready')
                setBusy(false)
              }
            })

          return true
        }

        case 'model':
          if (!arg) {
            sys('usage: /model <name>')

            return true
          }

          rpc('config.set', { key: 'model', value: arg }).then(() => sys(`model → ${arg}`))

          return true

        case 'skin':
          if (!arg) {
            sys('usage: /skin <name>')

            return true
          }

          rpc('config.set', { key: 'skin', value: arg }).then(() => sys(`skin → ${arg} (restart to apply)`))

          return true

        default:
          gw.request('command.dispatch', { name: name ?? '', arg, session_id: sid })
            .then((r: any) => {
              if (r.type === 'exec') {
                sys(r.output || '(no output)')
              } else if (r.type === 'alias') {
                slash(`/${r.target}${arg ? ' ' + arg : ''}`)
              } else if (r.type === 'plugin') {
                sys(r.output || '(no output)')
              } else if (r.type === 'skill') {
                sys(`⚡ loading skill: ${r.name}`)
                send(r.message)
              }
            })
            .catch(() => {
              gw.request('command.resolve', { name: name ?? '' })
                .then((r: any) => {
                  if (r.canonical && r.canonical !== name) {
                    sys(`/${name} → /${r.canonical}`)
                    slash(`/${r.canonical}${arg ? ' ' + arg : ''}`)
                  } else {
                    sys(`unknown command: /${name}`)
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

          {thinking && <Thinking reasoning={reasoning} t={theme} thinking={thinkingText} tools={tools} />}
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
                  setUsage(ZERO)
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

        {!blocked && input.startsWith('/') && <CommandPalette matches={paletteForLine(input, catalog)} t={theme} />}

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
