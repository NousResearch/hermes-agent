'use strict'

import { Box, render, Text, useApp, useInput, useStdout } from 'ink'
import TextInput from 'ink-text-input'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { AltScreen } from './altScreen.js'
import { caduceus, logo, LOGO_WIDTH } from './banner.js'
import { GatewayClient, type GatewayEvent } from './gatewayClient.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'

// ── Types ───────────────────────────────────────────────────────────

type Role = 'user' | 'assistant' | 'system' | 'tool'

interface Msg {
  role: Role
  text: string
}
interface SessionInfo {
  model: string
  tools: Record<string, string[]>
  skills: Record<string, string[]>
}
interface ActiveTool {
  id: string
  name: string
}
interface ClarifyReq {
  requestId: string
  question: string
  choices: string[] | null
}
interface ApprovalReq {
  command: string
  description: string
}
interface Usage {
  input: number
  output: number
  total: number
  calls: number
}

// ── Constants ───────────────────────────────────────────────────────

const ZERO: Usage = { input: 0, output: 0, total: 0, calls: 0 }
const MAX_CTX = 128_000
const LONG_MSG = 300

const COMMANDS: [string, string][] = [
  ['/help', 'commands & hotkeys'],
  ['/model', 'switch model'],
  ['/skin', 'change theme'],
  ['/clear', 'reset chat'],
  ['/new', 'new session'],
  ['/undo', 'drop last exchange'],
  ['/retry', 'resend last message'],
  ['/compact', 'toggle compact [focus]'],
  ['/cost', 'token usage stats'],
  ['/copy', 'copy last response'],
  ['/context', 'context window info'],
  ['/compress', 'compress context'],
  ['/skills', 'list skills'],
  ['/config', 'show config'],
  ['/status', 'session info'],
  ['/quit', 'exit hermes']
]

const HOTKEYS: [string, string][] = [
  ['Ctrl+C', 'interrupt / clear / exit'],
  ['Ctrl+D', 'exit'],
  ['Ctrl+L', 'clear screen'],
  ['↑/↓', 'queue edit (if queued) / input history'],
  ['PgUp/PgDn', 'scroll messages'],
  ['Esc', 'clear input'],
  ['\\+Enter', 'multi-line continuation'],
  ['!cmd', 'run shell command'],
  ['{!cmd}', 'interpolate shell output inline']
]

const PLACEHOLDERS = [
  'Ask me anything…',
  'Try "explain this codebase"',
  'Try "write a test for…"',
  'Try "refactor the auth module"',
  'Try "/help" for commands',
  'Try "fix the lint errors"',
  'Try "how does the config loader work?"'
]

const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

const FACES = [
  '(｡•́︿•̀｡)',
  '(◔_◔)',
  '(¬‿¬)',
  '( •_•)>⌐■-■',
  '(⌐■_■)',
  '(´･_･`)',
  '◉_◉',
  '(°ロ°)',
  '( ˘⌣˘)♡',
  'ヽ(>∀<☆)☆',
  '٩(๑❛ᴗ❛๑)۶',
  '(⊙_⊙)',
  '(¬_¬)',
  '( ͡° ͜ʖ ͡°)',
  'ಠ_ಠ'
]

const VERBS = [
  'pondering',
  'contemplating',
  'musing',
  'cogitating',
  'ruminating',
  'deliberating',
  'mulling',
  'reflecting',
  'processing',
  'reasoning',
  'analyzing',
  'computing',
  'synthesizing',
  'formulating',
  'brainstorming'
]

const TOOL_VERBS: Record<string, string> = {
  read_file: '📖 reading',
  write_file: '✏️ writing',
  search_code: '🔍 searching',
  run_command: '⚙️ running',
  execute_code: '⚡ executing',
  list_files: '📂 listing',
  web_search: '🌐 searching',
  create_file: '📝 creating',
  delete_file: '🗑️ deleting',
  memory: '🧠 remembering',
  clarify: '❓ asking',
  delegate_task: '🤖 delegating',
  browser: '🌐 browsing',
  terminal: '💻 terminal',
  patch: '🩹 patching',
  search_files: '🔍 searching',
  image_generate: '🎨 generating'
}

const ROLE: Record<Role, (t: Theme) => { glyph: string; prefix: string; body: string }> = {
  user: t => ({ glyph: t.brand.prompt, prefix: t.color.label, body: t.color.label }),
  assistant: t => ({ glyph: t.brand.tool, prefix: t.color.bronze, body: t.color.cornsilk }),
  system: t => ({ glyph: '!', prefix: t.color.error, body: t.color.error }),
  tool: t => ({ glyph: '⚡', prefix: t.color.dim, body: t.color.dim })
}

// ── Pure helpers ────────────────────────────────────────────────────

const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)]!
const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`)
const flat = (r: Record<string, string[]>) => Object.values(r).flat()

const estimateRows = (text: string, w: number) =>
  text.split('\n').reduce((s, l) => s + Math.max(1, Math.ceil(Math.max(1, l.length) / w)), 0)

const compactPreview = (s: string, max: number) => {
  const one = s.replace(/\s+/g, ' ').trim()

  return !one ? '' : one.length > max ? one.slice(0, max - 1) + '…' : one
}

const userDisplay = (text: string): string => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return `${prefix || '(message)'} [long message]`
}

const INTERPOLATION_RE = /\{!(.+?)\}/g
const hasInterpolation = (s: string) => INTERPOLATION_RE.test(s)

const PLACEHOLDER = pick(PLACEHOLDERS)

// ── Components ──────────────────────────────────────────────────────

function ArtLines({ lines }: { lines: [string, string][] }) {
  return (
    <>
      {lines.map(([c, text], i) => (
        <Text color={c} key={i}>
          {text}
        </Text>
      ))}
    </>
  )
}

function Banner({ t }: { t: Theme }) {
  const cols = useStdout().stdout?.columns ?? 80

  return (
    <Box flexDirection="column" marginBottom={1}>
      {cols >= LOGO_WIDTH ? (
        <ArtLines lines={logo(t.color)} />
      ) : (
        <Text bold color={t.color.gold}>
          {t.brand.icon} NOUS HERMES
        </Text>
      )}
      <Text />
      <Text>
        <Text color={t.color.amber}>{t.brand.icon} Nous Research</Text>
        <Text color={t.color.dim}> · Messenger of the Digital Gods</Text>
      </Text>
    </Box>
  )
}

function SessionPanel({ t, info }: { t: Theme; info: SessionInfo }) {
  const cols = useStdout().stdout?.columns ?? 100
  const wide = cols >= 90
  const w = wide ? cols - 46 : cols - 10
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''

    for (const item of items.sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > w) {
        return line ? `${line}, …+${items.length - line.split(', ').length}` : `${item}, …`
      }

      line = next
    }

    return line
  }

  const section = (title: string, data: Record<string, string[]>, max = 8) => {
    const entries = Object.entries(data).sort()
    const shown = entries.slice(0, max)
    const overflow = entries.length - max

    return (
      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.amber}>
          Available {title}
        </Text>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.dim}>{strip(k)}: </Text>
            <Text color={t.color.cornsilk}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && <Text color={t.color.dim}>(and {overflow} more…)</Text>}
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.bronze} borderStyle="round" marginBottom={1} paddingX={2} paddingY={1}>
      {wide && (
        <Box flexDirection="column" marginRight={2} width={34}>
          <ArtLines lines={caduceus(t.color)} />
          <Text />
          <Text color={t.color.dim}>Nous Research</Text>
        </Box>
      )}
      <Box flexDirection="column" width={w}>
        <Text bold color={t.color.gold}>
          {t.brand.icon} {t.brand.name}
        </Text>
        {section('Tools', info.tools)}
        {section('Skills', info.skills)}
        <Text />
        <Text color={t.color.cornsilk}>
          {flat(info.tools).length} tools{' · '}
          {flat(info.skills).length} skills
          {' · '}
          <Text color={t.color.dim}>/help for commands</Text>
        </Text>
        <Text color={t.color.dim}>
          {info.model.split('/').pop()}
          {' · '}Ctrl+C to interrupt
        </Text>
      </Box>
    </Box>
  )
}

function CommandPalette({ t, filter }: { t: Theme; filter: string }) {
  const m = COMMANDS.filter(([cmd]) => cmd.startsWith(filter))

  if (!m.length) {
    return null
  }

  return (
    <Box flexDirection="column">
      {m.map(([cmd, desc]) => (
        <Text key={cmd}>
          <Text bold color={t.color.amber}>
            {cmd}
          </Text>
          <Text color={t.color.dim}> — {desc}</Text>
        </Text>
      ))}
    </Box>
  )
}

function Thinking({ t, tools, reasoning }: { t: Theme; tools: ActiveTool[]; reasoning: string }) {
  const [frame, setFrame] = useState(0)
  const [verb] = useState(() => pick(VERBS))
  const [face] = useState(() => pick(FACES))

  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % SPINNER.length), 80)

    return () => clearInterval(id)
  }, [])

  return (
    <Box flexDirection="column">
      {tools.length ? (
        tools.map(tool => (
          <Text color={t.color.dim} key={tool.id}>
            {SPINNER[frame]} {TOOL_VERBS[tool.name] ?? '⚡ ' + tool.name}…
          </Text>
        ))
      ) : (
        <Text color={t.color.dim}>
          {SPINNER[frame]} {face} {verb}…
        </Text>
      )}
      {reasoning && (
        <Text color={t.color.dim} dimColor wrap="truncate-end">
          {'  💭 '}
          {reasoning.slice(-120).replace(/\n/g, ' ')}
        </Text>
      )}
    </Box>
  )
}

// ── Interactive prompts ─────────────────────────────────────────────

function ClarifyPrompt({ t, req, onAnswer }: { t: Theme; req: ClarifyReq; onAnswer: (s: string) => void }) {
  const [sel, setSel] = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const choices = req.choices ?? []

  useInput((ch, key) => {
    if (typing) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < choices.length) {
      setSel(s => s + 1)
    }

    if (key.return) {
      if (sel === choices.length) {
        setTyping(true)
      } else if (choices[sel]) {
        onAnswer(choices[sel]!)
      }
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= choices.length) {
      onAnswer(choices[n - 1]!)
    }
  })

  if (typing || !choices.length) {
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.amber}>
          ❓ {req.question}
        </Text>
        <Box>
          <Text color={t.color.label}>{'> '}</Text>
          <TextInput onChange={setCustom} onSubmit={onAnswer} value={custom} />
        </Box>
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>
        ❓ {req.question}
      </Text>
      {[...choices, 'Other (type your answer)'].map((c, i) => (
        <Text key={i}>
          <Text color={sel === i ? t.color.label : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            {i + 1}. {c}
          </Text>
        </Text>
      ))}
      <Text color={t.color.dim}>↑/↓ select · Enter confirm · 1-{choices.length} quick pick</Text>
    </Box>
  )
}

function ApprovalPrompt({ t, req, onChoice }: { t: Theme; req: ApprovalReq; onChoice: (s: string) => void }) {
  const [sel, setSel] = useState(3)
  const opts = ['once', 'session', 'always', 'deny'] as const
  const labels = { once: 'Allow once', session: 'Allow this session', always: 'Always allow', deny: 'Deny' } as const

  useInput((ch, key) => {
    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < 3) {
      setSel(s => s + 1)
    }

    if (key.return) {
      onChoice(opts[sel]!)
    }

    if (ch === 'o') {
      onChoice('once')
    }

    if (ch === 's') {
      onChoice('session')
    }

    if (ch === 'a') {
      onChoice('always')
    }

    if (ch === 'd' || key.escape) {
      onChoice('deny')
    }
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>
        ⚠️ DANGEROUS COMMAND: {req.description}
      </Text>
      <Text color={t.color.dim}> {req.command}</Text>
      <Text />
      {opts.map((o, i) => (
        <Text key={o}>
          <Text color={sel === i ? t.color.warn : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            [{o[0]}] {labels[o]}
          </Text>
        </Text>
      ))}
      <Text color={t.color.dim}>↑/↓ select · Enter confirm · o/s/a/d quick pick</Text>
    </Box>
  )
}

// ── Markdown ────────────────────────────────────────────────────────

function Md({ t, text, compact }: { t: Theme; text: string; compact?: boolean }) {
  const lines = text.split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]!
    const k = nodes.length

    if (compact && !line.trim()) {
      i++

      continue
    }

    if (line.startsWith('```')) {
      const lang = line.slice(3).trim()
      const block: string[] = []

      for (i++; i < lines.length && !lines[i]!.startsWith('```'); i++) {
        block.push(lines[i]!)
      }

      i++
      nodes.push(
        <Box flexDirection="column" key={k} paddingLeft={2}>
          {lang && <Text color={t.color.dim}>{'─ ' + lang}</Text>}
          {block.map((l, j) => (
            <Text color={t.color.cornsilk} key={j}>
              {l}
            </Text>
          ))}
        </Box>
      )

      continue
    }

    const hm = line.match(/^#{1,3}\s+(.*)/)

    if (hm) {
      nodes.push(
        <Text bold color={t.color.amber} key={k}>
          {hm[1]}
        </Text>
      )
      i++

      continue
    }

    const bm = line.match(/^\s*[-*]\s(.*)/)

    if (bm) {
      nodes.push(
        <Text key={k}>
          <Text color={t.color.dim}> • </Text>
          <MdInline t={t} text={bm[1]!} />
        </Text>
      )
      i++

      continue
    }

    const nm = line.match(/^\s*(\d+)\.\s(.*)/)

    if (nm) {
      nodes.push(
        <Text key={k}>
          <Text color={t.color.dim}> {nm[1]}. </Text>
          <MdInline t={t} text={nm[2]!} />
        </Text>
      )
      i++

      continue
    }

    nodes.push(<MdInline key={k} t={t} text={line} />)
    i++
  }

  return <Box flexDirection="column">{nodes}</Box>
}

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: React.ReactNode[] = []
  const re = /(\[(.+?)\]\((https?:\/\/[^\s)]+)\)|\*\*(.+?)\*\*|`([^`]+)`|\*(.+?)\*|(https?:\/\/[^\s]+))/g

  let last = 0,
    m: RegExpExecArray | null

  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(
        <Text color={t.color.cornsilk} key={parts.length}>
          {text.slice(last, m.index)}
        </Text>
      )
    }

    if (m[2] && m[3]) {
      parts.push(
        <Text color={t.color.amber} key={parts.length} underline>
          {m[2]}
        </Text>
      )
    } else if (m[4]) {
      parts.push(
        <Text bold color={t.color.cornsilk} key={parts.length}>
          {m[4]}
        </Text>
      )
    } else if (m[5]) {
      parts.push(
        <Text color={t.color.amber} dimColor key={parts.length}>
          {m[5]}
        </Text>
      )
    } else if (m[6]) {
      parts.push(
        <Text color={t.color.cornsilk} italic key={parts.length}>
          {m[6]}
        </Text>
      )
    } else if (m[7]) {
      parts.push(
        <Text color={t.color.amber} key={parts.length} underline>
          {m[7]}
        </Text>
      )
    }

    last = m.index + m[0].length
  }

  if (last < text.length) {
    parts.push(
      <Text color={t.color.cornsilk} key={parts.length}>
        {text.slice(last)}
      </Text>
    )
  }

  return <Text>{parts.length ? parts : <Text color={t.color.cornsilk}>{text}</Text>}</Text>
}

// ── Message ─────────────────────────────────────────────────────────

function MessageLine({ t, msg, compact }: { t: Theme; msg: Msg; compact?: boolean }) {
  const { glyph, prefix, body } = ROLE[msg.role](t)

  const content = (() => {
    if (msg.role === 'assistant') {
      return <Md compact={compact} t={t} text={msg.text} />
    }

    if (msg.role === 'user' && msg.text.length > LONG_MSG) {
      const d = userDisplay(msg.text)
      const [head, ...rest] = d.split('[long message]')

      return (
        <Text color={body}>
          {head}
          <Text color={t.color.dim} dimColor>
            [long message]
          </Text>
          {rest.join('')}
        </Text>
      )
    }

    return <Text color={body}>{msg.text}</Text>
  })()

  return (
    <Box>
      <Box width={3}>
        <Text bold={msg.role === 'user'} color={prefix}>
          {glyph}{' '}
        </Text>
      </Box>
      {content}
    </Box>
  )
}

// ── App ─────────────────────────────────────────────────────────────

function App({ gw }: { gw: GatewayClient }) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const cols = stdout?.columns ?? 80
  const rows = stdout?.rows ?? 24

  // ── State ─────────────────────────────────────────────────────────

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

  // ── Queue / history helpers ───────────────────────────────────────

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
    const [h, ...rest] = queueRef.current
    queueRef.current = rest
    syncQueue()

    return h
  }

  const replaceQ = (i: number, text: string) => {
    queueRef.current[i] = text
    syncQueue()
  }

  const pushHistory = (text: string) => {
    const t = text.trim()

    if (t && historyRef.current.at(-1) !== t) {
      historyRef.current.push(t)
    }
  }

  // ── Derived ───────────────────────────────────────────────────────

  useEffect(() => {
    if (stickyRef.current) {
      setScrollOffset(0)
    }
  }, [messages.length])

  const msgBudget = Math.max(3, rows - 2 - (empty ? 0 : 2) - (thinking ? 2 : 0) - 2)

  const viewport = useMemo(() => {
    if (!messages.length) {
      return { start: 0, end: 0, above: 0 }
    }

    const end = Math.max(0, messages.length - scrollOffset)
    const w = Math.max(20, cols - 5)

    let budget = msgBudget,
      start = end

    for (let i = end - 1; i >= 0 && budget > 0; i--) {
      const m = messages[i]!
      const margin = m.role === 'user' && i > 0 && messages[i - 1]?.role !== 'user' ? 1 : 0
      budget -= margin + estimateRows(m.role === 'user' ? userDisplay(m.text) : m.text, w)

      if (budget >= 0) {
        start = i
      }
    }

    if (start === end && end > 0) {
      start = end - 1
    }

    return { start, end, above: start }
  }, [messages, scrollOffset, msgBudget, cols])

  // ── Actions ───────────────────────────────────────────────────────

  const sys = useCallback((text: string) => setMessages(p => [...p, { role: 'system' as const, text }]), [])

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
    setScrollOffset(p => Math.min(Math.max(0, messages.length - 1), p + n))
    stickyRef.current = false
  }

  const scrollDown = (n: number) => {
    setScrollOffset(p => {
      const v = Math.max(0, p - n)

      if (!v) {
        stickyRef.current = true
      }

      return v
    })
  }

  const send = (text: string) => {
    setLastUserMsg(text)
    setMessages(p => [...p, { role: 'user', text }])
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
    setMessages(p => [...p, { role: 'user', text: `!${cmd}` }])
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

  // ── Hotkeys ───────────────────────────────────────────────────────

  useInput((ch, key) => {
    if (blocked) {
      if (key.ctrl && ch === 'c' && approval) {
        gw.request('approval.respond', { session_id: sid, choice: 'deny' }).catch(() => {})
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
        const h = historyRef.current
        const idx = historyIdx === null ? h.length - 1 : Math.max(0, historyIdx - 1)

        if (historyIdx === null) {
          historyDraftRef.current = input
        }

        setHistoryIdx(idx)
        setQueueEdit(null)
        setInput(h[idx] ?? '')
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
        const h = historyRef.current
        const next = historyIdx + 1

        if (next >= h.length) {
          setHistoryIdx(null)
          setInput(historyDraftRef.current)
        } else {
          setHistoryIdx(next)
          setInput(h[next] ?? '')
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

  // ── Gateway events ────────────────────────────────────────────────

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
          setClarify({ requestId: p.request_id, question: p.question, choices: p.choices })
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
  }, [gw, exit, onEvent])

  // ── Slash commands ────────────────────────────────────────────────

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
                setMessages(p => {
                  const q = [...p]

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

          setMessages(p => {
            const q = [...p]

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
          sys(
            `context: ${fmtK(usage.total)} / ${fmtK(MAX_CTX)} (${pct}%)\n[${'█'.repeat(bar)}${'░'.repeat(30 - bar)}] ${pct < 50 ? '✓' : pct < 80 ? '⚠' : '✗'}`
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
    [gw, sid, status, sys, compact, info, usage, messages, lastUserMsg]
  )

  // ── Submit ────────────────────────────────────────────────────────

  const submit = useCallback(
    (value: string) => {
      // double-enter flushes queue head
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

      // multi-line continuation
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

      // queue edit mode → replace, don't send
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

      // queue if busy (slash/shell bypass; interpolation resolves then queues)
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
    [gw, sid, slash, sys, inputBuf, busy]
  )

  // ── Render ────────────────────────────────────────────────────────

  const statusColor =
    status === 'ready'
      ? theme.color.ok
      : status.startsWith('error')
        ? theme.color.error
        : status === 'interrupted'
          ? theme.color.warn
          : theme.color.dim

  const qW = 3
  const qStart = queueEditIdx === null ? 0 : Math.max(0, Math.min(queueEditIdx - 1, queuedDisplay.length - qW))
  const qEnd = Math.min(queuedDisplay.length, qStart + qW)

  return (
    <AltScreen>
      <Box flexDirection="column" flexGrow={1} padding={1}>
        {/* ── Header ──────────────────────────────────────────────── */}

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

        {/* ── Messages ────────────────────────────────────────────── */}

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

        {/* ── Prompts / chrome ─────────────────────────────────────── */}

        {clarify && (
          <ClarifyPrompt
            onAnswer={answer => {
              gw.request('clarify.respond', { request_id: clarify.requestId, answer }).catch(() => {})
              setMessages(p => [...p, { role: 'user', text: answer }])
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
              gw.request('approval.respond', { session_id: sid, choice }).catch(() => {})
              setApproval(null)
              sys(choice === 'deny' ? 'denied' : `approved (${choice})`)
              setStatus('running…')
            }}
            req={approval}
            t={theme}
          />
        )}

        {!blocked && input.startsWith('/') && <CommandPalette filter={input} t={theme} />}

        {queuedDisplay.length > 0 && (
          <Box flexDirection="column">
            <Text color={theme.color.dim} dimColor>
              queued ({queuedDisplay.length}){queueEditIdx !== null ? ` · editing ${queueEditIdx + 1}` : ''}
            </Text>
            {qStart > 0 && (
              <Text color={theme.color.dim} dimColor>
                {' '}
                …
              </Text>
            )}
            {queuedDisplay.slice(qStart, qEnd).map((q, i) => {
              const idx = qStart + i,
                active = queueEditIdx === idx

              return (
                <Text color={active ? theme.color.amber : theme.color.dim} dimColor key={`${idx}-${q.slice(0, 16)}`}>
                  {active ? '▸' : ' '} {idx + 1}. {compactPreview(q, Math.max(16, cols - 10))}
                </Text>
              )
            })}
            {qEnd < queuedDisplay.length && (
              <Text color={theme.color.dim} dimColor>
                {'  '}…and {queuedDisplay.length - qEnd} more
              </Text>
            )}
          </Box>
        )}

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

// ── Helpers ──────────────────────────────────────────────────────────

function upsert(prev: Msg[], role: Role, text: string): Msg[] {
  return prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
}

// ── Boot ────────────────────────────────────────────────────────────

if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY')
  process.exit(0)
}

const gw = new GatewayClient()
gw.start()
render(<App gw={gw} />, { exitOnCtrlC: false })
