'use strict'

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { render, Box, Text, useApp, useStdout, useInput } from 'ink'
import TextInput from 'ink-text-input'

import { GatewayClient, type GatewayEvent } from './gatewayClient.js'
import { DEFAULT_THEME, fromSkin, type Theme } from './theme.js'
import { logo, caduceus, LOGO_WIDTH } from './banner.js'
import { AltScreen } from './altScreen.js'


type Role = 'user' | 'assistant' | 'system' | 'tool'

interface Msg { role: Role; text: string }
interface SessionInfo { model: string; tools: Record<string, string[]>; skills: Record<string, string[]> }
interface ActiveTool { id: string; name: string }
interface ClarifyReq { requestId: string; question: string; choices: string[] | null }
interface ApprovalReq { command: string; description: string }
interface Usage { input: number; output: number; total: number; calls: number }

const ZERO: Usage = { input: 0, output: 0, total: 0, calls: 0 }
const MAX_CTX = 128_000

const COMMANDS: [string, string][] = [
  ['/help',     'commands & hotkeys'],
  ['/model',    'switch model'],
  ['/skin',     'change theme'],
  ['/clear',    'reset chat'],
  ['/new',      'new session'],
  ['/undo',     'drop last exchange'],
  ['/retry',    'resend last message'],
  ['/compact',  'toggle compact [focus]'],
  ['/cost',     'token usage stats'],
  ['/copy',     'copy last response'],
  ['/context',  'context window info'],
  ['/compress', 'compress context'],
  ['/skills',   'list skills'],
  ['/config',   'show config'],
  ['/status',   'session info'],
  ['/quit',     'exit hermes'],
]

const HOTKEYS: [string, string][] = [
  ['Ctrl+C',   'interrupt / clear / exit'],
  ['Ctrl+D',   'exit'],
  ['Ctrl+L',   'clear screen'],
  ['↑/↓',      'queue edit (if queued) / input history'],
  ['PgUp/PgDn','scroll messages'],
  ['Ctrl+J',   'newline in input'],
  ['Esc',      'clear input'],
  ['\\+Enter', 'multi-line continuation'],
  ['!cmd',     'run shell command'],
  ['{!cmd}',   'interpolate shell output inline'],
]

const PLACEHOLDERS = [
  'Ask me anything…', 'Try "explain this codebase"', 'Try "write a test for…"',
  'Try "refactor the auth module"', 'Try "/help" for commands',
  'Try "fix the lint errors"', 'Try "how does the config loader work?"',
]

const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

const FACES = [
  '(｡•́︿•̀｡)', '(◔_◔)', '(¬‿¬)', '( •_•)>⌐■-■', '(⌐■_■)',
  '(´･_･`)', '◉_◉', '(°ロ°)', '( ˘⌣˘)♡', 'ヽ(>∀<☆)☆',
  '٩(๑❛ᴗ❛๑)۶', '(⊙_⊙)', '(¬_¬)', '( ͡° ͜ʖ ͡°)', 'ಠ_ಠ',
]

const VERBS = [
  'pondering', 'contemplating', 'musing', 'cogitating', 'ruminating',
  'deliberating', 'mulling', 'reflecting', 'processing', 'reasoning',
  'analyzing', 'computing', 'synthesizing', 'formulating', 'brainstorming',
]

const TOOL_VERBS: Record<string, string> = {
  read_file: '📖 reading', write_file: '✏️ writing', search_code: '🔍 searching',
  run_command: '⚙️ running', execute_code: '⚡ executing', list_files: '📂 listing',
  web_search: '🌐 searching', create_file: '📝 creating', delete_file: '🗑️ deleting',
  memory: '🧠 remembering', clarify: '❓ asking', delegate_task: '🤖 delegating',
  browser: '🌐 browsing', terminal: '💻 terminal', patch: '🩹 patching',
  search_files: '🔍 searching', image_generate: '🎨 generating',
}

const ROLE: Record<Role, (t: Theme) => [string, string, string]> = {
  user:      t => [t.brand.prompt + ' ', t.color.label,   t.color.label],
  assistant: t => [t.brand.tool + ' ',   t.color.bronze,  t.color.cornsilk],
  system:    t => ['! ',                  t.color.error,   t.color.error],
  tool:      t => ['⚡ ',                 t.color.dim,     t.color.dim],
}

const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)]!
const fmtK = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`
const flat = (r: Record<string, string[]>) => Object.values(r).flat()
const estimateRows = (text: string, w: number) =>
  text.split('\n').reduce((s, l) => s + Math.max(1, Math.ceil(Math.max(1, l.length) / w)), 0)
const compactPreview = (s: string, max: number) => {
  const one = s.replace(/\s+/g, ' ').trim()
  if (!one) return ''
  return one.length > max ? one.slice(0, max - 1) + '…' : one
}
const PLACEHOLDER = pick(PLACEHOLDERS)


// ── Components ──────────────────────────────────────────────────────

function ArtLines({ lines }: { lines: [string, string][] }) {
  return <>{lines.map(([c, text], i) => <Text key={i} color={c}>{text}</Text>)}</>
}

function Banner({ t }: { t: Theme }) {
  const cols = useStdout().stdout?.columns ?? 80

  return (
    <Box flexDirection="column" marginBottom={1}>
      {cols >= LOGO_WIDTH
        ? <ArtLines lines={logo(t.color)} />
        : <Text bold color={t.color.gold}>{t.brand.icon} NOUS HERMES</Text>}

      <Text />

      <Text>
        <Text color={t.color.amber}>{t.brand.icon} Nous Research</Text>
        <Text color={t.color.dim}> · Messenger of the Digital Gods</Text>
      </Text>
    </Box>
  )
}

function truncLine(pfx: string, items: string[], max: number): string {
  let line = ''
  for (const item of items.sort()) {
    const next = line ? `${line}, ${item}` : item
    if (pfx.length + next.length > max)
      return line ? `${line}, …+${items.length - line.split(', ').length}` : `${item}, …`
    line = next
  }
  return line
}

function SessionPanel({ t, info }: { t: Theme; info: SessionInfo }) {
  const cols = useStdout().stdout?.columns ?? 100
  const wide = cols >= 90
  const w = wide ? cols - 46 : cols - 10
  const strip = (s: string) => s.endsWith('_tools') ? s.slice(0, -6) : s

  const section = (title: string, data: Record<string, string[]>, max = 8) => {
    const entries = Object.entries(data).sort()
    const shown = entries.slice(0, max)
    const overflow = entries.length - max

    return (
      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.amber}>Available {title}</Text>

        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.dim}>{strip(k)}: </Text>
            <Text color={t.color.cornsilk}>{truncLine(strip(k) + ': ', vs, w)}</Text>
          </Text>
        ))}

        {overflow > 0 && (
          <Text color={t.color.dim}>(and {overflow} more…)</Text>
        )}
      </Box>
    )
  }

  return (
    <Box marginBottom={1} borderStyle="round" borderColor={t.color.bronze} paddingX={2} paddingY={1}>

      {wide && (
        <Box flexDirection="column" width={34} marginRight={2}>
          <ArtLines lines={caduceus(t.color)} />
          <Text />
          <Text color={t.color.dim}>Nous Research</Text>
        </Box>
      )}

      <Box flexDirection="column" width={w}>
        <Text bold color={t.color.gold}>{t.brand.icon} {t.brand.name}</Text>

        {section('Tools', info.tools)}
        {section('Skills', info.skills)}

        <Text />

        <Text color={t.color.cornsilk}>
          {flat(info.tools).length} tools
          {' · '}{flat(info.skills).length} skills
          {' · '}<Text color={t.color.dim}>/help for commands</Text>
        </Text>

        <Text color={t.color.dim}>
          {info.model.split('/').pop()}
          {' · '}<Text color={t.color.dim}>Ctrl+C to interrupt</Text>
        </Text>
      </Box>
    </Box>
  )
}

function CommandPalette({ t, filter }: { t: Theme; filter: string }) {
  const m = COMMANDS.filter(([cmd]) => cmd.startsWith(filter))
  if (!m.length) return null

  return (
    <Box flexDirection="column">
      {m.map(([cmd, desc]) => (
        <Text key={cmd}>
          <Text bold color={t.color.amber}>{cmd}</Text>
          <Text color={t.color.dim}> — {desc}</Text>
        </Text>
      ))}
    </Box>
  )
}

function Thinking({ t, tools, reasoning }: { t: Theme; tools: ActiveTool[]; reasoning: string }) {
  const [frame, setFrame] = useState(0)
  const [verb]  = useState(() => pick(VERBS))
  const [face]  = useState(() => pick(FACES))

  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % SPINNER.length), 80)
    return () => clearInterval(id)
  }, [])

  return (
    <Box flexDirection="column">
      {tools.length
        ? tools.map(tool => (
            <Text key={tool.id} color={t.color.dim}>
              {SPINNER[frame]} {TOOL_VERBS[tool.name] ?? '⚡ ' + tool.name}…
            </Text>
          ))
        : <Text color={t.color.dim}>{SPINNER[frame]} {face} {verb}…</Text>}

      {reasoning && (
        <Text color={t.color.dim} dimColor wrap="truncate-end">
          {'  💭 '}{reasoning.slice(-120).replace(/\n/g, ' ')}
        </Text>
      )}
    </Box>
  )
}


// ── Interactive Prompts ─────────────────────────────────────────────

function ClarifyPrompt({ t, req, onAnswer }: { t: Theme; req: ClarifyReq; onAnswer: (s: string) => void }) {
  const [sel, setSel]       = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const choices = req.choices ?? []

  useInput((ch, key) => {
    if (typing) return
    if (key.upArrow && sel > 0)               setSel(s => s - 1)
    if (key.downArrow && sel < choices.length) setSel(s => s + 1)
    if (key.return) {
      if (sel === choices.length) setTyping(true)
      else if (choices[sel])      onAnswer(choices[sel]!)
    }
    const n = parseInt(ch)
    if (n >= 1 && n <= choices.length) onAnswer(choices[n - 1]!)
  })

  if (typing || !choices.length)
    return (
      <Box flexDirection="column">
        <Text bold color={t.color.amber}>❓ {req.question}</Text>
        <Box>
          <Text color={t.color.label}>{'> '}</Text>
          <TextInput value={custom} onChange={setCustom} onSubmit={onAnswer} />
        </Box>
      </Box>
    )

  const row = (i: number, label: string) => (
    <Text key={i}>
      <Text color={sel === i ? t.color.label : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
      <Text color={sel === i ? t.color.cornsilk : t.color.dim}>{i + 1}. {label}</Text>
    </Text>
  )

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>❓ {req.question}</Text>
      {choices.map((c, i) => row(i, c))}
      {row(choices.length, 'Other (type your answer)')}
      <Text color={t.color.dim}>↑/↓ select · Enter confirm · 1-{choices.length} quick pick</Text>
    </Box>
  )
}

function ApprovalPrompt({ t, req, onChoice }: { t: Theme; req: ApprovalReq; onChoice: (s: string) => void }) {
  const [sel, setSel] = useState(3)
  const opts = ['once', 'session', 'always', 'deny'] as const

  useInput((ch, key) => {
    if (key.upArrow && sel > 0)   setSel(s => s - 1)
    if (key.downArrow && sel < 3) setSel(s => s + 1)
    if (key.return) onChoice(opts[sel]!)
    if (ch === 'o') onChoice('once')
    if (ch === 's') onChoice('session')
    if (ch === 'a') onChoice('always')
    if (ch === 'd' || key.escape) onChoice('deny')
  })

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.warn}>⚠️  DANGEROUS COMMAND: {req.description}</Text>
      <Text color={t.color.dim}>  {req.command}</Text>
      <Text />
      {opts.map((o, i) => (
        <Text key={o}>
          <Text color={sel === i ? t.color.warn : t.color.dim}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? t.color.cornsilk : t.color.dim}>
            [{o[0]}] {o === 'once' ? 'Allow once' : o === 'session' ? 'Allow this session' : o === 'always' ? 'Always allow' : 'Deny'}
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

    if (compact && !line.trim()) { i++; continue }

    if (line.startsWith('```')) {
      const lang = line.slice(3).trim()
      const block: string[] = []
      for (i++; i < lines.length && !lines[i]!.startsWith('```'); i++)
        block.push(lines[i]!)
      i++
      nodes.push(
        <Box key={k} flexDirection="column" paddingLeft={2}>
          {lang && <Text color={t.color.dim}>{'─ ' + lang}</Text>}
          {block.map((l, j) => <Text key={j} color={t.color.cornsilk}>{l}</Text>)}
        </Box>,
      )
      continue
    }

    const hm = line.match(/^#{1,3}\s+(.*)/)
    if (hm) { nodes.push(<Text key={k} bold color={t.color.amber}>{hm[1]}</Text>); i++; continue }

    const bm = line.match(/^\s*[-*]\s(.*)/)
    if (bm) { nodes.push(<Text key={k}><Text color={t.color.dim}>  • </Text><MdInline t={t} text={bm[1]!} /></Text>); i++; continue }

    const nm = line.match(/^\s*(\d+)\.\s(.*)/)
    if (nm) { nodes.push(<Text key={k}><Text color={t.color.dim}>  {nm[1]}. </Text><MdInline t={t} text={nm[2]!} /></Text>); i++; continue }

    nodes.push(<MdInline key={k} t={t} text={line} />)
    i++
  }

  return <Box flexDirection="column">{nodes}</Box>
}

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: React.ReactNode[] = []
  const re = /(\[(.+?)\]\((https?:\/\/[^\s)]+)\)|\*\*(.+?)\*\*|`([^`]+)`|\*(.+?)\*|(https?:\/\/[^\s]+))/g
  let last = 0
  let m: RegExpExecArray | null

  while ((m = re.exec(text)) !== null) {
    if (m.index > last)
      parts.push(<Text key={parts.length} color={t.color.cornsilk}>{text.slice(last, m.index)}</Text>)

    if (m[2] && m[3]) {
      parts.push(<Text key={parts.length} color={t.color.amber} underline>{m[2]}</Text>)
    } else if (m[4]) {
      parts.push(<Text key={parts.length} bold color={t.color.cornsilk}>{m[4]}</Text>)
    } else if (m[5]) {
      parts.push(<Text key={parts.length} color={t.color.amber} dimColor>{m[5]}</Text>)
    } else if (m[6]) {
      parts.push(<Text key={parts.length} italic color={t.color.cornsilk}>{m[6]}</Text>)
    } else if (m[7]) {
      parts.push(<Text key={parts.length} color={t.color.amber} underline>{m[7]}</Text>)
    }

    last = m.index + m[0].length
  }

  if (last < text.length)
    parts.push(<Text key={parts.length} color={t.color.cornsilk}>{text.slice(last)}</Text>)

  return <Text>{parts.length ? parts : <Text color={t.color.cornsilk}>{text}</Text>}</Text>
}


// ── Message ─────────────────────────────────────────────────────────

function MessageLine({ t, msg, compact }: { t: Theme; msg: Msg; compact?: boolean }) {
  const [, pc, tc] = ROLE[msg.role](t)
  const glyph = msg.role === 'user' ? t.brand.prompt
    : msg.role === 'assistant' ? t.brand.tool
    : msg.role === 'tool' ? '⚡' : '!'

  return (
    <Box>
      <Box width={3}><Text bold={msg.role === 'user'} color={pc}>{glyph} </Text></Box>

      {msg.role === 'assistant'
        ? <Md t={t} text={msg.text} compact={compact} />
        : <Text color={tc}>{msg.text}</Text>}
    </Box>
  )
}


// ── App ─────────────────────────────────────────────────────────────

function App({ gw }: { gw: GatewayClient }) {
  const { exit }  = useApp()
  const { stdout } = useStdout()
  const cols       = stdout?.columns ?? 80

  const [input, setInput]       = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [messages, setMessages] = useState<Msg[]>([])
  const [status, setStatus]     = useState('summoning hermes…')
  const [sid, setSid]           = useState<string | null>(null)
  const [theme, setTheme]       = useState<Theme>(DEFAULT_THEME)
  const [info, setInfo]         = useState<SessionInfo | null>(null)
  const [thinking, setThinking] = useState(false)
  const [tools, setTools]       = useState<ActiveTool[]>([])
  const [busy, setBusy]         = useState(false)
  const [compact, setCompact]   = useState(false)
  const [usage, setUsage]       = useState<Usage>(ZERO)
  const [clarify, setClarify]   = useState<ClarifyReq | null>(null)
  const [approval, setApproval] = useState<ApprovalReq | null>(null)
  const [reasoning, setReasoning] = useState('')
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)
  const [scrollOffset, setScrollOffset] = useState(0)

  const buf      = useRef('')
  const stickyRef = useRef(true)
  const queueRef = useRef<string[]>([])
  const historyRef = useRef<string[]>([])
  const historyDraftRef = useRef('')
  const [historyIdx, setHistoryIdx] = useState<number | null>(null)
  const queueEditIdxRef = useRef<number | null>(null)
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const lastEmptySubmitAt = useRef(0)
  const empty = !messages.length

  const setQueueEdit = (idx: number | null) => {
    queueEditIdxRef.current = idx
    setQueueEditIdx(idx)
  }

  const enqueue = (text: string) => {
    queueRef.current.push(text)
    setQueuedDisplay([...queueRef.current])
  }

  const pushHistory = (text: string) => {
    const t = text.trim()
    if (!t) return
    const h = historyRef.current
    if (h.at(-1) !== t) h.push(t)
  }

  const replaceQueued = (idx: number, text: string) => {
    if (idx < 0 || idx >= queueRef.current.length) return
    queueRef.current[idx] = text
    setQueuedDisplay([...queueRef.current])
  }

  const removeQueued = (idx: number) => {
    if (idx < 0 || idx >= queueRef.current.length) return
    queueRef.current = queueRef.current.filter((_, i) => i !== idx)
    setQueuedDisplay([...queueRef.current])
  }

  const dequeue = () => {
    const [next, ...rest] = queueRef.current
    queueRef.current = rest
    setQueuedDisplay([...rest])
    return next
  }

  useEffect(() => { if (stickyRef.current) setScrollOffset(0) }, [messages.length])

  const termRows = stdout?.rows ?? 24
  const chromeRows = 2 + (empty ? 0 : 2) + (thinking ? 2 : 0) + 2
  const msgBudget = Math.max(3, termRows - chromeRows)

  const visibleSlice = useMemo(() => {
    if (!messages.length) return { start: 0, end: 0, above: 0 }
    const end = Math.max(0, messages.length - scrollOffset)
    const w = Math.max(20, cols - 5)
    let budget = msgBudget
    let start = end
    for (let i = end - 1; i >= 0 && budget > 0; i--) {
      const margin = messages[i]!.role === 'user' && i > 0 && messages[i - 1]?.role !== 'user' ? 1 : 0
      budget -= margin + estimateRows(messages[i]!.text, w)
      if (budget >= 0) start = i
    }
    return { start, end, above: start }
  }, [messages, scrollOffset, msgBudget, cols])

  const scrollUp = (n: number) => {
    setScrollOffset(prev => Math.min(Math.max(0, messages.length - 1), prev + n))
    stickyRef.current = false
  }
  const scrollDown = (n: number) => {
    setScrollOffset(prev => { const next = Math.max(0, prev - n); if (next === 0) stickyRef.current = true; return next })
  }
  const scrollBottom = () => { setScrollOffset(0); stickyRef.current = true }

  const sys      = useCallback((text: string) => setMessages(p => [...p, { role: 'system' as const, text }]), [])
  const idle     = () => { setThinking(false); setTools([]); setBusy(false); setClarify(null); setApproval(null); setReasoning('') }
  const die      = () => { gw.kill(); exit() }
  const clearIn  = () => { setInput(''); setInputBuf([]); setQueueEdit(null); setHistoryIdx(null); historyDraftRef.current = '' }
  const blocked  = !!(clarify || approval)

  // ── Hotkeys ───────────────────────────────────────────────────────

  useInput((ch, key) => {
    if (blocked) {
      if (key.ctrl && ch === 'c' && approval) {
        gw.request('approval.respond', { session_id: sid, choice: 'deny' }).catch(() => {})
        setApproval(null); sys('denied')
      }
      return
    }

    if (key.pageUp) { scrollUp(5); return }
    if (key.pageDown) { scrollDown(5); return }

    if (key.upArrow && !inputBuf.length) {
      if (queueRef.current.length) {
        const len = queueRef.current.length
        const idx = queueEditIdx === null ? 0 : (queueEditIdx + 1) % len
        setQueueEdit(idx)
        setHistoryIdx(null)
        setInput(queueRef.current[idx] ?? '')
        return
      }

      const h = historyRef.current
      if (!h.length) return
      const idx = historyIdx === null ? h.length - 1 : Math.max(0, historyIdx - 1)
      if (historyIdx === null) historyDraftRef.current = input
      setHistoryIdx(idx)
      setQueueEdit(null)
      setInput(h[idx] ?? '')
      return
    }

    if (key.downArrow && !inputBuf.length) {
      if (queueRef.current.length) {
        const len = queueRef.current.length
        const idx = queueEditIdx === null ? len - 1 : (queueEditIdx - 1 + len) % len
        setQueueEdit(idx)
        setHistoryIdx(null)
        setInput(queueRef.current[idx] ?? '')
        return
      }

      if (historyIdx === null) return
      const h = historyRef.current
      const next = historyIdx + 1
      if (next >= h.length) {
        setHistoryIdx(null)
        setInput(historyDraftRef.current)
      } else {
        setHistoryIdx(next)
        setInput(h[next] ?? '')
      }
      return
    }

    if (key.ctrl && ch === 'c') {
      if (busy && sid) {
        gw.request('session.interrupt', { session_id: sid }).catch(() => {})
        idle(); setStatus('interrupted'); sys('interrupted by user')
        setTimeout(() => setStatus('ready'), 1500)
      } else if (input || inputBuf.length) {
        clearIn()
      } else {
        die()
      }
      return
    }

    if (key.ctrl && ch === 'd') die()
    if (key.ctrl && ch === 'l') setMessages([])
    if (key.escape) clearIn()
  })

  // ── Gateway events ────────────────────────────────────────────────

  const onEvent = useCallback((ev: GatewayEvent) => {
    const p = ev.payload as any

    switch (ev.type) {
      case 'gateway.ready':
        if (p?.skin) setTheme(fromSkin(p.skin.colors ?? {}, p.skin.branding ?? {}))
        setStatus('forging session…')
        gw.request('session.create')
          .then((r: any) => { setSid(r.session_id); setStatus('ready') })
          .catch((e: Error) => setStatus(`error: ${e.message}`))
        break

      case 'session.info':    setInfo(p as SessionInfo); break
      case 'thinking.delta':  break
      case 'message.start':   setThinking(true); setBusy(true); setReasoning(''); setStatus('thinking…'); break
      case 'status.update':   if (p?.text) setStatus(p.text); break

      case 'reasoning.delta':
        if (p?.text) setReasoning(prev => prev + p.text)
        break

      case 'tool.generating':
        if (p?.name) setStatus(`preparing ${p.name}…`)
        break

      case 'tool.progress':
        if (p?.preview) setMessages(prev => {
          if (prev.at(-1)?.role === 'tool') return [...prev.slice(0, -1), { role: 'tool' as const, text: `${p.name}: ${p.preview}` }]
          return [...prev, { role: 'tool' as const, text: `${p.name}: ${p.preview}` }]
        })
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
        if (!p?.text) break
        buf.current += p.text
        setThinking(false); setTools([]); setReasoning('')
        setMessages(prev => upsert(prev, 'assistant', buf.current.trimStart()))
        break

      case 'message.complete':
        idle()
        setMessages(prev => upsert(prev, 'assistant', (p?.text ?? buf.current).trimStart()))
        buf.current = ''; setStatus('ready')
        if (p?.usage) setUsage(p.usage)
        if (p?.status === 'interrupted') sys('response interrupted')
        if (queueEditIdxRef.current !== null) break

        // drain queued message
        const next = dequeue()
        if (next) {
          setLastUserMsg(next)
          setMessages(prev => [...prev, { role: 'user' as const, text: next }])
          setStatus('thinking…'); setBusy(true); buf.current = ''
          gw.request('prompt.submit', { session_id: ev.session_id, text: next })
            .catch((e: Error) => { sys(`error: ${e.message}`); setStatus('ready'); setBusy(false) })
        }
        break

      case 'error':
        sys(`error: ${p?.message}`)
        idle(); setStatus('ready')
        break
    }
  }, [gw, sys])

  useEffect(() => {
    gw.on('event', onEvent)
    gw.on('exit', () => { setStatus('gateway exited'); exit() })
    return () => { gw.off('event', onEvent) }
  }, [gw, exit, onEvent])

  // ── Slash commands ────────────────────────────────────────────────

  const slash = useCallback((cmd: string): boolean => {
    const [name, ...rest] = cmd.slice(1).split(/\s+/)
    const arg = rest.join(' ')

    switch (name) {
      case 'help':
        sys([
          '  Commands:',
          ...COMMANDS.map(([c, d]) => `    ${c.padEnd(12)} ${d}`),
          '', '  Hotkeys:',
          ...HOTKEYS.map(([k, d]) => `    ${k.padEnd(12)} ${d}`),
        ].join('\n'))
        return true

      case 'clear':   setMessages([]); return true
      case 'quit': case 'exit': die(); return true

      case 'new':
        setStatus('forging session…')
        gw.request('session.create')
          .then((r: any) => {
            setSid(r.session_id); setMessages([]); setUsage(ZERO)
            setStatus('ready'); sys('new session started')
          })
          .catch((e: Error) => setStatus(`error: ${e.message}`))
        return true

      case 'undo':
        if (!sid) return true
        gw.request('session.undo', { session_id: sid })
          .then((r: any) => {
            if (r.removed > 0) {
              setMessages(p => { const q = [...p]; while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') q.pop(); if (q.at(-1)?.role === 'user') q.pop(); return q })
              sys(`undid ${r.removed} messages`)
            } else sys('nothing to undo')
          })
          .catch((e: Error) => sys(`error: ${e.message}`))
        return true

      case 'retry':
        if (!lastUserMsg) { sys('nothing to retry'); return true }
        setMessages(p => { const q = [...p]; while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') q.pop(); return q })
        setMessages(p => [...p, { role: 'user', text: lastUserMsg }])
        setStatus('thinking…'); setBusy(true); buf.current = ''
        gw.request('prompt.submit', { session_id: sid, text: lastUserMsg })
          .catch((e: Error) => { sys(`error: ${e.message}`); setStatus('ready'); setBusy(false) })
        return true

      case 'compact':
        if (arg) { setCompact(true); sys(`compact on, focus: ${arg}`) }
        else     { setCompact(c => !c); sys(`compact ${compact ? 'off' : 'on'}`) }
        return true

      case 'compress':
        if (!sid) return true
        gw.request('session.compress', { session_id: sid })
          .then((r: any) => { sys('context compressed'); if (r.usage) setUsage(r.usage) })
          .catch((e: Error) => sys(`error: ${e.message}`))
        return true

      case 'cost': case 'usage':
        sys(`in: ${fmtK(usage.input)}  out: ${fmtK(usage.output)}  total: ${fmtK(usage.total)}  calls: ${usage.calls}`)
        return true

      case 'copy': {
        const msgs = messages.filter(m => m.role === 'assistant')
        const target = msgs[arg ? Math.min(parseInt(arg), msgs.length) - 1 : msgs.length - 1]
        if (!target) { sys('nothing to copy'); return true }
        process.stdout.write(`\x1b]52;c;${Buffer.from(target.text).toString('base64')}\x07`)
        sys('copied to clipboard')
        return true
      }

      case 'context': {
        const pct = Math.min(100, Math.round((usage.total / MAX_CTX) * 100))
        const filled = Math.round((pct / 100) * 30)
        sys(`context: ${fmtK(usage.total)} / ${fmtK(MAX_CTX)} (${pct}%)\n[${'█'.repeat(filled)}${'░'.repeat(30 - filled)}] ${pct < 50 ? '✓' : pct < 80 ? '⚠' : '✗'}`)
        return true
      }

      case 'config':
        sys(`model: ${info?.model ?? '?'}  session: ${sid ?? 'none'}  compact: ${compact}\ntools: ${flat(info?.tools ?? {}).length}  skills: ${flat(info?.skills ?? {}).length}`)
        return true

      case 'status':
        sys(`session: ${sid ?? 'none'}  status: ${status}  tokens: ${fmtK(usage.input)}↑ ${fmtK(usage.output)}↓ (${usage.calls} calls)`)
        return true

      case 'skills':
        if (!info?.skills || !Object.keys(info.skills).length) { sys('no skills loaded'); return true }
        sys(Object.entries(info.skills).map(([k, vs]) => `${k}: ${vs.join(', ')}`).join('\n'))
        return true

      case 'model':
        if (!arg) { sys('usage: /model <name>'); return true }
        gw.request('config.set', { key: 'model', value: arg })
          .then(() => sys(`model → ${arg}`))
          .catch((e: Error) => sys(`error: ${e.message}`))
        return true

      case 'skin':
        if (!arg) { sys('usage: /skin <name>'); return true }
        gw.request('config.set', { key: 'skin', value: arg })
          .then(() => sys(`skin → ${arg} (restart to apply)`))
          .catch((e: Error) => sys(`error: ${e.message}`))
        return true

      default: return false
    }
  }, [gw, sid, status, sys, compact, info, usage, messages, lastUserMsg])

  // ── Submit ────────────────────────────────────────────────────────

  const submit = useCallback((value: string) => {
    if (!value.trim() && !inputBuf.length) {
      const now = Date.now()
      const dbl = now - lastEmptySubmitAt.current < 450
      lastEmptySubmitAt.current = now

      if (dbl && queueRef.current.length) {
        if (busy && sid) {
          gw.request('session.interrupt', { session_id: sid }).catch(() => {})
          setStatus('interrupting…')
          return
        }

        const next = dequeue()
        if (next && sid) {
          setQueueEdit(null)
          setLastUserMsg(next)
          setMessages(p => [...p, { role: 'user', text: next }])
          setStatus('thinking…'); setBusy(true); buf.current = ''
          gw.request('prompt.submit', { session_id: sid, text: next })
            .catch((e: Error) => { sys(`error: ${e.message}`); setStatus('ready'); setBusy(false) })
        }
      }
      return
    }
    lastEmptySubmitAt.current = 0

    if (value.endsWith('\\')) {
      setInputBuf(prev => [...prev, value.slice(0, -1)])
      setInput('')
      return
    }

    const full = [...inputBuf, value].join('\n')
    setInputBuf([])
    if (!full.trim() || !sid) return
    setInput('')
    setHistoryIdx(null)
    historyDraftRef.current = ''

    const editIdx = queueEditIdxRef.current

    // editing an already queued entry
    if (editIdx !== null && !full.startsWith('/') && !full.startsWith('!')) {
      replaceQueued(editIdx, full)
      setQueueEdit(null)
      return
    } else if (editIdx !== null) {
      setQueueEdit(null)
    }
    pushHistory(full)

    // queue if busy (slash commands still run immediately)
    if (busy && !full.startsWith('/') && !full.startsWith('!')) {
      enqueue(full)
      return
    }

    // !command → direct shell exec (entire line after !)
    if (full.startsWith('!')) {
      setMessages(p => [...p, { role: 'user', text: full }])
      setBusy(true); setStatus('running…')
      gw.request('shell.exec', { command: full.slice(1).trim() })
        .then((r: any) => {
          const out = [r.stdout, r.stderr].filter(Boolean).join('\n').trim()
          sys(out || `exit ${r.code}`)
          if (r.code !== 0 && out) sys(`exit ${r.code}`)
        })
        .catch((e: Error) => sys(`error: ${e.message}`))
        .finally(() => { setStatus('ready'); setBusy(false) })
      return
    }

    if (full.startsWith('/') && slash(full)) return

    // {!cmd} inline shell interpolation
    if (/\{!.+?\}/.test(full)) {
      setBusy(true); setStatus('interpolating…')
      const re = /\{!(.+?)\}/g
      const matches = [...full.matchAll(re)]
      Promise.all(matches.map(m =>
        gw.request('shell.exec', { command: m[1]! })
          .then((r: any) => [r.stdout, r.stderr].filter(Boolean).join('\n').trim().split('\n')[0] ?? '')
          .catch(() => '(error)')
      )).then(results => {
        let text = full
        for (let i = matches.length - 1; i >= 0; i--)
          text = text.slice(0, matches[i]!.index!) + results[i] + text.slice(matches[i]!.index! + matches[i]![0].length)
        setLastUserMsg(text)
        setMessages(p => [...p, { role: 'user', text }])
        setStatus('thinking…'); buf.current = ''
        gw.request('prompt.submit', { session_id: sid, text })
          .catch((e: Error) => { sys(`error: ${e.message}`); setStatus('ready'); setBusy(false) })
      })
      return
    }

    setLastUserMsg(full)
    setMessages(p => [...p, { role: 'user', text: full }])
    scrollBottom()
    setStatus('thinking…'); setBusy(true); buf.current = ''
    gw.request('prompt.submit', { session_id: sid, text: full })
      .catch((e: Error) => { sys(`error: ${e.message}`); setStatus('ready'); setBusy(false) })
  }, [gw, sid, slash, sys, inputBuf, busy])

  // ── Render ────────────────────────────────────────────────────────

  const statusColor = status === 'ready' ? theme.color.ok
    : status.startsWith('error') ? theme.color.error
    : status === 'interrupted' ? theme.color.warn
    : theme.color.dim
  const queueWindow = 3
  const queueStart = queueEditIdx === null
    ? 0
    : Math.max(0, Math.min(queueEditIdx - 1, Math.max(0, queuedDisplay.length - queueWindow)))
  const queueEnd = Math.min(queuedDisplay.length, queueStart + queueWindow)
  const queueShown = queuedDisplay.slice(queueStart, queueEnd)

  return (
    <AltScreen>
      <Box flexDirection="column" padding={1} flexGrow={1}>

        {empty ? (
          <>
            <Banner t={theme} />
            {info && <SessionPanel t={theme} info={info} />}

            {!sid
              ? <Text color={theme.color.dim}>⚕ {status}</Text>
              : <Text color={theme.color.dim}>
                  type <Text color={theme.color.amber}>/</Text> for commands
                  {' · '}<Text color={theme.color.amber}>!</Text> for shell
                  {' · '}<Text color={theme.color.amber}>Ctrl+C</Text> to interrupt
                </Text>}
          </>
        ) : (
          <Box marginBottom={1}>
            <Text bold color={theme.color.gold}>{theme.brand.icon} </Text>
            <Text bold color={theme.color.amber}>{theme.brand.name}</Text>

            <Text color={theme.color.dim}>
              {info?.model ? ` · ${info.model.split('/').pop()}` : ''}
              {' · '}<Text color={statusColor}>{status}</Text>
              {busy && ' · Ctrl+C to stop'}
            </Text>

            {usage.total > 0 && (
              <Text color={theme.color.dim}>
                {' · '}{fmtK(usage.input)}↑ {fmtK(usage.output)}↓ ({usage.calls} calls)
              </Text>
            )}
          </Box>
        )}

        <Box flexDirection="column" flexGrow={1} overflow="hidden">
          {visibleSlice.above > 0 && (
            <Text dimColor color={theme.color.dim}>↑ {visibleSlice.above} above · PgUp/PgDn to scroll</Text>
          )}
          {messages.slice(visibleSlice.start, visibleSlice.end).map((m, i) => {
            const ri = visibleSlice.start + i
            return (
              <Box key={ri} flexDirection="column" marginTop={m.role === 'user' && ri > 0 && messages[ri - 1]!.role !== 'user' ? 1 : 0}>
                <MessageLine t={theme} msg={m} compact={compact} />
              </Box>
            )
          })}
          {scrollOffset > 0 && (
            <Text dimColor color={theme.color.dim}>↓ {scrollOffset} below · PgDn or Enter to return</Text>
          )}
          {thinking && <Thinking t={theme} tools={tools} reasoning={reasoning} />}
        </Box>

        {clarify && (
          <ClarifyPrompt t={theme} req={clarify} onAnswer={answer => {
            gw.request('clarify.respond', { request_id: clarify.requestId, answer }).catch(() => {})
            setMessages(p => [...p, { role: 'user', text: answer }])
            setClarify(null); setStatus('thinking…')
          }} />
        )}

        {approval && (
          <ApprovalPrompt t={theme} req={approval} onChoice={choice => {
            gw.request('approval.respond', { session_id: sid, choice }).catch(() => {})
            setApproval(null)
            sys(choice === 'deny' ? 'denied' : `approved (${choice})`)
            setStatus('running…')
          }} />
        )}

        {!blocked && input.startsWith('/') && <CommandPalette t={theme} filter={input} />}

        {queuedDisplay.length > 0 && (
          <Box flexDirection="column">
            <Text dimColor color={theme.color.dim}>
              queued ({queuedDisplay.length}){queueEditIdx !== null ? ` · editing ${queueEditIdx + 1}` : ''}
            </Text>
            {queueStart > 0 && (
              <Text dimColor color={theme.color.dim}>  …</Text>
            )}
            {queueShown.map((q, i) => {
              const idx = queueStart + i
              const active = queueEditIdx === idx
              return (
                <Text key={`${idx}-${q.slice(0, 16)}`} dimColor color={active ? theme.color.amber : theme.color.dim}>
                  {active ? '▸' : ' '} {idx + 1}. {compactPreview(q, Math.max(16, cols - 10))}
                </Text>
              )
            })}
            {queueEnd < queuedDisplay.length && (
              <Text dimColor color={theme.color.dim}>
                {'  '}…and {queuedDisplay.length - queueEnd} more
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
              value={input}
              placeholder={empty ? PLACEHOLDER : busy ? 'Ctrl+C to interrupt…' : inputBuf.length ? 'continue (or Enter to send)' : ''}
              onChange={setInput}
              onSubmit={submit}
            />
          </Box>
        )}

      </Box>
    </AltScreen>
  )
}


function upsert(prev: Msg[], role: Role, text: string): Msg[] {
  return prev.at(-1)?.role === role
    ? [...prev.slice(0, -1), { role, text }]
    : [...prev, { role, text }]
}


if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY (run in a real terminal)')
  process.exit(0)
}

const gw = new GatewayClient()
gw.start()
render(<App gw={gw} />, { exitOnCtrlC: false })
