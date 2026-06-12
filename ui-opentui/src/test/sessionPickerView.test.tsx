/**
 * Resume-picker overlay tests (design doc §B item 6) — headless frames with a
 * simulated keyboard through the REAL component against fake gateway ops:
 * opens on the Recent tab querying the interactive sources; Tab cycling
 * re-queries with each tab's `sources`; typing filters CLIENT-side (no
 * re-query); Space opens the peek preview and issues exactly ONE peek per
 * settled highlight (stale responses dropped); Enter resumes the highlighted
 * id; Ctrl+R renames inline via the rename op; `initialTab` pre-selects;
 * `truncated: true` renders the honesty row; a full page offers "load more"
 * which fetches the next offset; Esc closes (but only cancels an open rename).
 */
import { describe, expect, test } from 'vitest'

import { INTERACTIVE_SOURCES, PLATFORM_SOURCES } from '../logic/sessionPicker.ts'
import { DEFAULT_THEME } from '../logic/theme.ts'
import { SessionPicker, type SessionPickerOps } from '../view/overlays/sessionPicker.tsx'
import { ThemeProvider } from '../view/theme.tsx'
import { renderProbe, type RenderProbe } from './lib/render.ts'

const NOW_S = Math.floor(Date.now() / 1000)

interface FakeSession {
  id: string
  title?: string
  preview?: string
  source?: string
  message_count?: number
  started_at?: number
  last_active?: number
  cwd?: string
}

const SESSIONS: FakeSession[] = [
  {
    cwd: '/home/u/worktrees/lively-thrush/hermes-agent',
    id: 's1',
    last_active: NOW_S - 3600,
    message_count: 142,
    preview: 'rebuild the TUI',
    source: 'tui',
    started_at: NOW_S - 7200,
    title: 'Adopt OpenTUI paradigm'
  },
  {
    id: 's2',
    last_active: NOW_S - 60,
    message_count: 9,
    preview: 'goal work',
    source: 'cli',
    started_at: NOW_S - 120,
    title: 'goal-v4'
  },
  {
    id: 's3',
    last_active: NOW_S - 86_400,
    message_count: 3,
    preview: 'an older chat',
    source: 'tui',
    started_at: NOW_S - 90_000,
    title: 'older session'
  }
]

interface Harness {
  probe: RenderProbe
  listCalls: Record<string, unknown>[]
  peekCalls: string[]
  renameCalls: Array<{ id: string; title: string }>
  resumed: string[]
  closed: { value: boolean }
}

interface MountOptions {
  sessions?: FakeSession[]
  truncated?: boolean
  initialTab?: 'recent' | 'cron' | 'gateways' | 'all'
  /** The TUI's cwd — drives the this-directory-first grouping. */
  currentCwd?: string
  /** Per-id peek delay (ms) — drives the stale-cancellation test. */
  peekDelay?: (id: string) => number
  /** Paged list: serve `sessions` windowed by offset/limit instead of whole. */
  paged?: boolean
}

async function mountPicker(options: MountOptions = {}): Promise<Harness> {
  const sessions = options.sessions ?? SESSIONS
  const listCalls: Record<string, unknown>[] = []
  const peekCalls: string[] = []
  const renameCalls: Array<{ id: string; title: string }> = []
  const resumed: string[] = []
  const closed = { value: false }
  const ops: SessionPickerOps = {
    list: params => {
      listCalls.push(params)
      const offset = typeof params.offset === 'number' ? params.offset : 0
      const limit = typeof params.limit === 'number' ? params.limit : sessions.length
      const page = options.paged ? sessions.slice(offset, offset + limit) : sessions
      return Promise.resolve({ sessions: page, truncated: options.truncated ?? false })
    },
    peek: id => {
      peekCalls.push(id)
      const payload = {
        head: [{ content: `peek-head-${id}`, role: 'user', truncated: false }],
        session: { cwd: '/home/u/proj', id, message_count: 5, model: 'hermes-4', title: 'x' },
        tail: [{ content: `peek-tail-${id}`, role: 'assistant', truncated: false }],
        total_messages: 5
      }
      const delay = options.peekDelay?.(id) ?? 0
      if (!delay) return Promise.resolve(payload)
      return new Promise(resolve => setTimeout(() => resolve(payload), delay))
    },
    rename: (id, title) => {
      renameCalls.push({ id, title })
      return Promise.resolve()
    }
  }
  const probe = await renderProbe(
    () => (
      <ThemeProvider theme={() => DEFAULT_THEME}>
        <SessionPicker
          ops={ops}
          initialTab={options.initialTab ?? 'recent'}
          currentCwd={() => options.currentCwd}
          peekDebounceMs={20}
          onResume={id => resumed.push(id)}
          onClose={() => (closed.value = true)}
        />
      </ThemeProvider>
    ),
    // kitty keyboard so a SIMULATED lone Esc parses (see lib/render.ts)
    { height: 30, kittyKeyboard: true, width: 90 }
  )
  // the initial session.list resolves async — wait for the rows (or the empty
  // state) to paint before handing the probe to the test.
  await probe.waitForFrame(f => !f.includes('loading…'))
  return { closed, listCalls, peekCalls, probe, renameCalls, resumed }
}

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

describe('SessionPicker — open + rows', () => {
  test('opens on the Recent tab, queries the interactive sources, renders title + meta rows', async () => {
    const h = await mountPicker()
    try {
      expect(h.listCalls).toHaveLength(1)
      expect(h.listCalls[0]).toEqual({ limit: 100, offset: 0, sources: [...INTERACTIVE_SOURCES] })
      const frame = h.probe.frame()
      expect(frame).toContain('Resume session (3 of 3)')
      expect(frame).toContain('[ Recent ]') // active chip
      expect(frame).toContain('❯ Adopt OpenTUI paradigm') // first row selected
      expect(frame).toContain('1 hour ago · tui · 142 msgs · …') // meta line w/ truncated cwd
      expect(frame).toContain('lively-thrush/hermes-agent')
      expect(frame).toContain('1 minute ago · cli · 9 msgs') // cwd-less meta
      expect(frame).toContain('↑↓ select · Enter resume · Tab scope · Space preview · Ctrl+R rename · Esc cancel')
    } finally {
      h.probe.destroy()
    }
  })

  test('initialTab cron pre-selects the chip and queries sources [cron] (/sessions cron)', async () => {
    const h = await mountPicker({ initialTab: 'cron', sessions: [] })
    try {
      expect(h.listCalls[0]).toEqual({ limit: 100, offset: 0, sources: ['cron'] })
      const frame = h.probe.frame()
      expect(frame).toContain('[ Cron ]')
      expect(frame).toContain('(no sessions on this tab)')
    } finally {
      h.probe.destroy()
    }
  })

  test('truncated: true renders the honesty row', async () => {
    const h = await mountPicker({ truncated: true })
    try {
      expect(h.probe.frame()).toContain('results truncated — narrow the search')
    } finally {
      h.probe.destroy()
    }
  })
})

describe('SessionPicker — tab cycling re-queries', () => {
  test('Tab walks Recent→Cron→Gateways→All with each tab’s sources; Shift+Tab wraps back', async () => {
    const h = await mountPicker()
    try {
      h.probe.keys.pressTab()
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('[ Cron ]'))
      expect(h.listCalls[1]).toMatchObject({ offset: 0, sources: ['cron'] })
      h.probe.keys.pressTab()
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('[ Gateways ]'))
      expect(h.listCalls[2]).toMatchObject({ sources: [...PLATFORM_SOURCES] })
      h.probe.keys.pressTab()
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('[ All ]'))
      expect(h.listCalls[3]).toEqual({ limit: 100, offset: 0 }) // All: no sources filter
      h.probe.keys.pressTab({ shift: true }) // wrap back → Gateways
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('[ Gateways ]'))
      expect(h.listCalls).toHaveLength(5)
    } finally {
      h.probe.destroy()
    }
  })
})

describe('SessionPicker — search filters client-side', () => {
  test('typing narrows rows (title/cwd match) without re-querying the gateway', async () => {
    const h = await mountPicker()
    try {
      await h.probe.keys.typeText('goal')
      await h.probe.settle()
      const frame = await h.probe.waitForFrame(f => !f.includes('Adopt OpenTUI'))
      expect(frame).toContain('❯ goal-v4')
      expect(frame).toContain('Resume session (1 of 3)')
      expect(h.listCalls).toHaveLength(1) // no re-query — pure client filter
      for (let i = 0; i < 4; i++) h.probe.keys.pressBackspace()
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('Adopt OpenTUI'))
    } finally {
      h.probe.destroy()
    }
  })
})

describe('SessionPicker — Enter resumes', () => {
  test('↓ then Enter resumes the highlighted session id', async () => {
    const h = await mountPicker()
    try {
      h.probe.keys.pressArrow('down')
      await h.probe.settle()
      expect(h.probe.frame()).toContain('❯ goal-v4')
      h.probe.keys.pressEnter()
      await h.probe.settle()
      expect(h.resumed).toEqual(['s2'])
    } finally {
      h.probe.destroy()
    }
  })
})

describe('SessionPicker — Space preview (peek)', () => {
  test('Space opens the pane with ONE immediate peek; Space again closes it', async () => {
    const h = await mountPicker()
    try {
      h.probe.keys.pressKey(' ')
      await h.probe.settle()
      const frame = await h.probe.waitForFrame(f => f.includes('peek-head-s1'))
      expect(frame).toContain('preview (Space)')
      expect(frame).toContain('hermes-4') // session meta line
      expect(frame).toContain('peek-tail-s1')
      expect(h.peekCalls).toEqual(['s1'])
      // Space never types into the search input (it toggles the pane instead)
      expect(frame).toContain('Resume session (3 of 3)')
      h.probe.keys.pressKey(' ')
      await h.probe.settle()
      expect(h.probe.frame()).not.toContain('preview (Space)')
      expect(h.peekCalls).toEqual(['s1'])
    } finally {
      h.probe.destroy()
    }
  })

  test('scrolling while open debounces to ONE peek for the settled row; stale responses are dropped', async () => {
    // s1's response is SLOW — it resolves after s3's and must be discarded.
    const h = await mountPicker({ peekDelay: id => (id === 's1' ? 120 : 0) })
    try {
      h.probe.keys.pressKey(' ') // immediate peek for s1 (slow)
      await h.probe.settle()
      h.probe.keys.pressArrow('down') // s2 — debounced…
      h.probe.keys.pressArrow('down') // …superseded by s3 within the window
      await h.probe.settle()
      await wait(60) // debounce (20ms) fires once for s3
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('peek-head-s3'))
      expect(h.peekCalls).toEqual(['s1', 's3']) // exactly one peek per settled highlight
      await wait(100) // let the stale s1 response land
      await h.probe.settle()
      expect(h.probe.frame()).toContain('peek-head-s3') // not clobbered by stale s1
      expect(h.probe.frame()).not.toContain('peek-head-s1')
    } finally {
      h.probe.destroy()
    }
  })
})

describe('SessionPicker — Ctrl+R inline rename', () => {
  test('Ctrl+R opens the prefilled rename line; Enter commits via session.title and updates the row', async () => {
    const h = await mountPicker()
    try {
      h.probe.keys.pressKey('r', { ctrl: true })
      await h.probe.settle()
      let frame = await h.probe.waitForFrame(f => f.includes('rename:'))
      expect(frame).toContain('Enter save · Esc cancel rename')
      await h.probe.keys.typeText(' v2') // appends to the prefilled current title
      await h.probe.settle()
      h.probe.keys.pressEnter()
      await h.probe.settle()
      expect(h.renameCalls).toEqual([{ id: 's1', title: 'Adopt OpenTUI paradigm v2' }])
      expect(h.resumed).toEqual([]) // Enter committed the rename, not a resume
      frame = await h.probe.waitForFrame(f => f.includes('Adopt OpenTUI paradigm v2'))
      expect(frame).not.toContain('rename:')
    } finally {
      h.probe.destroy()
    }
  })

  test('Esc cancels the rename WITHOUT closing the picker; a later Esc closes it', async () => {
    const h = await mountPicker()
    try {
      h.probe.keys.pressKey('r', { ctrl: true })
      await h.probe.settle()
      await h.probe.waitForFrame(f => f.includes('rename:'))
      h.probe.keys.pressEscape()
      await h.probe.settle()
      const frame = await h.probe.waitForFrame(f => !f.includes('rename:'))
      expect(h.closed.value).toBe(false) // picker stayed open
      expect(frame).toContain('Resume session')
      await wait(60) // the once-per-press Esc guard window
      h.probe.keys.pressEscape()
      await h.probe.settle()
      await h.probe.settle()
      expect(h.closed.value).toBe(true)
      expect(h.renameCalls).toEqual([])
    } finally {
      h.probe.destroy()
    }
  })
})

describe('SessionPicker — pagination', () => {
  test('a full page offers “load more”; Enter on it fetches the next offset', async () => {
    // 100 rows fill page one exactly → the load-more row appears.
    const many: FakeSession[] = Array.from({ length: 120 }, (_, i) => ({
      id: `bulk-${i}`,
      last_active: NOW_S - i * 60,
      message_count: i,
      source: 'tui',
      started_at: NOW_S - i * 60,
      title: `bulk session ${i}`
    }))
    const h = await mountPicker({ paged: true, sessions: many })
    try {
      let frame = h.probe.frame()
      expect(frame).toContain('Resume session (100 of 100+)') // page full → maybe more
      h.probe.keys.pressArrow('up') // wrap to the LAST selectable = the load-more row (scrolls into view)
      await h.probe.settle()
      expect(h.probe.frame()).toContain('❯ ↓ load more (100 loaded)')
      h.probe.keys.pressEnter()
      await h.probe.settle()
      frame = await h.probe.waitForFrame(f => f.includes('(120 of 120)'))
      expect(h.listCalls).toHaveLength(2)
      expect(h.listCalls[1]).toMatchObject({ limit: 100, offset: 100 })
      expect(frame).not.toContain('load more') // short second page → exhausted
      expect(h.resumed).toEqual([]) // Enter loaded, never resumed
    } finally {
      h.probe.destroy()
    }
  })
})

describe('this-directory grouping', () => {
  const CWD = '/home/u/projects/alpha'
  const GROUPED: FakeSession[] = [
    { cwd: '/elsewhere/beta', id: 'g1', last_active: NOW_S - 60, message_count: 5, source: 'tui', title: 'beta work' },
    { cwd: CWD, id: 'g2', last_active: NOW_S - 120, message_count: 8, source: 'tui', title: 'alpha here' },
    { id: 'g3', last_active: NOW_S - 180, message_count: 2, source: 'cli', title: 'no cwd at all' },
    {
      cwd: CWD + '/',
      id: 'g4',
      last_active: NOW_S - 240,
      message_count: 4,
      source: 'tui',
      title: 'alpha trailing slash'
    }
  ]

  test('sessions started in the current cwd group first under a caption', async () => {
    const h = await mountPicker({ currentCwd: CWD, sessions: GROUPED })
    try {
      const frame = h.probe.frame()
      const at = (needle: string) => frame.indexOf(needle)
      expect(at('▾ this directory (2)')).toBeGreaterThanOrEqual(0)
      expect(at('▾ other directories')).toBeGreaterThan(at('▾ this directory (2)'))
      // here-rows (incl. trailing-slash normalization) above the caption split,
      // elsewhere rows below it; selection starts on the first here-row.
      expect(at('alpha here')).toBeLessThan(at('▾ other directories'))
      expect(at('alpha trailing slash')).toBeLessThan(at('▾ other directories'))
      expect(at('beta work')).toBeGreaterThan(at('▾ other directories'))
      expect(at('no cwd at all')).toBeGreaterThan(at('▾ other directories'))
      expect(frame).toContain('❯ alpha here')
      // Enter resumes the first here-session
      h.probe.keys.pressEnter()
      await h.probe.settle()
      expect(h.resumed).toEqual(['g2'])
    } finally {
      h.probe.destroy()
    }
  })

  test('search drops the grouping (fuzzy relevance owns the order)', async () => {
    const h = await mountPicker({ currentCwd: CWD, sessions: GROUPED })
    try {
      await h.probe.keys.typeText('beta')
      await h.probe.settle()
      const frame = await h.probe.waitForFrame(f => f.includes('beta work'))
      expect(frame).not.toContain('▾ this directory')
      expect(frame).not.toContain('▾ other directories')
    } finally {
      h.probe.destroy()
    }
  })

  test('no current cwd (or no matches) → plain recency list, no captions', async () => {
    const h = await mountPicker({ sessions: GROUPED })
    try {
      expect(h.probe.frame()).not.toContain('▾ this directory')
    } finally {
      h.probe.destroy()
    }
  })
})
