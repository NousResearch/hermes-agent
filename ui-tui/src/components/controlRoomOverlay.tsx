import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { type MenuRowSpec, useMenu } from './overlayPrimitives.js'

// ── Control Room snapshot wire types (mirror of control_room contract) ──

interface CrCounts {
  needs_you?: number
  agents_active?: number
  tasks_running?: number
  messages_unread?: number
  system_severity?: number
}

interface CrAttention {
  kind: string
  id: string
  severity: number
  title: string
  detail?: string
  profile?: string
}

interface CrRow {
  kind?: string
  id: string
  name?: string
  title?: string
  status?: string
  state?: string
  sender?: string
  detail?: string
  profile?: string
  available_actions?: string[]
}

interface CrCapabilities {
  approvals?: boolean
  peer_messages?: boolean
  kanban_actions?: boolean
  process_control?: boolean
  delegation_control?: boolean
}

interface CrSystem {
  state?: string
  severity?: number
  detail?: string
}

interface CrSnapshot {
  version: number
  profile: string
  generated_at: string
  attention: CrAttention[]
  counts: CrCounts
  agents: CrRow[]
  tasks: CrRow[]
  messages: CrRow[]
  system: CrSystem
  capabilities: CrCapabilities
}

// ── Constants ─────────────────────────────────────────────────────────

const SECTIONS = [
  { key: 'needs-you', label: 'Needs You' },
  { key: 'agents', label: 'Agents' },
  { key: 'tasks', label: 'Tasks' },
  { key: 'messages', label: 'Messages' },
  { key: 'system', label: 'System' }
] as const

type SectionKey = (typeof SECTIONS)[number]['key']

const SEVERITY_GLYPH = ['!!', '!', '~', '·'] as const

// ── Component ─────────────────────────────────────────────────────────

export function ControlRoomOverlay({ gw, onClose, t }: { gw: GatewayClient; onClose: () => void; t: Theme }) {
  const [snapshot, setSnapshot] = useState<CrSnapshot | null>(null)
  const [error, setError] = useState<string>('')
  const [section, setSection] = useState<SectionKey | null>(null)
  const [loading, setLoading] = useState(true)

  // Fetch the read-only snapshot on open (bounded server-side cache).
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    gw.request<CrSnapshot>('control.room.snapshot', { profile: 'default' })
      .then(raw => {
        const snap = asRpcResult<CrSnapshot>(raw)

        if (snap && !cancelled) {
          setSnapshot(snap)
          setError('')
        } else if (!cancelled) {
          setError('snapshot unavailable')
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {setError(e instanceof Error ? e.message : String(e))}
      })
      .finally(() => {
        if (!cancelled) {setLoading(false)}
      })

    return () => {
      cancelled = true
    }
  }, [gw])

  // Ctrl+P / Esc close.
  useInput((ch, key) => {
    if (key.escape || (key.ctrl && ch.toLowerCase() === 'p')) {
      onClose()
    }
  })

  if (loading && !snapshot) {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text color={t.color.primary}>KENSEI › Control Room</Text>
        <Text color={t.color.muted}>loading…</Text>
      </Box>
    )
  }

  if (error && !snapshot) {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text color={t.color.primary}>KENSEI › Control Room</Text>
        <Text color={t.color.error}>unavailable: {error}</Text>
        <Text color={t.color.muted}>Esc to close</Text>
      </Box>
    )
  }

  if (!snapshot) {
    return (
      <Box paddingX={2} paddingY={1}>
        <Text color={t.color.muted}>Control Room unavailable</Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      {section ? (
        <SectionView onBack={() => setSection(null)} onClose={onClose} section={section} snapshot={snapshot} t={t} />
      ) : (
        <HomeView onClose={onClose} onOpen={setSection} snapshot={snapshot} t={t} />
      )}
    </Box>
  )
}

// ── Home ──────────────────────────────────────────────────────────────

function HomeView({
  snapshot,
  t,
  onOpen,
  onClose
}: {
  snapshot: CrSnapshot
  t: Theme
  onOpen: (s: SectionKey) => void
  onClose: () => void
}) {
  const rows: MenuRowSpec[] = useMemo(() => {
    const c = snapshot.counts ?? {}

    const countFor = (key: SectionKey): string => {
      switch (key) {
        case 'needs-you':
          return `${c.needs_you ?? 0}`

        case 'agents':
          return `${c.agents_active ?? 0} active`

        case 'tasks':
          return `${c.tasks_running ?? 0} running`

        case 'messages':
          return `${c.messages_unread ?? 0} unread`

        case 'system':
          return snapshot.system?.state ?? 'unknown'
      }
    }

    return SECTIONS.map(s => ({
      label: `› ${s.label.padEnd(12)} ${countFor(s.key)}`,
      run: () => onOpen(s.key)
    }))
  }, [snapshot, onOpen])

  const sel = useMenu(rows, onClose)

  return (
    <Box flexDirection="column">
      <Text color={t.color.primary}>KENSEI › Control Room (profile: {snapshot.profile})</Text>
      <Box marginY={1} />
      {rows.map((row, i) => (
        <Text color={i === sel ? t.color.accent : t.color.text} key={row.label}>
          {i === sel ? '›' : ' '} {row.label}
        </Text>
      ))}
      <Box marginY={1} />
      <Text color={t.color.muted}>↑↓ navigate · Enter open · Esc close</Text>
    </Box>
  )
}

// ── Sections ──────────────────────────────────────────────────────────

function SectionView({
  snapshot,
  section,
  t,
  onBack,
  onClose
}: {
  snapshot: CrSnapshot
  section: SectionKey
  t: Theme
  onBack: () => void
  onClose: () => void
}) {
  const rows = useMemo<MenuRowSpec[]>(() => {
    switch (section) {
      case 'needs-you':
        return snapshot.attention
          .filter(a => a.severity <= 1)
          .map(a => ({ label: `${SEVERITY_GLYPH[a.severity] ?? '·'} ${a.title}`, run: () => {} }))

      case 'agents':
        return snapshot.agents.map(a => ({
          label: `[${a.kind ?? 'agent'}] ${a.name ?? a.id} · ${a.status ?? ''}${a.detail ? ` · ${a.detail}` : ''}`,
          run: () => {}
        }))

      case 'tasks':
        return snapshot.tasks.map(task => ({
          label: `${task.id} · ${task.title ?? ''} · ${task.state ?? ''}`,
          run: () => {}
        }))

      case 'messages':
        return snapshot.messages.map(m => ({
          label: `[${m.state ?? 'queued'}] ${m.title ?? ''}${m.sender ? ` from ${m.sender}` : ''}`,
          run: () => {}
        }))

      case 'system':
        return [
          { label: `state: ${snapshot.system?.state ?? 'unknown'}`, run: () => {} },
          { label: snapshot.system?.detail ?? '', run: () => {} }
        ]
    }
  }, [snapshot, section])

  const sel = useMenu(rows, onBack)

  return (
    <Box flexDirection="column">
      <Text color={t.color.primary}>KENSEI › Control Room › {section}</Text>
      <Box marginY={1} />
      {rows.length === 0 ? (
        <Text color={t.color.muted}>nothing here</Text>
      ) : (
        rows.map((row, i) => (
          <Text color={i === sel ? t.color.accent : t.color.text} key={`${section}-${i}`}>
            {i === sel ? '›' : ' '} {row.label}
          </Text>
        ))
      )}
      <Box marginY={1} />
      <Text color={t.color.muted}>↑↓ navigate · Enter select · Esc back</Text>
    </Box>
  )
}
