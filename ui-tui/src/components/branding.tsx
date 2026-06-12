import { Box, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import { artWidth, caduceus, CADUCEUS_WIDTH, logo, LOGO_WIDTH } from '../banner.js'
import { flat } from '../lib/text.js'
import { useGateway } from '../app/gatewayContext.js'
import { $uiSessionId } from '../app/uiStore.js'
import type { Theme } from '../theme.js'
import type { PanelSection, QuotaInfo, QuotaModel, SessionInfo } from '../types.js'

const LOADER_TICK_MS = 120
// How often the SessionPanel re-fetches quota data. Backend caches the
// result for 60s; this client-side timer just controls the visible
// re-render cadence. 5s keeps the progress bar feeling "live" without
// hammering the upstream billing API.
const QUOTA_TICK_MS = 5000

function InlineLoader({ label, t }: { label: string; t: Theme }) {
  const [tick, setTick] = useState(0)
  const spinner = unicodeSpinners.braille
  const frame = spinner.frames[tick % spinner.frames.length] ?? '⠋'

  useEffect(() => {
    const id = setInterval(() => setTick(n => n + 1), Math.max(LOADER_TICK_MS, spinner.interval))

    return () => clearInterval(id)
  }, [spinner.interval])

  return (
    <Text color={t.color.muted} wrap="truncate">
      <Text color={t.color.accent}>{frame}</Text> {label}
    </Text>
  )
}

export function ArtLines({ lines }: { lines: [string, string][] }) {
  return (
    <Box flexDirection="column" height={lines.length} opaque width={artWidth(lines)}>
      {lines.map(([c, text], i) => (
        <Text color={c} key={i} wrap="truncate-end">
          {text}
        </Text>
      ))}
    </Box>
  )
}

// Responsive Banner: full art → compact rule → text → hidden.
//
// Terminals can't scale glyphs, so "responsive" means picking a layout that
// fits the available columns. Thresholds are picked so each tier reads
// comfortably without forcing wrap or truncation drift on box-drawing edges.
const TAG_FULL = 'Nous Research · Messenger of the Digital Gods'
const TAG_MID = 'Messenger of the Digital Gods'
const TAG_TINY = 'Nous Research'
const HIDE_BELOW = 34
const COMPACT_FROM = 58

const clip = (s: string, w: number) =>
  w <= 0 ? '' : s.length > w ? `${s.slice(0, Math.max(0, w - 1))}…` : s

const centerIn = (s: string, w: number) => {
  const f = clip(s, w)
  const slack = Math.max(0, w - f.length)
  const left = slack >> 1

  return `${' '.repeat(left)}${f}${' '.repeat(slack - left)}`
}

const ruleIn = (label: string, w: number) => {
  const f = clip(label, Math.max(1, w - 4))
  const slack = Math.max(0, w - f.length - 2)
  const left = slack >> 1

  return `${'─'.repeat(left)} ${f} ${'─'.repeat(slack - left)}`
}

function CompactBanner({ cols, t }: { cols: number; t: Theme }) {
  // -4 keeps a margin so exact-edge rows don't trip terminal pending-wrap.
  const w = Math.max(28, cols - 4)

  return (
    <Box flexDirection="column" height={3} marginBottom={1} opaque width={w}>
      <Text bold color={t.color.primary}>{ruleIn(t.brand.name, w)}</Text>
      <Text color={t.color.muted}>{centerIn(TAG_FULL, w)}</Text>
      <Text color={t.color.primary}>{'─'.repeat(w)}</Text>
    </Box>
  )
}

export function Banner({ maxWidth, t }: { maxWidth?: number; t: Theme }) {
  const term = useStdout().stdout?.columns ?? 80
  const cols = Math.max(1, Math.min(term, maxWidth ?? term))

  if (cols < HIDE_BELOW) {
    return null
  }

  const logoLines = logo(t.color, t.bannerLogo || undefined)
  const logoW = t.bannerLogo ? artWidth(logoLines) : LOGO_WIDTH

  if (cols >= logoW + 2) {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <ArtLines lines={logoLines} />
        <Text color={t.color.muted} wrap="truncate-end">
          {t.brand.icon} {TAG_FULL}
        </Text>
      </Box>
    )
  }

  if (cols >= COMPACT_FROM) {
    return <CompactBanner cols={cols} t={t} />
  }

  const name = cols >= 52 ? t.brand.name : (t.brand.name.split(' ')[0] ?? t.brand.name)
  const tag = cols >= 64 ? TAG_FULL : cols >= 46 ? TAG_MID : TAG_TINY

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color={t.color.primary} wrap="truncate-end">{t.brand.icon} {name}</Text>
      <Text color={t.color.muted} wrap="truncate-end">{t.brand.icon} {tag}</Text>
    </Box>
  )
}

// ── Quota line ──────────────────────────────────────────────────────
// Compact one-liner summarising the current model's plan quota.
// Renders nothing if quota is missing / unsupported / failed.
//
// Coding plan example:
//   ▰▰▰▰▰▰▰▰▰▱ 94% · 5h窗口 2h58m · 本周99%
// Credit (OpenRouter) example:
//   $7.42 / $50.00 used · 14.8% left

const STATUS_LABELS: Record<number, string> = {
  1: '正常',
  2: '告急',
  3: '充足',
  4: '耗尽',
}

function formatDuration(secs: number): string {
  secs = Math.max(0, Math.floor(secs))
  if (secs <= 0) return '已结束'
  const d = Math.floor(secs / 86400)
  const h = Math.floor((secs % 86400) / 3600)
  const m = Math.floor((secs % 3600) / 60)
  if (d > 0) return `${d}天${h}h`
  if (h > 0) return `${h}h${m}m`
  return `${m}m`
}

function progressBar(pct: number, width: number = 10): string {
  // Unicode block bar; avoids needing colored segments (one color for the bar).
  const clamped = Math.max(0, Math.min(100, pct))
  const filled = Math.round((clamped / 100) * width)
  return '▰'.repeat(filled) + '▱'.repeat(Math.max(0, width - filled))
}

function pctColor(pct: number | null, t: Theme): string {
  if (pct == null) return t.color.muted
  if (pct >= 60) return t.color.ok
  if (pct >= 30) return t.color.warn
  return t.color.error
}

function QuotaLine({ quota, t }: { quota?: QuotaInfo; t: Theme }) {
  if (!quota || !quota.supported) return null
  if (quota.kind === 'coding_plan') {
    const primary = quota.primary ?? (quota.models ?? [])[0]
    if (!primary) return null
    const ivPct = primary.interval_remaining_percent ?? 0
    const wkPct = primary.weekly_remaining_percent ?? 0
    const ivLeft = formatDuration(primary.interval_remaining_s)
    const wkLeft = formatDuration(primary.weekly_remaining_s)
    const status = STATUS_LABELS[primary.interval_status] ?? ''
    // The "since <timestamp>" suffix is the freshness indicator. Click on
    // this prefix (Ctrl+C not required) triggers an immediate refresh — the
    // SessionPanel wires that up via the click handler below.
    return (
      <Text wrap="truncate-end">
        <Text color={t.color.muted}>{quota.provider} · </Text>
        <Text color={pctColor(ivPct, t)} bold>{progressBar(ivPct, 10)}</Text>
        <Text color={pctColor(ivPct, t)}> {ivPct}%</Text>
        <Text color={t.color.muted}> · 5h </Text>
        <Text color={t.color.text}>{ivLeft}</Text>
        <Text color={t.color.muted}> · 本周 </Text>
        <Text color={pctColor(wkPct, t)}>{wkPct}%</Text>
        <Text color={t.color.muted}> ({wkLeft})</Text>
        {status ? <Text color={t.color.muted}> · {status}</Text> : null}
        <QuotaAge suffix={quota.fetched_at_s} t={t} />
      </Text>
    )
  }
  if (quota.kind === 'credit') {
    const used = quota.used ?? 0
    const limit = quota.limit
    if (limit == null) {
      return (
        <Text wrap="truncate-end">
          <Text color={t.color.muted}>{quota.provider} · </Text>
          <Text color={t.color.text}>${used.toFixed(4)}</Text>
          <Text color={t.color.muted}> used · no hard cap</Text>
          <QuotaAge suffix={quota.fetched_at_s} t={t} />
        </Text>
      )
    }
    const left = Math.max(0, limit - used)
    const leftPct = limit > 0 ? (left / limit) * 100 : 0
    return (
      <Text wrap="truncate-end">
        <Text color={t.color.muted}>{quota.provider} · </Text>
        <Text color={pctColor(leftPct, t)} bold>{progressBar(leftPct, 10)}</Text>
        <Text color={pctColor(leftPct, t)}> ${left.toFixed(2)}</Text>
        <Text color={t.color.muted}> / ${limit.toFixed(2)} left</Text>
        {quota.is_free_tier ? <Text color={t.color.muted}> · free tier</Text> : null}
        <QuotaAge suffix={quota.fetched_at_s} t={t} />
      </Text>
    )
  }
  return null
}

// Tiny " · 5s ago" suffix so users can tell how stale the data is.
// Hidden when fetched_at_s is missing or 0.
function QuotaAge({ suffix, t }: { suffix?: number; t: Theme }) {
  if (!suffix || suffix <= 0) return null
  const age = Math.max(0, Math.floor(Date.now() / 1000) - suffix)
  let label: string
  if (age < 5) label = 'just now'
  else if (age < 60) label = `${age}s ago`
  else if (age < 3600) label = `${Math.floor(age / 60)}m ago`
  else label = `${Math.floor(age / 3600)}h ago`
  return <Text color={t.color.muted}> · {label}</Text>
}

// ── Collapsible helpers ──────────────────────────────────────────────

function CollapseToggle({
  count,
  open,
  suffix,
  t,
  title,
  onToggle
}: {
  count?: number
  open: boolean
  suffix?: string
  t: Theme
  title: string
  onToggle: () => void
}) {
  return (
    <Box onClick={onToggle}>
      <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      {typeof count === 'number' ? (
        <Text color={t.color.muted}> ({count})</Text>
      ) : null}
      {suffix ? (
        <Text color={t.color.muted}> {suffix}</Text>
      ) : null}
    </Box>
  )
}

// ── SessionPanel ─────────────────────────────────────────────────────

const SKILLS_MAX = 8
const TOOLSETS_MAX = 8

export function SessionPanel({ info, maxWidth, sid, t }: SessionPanelProps) {
  const term = useStdout().stdout?.columns ?? 100
  const cols = Math.max(20, Math.min(term, maxWidth ?? term))
  const heroLines = caduceus(t.color, t.bannerHero || undefined)
  const leftW = Math.min((artWidth(heroLines) || CADUCEUS_WIDTH) + 4, Math.floor(cols * 0.4))
  const wide = cols >= 90 && leftW + 40 < cols
  const w = Math.max(20, wide ? cols - leftW - 14 : cols - 12)
  const lineBudget = Math.max(12, w - 2)
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

  // ── Local collapse state for each section ──
  const [toolsOpen, setToolsOpen] = useState(true)
  const [skillsOpen, setSkillsOpen] = useState(false)
  const [systemOpen, setSystemOpen] = useState(false)
  const [mcpOpen, setMcpOpen] = useState(false)

  // ── Quota auto-refresh ──
  // Poll quota.refresh every QUOTA_TICK_MS so the percentage bar ticks down
  // smoothly (MiniMax's 5-hour window advances every second; without this the
  // bar would only update on agent-init events). Backend has its own 60s
  // refresher, but the frontend timer drives the visible re-render cadence.
  //
  // The sid is sourced from the nanostore (not the prop) because the prop is
  // only passed when the intro message is rendered — it doesn't update as
  // the session id becomes known later. The store subscription ensures the
  // polling effect re-runs as soon as a sid becomes available.
  const sessionId = useStore($uiSessionId)
  const [liveQuota, setLiveQuota] = useState<QuotaInfo | undefined>(info.quota)
  const { rpc } = useGateway()
  const { stderr } = useStderr()
  useEffect(() => {
    stderr.write(`[quota] SessionPanel sessionId=${sessionId} info.quota.supported=${info.quota?.supported}\n`)
  })
  useEffect(() => {
    // Pick up changes to info.quota (e.g. after model switch)
    setLiveQuota(info.quota)
  }, [info.quota])

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const resp = await rpc<{ quota: QuotaInfo }>('quota.refresh', { session_id: sessionId ?? 'unknown' })
        if (cancelled || !resp?.quota) return
        setLiveQuota({ ...resp.quota })
      } catch {
        // Network blip; keep the previous data on screen.
      }
    }
    if (!sessionId) {
      const t = setTimeout(refresh, QUOTA_TICK_MS)
      return () => clearTimeout(t)
    }
    const id = setInterval(refresh, QUOTA_TICK_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, info.quota?.supported, rpc])

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''
    let shown = 0

    for (const item of [...items].sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > lineBudget) {
        return line ? `${line}, …+${items.length - shown}` : `${item}, …`
      }

      line = next
      shown++
    }

    return line
  }

  // ── Collapsible skills section ──
  const skillEntries = Object.entries(info.skills).sort()
  const skillsTotal = flat(info.skills).length
  const skillsCatCount = skillEntries.length

  const skillsBody = () => {
    if (info.lazy && skillEntries.length === 0) {
      return <InlineLoader label="scanning skills" t={t} />
    }

    const shown = skillEntries.slice(0, SKILLS_MAX)
    const overflow = skillEntries.length - SKILLS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && (
          <Text color={t.color.muted}>(and {overflow} more categories…)</Text>
        )}
      </>
    )
  }

  // ── Collapsible tools section ──
  const toolEntries = Object.entries(info.tools).sort()
  const toolsTotal = flat(info.tools).length

  const toolsBody = () => {
    const shown = toolEntries.slice(0, TOOLSETS_MAX)
    const overflow = toolEntries.length - TOOLSETS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && (
          <Text color={t.color.muted}>(and {overflow} more toolsets…)</Text>
        )}
      </>
    )
  }

  // ── Collapsible MCP section ──
  const mcpBody = () => (
    <>
      {(info.mcp_servers ?? []).map(s => (
        <Text key={s.name} wrap="truncate">
          <Text color={t.color.muted}>{`  ${s.name} `}</Text>
          <Text color={t.color.muted}>{`[${s.transport}]`}</Text>
          <Text color={t.color.muted}>: </Text>
          {s.connected ? (
            <Text color={t.color.text}>
              {s.tools} tool{s.tools === 1 ? '' : 's'}
            </Text>
          ) : s.disabled || s.status === 'disabled' ? (
            <Text color={t.color.muted}>disabled</Text>
          ) : s.status === 'connecting' ? (
            <Text color={t.color.warn}>connecting</Text>
          ) : s.status === 'configured' ? (
            <Text color={t.color.muted}>configured</Text>
          ) : (
            <Text color={t.color.error}>failed</Text>
          )}
        </Text>
      ))}
    </>
  )

  // ── System prompt body ──
  const sysPromptLen = (info.system_prompt ?? '').length

  const systemBody = () => {
    if (sysPromptLen === 0) {
      return <Text color={t.color.muted}>No system prompt loaded.</Text>
    }

    return (
      <Text color={t.color.muted}>
        {info.system_prompt}
      </Text>
    )
  }

  return (
    <Box borderColor={t.color.border} borderStyle="round" marginBottom={1} paddingX={2} paddingY={1}>
      {wide && (
        <Box flexDirection="column" marginRight={2} width={leftW}>
          <ArtLines lines={heroLines} />
          <Text />

          <Text color={t.color.accent}>
            {info.model.split('/').pop()}
            <Text color={t.color.muted}> · Nous Research</Text>
          </Text>

          <Text color={t.color.muted} wrap="truncate-end">
            {info.cwd || process.cwd()}
          </Text>

          {sid && (
            <Text>
              <Text color={t.color.sessionLabel}>Session: </Text>
              <Text color={t.color.sessionBorder}>{sid}</Text>
            </Text>
          )}
        </Box>
      )}

      <Box flexDirection="column" width={w}>
        {wide ? (
          <Box justifyContent="center" marginBottom={1}>
            <Text bold color={t.color.primary}>
              {t.brand.name}
              {info.version ? ` v${info.version}` : ''}
              {info.release_date ? ` (${info.release_date})` : ''}
            </Text>
          </Box>
        ) : (
          // Narrow layout hides the hero column; surface model/cwd/session
          // here so they aren't lost.
          <Box flexDirection="column" marginBottom={1}>
            <Text color={t.color.accent} wrap="truncate-end">
              {info.model.split('/').pop()}
              <Text color={t.color.muted}> · Nous Research</Text>
            </Text>
            <Text color={t.color.muted} wrap="truncate-end">
              {info.cwd || process.cwd()}
            </Text>
            {sid && (
              <Text wrap="truncate-end">
                <Text color={t.color.sessionLabel}>Session: </Text>
                <Text color={t.color.sessionBorder}>{sid}</Text>
              </Text>
            )}
          </Box>
        )}

        {/* ── Tools (expanded by default) ── */}
        <Box flexDirection="column" marginTop={1}>
          <CollapseToggle
            onToggle={() => setToolsOpen(v => !v)}
            open={toolsOpen}
            t={t}
            title="Available Tools"
          />
          {toolsOpen && toolsBody()}
        </Box>

        {/* ── Skills (collapsed by default) ── */}
        <Box flexDirection="column" marginTop={1}>
          <CollapseToggle
            count={skillsTotal}
            onToggle={() => setSkillsOpen(v => !v)}
            open={skillsOpen}
            suffix={skillsCatCount > 0 ? `in ${skillsCatCount} categor${skillsCatCount === 1 ? 'y' : 'ies'}` : undefined}
            t={t}
            title="Available Skills"
          />
          {skillsOpen && skillsBody()}
        </Box>

        {/* ── System Prompt (collapsed by default) ── */}
        {sysPromptLen > 0 && (
          <Box flexDirection="column" marginTop={1}>
            <CollapseToggle
              onToggle={() => setSystemOpen(v => !v)}
              open={systemOpen}
              suffix={`— ${sysPromptLen.toLocaleString()} chars`}
              t={t}
              title="System Prompt"
            />
            {systemOpen && systemBody()}
          </Box>
        )}

        {/* ── MCP Servers (collapsed by default) ── */}
        {info.mcp_servers && info.mcp_servers.length > 0 && (
          <Box flexDirection="column" marginTop={1}>
            <CollapseToggle
              count={info.mcp_servers.length}
              onToggle={() => setMcpOpen(v => !v)}
              open={mcpOpen}
              suffix="connected"
              t={t}
              title="MCP Servers"
            />
            {mcpOpen && mcpBody()}
          </Box>
        )}

        <Text />

        {/* Quota lives in the main column (not the hero column) so it has the
            full inner width — even in narrow layouts the hero column would
            truncate it after ~15 chars. `liveQuota` is the locally-pinned
            copy that ticks every 5s via the quota.refresh RPC; it falls
            back to info.quota (set at session startup) on first render. */}
        <QuotaLine quota={liveQuota} t={t} />

        <Text color={t.color.text}>
          {toolsTotal} tools{' · '}
          {skillsTotal} skills
          {info.mcp_servers?.length ? ` · ${info.mcp_servers.length} MCP` : ''}
          {' · '}
          <Text color={t.color.muted}>/help for commands</Text>
        </Text>

        {typeof info.update_behind === 'number' && info.update_behind > 0 && (
          <Text bold color={t.color.warn}>
            ! {info.update_behind} {info.update_behind === 1 ? 'commit' : 'commits'} behind
            <Text bold={false} color={t.color.warn} dimColor>
              {' '}
              - run{' '}
            </Text>
            <Text bold color={t.color.warn}>
              {info.update_command || 'hermes update'}
            </Text>
            <Text bold={false} color={t.color.warn} dimColor>
              {' '}
              to update
            </Text>
          </Text>
        )}
      </Box>
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.primary}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.accent}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.muted}>{k.padEnd(20)}</Text>
              <Text color={t.color.text}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.text} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.muted}>{sec.text}</Text>}
        </Box>
      ))}
    </Box>
  )
}

interface PanelProps {
  sections: PanelSection[]
  t: Theme
  title: string
}

interface SessionPanelProps {
  info: SessionInfo
  maxWidth?: number
  sid?: string | null
  setIntroInfo?: (updater: (prev: SessionInfo | null | undefined) => SessionInfo | null | undefined) => void
  t: Theme
}
