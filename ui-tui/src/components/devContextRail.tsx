import { Box, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo, type ReactNode } from 'react'

import { $delegationState } from '../app/delegationStore.js'
import type { AppLayoutStatusProps, UiState } from '../app/interfaces.js'
import { $overlayState } from '../app/overlayStore.js'
import { $turnState } from '../app/turnStore.js'
import { $uiState, $uiTheme } from '../app/uiStore.js'
import { DEV_CONTEXT_RAIL_WIDTH, devContextRailVisible } from '../domain/devContext.js'
import { EMPTY_GIT_WORKSPACE, truncateDisplay } from '../domain/workspaceHud.js'
import { countPendingTodos } from '../lib/liveProgress.js'
import { useAmbientRailWidth } from '../sdk/host.js'
import type { Theme } from '../theme.js'
import type { ActivityItem, SubagentProgress, TodoItem, Usage } from '../types.js'

export interface DevContextRailProps {
  cols: number
  queuedCount: number
  status: AppLayoutStatusProps
}

const SECRET_ASSIGNMENT = /\b(?:token|password|passwd|secret|api[_-]?key|authorization)\s*[:=]\s*[^\s]+/gi
const URL_OR_REMOTE = /(?:https?|ssh):\/\/[^\s]+|git@github\.com:[^\s]+/gi

const safeText = (value: string): string =>
  value
    .replace(URL_OR_REMOTE, '[url]')
    .replace(SECRET_ASSIGNMENT, '[redacted]')
    .split('')
    .map(char => {
      const code = char.codePointAt(0) ?? 0

      return code <= 0x1f || code === 0x7f ? ' ' : char
    })
    .join('')
    .replace(/\s+/g, ' ')
    .trim()

const displayText = (value: string, maxWidth = 120): string => truncateDisplay(safeText(value), maxWidth)

const formatTokens = (value: number | undefined): string => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '—'
  }

  const absolute = Math.abs(value)

  if (absolute >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}m`
  }

  if (absolute >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`
  }

  return `${Math.round(value)}`
}

const formatPercent = (value: number | undefined): string =>
  typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)}%` : '—'

const formatSeconds = (value: number | undefined): string =>
  typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)}s` : '—'

const formatMoney = (value: number | undefined): string =>
  typeof value === 'number' && Number.isFinite(value) ? `$${value.toFixed(2)}` : '—'

const formatMeter = (percent: number | undefined): string => {
  if (typeof percent !== 'number' || !Number.isFinite(percent)) {
    return '────────'
  }

  const filled = Math.max(0, Math.min(8, Math.round((percent / 100) * 8)))

  return `${'█'.repeat(filled)}${'░'.repeat(8 - filled)}`
}

const recordItemCount = (record: Record<string, string[]> | undefined): number =>
  Object.values(record ?? {}).reduce((count, items) => count + items.length, 0)

const statusGlyph = (status: SubagentProgress['status']): string => {
  switch (status) {
    case 'completed':
      return '✓'

    case 'error':

    case 'failed':

    case 'timeout':
      return '×'

    case 'interrupted':
      return '⏸'

    case 'queued':
      return '·'

    case 'running':
      return '▶'
  }
}

const statusColor = (status: SubagentProgress['status'], t: Theme): string => {
  switch (status) {
    case 'completed':
      return t.color.ok

    case 'error':

    case 'failed':

    case 'timeout':
      return t.color.error

    case 'interrupted':

    case 'queued':
      return t.color.muted

    case 'running':
      return t.color.accent
  }
}

const todoGlyph = (status: TodoItem['status']): string => {
  switch (status) {
    case 'completed':
      return '✓'

    case 'cancelled':
      return '×'

    case 'in_progress':
      return '▶'

    case 'pending':
      return '·'
  }
}

const todoColor = (status: TodoItem['status'], t: Theme): string => {
  switch (status) {
    case 'completed':
      return t.color.ok

    case 'cancelled':
      return t.color.muted

    case 'in_progress':
      return t.color.accent

    case 'pending':
      return t.color.text
  }
}

const activityColor = (item: ActivityItem, t: Theme): string => {
  switch (item.tone) {
    case 'error':
      return t.color.error

    case 'warn':
      return t.color.warn

    case 'info':
      return t.color.muted
  }
}

const activeSubagent = (status: SubagentProgress['status']): boolean => status === 'queued' || status === 'running'

const contextValue = (usage: Usage): string => {
  const percent = formatPercent(usage.context_percent)
  const used = formatTokens(usage.context_used)
  const max = formatTokens(usage.context_max)

  return `${formatMeter(usage.context_percent)} ${percent} · ${used}/${max}`
}

const usageValue = (usage: Usage): string =>
  `calls ${usage.calls} · in ${formatTokens(usage.input)} · out ${formatTokens(usage.output)}`

const performanceValue = (usage: Usage): string => {
  const values = [
    usage.avg_latency_s === undefined ? '' : `lat ${formatSeconds(usage.avg_latency_s)}`,
    usage.avg_tps === undefined ? '' : `out ${Math.round(usage.avg_tps)} tok/s`,
    usage.cache_hit_pct === undefined ? '' : `cache ${formatPercent(usage.cache_hit_pct)}`,
    usage.compressions === undefined ? '' : `compressed ${usage.compressions}×`,
    usage.cost_usd === undefined ? '' : `cost ${formatMoney(usage.cost_usd)}`
  ].filter(Boolean)

  return values.join(' · ') || 'no turn measurements'
}

const workspaceGitValue = (status: AppLayoutStatusProps['workspace']): string => {
  if (status.dirty === true) {
    return 'dirty'
  }

  if (status.dirty === false) {
    return 'clean'
  }

  return 'not detected'
}

const syncValue = (status: AppLayoutStatusProps['workspace']): string => {
  if (!status.upstream) {
    return 'sync unavailable'
  }

  return `↑${status.upstream.ahead} ↓${status.upstream.behind}`
}

const approvalValue = (ui: UiState, overlay: ReturnType<typeof $overlayState.get>): string => {
  if (overlay.approval) {
    return 'approval required'
  }

  if (overlay.confirm) {
    return 'confirmation required'
  }

  if (overlay.clarify) {
    return 'answer required'
  }

  if (overlay.secret) {
    return 'secret input required'
  }

  if (overlay.sudo) {
    return 'sudo input required'
  }

  return ui.busy ? 'working' : 'ready'
}

const Section = memo(function Section({ children, t, title }: { children: ReactNode; t: Theme; title: string }) {
  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1}>
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      {children}
    </Box>
  )
})

const Row = memo(function Row({
  color,
  label,
  t,
  value
}: {
  color?: string
  label: string
  t: Theme
  value: string
}) {
  return (
    <Text color={color ?? t.color.text} wrap="truncate-end">
      <Text color={t.color.label}>{label}: </Text>
      {value}
    </Text>
  )
})

const TodoRow = memo(function TodoRow({ t, todo }: { t: Theme; todo: TodoItem }) {
  return (
    <Text color={todoColor(todo.status, t)} wrap="truncate-end">
      {todo.parent ? '  ' : ''}
      {todoGlyph(todo.status)} <Text color={todo.status === 'cancelled' ? t.color.muted : t.color.text}>{displayText(todo.content)}</Text>
    </Text>
  )
})

const AgentRow = memo(function AgentRow({ agent, t }: { agent: SubagentProgress; t: Theme }) {
  return (
    <Text color={statusColor(agent.status, t)} wrap="truncate-end">
      {statusGlyph(agent.status)} <Text color={t.color.text}>{displayText(agent.goal)}</Text>
    </Text>
  )
})

const ActivityRow = memo(function ActivityRow({ item, t }: { item: ActivityItem; t: Theme }) {
  return (
    <Text color={activityColor(item, t)} wrap="truncate-end">
      {item.tone === 'error' ? '×' : item.tone === 'warn' ? '!' : '·'} {displayText(item.text)}
    </Text>
  )
})

export const DevContextRail = memo(function DevContextRail({ cols, queuedCount, status }: DevContextRailProps) {
  const ui = useStore($uiState)
  const overlay = useStore($overlayState)
  const turn = useStore($turnState)
  const delegation = useStore($delegationState)
  const t = useStore($uiTheme)
  const { stdout } = useStdout()
  const ambientRailColumns = useAmbientRailWidth('left') + useAmbientRailWidth('right')

  if (!devContextRailVisible(ui.devContext, cols, ambientRailColumns)) {
    return null
  }

  const info = ui.info
  const usage = ui.usage
  const workspace = status.workspace ?? { ...EMPTY_GIT_WORKSPACE, projectName: null }
  const dense = (stdout?.rows ?? 24) < 30
  const veryDense = (stdout?.rows ?? 24) < 22
  const activeAgents = turn.subagents.filter(agent => activeSubagent(agent.status))
  const completedTodos = turn.todos.filter(todo => todo.status === 'completed' || todo.status === 'cancelled').length
  const currentTodo = turn.todos.find(todo => todo.status === 'in_progress')
  const mcpServers = info?.mcp_servers ?? []
  const connectedMcp = mcpServers.filter(server => server.connected || server.status === 'connected').length
  const failedMcp = mcpServers.filter(server => server.status === 'failed').length

  const servicesValue = info
    ? `mcp ${connectedMcp}/${mcpServers.length}${failedMcp ? ` · failed ${failedMcp}` : ''} · tools ${recordItemCount(info.tools)} · skills ${recordItemCount(info.skills)}`
    : 'session services unavailable'

  const repositoryValue = workspace.github ? `GH ${workspace.github.fullName}` : 'GitHub not detected'

  const pullRequestValue = workspace.pullRequest
    ? `#${workspace.pullRequest.number} ${workspace.pullRequest.state} ${displayText(workspace.pullRequest.title)}`
    : 'no matching PR'

  const compactGitValue = [workspaceGitValue(workspace), workspace.upstream ? syncValue(workspace) : ''].filter(Boolean).join(' · ')

  const activeWorkValue = [
    `agents ${activeAgents.length}/${turn.subagents.length}`,
    `bg ${ui.bgTasks.size}`,
    turn.tools.length ? `tools ${turn.tools.length}` : '',
    queuedCount ? `queue ${queuedCount}` : ''
  ]
    .filter(Boolean)
    .join(' · ')

  const statusValue = displayText(ui.status || 'ready')

  const approval = approvalValue(ui, overlay)
  const compactRepositoryValue = repositoryValue

  return (
    <Box
      borderColor={t.color.border}
      borderStyle="round"
      flexDirection="column"
      flexShrink={0}
      overflow="hidden"
      width={DEV_CONTEXT_RAIL_WIDTH}
    >
      <Box justifyContent="space-between" paddingX={1}>
        <Text bold color={t.color.primary}>
          DEV CONTEXT
        </Text>
        <Text color={ui.busy ? t.color.accent : t.color.muted}>{ui.compacting ? '◌ compacting' : ui.busy ? '● live' : '○ ready'}</Text>
      </Box>

      <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} overflow="hidden" paddingX={1}>
        <Section t={t} title="REPO">
          <Row label="project" t={t} value={displayText(workspace.projectName ?? status.cwdLabel ?? 'workspace')} />
          <Row label="branch" t={t} value={displayText(workspace.branch ?? info?.branch ?? 'detached')} />
          <Row color={workspace.dirty ? t.color.warn : t.color.ok} label="git" t={t} value={dense ? compactGitValue : workspaceGitValue(workspace)} />
          <Row label="remote" t={t} value={dense ? compactRepositoryValue : repositoryValue} />
          {!dense && <Row label="sync" t={t} value={syncValue(workspace)} />}
          {workspace.pullRequest && <Row label="PR" t={t} value={pullRequestValue} />}
        </Section>

        <Section t={t} title="RUNTIME">
          <Row
            label="model"
            t={t}
            value={displayText(
              [info?.model, info?.provider, info?.reasoning_effort, info?.service_tier].filter(Boolean).join(' · ') || 'not connected'
            )}
          />
          <Row label="context" t={t} value={contextValue(usage)} />
          <Row label="usage" t={t} value={usageValue(usage)} />
          {!veryDense && <Row label="perf" t={t} value={performanceValue(usage)} />}
          {!veryDense && <Row label="services" t={t} value={servicesValue} />}
        </Section>

        <Section t={t} title="PLAN">
          {turn.todos.length ? (
            <>
              <Row
                color={completedTodos === turn.todos.length ? t.color.ok : t.color.text}
                label="progress"
                t={t}
                value={`${completedTodos}/${turn.todos.length} done · ${countPendingTodos(turn.todos)} open`}
              />
              {currentTodo && <TodoRow t={t} todo={currentTodo} />}
              {!dense && turn.todos.filter(todo => todo !== currentTodo).slice(0, 3).map(todo => <TodoRow key={todo.id} t={t} todo={todo} />)}
            </>
          ) : (
            <Row color={t.color.muted} label="progress" t={t} value="no active plan" />
          )}
        </Section>

        <Section t={t} title="WORK">
          <Row label="active" t={t} value={activeWorkValue || 'idle'} />
          {!dense && activeAgents.slice(0, 3).map(agent => <AgentRow agent={agent} key={agent.id} t={t} />)}
          {!dense && delegation.paused && <Row color={t.color.warn} label="delegation" t={t} value="paused" />}
        </Section>

        <Section t={t} title="GUARDRAILS">
          <Row color={ui.busy ? t.color.accent : t.color.ok} label="state" t={t} value={statusValue} />
          <Row color={approval.includes('required') ? t.color.warn : t.color.muted} label="input" t={t} value={approval} />
          <Row label="modes" t={t} value={`${ui.focusView ? 'focus on' : 'focus off'} · ${ui.destructiveSlashConfirm ? 'confirm on' : 'confirm off'}`} />
          {!veryDense && status.sessionTitle && <Row label="session" t={t} value={displayText(status.sessionTitle)} />}
        </Section>

        {!dense && turn.activity.length > 0 && (
          <Section t={t} title="ACTIVITY">
            {turn.activity.slice(-3).reverse().map(item => <ActivityRow item={item} key={item.id} t={t} />)}
          </Section>
        )}
      </Box>

      <Box paddingX={1}>
        <Text color={t.color.muted} wrap="truncate-end">
          /dev-context · /agents · /usage
        </Text>
      </Box>
    </Box>
  )
})
