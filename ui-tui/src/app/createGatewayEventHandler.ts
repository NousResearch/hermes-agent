import { STREAM_BATCH_MS } from '../config/timing.js'
import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import type { CommandsCatalogResponse, GatewayEvent, GatewaySkin } from '../gatewayTypes.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import { formatToolCall, stripAnsi } from '../lib/text.js'
import { fromSkin } from '../theme.js'
import type { Msg, SubagentProgress } from '../types.js'

import type { GatewayEventHandlerContext } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

const ERRLIKE_RE = /\b(error|traceback|exception|failed|spawn)\b/i
const NO_PROVIDER_RE = /\bNo (?:LLM|inference) provider configured\b/i

const statusFromBusy = () => (getUiState().busy ? 'running…' : 'ready')

const applySkin = (s: GatewaySkin) =>
  patchUiState({
    theme: fromSkin(
      s.colors ?? {},
      s.branding ?? {},
      s.banner_logo ?? '',
      s.banner_hero ?? '',
      s.tool_prefix ?? '',
      s.help_header ?? ''
    )
  })

const dropBgTask = (taskId: string) =>
  patchUiState(state => {
    const next = new Set(state.bgTasks)
    next.delete(taskId)

    return { ...state, bgTasks: next }
  })

const pushUnique =
  (max: number) =>
  <T>(xs: T[], x: T): T[] =>
    xs.at(-1) === x ? xs : [...xs, x].slice(-max)

const pushThinking = pushUnique(6)
const pushNote = pushUnique(6)
const pushTool = pushUnique(8)

export function createGatewayEventHandler(ctx: GatewayEventHandlerContext): (ev: GatewayEvent) => void {
  const { rpc } = ctx.gateway
  const { STARTUP_RESUME_ID, newSession, resumeById, setCatalog } = ctx.session
  const { bellOnComplete, stdout, sys } = ctx.system
  const { appendMessage, panel, setHistoryItems } = ctx.transcript

  let pendingThinkingStatus = ''
  let thinkingStatusTimer: null | ReturnType<typeof setTimeout> = null

  const setStatus = (status: string) => {
    pendingThinkingStatus = ''

    if (thinkingStatusTimer) {
      clearTimeout(thinkingStatusTimer)
      thinkingStatusTimer = null
    }

    patchUiState({ status })
  }

  const scheduleThinkingStatus = (status: string) => {
    pendingThinkingStatus = status

    if (thinkingStatusTimer) {
      return
    }

    thinkingStatusTimer = setTimeout(() => {
      thinkingStatusTimer = null
      patchUiState({ status: pendingThinkingStatus || statusFromBusy() })
    }, STREAM_BATCH_MS)
  }

  const restoreStatusAfter = (ms: number) => {
    turnController.clearStatusTimer()
    turnController.statusTimer = setTimeout(() => {
      turnController.statusTimer = null
      patchUiState({ status: statusFromBusy() })
    }, ms)
  }

  const keepCompletedElseRunning = (s: SubagentProgress['status']) => (s === 'completed' ? s : 'running')

  const handleReady = (skin?: GatewaySkin) => {
    if (skin) {
      applySkin(skin)
    }

    rpc<CommandsCatalogResponse>('commands.catalog', {})
      .then(r => {
        if (!r?.pairs) {
          return
        }

        setCatalog({
          canon: (r.canon ?? {}) as Record<string, string>,
          categories: r.categories ?? [],
          pairs: r.pairs as [string, string][],
          skillCount: (r.skill_count ?? 0) as number,
          sub: (r.sub ?? {}) as Record<string, string[]>
        })

        if (r.warning) {
          turnController.pushActivity(String(r.warning), 'warn')
        }
      })
      .catch((e: unknown) => turnController.pushActivity(`command catalog unavailable: ${rpcErrorMessage(e)}`, 'warn'))

    if (!STARTUP_RESUME_ID) {
      patchUiState({ status: 'forging session…' })
      newSession()

      return
    }

    patchUiState({ status: 'resuming…' })
    resumeById(STARTUP_RESUME_ID)
  }

  return (ev: GatewayEvent) => {
    const sid = getUiState().sid

    if (ev.session_id && sid && ev.session_id !== sid && !ev.type.startsWith('gateway.')) {
      return
    }

    switch (ev.type) {
      case 'gateway.ready':
        handleReady(ev.payload?.skin)

        return

      case 'skin.changed':
        if (ev.payload) {
          applySkin(ev.payload)
        }

        return
      case 'session.info': {
        const info = ev.payload

        patchUiState(state => ({
          ...state,
          info,
          status: state.status === 'starting agent…' ? 'ready' : state.status,
          usage: info.usage ? { ...state.usage, ...info.usage } : state.usage
        }))

        setHistoryItems(prev => prev.map(m => (m.kind === 'intro' ? { ...m, info } : m)))

        return
      }

      case 'thinking.delta': {
        const text = ev.payload?.text

        if (text !== undefined) {
          scheduleThinkingStatus(text ? String(text) : statusFromBusy())
        }

        return
      }

      case 'message.start':
        turnController.startMessage()

        return
      case 'status.update': {
        const p = ev.payload

        if (!p?.text) {
          return
        }

        setStatus(p.text)

        if (!p.kind || p.kind === 'status') {
          return
        }

        if (turnController.lastStatusNote !== p.text) {
          turnController.lastStatusNote = p.text
          turnController.pushActivity(
            p.text,
            p.kind === 'error' ? 'error' : p.kind === 'warn' || p.kind === 'approval' ? 'warn' : 'info'
          )
        }

        restoreStatusAfter(4000)

        return
      }

      case 'gateway.stderr': {
        const line = String(ev.payload.line).slice(0, 120)

        turnController.pushActivity(line, ERRLIKE_RE.test(line) ? 'error' : 'warn')

        return
      }

      case 'gateway.start_timeout': {
        const { cwd, python } = ev.payload ?? {}
        const trace = python || cwd ? ` · ${String(python || '')} ${String(cwd || '')}`.trim() : ''

        setStatus('gateway startup timeout')
        turnController.pushActivity(`gateway startup timed out${trace} · /logs to inspect`, 'error')

        return
      }

      case 'gateway.protocol_error':
        setStatus('protocol warning')
        restoreStatusAfter(4000)

        if (!turnController.protocolWarned) {
          turnController.protocolWarned = true
          turnController.pushActivity('protocol noise detected · /logs to inspect', 'warn')
        }

        if (ev.payload?.preview) {
          turnController.pushActivity(`protocol noise: ${String(ev.payload.preview).slice(0, 120)}`, 'warn')
        }

        return

      case 'reasoning.delta':
        if (ev.payload?.text) {
          turnController.recordReasoningDelta(ev.payload.text)
        }

        return

      case 'reasoning.available':
        turnController.recordReasoningAvailable(String(ev.payload?.text ?? ''))

        return

      case 'tool.progress':
        if (ev.payload?.preview && ev.payload.name) {
          turnController.recordToolProgress(ev.payload.name, ev.payload.preview)
        }

        return

      case 'tool.generating':
        if (ev.payload?.name) {
          turnController.pushTrail(`drafting ${ev.payload.name}…`)
        }

        return

      case 'tool.start':
        turnController.recordToolStart(ev.payload.tool_id, ev.payload.name ?? 'tool', ev.payload.context ?? '')

        return

      case 'tool.complete':
        {
          const inlineDiffText =
            ev.payload.inline_diff && getUiState().inlineDiffs ? stripAnsi(String(ev.payload.inline_diff)).trim() : ''

          turnController.recordToolComplete(
            ev.payload.tool_id,
            ev.payload.name,
            ev.payload.error,
            inlineDiffText ? '' : ev.payload.summary
          )

          if (!inlineDiffText) {
            return
          }

          // Keep inline diffs attached to the assistant completion body so
          // they render in the same message flow, not as a standalone system
          // artifact that can look out-of-place around tool rows.
          turnController.queueInlineDiff(inlineDiffText)

          return
        }

        return

      case 'clarify.request':
        patchOverlayState({
          clarify: { choices: ev.payload.choices, question: ev.payload.question, requestId: ev.payload.request_id }
        })
        setStatus('waiting for input…')

        return
      case 'approval.request': {
        const description = String(ev.payload.description ?? 'dangerous command')

        patchOverlayState({ approval: { command: String(ev.payload.command ?? ''), description } })
        turnController.pushActivity(`approval needed · ${description}`, 'warn')
        setStatus('approval needed')

        return
      }

      case 'sudo.request':
        patchOverlayState({ sudo: { requestId: ev.payload.request_id } })
        setStatus('sudo password needed')

        return

      case 'secret.request':
        patchOverlayState({
          secret: { envVar: ev.payload.env_var, prompt: ev.payload.prompt, requestId: ev.payload.request_id }
        })
        setStatus('secret input needed')

        return

      case 'background.complete':
        dropBgTask(ev.payload.task_id)
        sys(`[bg ${ev.payload.task_id}] ${ev.payload.text}`)

        return

      case 'btw.complete':
        dropBgTask('btw:x')
        sys(`[btw] ${ev.payload.text}`)

        return

      case 'subagent.start':
        turnController.upsertSubagent(ev.payload, () => ({ status: 'running' }))

        return
      case 'subagent.thinking': {
        const text = String(ev.payload.text ?? '').trim()

        if (!text) {
          return
        }

        turnController.upsertSubagent(ev.payload, c => ({
          status: keepCompletedElseRunning(c.status),
          thinking: pushThinking(c.thinking, text)
        }))

        return
      }

      case 'subagent.tool': {
        const line = formatToolCall(
          ev.payload.tool_name ?? 'delegate_task',
          ev.payload.tool_preview ?? ev.payload.text ?? ''
        )

        turnController.upsertSubagent(ev.payload, c => ({
          status: keepCompletedElseRunning(c.status),
          tools: pushTool(c.tools, line)
        }))

        return
      }

      case 'subagent.progress': {
        const text = String(ev.payload.text ?? '').trim()

        if (!text) {
          return
        }

        turnController.upsertSubagent(ev.payload, c => ({
          notes: pushNote(c.notes, text),
          status: keepCompletedElseRunning(c.status)
        }))

        return
      }

      case 'subagent.complete':
        turnController.upsertSubagent(ev.payload, c => ({
          durationSeconds: ev.payload.duration_seconds ?? c.durationSeconds,
          status: ev.payload.status ?? 'completed',
          summary: ev.payload.summary || ev.payload.text || c.summary
        }))

        return

      case 'message.delta':
        turnController.recordMessageDelta(ev.payload ?? {})

        return
      case 'message.complete': {
        const { finalMessages, finalText, wasInterrupted } = turnController.recordMessageComplete(ev.payload ?? {})

        if (!wasInterrupted) {
          const msgs: Msg[] = finalMessages.length ? finalMessages : [{ role: 'assistant', text: finalText }]
          msgs.forEach(appendMessage)

          if (bellOnComplete && stdout?.isTTY) {
            stdout.write('\x07')
          }
        }

        setStatus('ready')

        if (ev.payload?.usage) {
          patchUiState(state => ({ ...state, usage: { ...state.usage, ...ev.payload!.usage } }))
        }

        return
      }

      case 'error':
        turnController.recordError()

        {
          const message = String(ev.payload?.message || 'unknown error')

          turnController.pushActivity(message, 'error')

          if (NO_PROVIDER_RE.test(message)) {
            panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
            setStatus('setup required')

            return
          }

          sys(`error: ${message}`)
          setStatus('ready')
        }
    }
  }
}
