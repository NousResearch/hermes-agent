import { STARTUP_IMAGE, STARTUP_QUERY } from '../config/env.js'
import { STREAM_BATCH_MS } from '../config/timing.js'
import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import type {
  CommandsCatalogResponse,
  ConfigFullResponse,
  DelegationStatusResponse,
  GatewayEvent,
  GatewaySkin,
  SessionMostRecentResponse
} from '../gatewayTypes.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import { openExternalUrl } from '../lib/openExternalUrl.js'
import { topLevelSubagents } from '../lib/subagentTree.js'
import { formatAbandonedClarify, formatToolCall, stripAnsi } from '../lib/text.js'
import { fromSkin } from '../theme.js'
import type { Msg, SubagentProgress, SubagentStatus } from '../types.js'

import { applyDelegationStatus, getDelegationState } from './delegationStore.js'
import type { GatewayEventHandlerContext } from './interfaces.js'
import { getOverlayState, patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

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

const KNOWN_SUBAGENT_STATUSES = new Set<SubagentStatus>([
  'completed',
  'error',
  'failed',
  'interrupted',
  'queued',
  'running',
  'timeout'
])

const normalizeSubagentStatus = (status: unknown, fallback: SubagentStatus): SubagentStatus => {
  if (typeof status !== 'string') {
    return fallback
  }

  const normalized = status.toLowerCase() as SubagentStatus

  return KNOWN_SUBAGENT_STATUSES.has(normalized) ? normalized : fallback
}

export function createGatewayEventHandler(ctx: GatewayEventHandlerContext): (ev: GatewayEvent) => void {
  const { rpc } = ctx.gateway
  const { STARTUP_RESUME_ID, newSession, recoverSidRef, resumeById, setCatalog } = ctx.session
  const { bellOnComplete, stdout, sys } = ctx.system
  const { appendMessage, panel, setHistoryItems } = ctx.transcript
  const { setInput } = ctx.composer
  const { submitRef } = ctx.submission
  const { setProcessing: setVoiceProcessing, setRecording: setVoiceRecording, setVoiceEnabled } = ctx.voice

  let pendingThinkingStatus = ''
  let thinkingStatusTimer: null | ReturnType<typeof setTimeout> = null
  let startupPromptSubmitted = false

  // Request IDs of clarify prompts we've already flushed to the transcript as
  // an abandoned-prompt record, so the tool.complete and message.complete
  // paths can't both persist the same prompt twice.
  const persistedAbandonedClarify = new Set<string>()

  // When a clarify prompt is dismissed without an answer (the backend _block
  // timed out and returned an empty string), the live ClarifyPrompt overlay is
  // left set until the next turn's idle() silently nulls it — so the question
  // and options vanish from the screen while the agent's follow-up still refers
  // to them.  The reliable signal is the clarify tool's own tool.complete (and,
  // as a backstop, message.complete): at those points the overlay is provably
  // still set on a timeout, but already cleared by answerClarify() on a real
  // answer (so this no-ops there).  Flush the question + options into the
  // transcript as a persistent system line, then clear the overlay.
  const flushAbandonedClarify = () => {
    const { clarify } = getOverlayState()

    if (!clarify || persistedAbandonedClarify.has(clarify.requestId)) {
      return
    }

    persistedAbandonedClarify.add(clarify.requestId)
    appendMessage({
      role: 'system',
      text: formatAbandonedClarify(clarify.question, clarify.choices, 'timed out')
    })
    patchOverlayState({ clarify: null })
  }

  // Inject the disk-save callback into turnController so recordMessageComplete
  // can fire-and-forget a persist without having to plumb a gateway ref around.
  turnController.persistSpawnTree = async (subagents, sessionId) => {
    try {
      const startedAt = subagents.reduce<number>((min, s) => {
        if (!s.startedAt) {
          return min
        }

        return min === 0 ? s.startedAt : Math.min(min, s.startedAt)
      }, 0)

      const top = topLevelSubagents(subagents)
        .map(s => s.goal)
        .filter(Boolean)
        .slice(0, 2)

      const label = top.length ? top.join(' · ') : `${subagents.length} subagents`

      await rpc('spawn_tree.save', {
        finished_at: Date.now() / 1000,
        label: label.slice(0, 120),
        session_id: sessionId ?? 'default',
        started_at: startedAt ? startedAt / 1000 : null,
        subagents
      })
    } catch {
      // Persistence is best-effort; in-memory history is the authoritative
      // same-session source.  A write failure doesn't block the turn.
    }
  }

  // Refresh delegation caps at most every 5s so the status bar HUD can
  // render a /warning close to the configured cap without spamming the RPC.
  let lastDelegationFetchAt = 0

  // ── Shared full-config read ──────────────────────────────────────────
  //
  // Several concerns need `display.*` flags at startup (the /agents nudge
  // gate below, the auto-resume check in the `gateway.ready` handler).
  // Memoize the `config.get full` RPC so we make exactly one round-trip
  // instead of one per concern.  Resolves to null on RPC failure; callers
  // treat null as "use defaults".
  let fullConfigPromise: null | Promise<ConfigFullResponse | null> = null

  const getFullConfigOnce = (): Promise<ConfigFullResponse | null> => {
    fullConfigPromise ??= rpc<ConfigFullResponse>('config.get', { key: 'full' }).catch(() => null)

    return fullConfigPromise
  }

  // ── Nudge toward /agents on delegation ───────────────────────────────
  //
  // When `display.tui_agents_nudge` is enabled (default true), the first
  // time a turn starts delegating we drop a single transient activity hint
  // ("subagents working · /agents to watch live") so the user discovers the
  // spawn-tree dashboard instead of staring at a quiet transcript — without
  // hijacking the screen by force-opening an overlay.  Guards:
  //   • fires at most once per turn (`agentsNudgedThisTurn`)
  //   • silent if the overlay is already open (nothing to advertise)
  // Reset on `message.start`.  The config flag is fetched once, lazily;
  // until it resolves we assume the default (on).
  let agentsNudgeEnabled = true
  let agentsNudgeConfigFetched = false
  let agentsNudgedThisTurn = false

  const ensureAgentsNudgeConfig = () => {
    if (agentsNudgeConfigFetched) {
      return
    }

    agentsNudgeConfigFetched = true
    getFullConfigOnce().then(cfg => {
      // Only an explicit `false` disables it; absent/unknown keeps default on.
      if (cfg?.config?.display?.tui_agents_nudge === false) {
        agentsNudgeEnabled = false
      }
    })
  }

  const maybeNudgeAgents = () => {
    ensureAgentsNudgeConfig()

    if (!agentsNudgeEnabled || agentsNudgedThisTurn) {
      return
    }

    // Already watching → no point advertising the dashboard.  Don't burn the
    // turn's nudge credit here: if the user closes the overlay later in the
    // same turn while delegation is still ongoing, a subsequent event should
    // still be allowed to nudge.  The flag is only set once we actually push.
    if (getOverlayState().agents) {
      return
    }

    agentsNudgedThisTurn = true
    turnController.pushActivity('subagents working · /agents to watch live', 'info')
  }

  const resetAgentsNudgeTurnState = () => {
    agentsNudgedThisTurn = false
  }

  const refreshDelegationStatus = (force = false) => {
    const now = Date.now()

    if (!force && now - lastDelegationFetchAt < 5000) {
      return
    }

    lastDelegationFetchAt = now
    rpc<DelegationStatusResponse>('delegation.status', {})
      .then(r => applyDelegationStatus(r))
      .catch(() => {})
  }

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

  const scheduleStartupPrompt = () => {
    if (startupPromptSubmitted || (!STARTUP_QUERY && !STARTUP_IMAGE)) {
      return
    }

    startupPromptSubmitted = true
    setTimeout(async () => {
      let sid = getUiState().sid

      for (let i = 0; !sid && i < 40; i += 1) {
        await new Promise(resolve => setTimeout(resolve, 100))
        sid = getUiState().sid
      }

      if (!sid) {
        return sys('startup query skipped: no active session')
      }

      if (STARTUP_IMAGE) {
        try {
          await rpc('image.attach', { path: STARTUP_IMAGE, session_id: sid })
        } catch (e) {
          sys(`startup image attach failed: ${rpcErrorMessage(e)}`)
        }
      }

      submitRef.current(STARTUP_QUERY || 'What do you see in this image?')
    }, 0)
  }

  // Terminal statuses are never overwritten by late-arriving live events —
  // otherwise a stale `subagent.start` / `spawn_requested` can clobber a
  // terminal state from complete (failed/interrupted/timeout/error).
  const isTerminalStatus = (s: SubagentProgress['status']) =>
    s === 'completed' || s === 'error' || s === 'failed' || s === 'interrupted' || s === 'timeout'

  const keepTerminalElseRunning = (s: SubagentProgress['status']) => (isTerminalStatus(s) ? s : 'running')

  const handleReady = (skin?: GatewaySkin) => {
    if (skin) {
      applySkin(skin)
    }

    // Kick off the config fetch once the gateway is actually ready. If handler
    // construction does this during React render, a startup transport error can
    // report through sys(), mutate transcript state, and trip React's
    // "too many re-renders" guard in embedded dashboard PTYs.
    ensureAgentsNudgeConfig()

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
      .catch((e: unknown) => turnController.pushActivity(`command catalog unavailable: ${rpcErrorMessage(e)}`, 'info'))

    // Crash recovery: a respawn triggered by an unexpected gateway death
    // resumes the session that was live, not a brand-new one. One-shot — the
    // ref is cleared so an ordinary later restart still forges/resumes per
    // config. No startup prompt here (this is mid-session, not a cold boot).
    const recoverSid = recoverSidRef?.current

    if (recoverSidRef && recoverSid) {
      recoverSidRef.current = null
      resumeById(recoverSid)
      // After resumeById: it synchronously sets status to 'resuming…' on entry,
      // so override it here to keep the distinct "recovering" label visible for
      // the duration of the resume RPC (which later flips status to 'ready').
      patchUiState({ status: 'recovering session…' })

      return
    }

    if (STARTUP_RESUME_ID) {
      patchUiState({ status: 'resuming…' })
      resumeById(STARTUP_RESUME_ID)
      scheduleStartupPrompt()

      return
    }

    // Opt-in: when `display.tui_auto_resume_recent` is true, look up
    // the most recent human-facing session and resume it instead of
    // forging a brand-new one.  Mirrors classic CLI's `hermes -c` /
    // `hermes --tui` muscle memory and addresses the audit's "session
    // unrecoverable after disconnection" gap.  Default off so existing
    // users aren't surprised.  (Shares the memoized full-config read.)
    getFullConfigOnce()
      .then(cfg => {
        if (!cfg?.config?.display?.tui_auto_resume_recent) {
          patchUiState({ status: 'forging session…' })
          newSession()
          scheduleStartupPrompt()

          return
        }

        return rpc<SessionMostRecentResponse>('session.most_recent', {}).then(r => {
          const target = r?.session_id

          if (target) {
            patchUiState({ status: 'resuming most recent…' })
            resumeById(target)
            scheduleStartupPrompt()

            return
          }

          patchUiState({ status: 'forging session…' })
          newSession()
          scheduleStartupPrompt()
        })
      })
      .catch(() => {
        patchUiState({ status: 'forging session…' })
        newSession()
        scheduleStartupPrompt()
      })
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
        if (!getUiState().busy) {
          return
        }

        const text = ev.payload?.text

        if (text !== undefined) {
          const value = String(text)
          scheduleThinkingStatus(value || statusFromBusy())

          if (value) {
            turnController.recordReasoningDelta(value)
          }
        }

        return
      }

      case 'message.start':
        resetAgentsNudgeTurnState()
        turnController.startMessage()

        return
      case 'status.update': {
        const p = ev.payload

        if (!p?.text) {
          return
        }

        if (p.kind === 'goal') {
          sys(p.text)

          const brief = p.text.startsWith('✓')
            ? '✓ goal complete'
            : p.text.startsWith('↻')
              ? '↻ goal continuing'
              : p.text.startsWith('⏸')
                ? '⏸ goal paused'
                : 'ready'

          setStatus(brief)
          restoreStatusAfter(6000)

          return
        }

        setStatus(p.text)

        if (p.kind === 'compressing') {
          sys(p.text)

          return
        }

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

      case 'notification.show': {
        // Credits/usage notice from the gateway. Payload is snake_case on the
        // wire and stays snake_case in UiState.notice (no mapping layer). The
        // text already carries its own glyph; turnController decides whether to
        // show now or hold until turn end (FaceTicker wins while busy).
        const p = ev.payload

        if (!p?.text) {
          return
        }

        turnController.showNotice({
          id: p.id,
          key: p.key,
          kind: p.kind ?? 'sticky',
          level: p.level ?? 'info',
          text: p.text,
          ttl_ms: p.ttl_ms ?? null
        })

        return
      }

      case 'notification.clear':
        // Key-matched clear only — a stale/late clear must not wipe a newer
        // notice (turnController guards the key match).
        turnController.clearNotice(ev.payload?.key)

        return
      case 'billing.step_up.verification': {
        // The billing step-up device flow runs in the headless gateway, so it
        // can't open a browser or print the URL where the user sees it. Surface
        // the link here (clickable/copyable in the transcript) and best-effort
        // open it via the TUI process's own opener. This event arrives while the
        // billing.step_up RPC is still polling (and may even outlive the RPC's
        // 120s timeout), so the link — not the RPC result — is the source of truth.
        const url = ev.payload.verification_url
        const code = ev.payload.user_code

        if (!url) {
          return
        }

        sys('💳 Open this link to grant terminal billing access:')
        sys(url)
        if (code) {
          sys(`If prompted, enter code: ${code}`)
        }
        void openExternalUrl(url)

        return
      }
      case 'gateway.stderr': {
        const line = String(ev.payload.line).slice(0, 120)

        turnController.pushActivity(line, 'info')

        return
      }

      case 'browser.progress': {
        const message = String(ev.payload?.message ?? '').trim()

        if (message) {
          sys(message)
        }

        return
      }

      case 'voice.status': {
        // Continuous VAD loop reports its internal state so the status bar
        // can show listening / transcribing / idle without polling.
        const state = String(ev.payload?.state ?? '')

        if (state === 'listening') {
          setVoiceRecording(true)
          setVoiceProcessing(false)
        } else if (state === 'transcribing') {
          setVoiceRecording(false)
          setVoiceProcessing(true)
        } else {
          setVoiceRecording(false)
          setVoiceProcessing(false)
        }

        return
      }

      case 'voice.transcript': {
        // CLI parity: the 3-strikes silence detector flipped off automatically.
        // Mirror that on the UI side and tell the user why the mode is off.
        if (ev.payload?.no_speech_limit) {
          setVoiceEnabled(false)
          setVoiceRecording(false)
          setVoiceProcessing(false)
          sys('voice: no speech detected 3 times, continuous mode stopped')

          return
        }

        const text = String(ev.payload?.text ?? '').trim()

        if (!text) {
          return
        }

        // CLI parity: _pending_input.put(transcript) unconditionally feeds
        // the transcript to the agent as its next turn — draft handling
        // doesn't apply because voice-mode users are speaking, not typing.
        //
        // We can't branch on composer input from inside a setInput updater
        // (React strict mode double-invokes it, duplicating the submit).
        // Just clear + defer submit so the cleared input is committed before
        // submit reads it.
        setInput('')
        setTimeout(() => submitRef.current(text), 0)

        return
      }

      case 'gateway.start_timeout': {
        const { cwd, python, stderr_tail: stderrTail } = ev.payload ?? {}
        const trace = python || cwd ? ` · ${String(python || '')} ${String(cwd || '')}`.trim() : ''

        setStatus('gateway startup timeout')
        turnController.pushActivity(`gateway startup timed out${trace} · /logs to inspect`, 'error')

        // Surface the most useful stderr lines inline so users can tell
        // "wrong python", "missing dep", and "config parse failure"
        // apart without leaving the TUI.  Filter blank rows BEFORE
        // taking the last N so trailing empty lines in the buffer
        // don't crowd out actual content; truncate to match the
        // 120-char clip used for `gateway.stderr` activity entries.
        const STDERR_LINE_CAP = 120
        const STDERR_LINES_MAX = 8

        const tailLines = (stderrTail ?? '')
          .split('\n')
          .map(l => l.trim())
          .filter(Boolean)
          .slice(-STDERR_LINES_MAX)

        for (const line of tailLines) {
          turnController.pushActivity(line.slice(0, STDERR_LINE_CAP), 'error')
        }

        return
      }

      case 'gateway.protocol_error':
        setStatus('protocol warning')
        restoreStatusAfter(4000)

        if (!turnController.protocolWarned) {
          turnController.protocolWarned = true
          turnController.pushActivity('protocol noise detected · /logs to inspect', 'info')
        }

        if (ev.payload?.preview) {
          turnController.pushActivity(`protocol noise: ${String(ev.payload.preview).slice(0, 120)}`, 'info')
        }

        return

      case 'reasoning.delta':
        if (ev.payload?.text) {
          turnController.recordReasoningDelta(ev.payload.text, Boolean(ev.payload.verbose))
        }

        return

      case 'reasoning.available':
        turnController.recordReasoningAvailable(String(ev.payload?.text ?? ''), Boolean(ev.payload?.verbose))

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
        turnController.recordTodos(ev.payload.todos)
        turnController.recordToolStart(
          ev.payload.tool_id,
          ev.payload.name ?? 'tool',
          ev.payload.context ?? '',
          ev.payload.args_text ? stripAnsi(String(ev.payload.args_text)) : undefined
        )

        return
      case 'tool.complete': {
        // The clarify tool finishing with its overlay still live means it was
        // abandoned (backend _block timed out, empty answer). A real answer
        // clears the overlay in answerClarify() before this fires, so this
        // no-ops there. Persist the question + options so they don't vanish.
        if (ev.payload.name === 'clarify') {
          flushAbandonedClarify()
        }

        const inlineDiffText =
          ev.payload.inline_diff && getUiState().inlineDiffs ? stripAnsi(String(ev.payload.inline_diff)).trim() : ''

        const resultText = ev.payload.result_text ? stripAnsi(String(ev.payload.result_text)) : undefined

        if (inlineDiffText) {
          turnController.recordInlineDiffToolComplete(
            inlineDiffText,
            ev.payload.tool_id,
            ev.payload.name,
            ev.payload.error,
            ev.payload.duration_s,
            resultText
          )
        } else {
          turnController.recordToolComplete(
            ev.payload.tool_id,
            ev.payload.name,
            ev.payload.error,
            ev.payload.summary,
            ev.payload.duration_s,
            ev.payload.todos,
            resultText
          )
        }

        return
      }

      case 'clarify.request':
        patchOverlayState({
          clarify: { choices: ev.payload.choices, question: ev.payload.question, requestId: ev.payload.request_id }
        })
        setStatus('waiting for input…')

        return
      case 'approval.request': {
        const description = String(ev.payload.description ?? 'dangerous command')
        // Only an explicit false (tirith warning) drops the permanent-allow option.
        const allowPermanent = ev.payload.allow_permanent !== false

        patchOverlayState({
          approval: { allowPermanent, command: String(ev.payload.command ?? ''), description }
        })
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
      case 'review.summary': {
        // Self-improvement background review emitted a persistent summary
        // of what it saved to memory/skills. Surface it as a system line
        // in the transcript so it never gets lost to a transient status
        // flash. Python-side already formats it as "💾 Self-improvement
        // review: …".
        const text = String(ev.payload?.text ?? '').trim()

        if (text) {
          sys(text)
        }

        return
      }

      case 'subagent.spawn_requested':
        // Child built but not yet running (waiting on ThreadPoolExecutor slot).
        // Preserve completed state if a later event races in before this one.
        turnController.upsertSubagent(ev.payload, c => (isTerminalStatus(c.status) ? {} : { status: 'queued' }))

        // First sign of delegation this turn → nudge toward /agents.
        maybeNudgeAgents()

        // Prime the status-bar HUD: fetch caps (once every 5s) so we can
        // warn as depth/concurrency approaches the configured ceiling.
        if (getDelegationState().maxSpawnDepth === null) {
          refreshDelegationStatus(true)
        } else {
          refreshDelegationStatus()
        }

        return

      case 'subagent.start':
        turnController.upsertSubagent(ev.payload, c => (isTerminalStatus(c.status) ? {} : { status: 'running' }))

        // `subagent.start` is the first delegation event the TUI reliably
        // receives (the delegate callback drops `spawn_requested` in the
        // CLI→gateway path), so nudge here too.  Once-per-turn guarded, so
        // hooking both events is safe.
        maybeNudgeAgents()

        return
      case 'subagent.thinking': {
        const text = String(ev.payload.text ?? '').trim()

        if (!text) {
          return
        }

        // Update-only: never resurrect subagents whose spawn_requested/start
        // we missed or that already flushed via message.complete.
        turnController.upsertSubagent(
          ev.payload,
          c => ({
            status: keepTerminalElseRunning(c.status),
            thinking: pushThinking(c.thinking, text)
          }),
          { createIfMissing: false }
        )

        return
      }

      case 'subagent.tool': {
        const line = formatToolCall(
          ev.payload.tool_name ?? 'delegate_task',
          ev.payload.tool_preview ?? ev.payload.text ?? ''
        )

        turnController.upsertSubagent(
          ev.payload,
          c => ({
            status: keepTerminalElseRunning(c.status),
            tools: pushTool(c.tools, line)
          }),
          { createIfMissing: false }
        )

        return
      }

      case 'subagent.progress': {
        const text = String(ev.payload.text ?? '').trim()

        if (!text) {
          return
        }

        turnController.upsertSubagent(
          ev.payload,
          c => ({
            notes: pushNote(c.notes, text),
            status: keepTerminalElseRunning(c.status)
          }),
          { createIfMissing: false }
        )

        return
      }

      case 'subagent.complete':
        turnController.upsertSubagent(
          ev.payload,
          c => ({
            durationSeconds: ev.payload.duration_seconds ?? c.durationSeconds,
            status: normalizeSubagentStatus(ev.payload.status, 'completed'),
            summary: ev.payload.summary || ev.payload.text || c.summary
          }),
          { createIfMissing: false }
        )

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
