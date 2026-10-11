import { randomUUID } from 'node:crypto'

import { t } from './i18n/runtime.js'

export interface SharedControl { session_id: string; execution_generation: number; prompt_id: string }
export const sharedControlParams = (prompt?: { sharedControl?: SharedControl } | null) => prompt?.sharedControl ?? {}

export interface CreationContract { sources: string[]; parameters: string[] }

export function canonicalRequest(method: string, original: Record<string, unknown>, contract?: CreationContract) {
  const params = { ...original }

  if (method === 'session.create') {
    delete params.cols
    params.source = 'tui'
    params.request_id ??= randomUUID()

    if (!contract?.sources.includes('tui')) { throw new Error(t('canonical.launch.noTuiPolicy')) }
    const unsupported = Object.keys(params).filter(key => !contract.parameters.includes(key))

    if (unsupported.length) { throw new Error(t('canonical.launch.unsupportedOptions', unsupported.join(', '))) }
  }

  if (['prompt.submit', 'session.steer', 'session.redirect'].includes(method) && params.submission_id) {
    params.input_id = params.submission_id
    delete params.submission_id
  }

  if (method === 'session.resume' || method === 'session.activate') {
    method = 'session.resume'
    delete params.cols
    delete params.omit_messages
  }

  return { method, params }
}

export function canonicalResult(method: string, value: any, request: Record<string, unknown> = {}): any {
  if (!value || typeof value !== 'object') { return value }

  // The JSON-RPC response correlates the prepared input; admission_id is a
  // separate server-issued identity and must not be rewritten to that input ID.
  if (['prompt.submit', 'session.steer', 'session.redirect'].includes(method) && value.ref) {
    return { ...value, input_id: request.input_id, target_profile_home: value.ref.profile_id, target_session_id: value.ref.session_id }
  }

  if (['session.create', 'session.resume', 'session.activate'].includes(method)) {
    // Canonical snapshots contain stored conversation rows (`content`), not
    // the legacy TUI's display rows (`text`). Without this translation only
    // live events render, making reconnect depend on catching the final delta.
    const messages = Array.isArray(value.messages) ? value.messages.map((row: any) =>
      row && typeof row === 'object'
        ? { ...row, text: row.content, name: row.tool_name }
        : row) : value.messages

    return { ...value, messages, info: { ...value.info, stored_session_id: value.stored_session_id,
      execution_epoch: String(value.authority_epoch), execution_generation: value.execution_generation,
      running: value.running, pending_submissions: pendingSubmissions(value.pending) } }
  }

  return value
}

// The authority's durable FIFO rows (`pending` on snapshots and session.info
// fanout) projected onto the client's receipt shape: one translation, so the
// queue panel, the receipt matcher and Desktop all read the same list.
export function pendingSubmissions(pending: unknown) {
  if (!Array.isArray(pending)) { return undefined }

  return pending.filter(row => row && typeof row === 'object').map(row => ({
    ...row, user: row.text, target_profile_home: row.ref?.profile_id, target_session_id: row.ref?.session_id }))
}

export function canonicalEvent<T extends { type: string; payload?: any }>(event: T): T {
  const pending = pendingSubmissions(event.payload?.pending)

  return pending ? { ...event, payload: { ...event.payload, pending_submissions: pending } } : event
}

export function localCreationOptions(env = process.env): Record<string, unknown> {
  const fields: Record<string, string | undefined> = {
    model: env.HERMES_MODEL, provider: env.HERMES_TUI_PROVIDER, cwd: env.HERMES_CWD
  }

  const options: Record<string, unknown> = Object.fromEntries(Object.entries(fields).filter(([, value]) => value))

  if (env.HERMES_TUI_TOOLSETS) { options.toolsets = env.HERMES_TUI_TOOLSETS.split(',') }

  // `hermes --tui -s a,b`: the owner renders these into the session's frozen prompt.
  const skills = [...new Set((env.HERMES_TUI_SKILLS ?? '').split(/[,\n]/).map(name => name.trim()).filter(Boolean))]

  if (skills.length) { options.skills = skills }

  // The launcher exports `--max-turns 5` as the string "5": numbers ride as
  // ints (0 / -1 = unlimited), and a word ("none", "unlimited") stays text
  // for the policy to read — `Number` would turn it into NaN, sent as null.
  if (env.HERMES_TUI_MAX_TURNS) {
    const turns = Number(env.HERMES_TUI_MAX_TURNS)
    options.max_turns = Number.isInteger(turns) ? turns : env.HERMES_TUI_MAX_TURNS.trim()
  }

  // `hermes --tui --yolo` / `--ignore-rules` export HERMES_YOLO_MODE / HERMES_IGNORE_RULES (the in-process
  // TUI read them); over the canonical gateway they must ride session.create as frozen launch flags or the
  // session prompts anyway / still loads rules, SOUL, memory and skills.auto_load.
  const truthy = (value?: string) => ['1', 'true', 'yes', 'on'].includes((value ?? '').trim().toLowerCase())

  // `--checkpoints` / `--pass-session-id` / `--accept-hooks` (and an exported HERMES_ACCEPT_HOOKS=1)
  // are this session's creation policy too: the owner applies them, nothing reads them in-process.
  for (const [field, value] of [
    ['yolo', env.HERMES_YOLO_MODE], ['ignore_rules', env.HERMES_IGNORE_RULES],
    ['safe_mode', env.HERMES_SAFE_MODE], ['ignore_user_config', env.HERMES_IGNORE_USER_CONFIG],
    ['checkpoints', env.HERMES_TUI_CHECKPOINTS], ['pass_session_id', env.HERMES_TUI_PASS_SESSION_ID],
    ['accept_hooks', env.HERMES_ACCEPT_HOOKS]
  ] as const) {
    if (truthy(value)) { options[field] = true }
  }

  return options
}

const TOOL_PROGRESS_MODES = new Set(['off', 'new', 'all', 'verbose'])

/** `hermes --tui -v` / `-Q` export HERMES_TUI_TOOL_PROGRESS (verbose / off). The owner carries it as the
 * session-scoped tool-progress mode (`config.set key=verbose`, the same control `/verbose` writes), so a
 * fresh session is pinned right after creation. An unreadable value refuses the launch, never drops it. */
export function launchToolProgress(env = process.env): string | undefined {
  const raw = env.HERMES_TUI_TOOL_PROGRESS?.trim().toLowerCase()

  if (!raw) { return undefined }

  if (!TOOL_PROGRESS_MODES.has(raw)) { throw new Error(t('canonical.launch.invalidToolProgress', raw)) }

  return raw
}
