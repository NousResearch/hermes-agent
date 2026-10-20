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

    // The shell's environment is context, not a launch option: a gateway predating it runs the session
    // with its own environment, as before.
    if (!contract.parameters.includes('client_env')) { delete params.client_env }
    // Same for the launch-directory provenance flag: an older owner still accepts cwd, and
    // refusing the whole create would block `hermes --tui` against that gateway.
    if (!contract.parameters.includes('cwd_explicit')) { delete params.cwd_explicit }
    const unsupported = Object.keys(params).filter(key => !contract.parameters.includes(key))

    if (unsupported.length) { throw new Error(t('canonical.launch.unsupportedOptions', unsupported.join(', '))) }
  }

  if (['prompt.submit', 'session.steer', 'session.redirect'].includes(method) && params.submission_id) {
    params.input_id = params.submission_id
    delete params.submission_id
  }

  if (method === 'session.resume' || method === 'session.activate') {
    method = 'session.resume'
    // A rendering resume asks for main's display projection of the same rows (`display_messages`);
    // a metadata-only activation (`omit_messages`) never paints a transcript.
    const renders = !params.omit_messages
    delete params.cols
    delete params.omit_messages

    if (renders) { params.display_projection = true }
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
    const stored = Array.isArray(value.messages) ? value.messages.map((row: any) =>
      row && typeof row === 'object'
        ? { ...row, text: rowText(row.content), name: row.tool_name }
        : row) : value.messages

    // Renderers read `messages`: main's display projection when the owner sent it (compaction
    // handoffs stripped, /steer unwrapped, tool rows labelled), as the sidecar's resume did.
    // `stored_messages` keeps the raw rows (content + row_id) the rewind controls target.
    const { display_messages: display, ...rest } = value
    const messages = Array.isArray(display) ? display : stored

    return { ...rest, messages, stored_messages: stored, info: { ...value.info, stored_session_id: value.stored_session_id,
      execution_epoch: String(value.authority_epoch), execution_generation: value.execution_generation,
      running: value.running, pending_submissions: pendingSubmissions(value.pending) } }
  }

  return value
}

// A stored multipart row (`content` is a parts list) as display text: text parts joined, an image
// part as `[image]` (N94). `content` itself stays on the row for /undo targeting.
function rowText(content: unknown): unknown {
  if (!Array.isArray(content)) { return content }

  return content.map(part => typeof part === 'string' ? part : typeof part?.text === 'string' ? part.text
    : part?.type ? (String(part.type).includes('image') ? '[image]' : `[${part.type}]`) : '').filter(Boolean).join('\n')
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

  // `hermes --tui` sets HERMES_CWD to the directory the process was launched from
  // (hermes_cli/main_tui_launch.py). That is a deliberate workspace, not the Desktop's
  // inherited app-global default: mark it explicit so a named profile's terminal.cwd
  // does not replace it, and the shared gateway does not fall back to $HOME (#84015, #52589).
  if (options.cwd) { options.cwd_explicit = true }

  if (env.HERMES_TUI_TOOLSETS) { options.toolsets = env.HERMES_TUI_TOOLSETS.split(',') }

  // `hermes --tui -s a,b`: the owner renders these into the session's frozen prompt.
  const skills = [...new Set((env.HERMES_TUI_SKILLS ?? '').split(/[,\n]/).map(name => name.trim()).filter(Boolean))]

  if (skills.length) { options.skills = skills }

  // The launcher's shell environment (allowlisted by `hermes --tui`): this session's HERMES_* behaviour
  // values and tool PATH/venv on the shared gateway, as main's own process had them.
  const clientEnv = parseClientEnv(env.HERMES_TUI_CLIENT_ENV)

  if (clientEnv) { options.client_env = clientEnv }

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

function parseClientEnv(raw?: string): Record<string, string> | undefined {
  if (!raw) { return undefined }

  try {
    const value: unknown = JSON.parse(raw)

    if (value && typeof value === 'object' && !Array.isArray(value)
      && Object.values(value).every(item => typeof item === 'string')) {
      return value as Record<string, string>
    }
  } catch {
    // An unreadable hand-off sends nothing: the session keeps the gateway's environment.
  }

  return undefined
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
