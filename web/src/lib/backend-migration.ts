/**
 * The System page's "move to a single Hermes" card: `/api/gateway/migrate/backends/*` (hermes gateway
 * migrate --backends). The backend owns every fact; this module turns its JSON into plain words for a
 * non-technical reader and reads one status poll into one of the card's phases.
 *
 * The poll reading follows the same route contract as Desktop's Web Access banner; both copies are
 * meant to move into @hermes/shared together.
 *
 * The click may come from the old dashboard, which the move stops: after the swap the SAME address is
 * answered by the gateway (same routes, different sign-in cookies), so the status poll answers 401
 * there, and also when a rollback restarted the old dashboard with a new signing key. A 401 after the
 * move started means "sign in again at this same address", never "failed".
 */
import { authedFetch } from "@/lib/api";

export type RunState = "none" | "running" | "done" | "rolled_back" | "failed" | "restored";

export interface BackendMigrationStatus {
  state: RunState;
  step: string;
  steps: { name: string; state: string }[];
  message: string;
  /** May be null or empty (a refused run, or a record whose address is unknown). */
  old_address: string | null;
  new_address: string | null;
  /** Server time the run started. */
  started_at: number;
  finished_at: number | null;
  manual_steps?: string[];
  /** The background run refused to start (nothing changed); `message` says why. */
  refused?: boolean;
  /** A `running` record only: whether its process still lives. */
  alive?: boolean;
  /** A failed or rolled-back run: why, without the rollback suffix. */
  reason?: string | null;
  log_tail?: string[];
  served_by?: string;
  /** The profile the answer is about (the name `hermes -p` takes), when the home is a profile's. */
  profile?: string | null;
  /** The engine's own spelling of the commands the card shows (with its root pin). */
  commands?: MigrateCommands;
}

/** `hermes [-p NAME] gateway migrate --backends --status|--restore`, as the server spells them for this
 *  home: a profile outside the standard root needs `HERMES_HOME=ROOT` pinned, which only the engine knows. */
export interface MigrateCommands {
  status: string;
  restore: string;
}

export interface BackendMigrationPlan {
  old_address: string | null;
  new_address: string | null;
  old_backend: {
    supervisor: string;
    unit: string | null;
    scope: string | null;
    pid: number | null;
    was_active: boolean;
  } | null;
  keep_webapp_address: boolean;
  copy_env_keys: string[];
  generate_secret: boolean;
  public_url: string | null;
  auth_before: { auth_required: boolean; providers: string[] } | null;
  gateway: { installed?: boolean; active?: boolean };
  linger: boolean | null;
  webapp_bundle_missing: string | null;
  /** null: the old backend's turn activity could not be read (never 0, never idle). */
  running_turns: number | null;
  blockers: string[];
  unsupported: { supervisor: string | null; unit: string | null; instructions: string; steps?: string[] }[];
  notices: string[];
  nothing_to_do: boolean;
  /** An old backend was found (movable or not). A blocker alone (an unreadable config.yaml) is no
   *  finding: such a plan is not "nothing to do", yet nothing runs here. */
  found?: boolean;
  text: string[];
  /** Opaque token of what this plan would move; sent back with the apply as `expect` (the route, and the
   *  background run under its lock, refuse with `plan_changed` when the fresh plan's token differs). */
  expect_token?: string;
}

export interface BackendMigrationPlanResponse {
  exit_code: number;
  plan: BackendMigrationPlan;
  run: BackendMigrationStatus;
  profile?: string | null;
  commands?: MigrateCommands;
}

export interface ApplyFlags {
  address?: string;
  keep_webapp_address?: boolean;
  force_interrupt?: boolean;
  expect?: string;
}

export interface Refusal {
  ok: false;
  error: string;
  message: string;
  blockers?: string[];
  command?: string;
  /** `plan_changed`: the fresh plan, to show and confirm again. */
  plan?: BackendMigrationPlan;
}

export interface Started {
  ok: true;
  /** Server time the move was asked for. */
  requested_at: number;
  /** False (HTTP 202): the launcher got no answer from the background run, so it may or may not run. */
  confirmed?: boolean;
  old_address: string | null;
  new_address: string | null;
}

const ROOT = "/api/gateway/migrate/backends";

/** `localhost:9119` / `0.0.0.0:9119` → the port, for words like "on port 9119". */
export function portOf(address: string | null | undefined): string {
  const port = (address ?? "").split(":").pop() ?? "";
  return /^\d+$/.test(port) ? port : "";
}

const REPORTED_RUNS: RunState[] = ["done", "rolled_back", "failed", "running"];

/** Evidence of an old backend in the plan: never just a blocker. */
export function foundBackend(plan: BackendMigrationPlan): boolean {
  return plan.found ?? (!plan.nothing_to_do && (plan.old_backend !== null || plan.unsupported.length > 0));
}

/** True when the card should show at all: an old server was found or the last run needs reporting. */
export function cardVisible(plan: BackendMigrationPlanResponse | null): boolean {
  if (!plan) return false;
  return foundBackend(plan.plan) || REPORTED_RUNS.includes(plan.run.state);
}

/** Every `hermes …` command in *text* with `-p <profile>`: a manual step for this home must never act on
 *  another one. Commands that already name a profile, paths and unit names are left alone. */
export function qualifyCommands(text: string, profile: string | null | undefined): string {
  if (!profile) return text;
  return text.replace(/(^|[^\w./-])hermes (?!-p |--profile)/g, `$1hermes -p ${profile} `);
}

/** `hermes -p <profile> gateway migrate --backends <flag>`. `-p` is given whenever the profile is known:
 *  without it a sticky `hermes profile use` choice would pick another home. */
export function migrateCommand(
  profile: string | null | undefined,
  flag: "--restore" | "--status",
  commands?: MigrateCommands | null,
): string {
  const given = commands?.[flag === "--status" ? "status" : "restore"];
  if (given) return given;
  return `hermes ${profile ? `-p ${profile} ` : ""}gateway migrate --backends ${flag}`;
}

/** The finding, in plain words. */
export function findingSentence(plan: BackendMigrationPlan): string {
  const port = portOf(plan.old_address);
  const where = port ? `on port ${port}` : "on this computer";
  return `An older Hermes server is running ${where}. Phones and browsers that use it talk to a second copy of Hermes.`;
}

/** What the move does and what people will notice, one line each. */
export function impactLines(plan: BackendMigrationPlan): string[] {
  const lines: string[] = [];
  const same = plan.old_address && plan.old_address === plan.new_address;
  // An address the engine could not read is null: never printed, never guessed.
  const at = plan.new_address ? ` ${plan.new_address}` : " its address";
  lines.push(
    plan.keep_webapp_address
      ? `The older server is switched off; the web app stays${plan.new_address ? ` at ${plan.new_address}` : " where it is"}.`
      : same
        ? `Hermes itself takes over the same address (${plan.old_address}); nothing becomes reachable from anywhere new.`
        : `Hermes itself takes over${at}.`,
  );
  if (plan.auth_before?.auth_required) {
    lines.push("The same username and password keep working. Everyone signs in once more afterwards.");
  } else {
    lines.push("Everyone signs in once more afterwards.");
  }
  lines.push("Browsers get the new Hermes web app instead of this dashboard page.");
  if (plan.webapp_bundle_missing) {
    lines.push("The web app is built first; this can take a few minutes.");
  }
  if (!plan.gateway.installed) {
    lines.push("Hermes is set up to start by itself when this computer starts.");
  }
  lines.push("The address is unavailable for a few seconds during the switch.");
  lines.push("If anything goes wrong, everything is put back the way it was.");
  return lines;
}

/** Blocker text from the engine, with what to do. The engine's sentences already end in the remedy
 *  (`Wait…`, `Re-run with…`); flags are translated into what this card offers. */
export function blockerAdvice(blocker: string, byHand = false): { text: string; canForce: boolean } {
  const canForce = blocker.includes("--force-interrupt");
  // A by-hand plan cannot be moved from the card: forcing it only shows its steps.
  const running = byHand ? ". Or show the steps anyway (following them cuts those replies off)."
    : ". Or move anyway (those replies are cut off).";
  const unknown = byHand ? ". Or show the steps anyway (following them cuts off any reply being written)."
    : ". Or move anyway (any reply being written is cut off).";
  // "To move it, confirm it is idle, then re-run with --force-interrupt": the card's own action instead.
  const unknownThen = byHand ? ", then show the steps anyway (following them cuts off any reply being written)."
    : ", then move anyway (any reply being written is cut off).";
  const text = blocker
    .replace(/,? ?or re-run with --force-interrupt \(they are interrupted\)\.?/, running)
    .replace(/,? ?or re-run with --force-interrupt \(any running turn is interrupted\)\.?/, unknown)
    .replace(/,? then re-run with --force-interrupt( \(any running turn is interrupted\))?\.?/, unknownThen)
    .replace(/Re-run with --detach\.?/, "");
  return { text: text.trim(), canForce };
}

/** The reason of a failed or rolled-back run: the route's `reason`, else the engine's message
 *  (`<reason>; rolled back, old backend verified on …`) without its rollback suffix. */
export function reasonOf(status: BackendMigrationStatus): string {
  if (status.reason) return status.reason;
  return (status.message.split("; rolled back")[0] ?? status.message).split("; rollback incomplete")[0].trim();
}

/** A rolled-back run's reason as the card shows it. A move the engine stopped before the old backend was
 *  stopped (a chat started after the preflight) gives the preflight's turn blocker: without the step
 *  prefix, and with "try again" (the card offers it) in place of the CLI flag. Other reasons are as given. */
export function rolledBackReason(reason: string): string {
  if (!reason.endsWith("The old backend has not been stopped.")) return reason;
  return reason
    .replace(/^[a-z_]+ failed: /, "")
    .replace(/,? ?or re-run with --force-interrupt \(they are interrupted\)\./, ", then try again.")
    .replace(/,? then re-run with --force-interrupt( \(any running turn is interrupted\))?\./, ", then try again.");
}

/** The engine's per-file manual-step notes (a file put back, kept for its edits, whose backup is missing or
 *  unreadable, or that the migration stopped while writing): words to follow, not a command to copy. */
export function isFileNote(step: string): boolean {
  return / was restored from its backup \(nothing to do for it\)$/.test(step)
    || /^keep \S.* as it is \(it was edited after the migration wrote it\): /.test(step)
    || /^the backup \S.* of \S+ is missing or cannot be read: /.test(step)
    || /^the migration stopped while writing \S.*, so it may hold the migration's changes: /.test(step);
}

/** The engine's account of an incomplete rollback (`<reason>; rollback incomplete: <detail>`), or "". */
export function incompleteDetail(status: BackendMigrationStatus): string {
  return (status.message.split("; rollback incomplete: ")[1] ?? "").trim();
}

/** A `running` record whose process is gone: `alive: false`, or, from a server without `alive`, the
 *  engine's own "interrupted" note on the message. */
export function runDied(status: BackendMigrationStatus): boolean {
  return status.alive === undefined ? status.message.includes("interrupted: the next run rolls it back") : !status.alive;
}

/** The card's phase. `checking`: the apply answer was lost (or the server said a move was already
 *  starting), so which run is ours is not known yet; `unknown`: still not known after the wait (check
 *  the status; never "did not start"). `stalled`: the record says running but its process is gone.
 *  `unreachable`: no answer about this run for longer than a restart takes. */
export type PollView =
  | { kind: "starting" }
  | { kind: "checking" }
  | { kind: "running"; step: string }
  | { kind: "switching" }
  | { kind: "signInAgain" }
  | { kind: "done"; address: string }
  | { kind: "rolledBack"; reason: string }
  | { kind: "manual"; reason: string; detail: string; steps: string[] }
  | { kind: "restored" }
  | { kind: "notStarted"; lines: string[] }
  | { kind: "stalled"; step: string }
  | { kind: "unknown" }
  | { kind: "unreachable" };

/** A status that belongs to the run being shown. */
export function viewOf(st: BackendMigrationStatus): PollView {
  switch (st.state) {
    case "done":
      return { kind: "done", address: st.new_address ?? "" };
    case "rolled_back":
      return { kind: "rolledBack", reason: rolledBackReason(reasonOf(st)) };
    case "failed":
      // A refused background run changed nothing: say why it did not start, not "could not be undone".
      if (st.refused) {
        const steps = st.manual_steps ?? [];
        // The CLI's own advice, true for a refused apply too: the steps may replace config.yaml and .env.
        // Per-file notes already say what to do with each file; the general warning would then repeat them,
        // unless another step still copies a backup over config.yaml or .env.
        const copiesBack = steps.some((s) => !isFileNote(s) && /config\.yaml|\.env\b/.test(s));
        const warning = steps.length && (!steps.some(isFileNote) || copiesBack)
          ? ["Before running these steps, save a copy of your current config.yaml and .env: the steps may "
             + "replace or remove them. To keep changes you made to those files, merge them into any backup "
             + "copy a step copies back (before you run it), or re-apply them after the last step."]
          : [];
        return { kind: "notStarted", lines: [st.message, ...warning, ...steps] };
      }
      // What the rollback could not do (a file kept for its edits, an old backend left down), as the engine says it.
      return { kind: "manual", reason: reasonOf(st), detail: incompleteDetail(st), steps: st.manual_steps ?? [] };
    case "restored":
      return { kind: "restored" };
    case "running":
      return runDied(st) ? { kind: "stalled", step: st.step } : { kind: "running", step: st.step };
    default:
      return { kind: "starting" };
  }
}

/** Seconds after the request with no record of the new run before its outcome counts as unknown. */
export const NOT_STARTED_AFTER_SECONDS = 60;
/** Seconds without an answer about the followed run before a poll stops calling it the restart (the
 *  move's own verify waits 90 s for the gateway, and its rollback starts the older server again). */
export const UNREACHABLE_AFTER_SECONDS = 150;

/** Which run a poll follows. `requestedAt`: the server time the apply answer named, or `null` when that
 *  answer was lost (or the server said a move was already starting); then any record newer than
 *  `previousStartedAt` (the last record seen before asking) is the run. `confirmed`: the background run
 *  said it holds the run lock; without that, "starting" reads as "finding out". */
export interface PollRef {
  previousStartedAt: number;
  requestedAt: number | null;
  confirmed?: boolean;
}

export type StatusRead =
  | { kind: "status"; status: BackendMigrationStatus }
  | { kind: "auth" }
  | { kind: "unreachable" };

export function isThisRun(status: BackendMigrationStatus, ref: PollRef): boolean {
  return ref.requestedAt === null ? status.started_at > ref.previousStartedAt : status.started_at >= ref.requestedAt - 1;
}

/** One poll. Until the new run writes its record the previous run's status is still on file, so only a
 *  newer record counts. Only a record says how a run went: its own refusal record ("did not start"), or
 *  its progress. No record for a minute is never taken as "did not start": it is unknown, and the card
 *  offers a status check. `quietSeconds`: time since the last answer about this run. */
export function readPoll(
  read: StatusRead,
  ref: PollRef,
  timing: { quietSeconds: number; sinceSeconds: number },
): PollView {
  if (read.kind === "auth") return { kind: "signInAgain" };
  if (read.kind === "status" && isThisRun(read.status, ref)) return viewOf(read.status);
  if (timing.quietSeconds > UNREACHABLE_AFTER_SECONDS) return { kind: "unreachable" };
  if (read.kind === "unreachable") return { kind: "switching" };
  if (timing.sinceSeconds > NOT_STARTED_AFTER_SECONDS) return { kind: "unknown" };
  return ref.requestedAt !== null && ref.confirmed !== false ? { kind: "starting" } : { kind: "checking" };
}

/** Still moving, or still finding out: keep polling and offer no new move. */
export function isInFlight(view: PollView | null): boolean {
  return view?.kind === "starting" || view?.kind === "running" || view?.kind === "switching" || view?.kind === "checking";
}

/** The outcome is not known: offer a check, never a new move. */
export function isUnresolved(view: PollView | null): boolean {
  return view?.kind === "signInAgain" || view?.kind === "stalled" || view?.kind === "unknown" || view?.kind === "unreachable";
}

/** What an apply answer means. `started`: the run was asked for at server time `requestedAt`;
 *  `confirmed` false (HTTP 202) when nobody confirmed it holds the run lock.
 *  `refused`: the route's own 4xx refusal body proves nothing started (an `unfinished_run` refusal names
 *  the terminal command that undoes the earlier run). `running`: a move is already starting or running
 *  here (a second click, another window), so follow the status and never apply again. `lost`: no
 *  answer, or one that proves nothing (a dropped connection, a proxy's 502, a 5xx): the status decides. */
export type ApplyOutcome =
  | { kind: "started"; requestedAt: number; confirmed: boolean }
  | { kind: "refused"; refusal: Refusal }
  | { kind: "running" }
  | { kind: "lost" };

export function applyOutcome(res: { status: number; body: Started | Refusal | null }): ApplyOutcome {
  const body = res.body;
  if ((res.status === 200 || res.status === 202) && body?.ok === true && typeof body.requested_at === "number") {
    return { kind: "started", requestedAt: body.requested_at, confirmed: res.status === 200 && body.confirmed !== false };
  }
  if (res.status < 400 || res.status >= 500 || !body || body.ok !== false || typeof body.error !== "string") {
    return { kind: "lost" };
  }
  return body.error === "already_running" ? { kind: "running" } : { kind: "refused", refusal: body };
}

const STEP_WORDS: Record<string, string> = {
  preflight: "Checking",
  manifest: "Saving how things are now",
  webapp_bundle: "Building the web app",
  write_config: "Writing the settings",
  gateway_service: "Setting Hermes up to start by itself",
  stop_old_backend: "Switching off the older server",
  start_gateway: "Starting Hermes on the same address",
  verify: "Checking the address answers",
};

export function stepWords(step: string): string {
  return STEP_WORDS[step] ?? step.replace(/_/g, " ");
}

// ── HTTP ─────────────────────────────────────────────────────────────
// authedFetch adds the page's management profile as `?profile=` (`/api/gateway` is a profile-scoped
// family), so every call acts on the profile the dashboard manages.

async function call<T>(method: string, path: string, body?: unknown): Promise<{ status: number; body: T | null }> {
  // authedFetch, not fetchJSON: a 401 here is expected after the swap and must not navigate away.
  let res: Response;
  try {
    res = await authedFetch(`${ROOT}${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    return { status: 0, body: null };
  }
  const parsed = (await res.json().catch(() => null)) as T | null;
  return { status: res.status, body: parsed };
}

async function readStatus(): Promise<StatusRead> {
  const res = await call<BackendMigrationStatus>("GET", "/status");
  if (res.status === 401 || res.status === 403) return { kind: "auth" };
  if (res.status === 200 && res.body) return { kind: "status", status: res.body };
  return { kind: "unreachable" };
}

export const backendMigrationApi = {
  /** `force_interrupt`: the read-only plan past a turn-activity blocker (a by-hand plan then lists its steps). */
  plan: (flags?: { force_interrupt?: boolean }) =>
    call<BackendMigrationPlanResponse>("GET", flags?.force_interrupt ? "/plan?force_interrupt=true" : "/plan"),
  apply: (flags: ApplyFlags) => call<Started | Refusal>("POST", "/apply", flags),
  status: readStatus,
};
