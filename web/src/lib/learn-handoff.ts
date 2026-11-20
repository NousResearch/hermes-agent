export interface LearnNavigationState {
  readonly learnSeed: string;
}

export interface LearnSocketLike {
  readonly readyState: number;
  send(data: string): void;
}

export interface LearnHandoffController {
  enqueue(seed: string | null | undefined): boolean;
  onSocketOpen(socket: LearnSocketLike): void;
  onSocketClose(): void;
  reset(): void;
}

export interface LearnHandoffControllerOptions {
  readonly delayMs?: number;
  readonly setTimer?: (callback: () => void, ms: number) => ReturnType<typeof setTimeout>;
  readonly clearTimer?: (timer: ReturnType<typeof setTimeout>) => void;
}

// Linear-time ECMA-48 / VT escape sequence matchers run before C0/C1 cleanup
// so escape parameters and OSC/DCS payloads are dropped rather than left behind
// as printable text in the PTY composer.
// eslint-disable-next-line no-control-regex -- intentional OSC sequence stripping for PTY-bound text
const OSC_SEQUENCE_RE = /(?:\u001b\]|\u009d)[^\u0007\u001b\u009c]*(?:\u0007|\u001b\\|\u009c)?/g;
// eslint-disable-next-line no-control-regex -- intentional DCS/SOS/PM/APC sequence stripping for PTY-bound text
const STRING_ESCAPE_RE = /(?:\u001b[PX^_]|[\u0090\u0098\u009e\u009f])[^\u001b\u009c]*(?:\u001b\\|\u009c)?/g;
// eslint-disable-next-line no-control-regex -- intentional 7-bit/8-bit CSI sequence stripping for PTY-bound text
const CSI_SEQUENCE_RE = /(?:\u001b\[|\u009b)[0-?]*[ -/]*[@-~]/g;
// eslint-disable-next-line no-control-regex -- intentional two-byte and nF ESC sequence stripping for PTY-bound text
const SINGLE_ESCAPE_RE = /\u001b(?:[ -/]+[@-~]|[@-Z\\-_])/g;
// eslint-disable-next-line no-control-regex -- intentional C0/C1, DEL, line-separator, and BiDi cleanup
const CONTROL_AND_BIDI_RE = /[\u0000-\u001f\u007f-\u009f\u2028\u2029\u202a-\u202e\u2066-\u2069]+/g;
const WHITESPACE_RE = /\s+/g;

const LEARN_SEED_KEY = "learnSeed";
const WS_READY_STATE_OPEN = 1;
const DEFAULT_COMPOSER_READY_DELAY_MS = 800;

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Collapse an in-app `/learn` seed into one printable terminal line by
 * removing terminal escape sequences, control characters, and BiDi overrides.
 */
export function normalizeLearnSeed(seed: string | null | undefined): string | null {
  if (!seed) {
    return null;
  }

  const normalized = seed
    .replace(OSC_SEQUENCE_RE, " ")
    .replace(STRING_ESCAPE_RE, " ")
    .replace(CSI_SEQUENCE_RE, " ")
    .replace(SINGLE_ESCAPE_RE, " ")
    .replace(CONTROL_AND_BIDI_RE, " ")
    .replace(WHITESPACE_RE, " ")
    .trim();

  return normalized || null;
}

/** Format a normalized seed into a single-line `/learn` composer command. */
export function formatLearnCommand(seed: string | null | undefined): string | null {
  const normalized = normalizeLearnSeed(seed);
  return normalized ? `/learn ${normalized}` : null;
}

/** Build the React Router location state payload for an in-app Skills -> Chat handoff. */
export function buildLearnNavigationState(
  seed: string | null | undefined,
): LearnNavigationState | null {
  const normalized = normalizeLearnSeed(seed);
  return normalized ? { learnSeed: normalized } : null;
}

/** True when router location state carries a `learnSeed` handoff property. */
export function hasLearnSeedState(state: unknown): boolean {
  return isPlainRecord(state) && Object.prototype.hasOwnProperty.call(state, LEARN_SEED_KEY);
}

/** Extract and normalize `learnSeed` from React Router location state. */
export function readLearnSeedFromState(state: unknown): string | null {
  if (!hasLearnSeedState(state)) {
    return null;
  }
  const raw = (state as Record<string, unknown>)[LEARN_SEED_KEY];
  return typeof raw === "string" ? normalizeLearnSeed(raw) : null;
}

/** Remove `learnSeed` from router location state while preserving any sibling keys. */
export function stripLearnSeedFromState(state: unknown): Record<string, unknown> | null {
  if (!isPlainRecord(state)) {
    return null;
  }
  if (!Object.prototype.hasOwnProperty.call(state, LEARN_SEED_KEY)) {
    return state;
  }
  const rest: Record<string, unknown> = {};
  let hasKeys = false;
  for (const [key, value] of Object.entries(state)) {
    if (key === LEARN_SEED_KEY) continue;
    rest[key] = value;
    hasKeys = true;
  }
  return hasKeys ? rest : null;
}

/**
 * Coordinate one-shot `/learn` command delivery against the chat PTY socket
 * lifecycle so Ink's composer has mounted before initial submission, while
 * warm in-app handoffs deliver immediately and closed sockets cancel timers.
 */
export function createLearnHandoffController(
  options: LearnHandoffControllerOptions = {},
): LearnHandoffController {
  const delayMs = options.delayMs ?? DEFAULT_COMPOSER_READY_DELAY_MS;
  const setTimer = options.setTimer ?? ((cb, ms) => setTimeout(cb, ms));
  const clearTimer = options.clearTimer ?? ((timer) => clearTimeout(timer));

  let activeSocket: LearnSocketLike | null = null;
  let composerReady = false;
  let pendingCommand: string | null = null;
  let flushTimer: ReturnType<typeof setTimeout> | null = null;

  const cancelTimer = (): void => {
    if (flushTimer !== null) {
      clearTimer(flushTimer);
      flushTimer = null;
    }
  };

  const flush = (): void => {
    if (!composerReady || !pendingCommand || !activeSocket) return;
    if (activeSocket.readyState !== WS_READY_STATE_OPEN) return;
    const cmd = pendingCommand;
    pendingCommand = null;
    try {
      activeSocket.send(`${cmd}\r`);
    } catch {
      /* PTY not ready / closed — user can retype */
    }
  };

  return {
    enqueue(seed) {
      const cmd = formatLearnCommand(seed);
      if (!cmd) return false;
      pendingCommand = cmd;
      flush();
      return true;
    },

    onSocketOpen(socket) {
      cancelTimer();
      activeSocket = socket;
      composerReady = false;
      flushTimer = setTimer(() => {
        flushTimer = null;
        if (activeSocket !== socket || socket.readyState !== WS_READY_STATE_OPEN) {
          return;
        }
        composerReady = true;
        flush();
      }, delayMs);
    },

    onSocketClose() {
      cancelTimer();
      activeSocket = null;
      composerReady = false;
    },

    reset() {
      cancelTimer();
      activeSocket = null;
      composerReady = false;
      pendingCommand = null;
    },
  };
}
