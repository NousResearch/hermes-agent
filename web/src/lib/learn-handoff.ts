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
  onSocketClose(socket: LearnSocketLike): void;
  onTransportTeardown(): void;
  onUserInput(): void;
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
// eslint-disable-next-line no-control-regex -- intentional Fp/Fe/Fs two-byte and nF ESC sequence stripping
const SINGLE_ESCAPE_RE = /\u001b(?:[ -/]+[@-~]|[0-~])/g;
// eslint-disable-next-line no-control-regex -- intentional C0/C1, DEL, line-separator, and BiDi cleanup
const CONTROL_AND_BIDI_RE = /[\u0000-\u001f\u007f-\u009f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]+/g;
const WHITESPACE_RE = /\s+/g;

const LEARN_SEED_KEY = "learnSeed";
const WS_READY_STATE_OPEN = 1;
const DEFAULT_COMPOSER_READY_DELAY_MS = 800;

// Matches terminal-generated reports/replies and mouse/focus events emitted
// through xterm's onData (CPR/DSR, DA1/DA2/DA3, DECRPM, XTWINOPS, Kitty kbd
// query reply, focus in/out, OSC/DCS replies, SGR/X10 mouse). User typing,
// editing keys, and bracketed paste (\x1b[200~...) never match.
const TERMINAL_REPLY_ONLY_RE =
  // eslint-disable-next-line no-control-regex -- intentional terminal-reply detection for PTY input
  /^(?:\u001b\[(?:[IO]|<\d+;\d+;\d+[Mm]|M[\s\S]{3}|[?>=][0-9;]*c|\?\d+u|\??[0-9;]+(?:[Rnt]|\$y))|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)|\u001bP[^\u001b]*\u001b\\)+$/;

/**
 * True when an xterm `onData` chunk represents user-initiated input (typing,
 * editing keys, or bracketed paste) rather than terminal-generated replies or
 * mouse/focus reports.
 */
export function isUserInitiatedTerminalInput(data: string): boolean {
  if (!data) {
    return false;
  }
  return !TERMINAL_REPLY_ONLY_RE.test(data);
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Collapse an in-app `/learn` seed into one printable terminal line by
 * removing terminal escape sequences, control characters, and BiDi controls.
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
 * Coordinate one-shot `/learn` delivery against the chat PTY socket lifecycle:
 * - Only queues when no PTY socket is already open (avoids submitting or
 *   corrupting an unfinished draft in an already-live composer).
 * - Waits for Ink's composer mount delay after socket open before sending once.
 * - Retains an unsent command across transient transport reconnects, while
 *   ignoring stale close events from retired sockets and canceling on user
 *   input or scope change/unmount.
 */
export function createLearnHandoffController(
  options: LearnHandoffControllerOptions = {},
): LearnHandoffController {
  const delayMs = options.delayMs ?? DEFAULT_COMPOSER_READY_DELAY_MS;
  const setTimer = options.setTimer ?? ((cb, ms) => setTimeout(cb, ms));
  const clearTimer = options.clearTimer ?? ((timer) => clearTimeout(timer));

  let activeSocket: LearnSocketLike | null = null;
  let pendingCommand: string | null = null;
  let flushTimer: ReturnType<typeof setTimeout> | null = null;

  const cancelTimer = (): void => {
    if (flushTimer !== null) {
      clearTimer(flushTimer);
      flushTimer = null;
    }
  };

  return {
    enqueue(seed) {
      const cmd = formatLearnCommand(seed);
      if (!cmd) return false;
      // Never inject into an already-open PTY session: the user may have an
      // unfinished draft or active turn in Ink's composer.
      if (activeSocket !== null && activeSocket.readyState === WS_READY_STATE_OPEN) {
        return false;
      }
      pendingCommand = cmd;
      return true;
    },

    onSocketOpen(socket) {
      cancelTimer();
      activeSocket = socket;
      if (!pendingCommand) return;
      flushTimer = setTimer(() => {
        flushTimer = null;
        if (
          activeSocket !== socket ||
          socket.readyState !== WS_READY_STATE_OPEN ||
          !pendingCommand
        ) {
          return;
        }
        const cmd = pendingCommand;
        pendingCommand = null;
        try {
          socket.send(`${cmd}\r`);
        } catch {
          /* PTY not ready / closed — user can retype */
        }
      }, delayMs);
    },

    onSocketClose(socket) {
      if (activeSocket !== socket) return;
      cancelTimer();
      activeSocket = null;
    },

    onTransportTeardown() {
      cancelTimer();
      activeSocket = null;
    },

    onUserInput() {
      cancelTimer();
      pendingCommand = null;
    },

    reset() {
      cancelTimer();
      activeSocket = null;
      pendingCommand = null;
    },
  };
}
