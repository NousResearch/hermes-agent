const DELETE = "\x7f";

// How long (ms) after a mobile IME / replacement event we treat subsequent
// terminal input as a candidate line-replacement rather than a plain append.
// Exported so the ChatPage integration and tests share one tunable value.
export const MOBILE_REPLACEMENT_WINDOW_MS = 350;

// Slash-command autocomplete list for the mobile textarea, mirroring
// hermes_cli/commands.py (COMMAND_REGISTRY) — the same source `/help`,
// Telegram and Slack menus derive from. Open-WebUI-style: typing "/" pops a
// matching command dropdown; Tab/Enter inserts the command into the buffer.
export const MOBILE_SLASH_COMMANDS: ReadonlyArray<{ cmd: string; hint: string }> = [
  { cmd: "/help", hint: "Show all commands" },
  { cmd: "/new", hint: "Fresh session (alias /reset)" },
  { cmd: "/clear", hint: "Clear screen + new session" },
  { cmd: "/retry", hint: "Resend last message" },
  { cmd: "/undo", hint: "Back up N user turns, re-prompt" },
  { cmd: "/title", hint: "Name this session" },
  { cmd: "/compress", hint: "Compress context (alias /compact)" },
  { cmd: "/stop", hint: "Kill background processes" },
  { cmd: "/status", hint: "Session, model, token info" },
  { cmd: "/model", hint: "Switch model" },
  { cmd: "/config", hint: "Show config" },
  { cmd: "/personality", hint: "Set a personality" },
  { cmd: "/reasoning", hint: "Reasoning effort/display" },
  { cmd: "/tools", hint: "Manage tools" },
  { cmd: "/skills", hint: "Search/install/manage skills" },
  { cmd: "/learn", hint: "Learn a reusable skill" },
  { cmd: "/memory", hint: "Review pending memory writes" },
  { cmd: "/cron", hint: "Manage scheduled tasks" },
  { cmd: "/bg", hint: "Run prompt in background session" },
  { cmd: "/queue", hint: "Queue prompt for next turn" },
  { cmd: "/steer", hint: "Inject message after next tool call" },
  { cmd: "/branch", hint: "Branch the session" },
  { cmd: "/resume", hint: "Resume a named session" },
  { cmd: "/sessions", hint: "Browse previous sessions" },
  { cmd: "/copy", hint: "Copy last response to clipboard" },
  { cmd: "/paste", hint: "Attach clipboard image" },
  { cmd: "/image", hint: "Attach a local image file" },
  { cmd: "/update", hint: "Update Hermes to latest" },
  { cmd: "/version", hint: "Show version" },
  { cmd: "/quit", hint: "Exit CLI (alias /exit)" },
];

// Pure helpers for the slash-command autocomplete — extracted so the
// behaviour is unit-testable without a DOM (the dropdown rendering lives in
// createMobileTextareaInput; these compute matches and edits).
export function mobileToken(value: string): string {
  const m = value.match(/(^|\s)(\/[\w-]*)\s*$/);
  return m ? m[2] : "";
}

export function mobileSlashMatches(token: string): ReadonlyArray<{ cmd: string; hint: string }> {
  if (!token.startsWith("/")) return [];
  const q = token.toLowerCase();
  let out = MOBILE_SLASH_COMMANDS.filter((c) =>
    c.cmd.toLowerCase().startsWith(q),
  );
  if (!out.length) {
    out = MOBILE_SLASH_COMMANDS.filter((c) =>
      c.cmd.toLowerCase().includes(q.slice(1)),
    );
  }
  return out;
}

export function mobileCommitToken(value: string, cmd: string): string {
  return value.replace(/(^|\s)\/[\w-]*\s*$/, (_whole: string, pre: string) => `${pre}${cmd} `);
}

function chars(text: string): string[] {
  return Array.from(text);
}

function removeLastChar(text: string): string {
  const c = chars(text);
  c.pop();
  return c.join("");
}

 
function isPlainText(data: string): boolean {
  // eslint-disable-next-line no-control-regex -- terminal data may contain control chars
  return !/[\x00-\x1f\x7f]/.test(data);
}

function lastWordMatch(line: string): RegExpMatchArray | null {
  return line.match(/^(.*?)(\S+)(\s*)$/u);
}

function collapseDuplicatedFinalWord(text: string, previousLine: string): string {
  const match = text.match(/^(.*?)(\S+)(\s+)(\S+)(\s*)$/u);
  if (!match) return text;

  const [, prefix, first, , second, trailing] = match;
  if (first.toLocaleLowerCase() !== second.toLocaleLowerCase()) return text;
  // Only collapse a duplication the tracked line already ended with — i.e.
  // Gboard re-emitted the final word. Requiring a >=2-char word avoids
  // eating legitimate single-letter reduplication ("a a", "i i") that a
  // user may genuinely type inside the replacement window.
  if (first.length < 2) return text;
  if (!previousLine.trimEnd().toLocaleLowerCase().endsWith(first.toLocaleLowerCase())) {
    return text;
  }
  return `${prefix}${first}${trailing}`;
}

function replacementLineForMobileInput(
  currentLine: string,
  incoming: string,
): string | null {
  if (!currentLine || currentLine.length < 2 || !incoming) return null;

  const currentLower = currentLine.toLocaleLowerCase();
  const incomingLower = incoming.toLocaleLowerCase();

  if (incomingLower.startsWith(currentLower)) {
    return collapseDuplicatedFinalWord(incoming, currentLine);
  }

  const word = lastWordMatch(currentLine);
  if (!word) return null;

  const [, prefix, last, trailing] = word;
  if (trailing) return null;

  const incomingFirst = incoming.trimStart().split(/\s+/u)[0] ?? "";
  if (
    incomingFirst &&
    incomingFirst.toLocaleLowerCase() === last.toLocaleLowerCase()
  ) {
    return `${prefix}${collapseDuplicatedFinalWord(incoming, currentLine)}`;
  }

  return null;
}

export function shouldTreatInputAsMobileReplacement(
  inputType: string | undefined,
  data: string | null | undefined,
  isMobileLike: boolean,
): boolean {
  if (
    inputType === "insertReplacementText" ||
    inputType === "insertFromComposition" ||
    inputType === "insertCompositionText"
  ) {
    return true;
  }
  return isMobileLike && inputType === "insertText" && (data?.length ?? 0) > 1;
}

export function updatePtyInputLine(currentLine: string, data: string): string {
  // Escape sequences (arrow keys, home/end, function keys, paste guards)
  // move the cursor or edit the line in ways this flat tracker cannot
  // model — and the per-char loop below would append their printable
  // payload (e.g. the "[D" of a left-arrow) as if it were typed text.
  // Reset instead: an unknown cursor position must disarm replacement
  // normalization until the user starts a fresh, cleanly-tracked line.
  if (data.includes("\x1b")) {
    return "";
  }
  let next = currentLine;
  // A fused CRLF/LFCR chunk is one Enter (see parse-keypress), not two
  // line endings — reset the tracked line once rather than twice so the
  // replacement tracker stays consistent with the enter fuser.
  if (data === "\r\n" || data === "\n\r") {
    return "";
  }
  for (const ch of chars(data)) {
    if (ch === "\r" || ch === "\n") {
      next = "";
    } else if (ch === DELETE || ch === "\b") {
      next = removeLastChar(next);
    } else if (ch === "\x15") {
      next = "";
    } else if (isPlainText(ch)) {
      next += ch;
    }
  }
  return next;
}

export function normalizePtyMobileInput(
  data: string,
  currentLine: string,
  replacementActive: boolean,
): { data: string; nextLine: string; normalized: boolean } {
  if (replacementActive && isPlainText(data)) {
    const replacementLine = replacementLineForMobileInput(currentLine, data);
    if (replacementLine !== null) {
      return {
        data: DELETE.repeat(chars(currentLine).length) + replacementLine,
        nextLine: replacementLine,
        normalized: true,
      };
    }
  }

  return {
    data,
    nextLine: updatePtyInputLine(currentLine, data),
    normalized: false,
  };
}

// How long (ms) a CR and its trailing LF may arrive apart and still count as
// one split CRLF Enter rather than two Enters. A human pressing Enter twice
// is far slower than xterm's two onData callbacks for one CRLF, and two
// identical bytes (CR CR / LF LF) are never fused — only CR+LF / LF+CR.
export const ENTER_SPLIT_WINDOW_MS = 60;

/**
 * Fuses a CRLF that Android/mobile terminals split across two xterm onData
 * callbacks (CR in one callback, LF in the next) into one Enter.
 *
 * The first line-ending byte is sent immediately (no latency added); only
 * the *second, opposite* line-ending byte arriving within the window is
 * swallowed — it is the carriage-return/newline half of the same Enter, not
 * a new keypress. Two identical bytes (CR CR / LF LF) are real double-Enters
 * and are never fused; a fused CRLF/LFCR chunk is one Enter and passes whole
 * (the parser already treats it as one, see parse-keypress).
 *
 * Also fuses the sticky-modifier case: a modified Enter (ESC + CR/LF, i.e.
 * Alt+Enter) can arrive as `\x1b\r` / `\x1b\n` in one callback with its
 * opposite line-ending half in the next. Without this the PTY sees a lone
 * Alt+Enter plus a separate Enter — the composer submits twice. The
 * opposite half is swallowed the same way as a bare split CRLF.
 */
export function createPtyEnterFuser(windowMs = ENTER_SPLIT_WINDOW_MS) {
  let last: { byte: string; at: number; expectHalf?: string } | null = null;

  return {
    /**
     * Process one chunk of PTY-bound input. Returns the chunk to actually
     * send, or "" when the chunk is the second half of a CRLF that was split
     * across callbacks (its CR/LF counterpart was already delivered).
     */
    process(data: string): string {
      // Modified-Enter (ESC + CR/LF) window: swallow exactly the CRLF half
      // we are waiting for; a non-matching CR/LF is a fresh Enter.
      if (last?.expectHalf && (data === "\r" || data === "\n")) {
        if (data === last.expectHalf && Date.now() - last.at <= windowMs) {
          return "";
        }
        last = null;
        return data;
      }
      if (data === "\r" || data === "\n") {
        if (
          last &&
          data !== last.byte &&
          Date.now() - last.at <= windowMs
        ) {
          return "";
        }
        last = { byte: data, at: Date.now() };
        return data;
      }
      if (data === "\r\n" || data === "\n\r") {
        last = { byte: data, at: Date.now() };
        return data;
      }
      // A modified Enter (ESC + CR or ESC + LF) split across callbacks:
      // keep the window open so its opposite CR/LF half is swallowed next.
      if (data === "\x1b\r" || data === "\x1b\n") {
        last = {
          byte: data,
          at: Date.now(),
          expectHalf: data === "\x1b\r" ? "\n" : "\r",
        };
        return data;
      }
      // Any other input ends the split-window: a later CR/LF is a real
      // Enter, not the echo of this chunk.
      last = null;
      return data;
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Android raw-mode input workaround (`type="password"`)
//
// Android Chrome delivers typed input through a *hidden password input*
// instead of xterm's own textarea. A `<input type="password">` forces the
// Android keyboard into raw mode: every tap arrives as a per-character
// `input` event with `inputType === "insertText"` and `data` = that one
// character — no IME composition, no word-buffering-to-space, no
// `insertCompositionText`. This is the proven fix for the "no text types
// at all" Android problem (#137986): modern keyboards (SwiftKey, Gboard)
// hold letters in a composition until space, and xterm's `_inputEvent`
// handler (PR #6024, 6.1.0-beta.315) only sees `insertText`, not
// `insertCompositionText`. The password input sidesteps composition
// entirely, so per-character input reaches the PTY on ALL keyboards.
//
// `type="password"` only works on `<input>` elements — xterm's internal
// textarea (`createElement('textarea')`) cannot carry it (it's a no-op
// there), so we add a *separate* hidden password input and route focus +
// input through it, leaving xterm's textarea untouched.
// ─────────────────────────────────────────────────────────────────────────

/**
 * Map one KeyboardEvent to the PTY byte(s) it should send. Returns "" when
 * the keydown must be ignored because the character arrives separately as
 * an `input` event (letters/digits on a password input, which the browser
 * delivers per-tap).
 *
 * Only structural keys are handled here (Enter, backspace/delete, arrows,
 * Home/End, Tab, Ctrl+letter, and the IME "process" key that must be
 * swallowed so React's delegation doesn't mis-handle it).
 */
export function mapKeydownToPty(e: {
  key: string;
  keyCode?: number;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
}): string {
  const key = e.key;
  const ctrl = !!e.ctrlKey || !!e.metaKey;
  const alt = !!e.altKey;
  const shift = !!e.shiftKey;

  // IME/composition-in-progress signal: never let this reach React's
  // synthetic onCompositionStart (the #52111 guard), and never send it.
  if (e.keyCode === 229 || key === "Process") return "";

  if (ctrl && key === "Enter") return "\x0a"; // Ctrl+Enter -> LF
  if (key === "Enter" || key === "Return") return "\r";
  if (key === "Backspace") return "\x7f";
  if (key === "Delete") return "\x1b[3~";
  if (key === "Tab") return shift ? "\x1b[Z" : "\t";

  if (ctrl && key.length === 1 && /[a-zA-Z]/.test(key)) {
    // Ctrl+letter -> control character (Ctrl+A = \x01 ... Ctrl+Z = \x1a).
    const code = key.toUpperCase().charCodeAt(0) - 64;
    return String.fromCharCode(code);
  }
  if (ctrl && key === "Space") return "\x00";

  if (key === "ArrowUp") return alt ? "\x1b[1;3A" : "\x1b[A";
  if (key === "ArrowDown") return alt ? "\x1b[1;3B" : "\x1b[B";
  if (key === "ArrowRight") return alt ? "\x1b[1;3C" : "\x1b[C";
  if (key === "ArrowLeft") return alt ? "\x1b[1;3D" : "\x1b[D";

  if (key === "Home") return "\x1b[H";
  if (key === "End") return "\x1b[F";
  if (key === "PageUp") return "\x1b[5~";
  if (key === "PageDown") return "\x1b[6~";

  if (key === "Escape") return "\x1b";

  // Printable characters are delivered via the `input` event on a password
  // input — do not double-send them from keydown.
  if (key.length === 1) return "";
  // Function keys (F1-F12): send their CSI sequences.
  if (/^F([1-9]|1[0-2])$/.test(key)) {
    const n = parseInt(key.slice(1), 10);
    const code = n <= 9 ? n : n === 10 ? 21 : n === 11 ? 23 : 24;
    return `\x1b[${code}~`;
  }
  return "";
}

/**
 * Map one `input` event (with its InputEvent `inputType` + `data`) to the
 * PTY bytes to send, given the element's current value. Returns "" when the
 * event carries no typed input.
 *
 * On a password input Android delivers each tap as `inputType ===
 * "insertText"` with `data` = that one character (no composition). We send
 * it immediately and (in the DOM wrapper) reset the value so nothing
 * accumulates. Backspace arrives as `deleteContentBackward`.
 */
export function mapInputEventToPty(
  inputType: string | undefined,
  data: string | null | undefined,
  value: string,
): string {
  if (inputType === "insertText" || inputType === "insertReplacementText") {
    return data ?? "";
  }
  if (inputType === "insertFromPaste" || inputType === "insertFromDrop") {
    // Paste on a password input: the whole accumulated value is the paste.
    return value;
  }
  if (
    inputType === "deleteContentBackward" ||
    inputType === "deleteContentForward"
  ) {
    return "\x7f";
  }
  // insertCompositionText / insertFromComposition should not occur on a
  // password input (that is the entire point of raw mode); if one does,
  // forward its data defensively so nothing is dropped.
  if (
    inputType === "insertCompositionText" ||
    inputType === "insertFromComposition"
  ) {
    return data ?? "";
  }
  return "";
}

const PASSWORD_INPUT_ID = "hermes-mobile-raw-input";
const MOBILE_TEXTAREA_ID = "hermes-mobile-textarea-input";

/**
 * Create a hidden `<input type="password">` that captures Android raw-mode
 * input and forwards it via `onInput(data)`. Returns null when the browser
 * is not a touch/mobile client (desktop keeps xterm's own textarea path).
 *
 * The element is visually hidden but focusable; focusing it opens the soft
 * keyboard in raw mode. Every typed character is forwarded per-tap and the
 * element is reset to empty so nothing accumulates. `dispose()` removes the
 * element and all listeners.
 */
export function createMobilePasswordInput(
  isMobile: boolean,
  onInput: (data: string) => void,
): {
  element: HTMLInputElement;
  focus(): void;
  dispose(): void;
} | null {
  if (!isMobile || typeof document === "undefined") return null;

  // Wrap the input in a dedicated <form autocomplete="off" data-form-type="other">.
  // Chrome's password manager IGNORES `autocomplete="off"` on the input itself
  // by design (so sites can't disable password saving), but it DOES respect it
  // on an enclosing form. Setting `data-form-type="other"` on the FORM (not just
  // the input) additionally classifies the whole form as non-password, which
  // suppresses BOTH the "save password?" prompt AND the "suggest a strong
  // password" bar. `new-password` on the input is the proven save-prompt
  // suppressor (fcead6fe4e) — combined with the form-level `other` classification
  // it no longer triggers the strong-password suggestion.
  const form = document.createElement("form");
  form.setAttribute("autocomplete", "off");
  form.setAttribute("data-form-type", "other");
  form.style.position = "fixed";
  form.style.left = "-9999px";
  form.style.width = "1px";
  form.style.height = "1px";
  form.style.opacity = "0";

  const el = document.createElement("input");
  el.type = "password";
  el.id = PASSWORD_INPUT_ID;
  el.setAttribute("aria-label", "Terminal input (raw mode)");
  el.setAttribute("autocomplete", "off");
  el.setAttribute("data-form-type", "other");
  el.setAttribute("autocapitalize", "off");
  // `readonly` is the ONLY thing that definitively stops Chrome's password
  // manager from touching this field at all: it skips readonly password
  // inputs entirely — no "save password?" prompt AND no "suggest a strong
  // password" bar. autocomplete juggling (off / new-password) provably fails
  // on this Chrome (one always comes back). We clear it right before focus
  // so the keyboard opens and raw-mode typing works, then re-set it on blur
  // so Chrome never sees a fillable password field holding a value.
  el.readOnly = true;
  // Visually hidden but focusable — the keyboard opens on this element.
  el.style.position = "fixed";
  el.style.left = "-9999px";
  el.style.width = "1px";
  el.style.height = "1px";
  el.style.opacity = "0";
  el.style.border = "0";
  el.style.padding = "0";
  el.tabIndex = 0;
  form.appendChild(el);
  document.body.appendChild(form);

  const handleInput = (ev: Event) => {
    const input = ev as InputEvent;
    const data = mapInputEventToPty(input.inputType, input.data, el.value);
    // Reset the accumulator so the next tap starts clean; keep caret at 0.
    el.value = "";
    if (el.setSelectionRange) el.setSelectionRange(0, 0);
    if (data) onInput(data);
  };
  const handleKeydown = (ev: KeyboardEvent) => {
    const data = mapKeydownToPty({
      key: ev.key,
      keyCode: ev.keyCode,
      ctrlKey: ev.ctrlKey,
      metaKey: ev.metaKey,
      altKey: ev.altKey,
      shiftKey: ev.shiftKey,
    });
    if (data) {
      ev.preventDefault();
      onInput(data);
    } else if (ev.keyCode === 229 || ev.key === "Process") {
      ev.preventDefault();
    }
  };

  el.addEventListener("input", handleInput);
  el.addEventListener("keydown", handleKeydown);
  // On blur (keyboard closed) immediately re-lock the field: clear the value
  // and restore readonly so Chrome's password manager never sees a fillable
  // password field holding a value — this is what suppresses the save prompt.
  const handleBlur = () => {
    el.value = "";
    if (el.setSelectionRange) el.setSelectionRange(0, 0);
    el.readOnly = true;
  };
  el.addEventListener("blur", handleBlur);

  return {
    element: el,
    focus() {
      // Clear readonly BEFORE focusing so the soft keyboard opens and raw-mode
      // per-tap typing works; Chrome sees autocomplete="off" + data-form-type
      // and (mostly) readonly, so it won't raise password UI.
      el.readOnly = false;
      el.focus();
    },
    dispose() {
      el.removeEventListener("input", handleInput);
      el.removeEventListener("keydown", handleKeydown);
      el.removeEventListener("blur", handleBlur);
      // Remove the wrapping <form> — it owns the input, so removing it
      // detaches both from the DOM.
      form.remove();
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Visible mobile textarea input (user-preference fallback)
//
// The user chose a VISIBLE textarea for Chrome on Android over the hidden
// raw-mode password input: it carries NO password/autocomplete UI at all
// (a plain textarea), and lets Gboard/SwiftKey do normal word-prediction +
// swipe typing instead of per-tap raw mode. We do NOT forward per-tap
// `input` events (Android holds words in composition — the original bug),
// so the textarea simply accumulates the visible line; Enter sends the
// complete line to the PTY in one chunk. This trades full interactive
// per-keystroke fidelity (vim/htop) for a clean chat/command-line input,
// which is what the user wants ("dat zou vooruitgang zijn").
// ─────────────────────────────────────────────────────────────────────────

// What the textarea sends to the PTY when the user presses Enter: the
// accumulated line plus a CR. Empty line -> a bare CR (blank submit).
//
// Android IMEs often leave a trailing newline in the textarea value after an
// Enter commit (the browser's default newline insertion that `preventDefault`
// on keydown cannot always stop). Sending that residual `\n` alongside our CR
// would make the backend see TWO line endings (`\n` + `\r` = two Enters) and
// double-submit the composer. Strip any trailing CR/LF so exactly ONE
// line-ending byte (the CR we append) reaches the PTY — this is what the
// desktop path sends for Enter (`\r`), so mobile matches it.
export function mobileTextareaEnterData(line: string): string {
  const trimmed = line.replace(/\r?\n+$/u, "").replace(/\r+$/u, "");
  return trimmed ? trimmed + "\r" : "\r";
}

export interface MobileTextareaInput {
  element: HTMLTextAreaElement;
  focus(): void;
  dispose(): void;
}

export function createMobileTextareaInput(
  isMobile: boolean,
  onEnter: (line: string) => void,
): MobileTextareaInput | null {
  if (!isMobile || typeof document === "undefined") return null;

  const el = document.createElement("textarea");
  el.id = MOBILE_TEXTAREA_ID;
  el.setAttribute("aria-label", "Terminal input (mobile)");
  // Plain textarea => no password manager involvement. Suppress the browser's
  // own suggestions/autofill on the field as far as possible.
  el.setAttribute("autocomplete", "off");
  el.setAttribute("autocorrect", "off");
  el.setAttribute("autocapitalize", "off");
  el.setAttribute("spellcheck", "false");
  el.setAttribute("inputmode", "text");
  el.rows = 1;
  el.value = "";
  // Visible styling is applied by ChatPage (theme-aware colors); base here.
  el.style.resize = "none";
  el.style.overflow = "hidden";

  // ── Slash-command autocomplete (Open-WebUI-style "/" predictive text) ─────
  // Typing "/" pops a dropdown of matching slash commands above the input;
  // Tab/Enter inserts the highlighted command into the buffer, arrows move
  // the highlight, Esc closes. The dropdown is a fixed-position panel styled
  // to match the terminal, appended to <body> (removed on dispose).
  let matches: ReadonlyArray<{ cmd: string; hint: string }> = [];
  let activeIndex = 0;
  let dropdown: HTMLDivElement | null = null;

  const buildDropdown = () => {
    const box = document.createElement("div");
    box.style.position = "fixed";
    box.style.zIndex = "9999";
    box.style.minWidth = "16rem";
    box.style.maxHeight = "40vh";
    box.style.overflowY = "auto";
    box.style.boxSizing = "border-box";
    box.style.padding = "0.25rem 0.5rem";
    box.style.border = "1px solid rgba(148,163,184,0.5)";
    box.style.borderRadius = "0.5rem";
    box.style.background = "#0e1715";
    box.style.color = "#e2e8f0";
    box.style.font = "inherit";
    box.style.fontSize = "13px";
    box.style.boxShadow = "0 8px 32px rgba(0,0,0,0.5)";
    return box;
  };

  const positionDropdown = () => {
    if (!dropdown) return;
    const r = el.getBoundingClientRect();
    dropdown.style.left = `${Math.max(8, r.left)}px`;
    dropdown.style.width = `${Math.min(r.width, 20 * 16)}px`;
    dropdown.style.bottom = `${window.innerHeight - r.top + 8}px`;
    if (dropdown.style.bottom === "") {
      dropdown.style.bottom = `${r.top + r.height + 8}px`;
    }
  };

  const renderDropdown = () => {
    if (!matches.length) {
      if (dropdown) {
        dropdown.remove();
        dropdown = null;
      }
      return;
    }
    if (!dropdown) dropdown = buildDropdown();
    const dd = dropdown; // capture: TS control-flow narrowing doesn't flow into the forEach callback
    dd.innerHTML = "";
    matches.forEach((m, i) => {
      const row = document.createElement("div");
      row.style.padding = "0.15rem 0.4rem";
      row.style.borderRadius = "0.25rem";
      row.style.whiteSpace = "nowrap";
      row.style.overflow = "hidden";
      row.style.textOverflow = "ellipsis";
      if (i === activeIndex) {
        row.style.background = "rgba(148,163,184,0.25)";
        row.style.color = "#e2e8f0";
      }
      const name = document.createElement("span");
      name.style.fontWeight = "600";
      name.textContent = m.cmd;
      const hint = document.createElement("span");
      hint.style.marginLeft = "0.5rem";
      hint.style.opacity = "0.7";
      hint.textContent = m.hint;
      row.append(name, hint);
      row.addEventListener("mousedown", (e) => {
        e.preventDefault(); // keep textarea focus
        commitAutocomplete(m.cmd);
      });
      dd.appendChild(row);
    });
    positionDropdown();
    if (!document.body.contains(dd)) document.body.appendChild(dd);
  };

  const currentToken = (): string => mobileToken(el.value);

  const updateMatches = () => {
    matches = mobileSlashMatches(currentToken());
    activeIndex = 0;
    renderDropdown();
  };

  const commitAutocomplete = (cmd: string) => {
    el.value = mobileCommitToken(el.value, cmd);
    matches = [];
    activeIndex = 0;
    renderDropdown();
    // keep the caret at the end
    el.setSelectionRange(el.value.length, el.value.length);
    el.focus();
  };

  // ── end autocomplete setup ────────────────────────────────────────────────

  // Double-Enter fix: on Android some keyboards deliver the FIRST Enter as a
  // newline via the `input` event (keydown swallowed), so the keydown handler
  // never sees it and the user must press Enter twice. Track the last keydown
  // so we can tell a real newline-Enter (input) from a Shift+Enter newline.
  let lastEnterViaKeydown: "enter" | "shift-enter" | null = null;
  // Timestamp of the last keydown-driven submit, so an `input`-event newline
  // that echoes it within the same tick isn't double-sent as a blank Enter.
  let lastKeydownSendAt = 0;

  const handleKeydown = (ev: KeyboardEvent) => {
    if (ev.key === "Enter") {
      const isMod = ev.shiftKey || ev.ctrlKey || ev.altKey;
      // Autocomplete open + plain Enter/Tab => commit the highlighted command
      // into the buffer (do NOT send the line).
      if (matches.length && !isMod) {
        ev.preventDefault();
        commitAutocomplete(matches[activeIndex].cmd);
        return;
      }
      lastEnterViaKeydown = isMod ? "shift-enter" : "enter";
      if (!isMod) {
        ev.preventDefault();
        const line = el.value;
        el.value = "";
        lastKeydownSendAt = Date.now();
        // reset so the input-event echo of this keydown is not re-sent
        lastEnterViaKeydown = null;
        matches = [];
        renderDropdown();
        onEnter(line);
      }
      // shift/ctrl/alt+Enter: let the browser insert a newline (multi-line).
      return;
    }
    // Non-Enter keys: clear the marker so a later input newline counts as a
    // fresh Enter (e.g. the user's first Enter, delivered as input).
    lastEnterViaKeydown = null;
    // ── autocomplete navigation ─────────────────────────────────────────────
    if (matches.length) {
      if (ev.key === "ArrowDown") {
        ev.preventDefault();
        activeIndex = (activeIndex + 1) % matches.length;
        renderDropdown();
        return;
      }
      if (ev.key === "ArrowUp") {
        ev.preventDefault();
        activeIndex = (activeIndex - 1 + matches.length) % matches.length;
        renderDropdown();
        return;
      }
      if (ev.key === "Tab") {
        ev.preventDefault();
        commitAutocomplete(matches[activeIndex].cmd);
        return;
      }
      if (ev.key === "Escape") {
        ev.preventDefault();
        matches = [];
        renderDropdown();
        return;
      }
    }
    if (ev.key === "Tab") {
      ev.preventDefault();
      // Insert a literal tab into the buffer (sent as part of the line on
      // Enter), so Tab doesn't move focus out of the textarea.
      el.value = el.value + "\t";
      return;
    }
    // Swallow the IME "process" key (keyCode 229) so React's root-level
    // delegation doesn't synthesise a bogus onCompositionStart (#52111).
    if (ev.keyCode === 229 || ev.key === "Process") {
      ev.preventDefault();
    }
  };

  const handleInput = () => {
    // Android sometimes delivers Enter as a newline via `input` with NO
    // keydown. If the value gained a trailing newline and it wasn't an
    // intentional Shift+Enter, treat it as the Enter submit.
    const v = el.value;
    if (v.endsWith("\n")) {
      if (lastEnterViaKeydown === "shift-enter") {
        // Intentional Shift+Enter newline: keep it as part of the buffer.
        lastEnterViaKeydown = null;
        return;
      }
      // A newline arriving within the same tick as a keydown submit is that
      // submit's echo (keydown cleared the value; browser re-inserted "\n").
      if (Date.now() - lastKeydownSendAt < 150) return;
      const line = v.slice(0, -1);
      el.value = "";
      lastEnterViaKeydown = null;
      onEnter(line);
      return;
    }
    // Refresh the slash-command suggestions as the buffer changes.
    updateMatches();
  };

  el.addEventListener("keydown", handleKeydown);
  el.addEventListener("input", handleInput);

  // Close the dropdown when the input loses focus, and reposition it when the
  // visual viewport shifts (keyboard open/close changes the layout).
  const closeOnBlur = () => {
    matches = [];
    if (dropdown) {
      dropdown.remove();
      dropdown = null;
    }
  };
  const repositionOnViewport = () => {
    if (matches.length) positionDropdown();
  };
  el.addEventListener("blur", closeOnBlur);
  window.addEventListener("resize", repositionOnViewport);
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", repositionOnViewport);
  }

  return {
    element: el,
    focus() {
      el.focus();
    },
    dispose() {
      el.removeEventListener("keydown", handleKeydown);
      el.removeEventListener("input", handleInput);
      el.removeEventListener("blur", closeOnBlur);
      window.removeEventListener("resize", repositionOnViewport);
      if (window.visualViewport) {
        window.visualViewport.removeEventListener("resize", repositionOnViewport);
      }
      closeOnBlur();
      el.remove();
    },
  };
}
