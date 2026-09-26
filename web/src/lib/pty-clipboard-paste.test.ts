import { describe, expect, it, vi } from "vitest";

import { sendPtyShortcutSequence } from "./pty-keyboard-shortcuts";
import {
  DEFAULT_PASTE_MAX_CHARS,
  PASTE_PREVIEW_MAX_CHARS,
  formatImageUploadError,
  formatPasteConfirmation,
  pastePreview,
  runPtyClipboardPaste,
  type PasteDeps,
} from "./pty-clipboard-paste";

function clipboardFile(name = "clipboard.png", type = "image/png"): File {
  return new File(["bytes"], name, { type });
}

interface Stubs {
  pasteText?: (text: string) => void;
  sendBytes?: (bytes: string) => boolean;
  readImages?: () => Promise<File[]>;
  /** `null` = this environment has no clipboard text reader at all. */
  readText?: (() => Promise<string>) | null;
  isSecureContext?: boolean;
  maxChars?: number;
}

function makeDeps(stubs: Stubs = {}) {
  const pasteText = vi.fn(stubs.pasteText ?? (() => undefined));
  const sendBytes = vi.fn(stubs.sendBytes ?? (() => true));
  const deps: PasteDeps = {
    pasteText,
    sendBytes,
    isSecureContext: stubs.isSecureContext ?? true,
    maxChars: stubs.maxChars ?? DEFAULT_PASTE_MAX_CHARS,
  };

  let readText: ReturnType<typeof vi.fn> | undefined;
  if (stubs.readText !== null) {
    readText = vi.fn(stubs.readText ?? (async () => "hello"));
    deps.readText = readText;
  }

  let readImages: ReturnType<typeof vi.fn> | undefined;
  if (stubs.readImages) {
    readImages = vi.fn(stubs.readImages);
    deps.readImages = readImages;
  }

  return { deps, pasteText, sendBytes, readText, readImages };
}

describe("runPtyClipboardPaste text path", () => {
  it("sends single-line text through the xterm paste path", async () => {
    const { deps, pasteText, sendBytes } = makeDeps({
      readText: async () => "hermes status",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-text", chars: 13 });
    expect(pasteText).toHaveBeenCalledTimes(1);
    expect(pasteText).toHaveBeenCalledWith("hermes status");
    // Text must never travel as raw PTY bytes: that would bypass
    // `term.paste` → `onData` → `normalizePtyMobileInput` (FR-2/FR-4).
    expect(sendBytes).not.toHaveBeenCalledWith("hermes status");
  });

  it("asks for confirmation before a multi-line paste is sent", async () => {
    const { deps, pasteText } = makeDeps({
      readText: async () => "line one\nline two",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome.kind).toBe("needs-confirmation");
    if (outcome.kind !== "needs-confirmation") return;
    expect(outcome.text).toBe("line one\nline two");
    expect(outcome.preview).toContain("line one");
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("treats a carriage return as multi-line too", async () => {
    const { deps, pasteText } = makeDeps({ readText: async () => "a\rb" });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome.kind).toBe("needs-confirmation");
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("sends a confirmed multi-line paste", async () => {
    const { deps, pasteText } = makeDeps();

    const outcome = await runPtyClipboardPaste(deps, {
      confirmation: "confirm",
      pendingText: "a\nb",
    });

    expect(outcome).toEqual({ kind: "sent-text", chars: 3 });
    expect(pasteText).toHaveBeenCalledWith("a\nb");
  });

  it("returns cancelled and sends nothing when the user declines", async () => {
    const { deps, pasteText, sendBytes, readText } = makeDeps();

    const outcome = await runPtyClipboardPaste(deps, {
      confirmation: "cancel",
      pendingText: "a\nb",
    });

    expect(outcome).toEqual({ kind: "cancelled" });
    expect(pasteText).not.toHaveBeenCalled();
    expect(sendBytes).not.toHaveBeenCalled();
    expect(readText).not.toHaveBeenCalled();
  });
});

describe("runPtyClipboardPaste image path", () => {
  it("takes the image path for an image-only clipboard without reading text", async () => {
    const { deps, readText, pasteText } = makeDeps({
      readImages: async () => [clipboardFile()],
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-image", count: 1 });
    expect(readText).not.toHaveBeenCalled();
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("prefers the image path when the clipboard holds both", async () => {
    const { deps, readText } = makeDeps({
      readImages: async () => [
        clipboardFile("a.png"),
        clipboardFile("b.png"),
      ],
      readText: async () => "also some text",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-image", count: 2 });
    expect(readText).not.toHaveBeenCalled();
  });

  it("reports blocked for an image clipboard when the guarded sender refuses the socket", async () => {
    // B2 regression: the image branch used to return `sent-image` before ever
    // reaching the PTY gate, so the caller went on to write `/image <path>`
    // plus `\r` into a socket the reconnect logic considers unusable (NS-591
    // half-open mobile socket) — silently, with no banner. The image route
    // must be gated exactly like the text route.
    const { deps, pasteText, sendBytes } = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: () => false,
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "blocked", reason: "socket-closed" });
    expect(pasteText).not.toHaveBeenCalled();
    // The pre-flight still runs, so the caller never reaches the raw sends.
    expect(sendBytes).toHaveBeenCalledWith("");
  });

  it("blocks an image clipboard through the real reconnect guard while an open socket passes", async () => {
    const ws = { readyState: 1 as const, send: vi.fn() };

    const blocked = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "reconnecting", bytes),
    });
    expect(await runPtyClipboardPaste(blocked.deps)).toEqual({
      kind: "blocked",
      reason: "socket-closed",
    });
    // readyState is still OPEN during a reconnect — this is the exact window
    // the `readyState`-only guard let through.
    expect(ws.readyState).toBe(1);
    expect(ws.send).not.toHaveBeenCalled();

    const open = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "open", bytes),
    });
    expect(await runPtyClipboardPaste(open.deps)).toEqual({
      kind: "sent-image",
      count: 1,
    });
  });

  it("falls back to text when the image reader rejects", async () => {
    const { deps, pasteText } = makeDeps({
      readImages: async () => {
        throw new Error("clipboard image read refused");
      },
      readText: async () => "hello",
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "sent-text", chars: 5 });
    expect(pasteText).toHaveBeenCalledWith("hello");
  });
});

describe("runPtyClipboardPaste failures", () => {
  it("names the insecure context when no text reader exists over plain HTTP", async () => {
    const { deps } = makeDeps({ readText: null, isSecureContext: false });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({
      kind: "unsupported",
      reason: "insecure-context",
    });
  });

  it("reports a missing reader separately in a secure context", async () => {
    const { deps } = makeDeps({ readText: null, isSecureContext: true });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "unsupported", reason: "unsupported-api" });
  });

  it("maps a rejected clipboard read to permission-denied", async () => {
    const { deps, pasteText } = makeDeps({
      readText: async () => {
        throw new Error("NotAllowedError");
      },
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({
      kind: "unsupported",
      reason: "permission-denied",
    });
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("reports blocked when the guarded sender refuses the socket", async () => {
    const { deps, pasteText } = makeDeps({ sendBytes: () => false });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "blocked", reason: "socket-closed" });
    expect(pasteText).not.toHaveBeenCalled();
  });

  it("blocks through the real reconnect guard and lets an open socket through", async () => {
    const ws = { readyState: 1 as const, send: vi.fn() };

    const blocked = makeDeps({
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "reconnecting", bytes),
    });
    expect(await runPtyClipboardPaste(blocked.deps)).toEqual({
      kind: "blocked",
      reason: "socket-closed",
    });
    expect(ws.send).not.toHaveBeenCalled();

    const open = makeDeps({
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "open", bytes),
    });
    expect(await runPtyClipboardPaste(open.deps)).toEqual({
      kind: "sent-text",
      chars: 5,
    });
    expect(open.pasteText).toHaveBeenCalledWith("hello");
  });

  it("refuses text larger than maxChars without sending anything", async () => {
    const { deps, pasteText, sendBytes } = makeDeps({
      readText: async () => "x".repeat(11),
      maxChars: 10,
    });

    const outcome = await runPtyClipboardPaste(deps);

    expect(outcome).toEqual({ kind: "unsupported", reason: "too-large" });
    expect(pasteText).not.toHaveBeenCalled();
    expect(sendBytes).not.toHaveBeenCalled();
  });

  it("treats an empty or whitespace-only clipboard as a no-op", async () => {
    const empty = makeDeps({ readText: async () => "" });
    expect(await runPtyClipboardPaste(empty.deps)).toEqual({ kind: "empty" });

    const blank = makeDeps({ readText: async () => "   \n  " });
    expect(await runPtyClipboardPaste(blank.deps)).toEqual({ kind: "empty" });

    expect(empty.pasteText).not.toHaveBeenCalled();
    expect(blank.pasteText).not.toHaveBeenCalled();
  });
});

describe("runPtyClipboardPaste output direction", () => {
  it("never writes to the terminal output path (the PRD v1 triple-echo regression)", async () => {
    // `term.write` is PTY → screen. Writing pasted text there fakes a local
    // echo, the TUI echoes it again down the socket, and the paste renders
    // two to three times. Every text path must go through `term.paste`.
    const terminal = { paste: vi.fn(), write: vi.fn() };
    const { deps } = makeDeps({ pasteText: (text) => terminal.paste(text) });

    await runPtyClipboardPaste(deps);
    await runPtyClipboardPaste(deps, {
      confirmation: "confirm",
      pendingText: "a\nb",
    });

    expect(terminal.write).not.toHaveBeenCalled();
    expect(terminal.paste).toHaveBeenCalledTimes(2);
  });
});

describe("pastePreview", () => {
  it("normalizes CRLF and truncates a long clipboard", () => {
    expect(pastePreview("a\r\nb")).toBe("a\nb");

    const long = pastePreview("x".repeat(PASTE_PREVIEW_MAX_CHARS + 50));
    expect(long.length).toBeLessThanOrEqual(PASTE_PREVIEW_MAX_CHARS + 1);
    expect(long.endsWith("…")).toBe(true);
  });
});

// Security-relevant (B1): the FR-8 confirmation is the only thing between a
// crafted clipboard and the agent terminal, so the rendered preview MUST be
// byte-identical to the payload that `confirm` sends. `String.replace` with a
// *string* replacement expands `$&`, `` $` ``, `$'` and `$$`, which silently
// renders a different string than the one that executes.
describe("formatPasteConfirmation", () => {
  const template =
    "Paste {preview} into the terminal? A multi-line paste is submitted line by line.";

  it.each([
    ["$&", "matched"],
    ["$`", "head"],
    ["$'", "tail"],
    ["$$", "dollar"],
    ["$1", "capture"],
  ])("renders a preview containing %s verbatim", (token) => {
    const preview = `rm -rf /\n${token}`;
    const rendered = formatPasteConfirmation(template, preview);

    expect(rendered).toContain(preview);
    // And nothing but the template's own literal text surrounds it.
    expect(rendered).toBe(
      template.split("{preview}").join(preview),
    );
  });

  it("does not duplicate the prompt when the preview ends with $'", () => {
    const rendered = formatPasteConfirmation(template, "x\n$'y");

    expect(rendered).toBe(`Paste x\n$'y into the terminal? A multi-line paste is submitted line by line.`);
    // `$'` used to splice the WHOLE prompt in again, which reads as a
    // rendering bug and trains the user to skim past the confirmation.
    expect(rendered.split("into the terminal?").length - 1).toBe(1);
    expect(rendered.split("x\n$'y").length - 1).toBe(1);
  });

  it("fills every placeholder, so a duplicated token can never leak to the user", () => {
    expect(formatPasteConfirmation("a {preview} b {preview} c", "X")).toBe(
      "a X b X c",
    );
    expect(formatPasteConfirmation("no placeholder", "X")).toBe(
      "no placeholder",
    );
  });

  it("renders an empty preview without throwing", () => {
    expect(formatPasteConfirmation(template, "")).toBe(
      template.replace("{preview}", ""),
    );
  });
});

describe("formatImageUploadError", () => {
  it("renders the failure message verbatim, $-patterns included", () => {
    const template = "Image upload failed: {message}";
    const message = "EACCES: $& $` $' $$ /tmp/shots";

    expect(formatImageUploadError(template, message)).toBe(
      `Image upload failed: ${message}`,
    );
  });
});

// B2's second half — the `ws.send("\r")` follow-up in `driveImageAttach` — is
// component-owned, so it needs the ChatPage harness. Those tests live in
// `src/pages/ChatPage.test.tsx` under "ChatPage driveImageAttach reconnect gate
// (B2)": the gate is flipped to refusing *between* the `/image <path>` write and
// the `\r`, which is the only way to prove the second send is gated rather than
// merely riding along on the first send's check.

// ─────────────────────────────────────────────────────────────────────────────
// QA pass — adversarial attempts to BREAK B1 and B2.
//
// These are deliberately hostile. The implementer's own tests cover the
// single-token cases ($&, $`, $', $$, $1) in isolation; every one of them
// still passes against a naive `.replace()` for some templates. These rows
// attack the properties, not the tokens: no byte of untrusted content may ever
// influence the rendered string except as the literal value.
// ─────────────────────────────────────────────────────────────────────────────

describe("B1 adversarial: formatPasteConfirmation literalness", () => {
  // A template whose own text also contains `$`-sequences, so a `$`-expanding
  // implementation produces a string that differs from the intent in a way the
  // isolated single-token tests do not notice.
  const template = "Paste {preview} into the terminal? (head $& tail $')";

  /** The ONLY acceptable rendering: template with the token slot filled. */
  const expected = (preview: string) =>
    template.split("{preview}").join(preview);

  it.each([
    ["double dollar", "$$"],
    ["doubled $&", "$&$&"],
    ["doubled $`", "$`$`"],
    ["doubled $'", "$'$'"],
    ["trailing bare dollar", "payload$"],
    ["leading bare dollar", "$payload"],
    ["lone dollar", "$"],
    ["alternating storm", "$$`$'&$`$'$&"],
    ["dollar-digit after a real match", "$&$1$2"],
    ["$0 (whole match)", "$0"],
    ["named-ish group", "$<name>"],
    ["$$ before a token", "$${preview}"],
    ["token inside the value", "see {preview} below"],
    ["all patterns in one multi-line payload", "a\n$&\nb\n$`\nc\n$'\nd\n$$"],
  ])("renders %s verbatim", (_label, token) => {
    const preview = `rm -rf /\n${token}`;
    const rendered = formatPasteConfirmation(template, preview);

    // Exact equality: any $-expansion whatsoever changes the result, and an
    // exact comparison is the only assertion that cannot be satisfied by a
    // "contains" false positive.
    expect(rendered).toBe(expected(preview));
    // The template's own text survives untouched.
    expect(rendered.startsWith("Paste ")).toBe(true);
    expect(rendered.endsWith("(head $& tail $')")).toBe(true);
  });

  it("cannot recurse when the preview itself contains the token", () => {
    // A preview of "{preview}" is a fixed point for split/join. A replacer that
    // re-scans its own output (or a manual loop) would grow without bound.
    const preview = "{preview}";
    const rendered = formatPasteConfirmation(template, preview);

    expect(rendered).toBe(expected(preview));
    expect(rendered).toBe("Paste {preview} into the terminal? (head $& tail $')");
    // No runaway expansion.
    expect(rendered.split("{preview}").length - 1).toBe(1);
  });

  it("does not recurse on a self-referential nested-token payload", () => {
    const preview = "{preview}{preview}";
    const rendered = formatPasteConfirmation(template, preview);

    expect(rendered).toBe(expected(preview));
    expect(rendered.split("{preview}").length - 1).toBe(2);
  });

  it("keeps a $-expansion payload and its rendering byte-identical", () => {
    // The end-to-end FR-8 contract: what the user is shown is a prefix of what
    // actually reaches the terminal. Any divergence is a defacement.
    const clipboard = "sudo rm -rf /\n$&\n$`\n$'\n$$";
    const preview = pastePreview(clipboard);
    const rendered = formatPasteConfirmation(template, preview);

    // Every line the user sees is present, in order, in the payload.
    for (const line of rendered
      .replace(/^Paste /, "")
      .replace(/ into the terminal\? \(head \$& tail \$'\)$/, "")
      .split("\n")) {
      expect(clipboard).toContain(line);
    }
    // The rendered preview occurs exactly once — `$'` used to splice the whole
    // prompt in a second time, which reads as a glitch and trains the user to
    // skim past the one confirmation standing between them and execution.
    expect(rendered.split(preview).length - 1).toBe(1);
  });

  it("survives a preview made entirely of substitution metacharacters", () => {
    const preview = "$`$'&$\\$`$'&$\\";
    expect(formatPasteConfirmation(template, preview)).toBe(
      expected(preview),
    );
  });

  it("is a fixed point: formatting the rendered value again changes nothing", () => {
    const preview = "$&\n$`\n$'";
    const once = formatPasteConfirmation(template, preview);
    const twice = formatPasteConfirmation(once, preview);
    expect(twice).toBe(once);
  });
});

describe("B1 boundary: PASTE_PREVIEW_MAX_CHARS", () => {
  const template = "Paste {preview}?";

  it("renders a preview of exactly PASTE_PREVIEW_MAX_CHARS without truncating it", () => {
    const preview = pastePreview("x".repeat(PASTE_PREVIEW_MAX_CHARS));
    expect(preview).toHaveLength(PASTE_PREVIEW_MAX_CHARS);
    expect(preview.endsWith("…")).toBe(false);
    expect(formatPasteConfirmation(template, preview)).toBe(
      `Paste ${preview}?`,
    );
  });

  it("truncates exactly one char over the limit and marks it", () => {
    const preview = pastePreview("x".repeat(PASTE_PREVIEW_MAX_CHARS + 1));
    expect(preview).toHaveLength(PASTE_PREVIEW_MAX_CHARS + 1);
    expect(preview.endsWith("…")).toBe(true);
    expect(formatPasteConfirmation(template, preview)).toBe(
      `Paste ${preview}?`,
    );
  });

  it("keeps $-patterns literal at both the exact and over-the-limit boundary", () => {
    // Boundary + hostile content together: truncation must not reintroduce a
    // substitution opportunity at the cut point.
    for (const size of [PASTE_PREVIEW_MAX_CHARS, PASTE_PREVIEW_MAX_CHARS + 1]) {
      const filler = "$&$`$'$1".repeat(Math.ceil(size / 8));
      const clipboard = `${filler.slice(0, size - 1)}$`;
      const preview = pastePreview(clipboard);
      expect(formatPasteConfirmation(template, preview)).toBe(
        `Paste ${preview}?`,
      );
      // The rendered result embeds the preview as a literal substring.
      expect(`Paste ${preview}?`.includes(preview)).toBe(true);
    }
  });

  it("does not split a surrogate pair at the truncation boundary", () => {
    // `slice` cuts on UTF-16 code units, so a cut at an odd offset leaves a
    // lone surrogate in the preview. That renders as U+FFFD (a "tofu" box) in
    // the confirmation, i.e. the user sees a character they never pasted —
    // the same class of "preview ≠ payload" defect B1 is about.
    const clipboard = "\u{1F600}".repeat(140) + "TAIL";
    const preview = pastePreview(clipboard);
    const body = preview.endsWith("…") ? preview.slice(0, -1) : preview;
    const last = body.charCodeAt(body.length - 1);

    expect(last < 0xd800 || last > 0xdfff).toBe(true);
  });

  it("never splits a surrogate pair at ANY truncation boundary, not just the default", () => {
    // QA's case above is one boundary; the class of bug is a cut landing on an
    // odd code-unit offset. Sweeping every offset proves the cut is on code
    // points rather than merely lucking out on this input, and pins that the
    // preview is always a prefix of the clipboard in whole code points.
    for (let size = 1; size <= PASTE_PREVIEW_MAX_CHARS; size++) {
      const clipboard = "\u{1F600}\u{1F680}".repeat(200);
      const preview = pastePreview(clipboard, size);
      const body = preview.endsWith("…") ? preview.slice(0, -1) : preview;
      const points = Array.from(body);

      expect(Array.from(body).join("")).toBe(body); // no split pairs
      expect(points.length).toBeLessThanOrEqual(size);
      // Every retained code point comes from the payload, in order, unmodified.
      expect(Array.from(clipboard).slice(0, points.length)).toEqual(points);
      // Never a lone surrogate anywhere, not merely at the cut.
      for (const ch of points) {
        const code = ch.codePointAt(0)!;
        expect(code < 0xd800 || code > 0xdfff).toBe(true);
      }
    }
  });

  it("measures the limit in code points, so the cut is independent of UTF-16 width", () => {
    // 280 emoji = 560 code units but 280 code points: exactly at the limit, so
    // it must NOT be truncated. Under code-unit counting this same input was
    // cut in half.
    const atLimit = "\u{1F600}".repeat(PASTE_PREVIEW_MAX_CHARS);
    expect(pastePreview(atLimit)).toBe(atLimit);
    expect(pastePreview(atLimit).endsWith("…")).toBe(false);

    // One code point over → truncated to exactly PASTE_PREVIEW_MAX_CHARS code
    // points plus the ellipsis.
    const over = "\u{1F600}".repeat(PASTE_PREVIEW_MAX_CHARS + 1);
    const preview = pastePreview(over);
    const body = preview.endsWith("…") ? preview.slice(0, -1) : preview;
    expect(Array.from(body)).toHaveLength(PASTE_PREVIEW_MAX_CHARS);
    expect(body).toBe("\u{1F600}".repeat(PASTE_PREVIEW_MAX_CHARS));
  });

  it("keeps the ≤ max + 1 relationship when astral characters straddle the cut", () => {
    // The existing ASCII boundary test asserts `≤ PASTE_PREVIEW_MAX_CHARS + 1`
    // in UTF-16 units. With astral content the honest unit is code points, and
    // the same relationship must hold there too.
    for (const count of [300, 400, 1000]) {
      const preview = pastePreview("\u{1F600}".repeat(count) + "tail");
      const body = preview.endsWith("…") ? preview.slice(0, -1) : preview;
      expect(Array.from(body).length).toBeLessThanOrEqual(PASTE_PREVIEW_MAX_CHARS);
      expect(preview.endsWith("…")).toBe(true);
    }
  });
});

describe("B2 adversarial: the image route consults the reconnect gate", () => {
  /** A socket that reports OPEN — the half-open NS-591 state. */
  function openSocket() {
    return { readyState: 1 as const, send: vi.fn() };
  }

  it("sends NOTHING on the image route when reconnecting and readyState is OPEN", () => {
    // The exact B2 scenario: a socket the browser still reports OPEN while the
    // reconnect logic already considers it unusable. A readyState-only guard
    // passes this; the fix must not.
    const ws = openSocket();
    const { deps, pasteText, readText } = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "reconnecting", bytes),
    });

    const outcome = runPtyClipboardPaste(deps);

    expect(ws.readyState).toBe(WebSocket.OPEN);
    return outcome.then((resolved) => {
      expect(resolved).toEqual({ kind: "blocked", reason: "socket-closed" });
      // Not one byte — not even the zero-length pre-flight — reaches the wire.
      expect(ws.send).not.toHaveBeenCalled();
      expect(pasteText).not.toHaveBeenCalled();
      expect(readText).not.toHaveBeenCalled();
    });
  });

  it.each(["connecting", "reconnecting", "closed", "ended"] as const)(
    "blocks the image route with zero bytes while ptyState is %s",
    async (ptyState) => {
      const ws = openSocket();
      const { deps } = makeDeps({
        readImages: async () => [clipboardFile()],
        sendBytes: (bytes) => sendPtyShortcutSequence(ws, ptyState, bytes),
      });

      expect(await runPtyClipboardPaste(deps)).toEqual({
        kind: "blocked",
        reason: "socket-closed",
      });
      expect(ws.send).not.toHaveBeenCalled();
    },
  );

  it("blocks the CONFIRMED multi-line route too, with zero bytes", () => {
    // The confirm path is a second entry into `deliverText`; B2's gate must
    // not have been added only to the image branch.
    const ws = openSocket();
    const { deps, pasteText } = makeDeps({
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "reconnecting", bytes),
    });

    return runPtyClipboardPaste(deps, {
      confirmation: "confirm",
      pendingText: "rm -rf /\nsudo make me a sandwich",
    }).then((outcome) => {
      expect(outcome).toEqual({ kind: "blocked", reason: "socket-closed" });
      expect(ws.send).not.toHaveBeenCalled();
      expect(pasteText).not.toHaveBeenCalled();
    });
  });

  it("does not over-block: an open socket with ptyState open still attaches", () => {
    // Over-blocking is a real regression too: the paste affordance would go
    // dead on the happy path. The gate must be a no-op here.
    const ws = openSocket();
    const { deps } = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "open", bytes),
    });

    return runPtyClipboardPaste(deps).then((outcome) => {
      expect(outcome).toEqual({ kind: "sent-image", count: 1 });
      // The pre-flight is the one permitted zero-length send.
      expect(ws.send).toHaveBeenCalledTimes(1);
      expect(ws.send).toHaveBeenCalledWith("");
    });
  });

  it("does not over-block the text route on a healthy open socket", () => {
    const ws = openSocket();
    const { deps, pasteText } = makeDeps({
      readText: async () => "hello",
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "open", bytes),
    });

    return runPtyClipboardPaste(deps).then((outcome) => {
      expect(outcome).toEqual({ kind: "sent-text", chars: 5 });
      expect(pasteText).toHaveBeenCalledWith("hello");
      expect(ws.send).toHaveBeenCalledExactlyOnceWith("");
    });
  });

  it("blocks the image route when the socket is gone entirely (readyState CLOSED)", () => {
    const ws = { readyState: 3 as const, send: vi.fn() };
    const { deps } = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "open", bytes),
    });

    return runPtyClipboardPaste(deps).then((outcome) => {
      expect(outcome).toEqual({ kind: "blocked", reason: "socket-closed" });
      expect(ws.send).not.toHaveBeenCalled();
    });
  });

  it("blocks when there is no socket at all", async () => {
    const { deps } = makeDeps({
      readImages: async () => [clipboardFile()],
      sendBytes: (bytes) => sendPtyShortcutSequence(null, "open", bytes),
    });

    expect(await runPtyClipboardPaste(deps)).toEqual({
      kind: "blocked",
      reason: "socket-closed",
    });
  });

  it("reports blocked for a multi-image clipboard and attaches nothing", async () => {
    const ws = openSocket();
    const { deps } = makeDeps({
      readImages: async () => [clipboardFile("a.png"), clipboardFile("b.png")],
      sendBytes: (bytes) => sendPtyShortcutSequence(ws, "reconnecting", bytes),
    });

    expect(await runPtyClipboardPaste(deps)).toEqual({
      kind: "blocked",
      reason: "socket-closed",
    });
    expect(ws.send).not.toHaveBeenCalled();
  });
});

