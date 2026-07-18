import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Regression tests for the web chat composer's accessibility fixes (#36784).
 *
 * These are static-source assertions rather than behavioral tests because
 * ChatPage is a 1100+ line React component with refs, WebSocket effects,
 * and xterm-coupled event handlers — extracting a meaningful behavioral
 * test surface would require mocking a full module just to exercise one
 * branch. The structural invariants below directly encode what the
 * maintainer flagged: that there is exactly one key handler registration
 * (because xterm silently overwrites later ones), that screen-reader mode
 * is enabled on the terminal, and that the host element is an accessible
 * region with a label.
 */

const CHAT_PAGE_SRC = readFileSync(
  resolve(__dirname, "./ChatPage.tsx"),
  "utf8",
);

// Strip block AND line comments. Line-comment stripping was added after a
// review pass: a code comment like `// term.attachCustomKeyEventHandler(...)`
// would otherwise inflate the "exactly one registration" count and turn a
// helpful doc comment into a test failure. The identifiers and string
// literals we care about (`attachCustomKeyEventHandler`, `screenReaderMode`,
// `"Tab"`, `role`, `aria-label`) are unique enough that literal source
// counting is reliable after stripping comments.
function stripComments(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
}

const CHAT_PAGE_NO_COMMENTS = stripComments(CHAT_PAGE_SRC);

describe("ChatPage a11y — terminal screen-reader support (#36784)", () => {
  it("enables screenReaderMode on the xterm Terminal constructor", () => {
    expect(CHAT_PAGE_NO_COMMENTS).toMatch(/\bscreenReaderMode\s*:\s*true\b/);
  });

  it("marks the host div as an accessible region with an aria-label", () => {
    // Anchor on the hostRef div to make sure we are asserting against the
    // xterm host element, not any other role="region" in the file. Use
    // separate matchers (not a single contiguous regex) so attribute
    // reordering by a code formatter doesn't break the test.
    expect(CHAT_PAGE_NO_COMMENTS).toMatch(/ref=\{hostRef\}[\s\S]*?role="region"/);
    expect(CHAT_PAGE_NO_COMMENTS).toMatch(/ref=\{hostRef\}[\s\S]*?aria-label="[^"]+"/);
  });

  it("registers exactly one attachCustomKeyEventHandler (xterm silently overwrites later registrations)", () => {
    // xterm.js stores a single _customKeyEventHandler per Terminal and
    // assign-overwrites on attachCustomKeyEventHandler (see
    // node_modules/@xterm/xterm/src/browser/CoreBrowserTerminal.ts:914).
    // A second registration would silently disable the first. Issue #36784
    // review caught exactly this regression in the first revision of this PR.
    const matches = CHAT_PAGE_NO_COMMENTS.match(
      /\bterm\.attachCustomKeyEventHandler\s*\(/g,
    );
    expect(matches?.length ?? 0).toBe(1);
  });

  it("the single key handler routes Tab events to escape (returns false)", () => {
    // Tab escape is the WCAG 2.1.2 No Keyboard Trap guarantee for screen-reader
    // and keyboard users. We assert the early branch exists in the single
    // key handler rather than invoking it — the handler closes over xterm
    // and the navigator clipboard, which would need full module mocks.
    expect(CHAT_PAGE_NO_COMMENTS).toMatch(/attachCustomKeyEventHandler\([\s\S]*?if\s*\(\s*ev\.key\s*===\s*"Tab"\s*\)\s*return false/);
    // Sanity: not a keyup-only guard — Tab escape must run on keydown.
    expect(CHAT_PAGE_NO_COMMENTS).toMatch(/if\s*\(\s*ev\.type\s*!==\s*"keydown"\s*\)\s*return true/);
  });
});
