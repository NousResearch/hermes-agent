// eslint-disable-next-line no-control-regex -- intentional C0/C1 stripping for PTY-bound deep-link text
const TERMINAL_CONTROL_CHARS_RE = /[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g;
// eslint-disable-next-line no-control-regex -- strips complete ANSI/VT escape sequences before C0 cleanup
const ANSI_ESCAPE_RE = /\u001b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g;
const WHITESPACE_RE = /\s+/g;

export function normalizeLearnDeepLinkSeed(seed: string | null): string | null {
  if (!seed) {
    return null;
  }

  const normalized = seed
    .replace(ANSI_ESCAPE_RE, " ")
    .replace(TERMINAL_CONTROL_CHARS_RE, " ")
    .replace(WHITESPACE_RE, " ")
    .trim();

  return normalized || null;
}
