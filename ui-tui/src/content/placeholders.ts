import { pick } from '../lib/text.js'

const PLACEHOLDERS = [
  'Ask me anything\u2026',
  'Try "explain this codebase"',
  'Try "write a test for\u2026"',
  'Try "refactor the auth module"',
  'Try "/help" for commands',
  'Try "fix the lint errors"',
  'Try "how does the config loader work?"',
  '\u8bf7\u95ee\u4efb\u4f55\u95ee\u9898\u2026',
  '\u5c1d\u8bd5: "explain this codebase"',
  '\u5c1d\u8bd5: "write a test for\u2026"',
  '\u5c1d\u8bd5: "refactor the auth module"',
  '\u5c1d\u8bd5: "/help" \u67e5\u770b\u547d\u4ee4',
  '\u5c1d\u8bd5: "fix the lint errors"',
  '\u5c1d\u8bd5: "how does the config loader work?"',
]

export const PLACEHOLDER = pick(PLACEHOLDERS)
