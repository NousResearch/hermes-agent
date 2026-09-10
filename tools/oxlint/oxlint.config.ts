import { defineConfig } from "oxlint";

// Vendored anti-slop rules (tools/oxlint/anti-slop, from dmmulroy/anti-slop),
// ported from NousResearch/hermes-portal. Oxlint runs only these plugin rules;
// ESLint stays the primary linter. This config feeds an ADVISORY lane: the
// ratchet in slop-ratchet.mjs reports net-new findings on a PR against the
// committed baseline and never blocks a merge.
export default defineConfig({
  // Repo-wide ignores live in /.slopignore (passed as --ignore-path): patterns
  // here are rooted at tools/oxlint/ and cannot reach files above it.
  ignorePatterns: ["anti-slop/**", "node_modules/**"],
  jsPlugins: [
    { name: "anti-slop", specifier: "./anti-slop/index.ts" },
  ],
  // Only the anti-slop plugin plus complexity: oxlint's built-in correctness
  // category is switched off so the baseline measures slop, not the rules
  // ESLint already owns in `npm run check`.
  categories: {
    correctness: "off",
  },
  rules: {
    // Built-in oxlint port of eslint/complexity: cap McCabe cyclomatic
    // complexity per function at the upstream default of 20. Existing
    // offenders are grandfathered by the ratchet baseline like every other
    // rule here; only net-new complexity is reported on the diff lane.
    "complexity": ["error", { max: 20 }],
    "anti-slop/no-chained-type-assertions": "error",
    "anti-slop/no-conditional-empty-object-spread": "error",
    "anti-slop/no-known-value-widening": "error",
    // Module mocking and typeof narrowing are warnings, not errors: vi.mock is
    // this repo's universal test seam, and typeof is the only way to express
    // feature detection and SSR/Electron-vs-browser guards. The ratchet
    // counts findings regardless of severity.
    "anti-slop/no-module-mocking": "warn",
    "anti-slop/no-object-parameters": "error",
    "anti-slop/no-reflect-apply": "error",
    "anti-slop/no-reflect-get": "error",
    "anti-slop/no-runtime-typeof": "warn",
    "anti-slop/no-shape-in-symbol-names": "error",
    "anti-slop/no-unknown-parameters": "error",
    "anti-slop/no-unknown-returns": "error",
    "anti-slop/no-unknown-type-aliases": "error",
    "anti-slop/no-unsafe-dictionary-type": "error",
    "anti-slop/no-widen-then-assert": "error",
    "anti-slop/require-safety-comment-for-type-assertion": "error",
  },
  overrides: [
    {
      // Test code constructs its own fixtures, so the production trust
      // boundary these assertion rules police does not exist there — and
      // vitest's typing idioms (`as Mock<…>`, `as unknown as Partial<…>`)
      // are assertions by design. Keep them visible as warnings.
      files: [
        "**/*.test.ts",
        "**/*.test.tsx",
        "**/*.test.mjs",
        "**/*.spec.ts",
        "**/__mocks__/**",
        "**/__tests__/**",
        "**/e2e/**",
        "**/test/**",
        "tests-js/**",
      ],
      rules: {
        "anti-slop/no-chained-type-assertions": "warn",
        "anti-slop/no-unsafe-dictionary-type": "warn",
        "anti-slop/require-safety-comment-for-type-assertion": "warn",
      },
    },
  ],
});
