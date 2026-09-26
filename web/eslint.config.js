import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // Context providers and hook files commonly export both a component
      // (the Provider) and a hook (useContext). Allow constant exports so
      // these don't need to be split into separate files.
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      // Upgraded from 'warn' once the codebase was swept clean (render-phase
      // state adjustments, microtask boundaries in effects, helpers moved out
      // of component files). New violations now fail `npm run lint`, and the
      // lint script enforces --max-warnings 0 in CI.
      'react-hooks/set-state-in-effect': 'error',
      'react-hooks/refs': 'error',
      'react-hooks/preserve-manual-memoization': 'error',
      'react-hooks/static-components': 'error',
    },
  },
])
