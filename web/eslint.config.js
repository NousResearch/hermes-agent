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
      // react-hooks v7 introduces strict rules around setState in effects,
      // ref access during render, and manual memoization. These are real
      // concerns but the existing code uses common patterns (data loading on
      // mount, ref-as-instance-var) that need careful refactoring. Warn
      // rather than error so we get visibility without blocking.
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/refs': 'warn',
      'react-hooks/preserve-manual-memoization': 'warn',
      'react-hooks/static-components': 'warn',
    },
  },
])
