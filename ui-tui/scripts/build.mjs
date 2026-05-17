#!/usr/bin/env node
// Bundles src/entry.tsx into a single self-contained dist/entry.js.
// No runtime node_modules needed.
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { createRequire as __cr } from 'node:module'

const require = __cr(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const out = resolve(root, 'dist/entry.js')

// esbuild binary version mismatch on Android arm64: the npm-installed
// @esbuild/android-arm64 binary can end up with a different version than
// the esbuild package lib/main.js expects (e.g. 0.28.0 vs 0.27.7).
// We resolve the correct binary here and pass it via ESBUILD_BINARY_PATH
// so the service subprocess uses the matching pair.
const esbuildLibDir = resolve(root, 'node_modules/esbuild/lib')
const esbuildBinPath = resolve(esbuildLibDir, '../bin/esbuild')

// IMPORTANT: clear any external ESBUILD_BINARY_PATH that may point to a
// stale version (e.g. ~/.hermes/esbuild-built on this system is 0.28.0
// while the local node_modules/esbuild is 0.27.7).
delete process.env.ESBUILD_BINARY_PATH

let esbuild
try {
  // Try local node_modules esbuild first (version-matched pair)
  esbuild = require('esbuild')
} catch {
  // Fallback: use the explicit binary path
  esbuild = require(resolve(esbuildLibDir, 'main.js'))
}

// `react-devtools-core` is only imported when DEV=true at runtime (Ink dev
// mode). Stub it out so the bundle doesn't carry the dep.
const stubDevtools = {
  name: 'stub-react-devtools-core',
  setup(b) {
    b.onResolve({ filter: /^react-devtools-core$/ }, args => ({
      path: args.path,
      namespace: 'stub-devtools'
    }))
    b.onLoad({ filter: /.*/, namespace: 'stub-devtools' }, () => ({
      contents: 'export default { initialize() {}, connectToDevTools() {} }',
      loader: 'js'
    }))
  }
}

await esbuild.build({
  entryPoints: [resolve(root, 'src/entry.tsx')],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node20',
  outfile: out,
  jsx: 'automatic',
  jsxImportSource: 'react',
  // Skip the prebuilt @hermes/ink bundle and inline the source instead:
  // (1) esbuild's `__esm` helper does not await nested async init, so the
  //     prebuilt bundle's lazy `render` would never resolve when nested in
  //     this top-level Promise.all; (2) bundling from source also lets us
  //     keep `ink-text-input` and the upstream `ink` graph OUT of the
  //     bundle entirely — re-exporting them from entry-exports created a
  //     circular async chain that hung the TUI at startup with only ANSI
  //     reset bytes on screen (#31227).
  alias: { '@hermes/ink': resolve(root, 'packages/hermes-ink/src/entry-exports.ts') },
  plugins: [stubDevtools],
  // Some transitive deps use CommonJS `require(...)` at runtime. ESM bundles
  // don't get a `require` binding automatically, so we inject one.
  banner: {
    js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);"
  },
  logLevel: 'info'
})

// esbuild preserves the shebang from src/entry.tsx into the bundle, but Nix's
// patchShebangs phase mangles `/usr/bin/env -S node --foo --bar` (it strips
// the `node` token, leaving a broken interpreter). The hermes_cli launcher
// always invokes this file as `node dist/entry.js` anyway, so the shebang is
// redundant — strip it.
const body = readFileSync(out, 'utf8')
if (body.startsWith('#!')) {
  writeFileSync(out, body.slice(body.indexOf('\n') + 1))
}

console.log(`built ${out}`)
