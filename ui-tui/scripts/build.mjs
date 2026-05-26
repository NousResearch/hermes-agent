#!/usr/bin/env node
// Bundles src/entry.tsx into a single self-contained dist/entry.js.
// No runtime node_modules needed.
import { build } from 'esbuild'
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const out = resolve(root, 'dist/entry.js')

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

await build({
  entryPoints: [resolve(root, 'src/entry.tsx')],
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node20',
  outfile: out,
  jsx: 'automatic',
  jsxImportSource: 'react',
  // Skip the prebuilt @hermes/ink bundle — esbuild's __esm helper doesn't
  // await nested async init, which breaks lazy-initialized exports like
  // `render`. Bundling from source sidesteps that.
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
const bundled = body.startsWith('#!') ? body.slice(body.indexOf('\n') + 1) : body

// Guard release bundles against reintroducing OpenTUI's native extractor,
// which leaked hidden libopentui.so copies into /tmp on Linux (#32283).
const forbiddenNativeMarkers = ['@opentui/', 'libopentui', 'opentui']
for (const marker of forbiddenNativeMarkers) {
  if (bundled.toLowerCase().includes(marker)) {
    throw new Error(`TUI bundle contains forbidden native runtime marker ${JSON.stringify(marker)}`)
  }
}

if (bundled !== body) {
  writeFileSync(out, bundled)
}

console.log(`built ${out}`)
