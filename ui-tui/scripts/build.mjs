#!/usr/bin/env node
// Bundles src/entry.tsx into a single self-contained dist/entry.js.
// No runtime node_modules needed.
import { build } from 'esbuild'
import { readFileSync, writeFileSync, renameSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const out = resolve(root, 'dist/entry.js')

// Build to a private temp file and rename into place at the very end.
// esbuild writes `outfile` NON-atomically: it truncates the existing bundle to
// 0 bytes and streams ~3.7MB back in. Any process that runs
// `node dist/entry.js` during that window loads a half-written module and dies
// with `SyntaxError: Unexpected end of input` at whatever byte the writer had
// reached. That is not hypothetical — launching two Hermes panes at once (e.g.
// an AgentGrid grid, or a dashboard Chat tab racing a terminal launch) makes
// one pane rebuild while the other execs the same path.
//
// rename(2) within a directory is atomic on POSIX and on Windows via
// MoveFileEx's replace semantics, so a concurrent reader now always sees
// either the complete previous bundle or the complete new one — never a
// prefix. The pid suffix keeps two concurrent builders off each other's temp.
const tmp = `${out}.tmp-${process.pid}`

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

try {
  await build({
    entryPoints: [resolve(root, 'src/entry.tsx')],
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node20',
    outfile: tmp,
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
  //
  // This rewrite happens on the temp file, BEFORE the rename. Doing it after
  // publishing would reintroduce the same truncation window it is meant to
  // avoid.
  const body = readFileSync(tmp, 'utf8')
  if (body.startsWith('#!')) {
    writeFileSync(tmp, body.slice(body.indexOf('\n') + 1))
  }

  // Publish. Atomic: readers see old-or-new, never a partial bundle.
  renameSync(tmp, out)
} finally {
  // A failed build (or a crash between build and rename) must not leave temp
  // bundles behind; dist/ is on the launcher's hot path. Already-renamed is a
  // no-op thanks to force.
  rmSync(tmp, { force: true })
}

console.log(`built ${out}`)
