const fs = require('fs')
const path = require('path')
const ts = require('/root/hermes-agent/node_modules/typescript')

const ROOT = '/tmp/sync'

function load(file, name) {
  const src = fs.readFileSync(file, 'utf8')
  const out = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.React }
  }).outputText
  const m = { exports: {} }
  const req = (id) => ({ defineLocale: (o) => o, mergeTranslations: (o) => o })
  new Function('require', 'module', 'exports', out)(req, m, m.exports)
  return m.exports[name]
}

function flat(o, p, a) {
  p = p || []; a = a || {}
  if (o && typeof o === 'object' && !Array.isArray(o) && '__raw' in o) { a[p.join('.')] = { kind: 'raw' }; return a }
  if (typeof o === 'string') { a[p.join('.')] = { kind: 'string', v: o }; return a }
  if (typeof o === 'function') { a[p.join('.')] = { kind: 'function', v: o, arity: o.length }; return a }
  if (Array.isArray(o)) { a[p.join('.')] = { kind: 'array', v: o }; return a }
  if (o && typeof o === 'object') { for (const k of Object.keys(o)) flat(o[k], p.concat(k), a); return a }
  a[p.join('.')] = { kind: '?' }; return a
}

const en = flat(load(path.join(ROOT, 'web/src/i18n/en.ts'), 'en'))
const cs = flat(load(path.join(ROOT, 'web/src/i18n/cs.ts'), 'cs'))
const enk = Object.keys(en), csk = Object.keys(cs)
const miss = enk.filter((k) => !(k in cs))
const obs = csk.filter((k) => !(k in en))
console.log('WEB en=', enk.length, 'cs=', csk.length, 'missing=', miss.length, 'obsolete=', obs.length)
if (miss.length) console.log('MISSING: ' + miss.slice(0, 80).join(', '))
if (obs.length) console.log('OBSOLETE: ' + obs.slice(0, 80).join(', '))
const kind = enk.filter((k) => k in cs && en[k].kind !== cs[k].kind)
console.log('kindMismatch=' + kind.length + (kind.length ? ': ' + kind.slice(0, 20).join(', ') : ''))
