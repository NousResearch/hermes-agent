import type { Theme } from '../theme.js'

export type Token = [string, string]

interface LangSpec {
  comment: null | string
  keywords: Set<string>
}

const KW = (s: string) => new Set(s.split(/\s+/).filter(Boolean))

const TS = KW(`
  abstract as async await break case catch class const continue debugger default delete do else enum export extends
  false finally for from function get if implements import in instanceof interface is let new null of package private
  protected public readonly return set static super switch this throw true try type typeof undefined var void while
  with yield
`)

const PY = KW(`
  False None True and as assert async await break class continue def del elif else except finally for from global if
  import in is lambda nonlocal not or pass raise return try while with yield
`)

const SH = KW(`
  if then else elif fi for in do done while until case esac function return break continue local export readonly
  declare typeset
`)

const GO = KW(`
  break case chan const continue default defer else fallthrough for func go goto if import interface map package range
  return select struct switch type var nil true false
`)

const RUST = KW(`
  as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut
  pub ref return self Self static struct super trait true type unsafe use where while yield
`)

const SQL = KW(`
  select from where and or not in is null as by group order limit offset insert into values update set delete create
  table drop alter add column primary key foreign references join left right inner outer on
`)

const LANGS: Record<string, LangSpec> = {
  go: { comment: '//', keywords: GO },
  json: { comment: null, keywords: KW('true false null') },
  py: { comment: '#', keywords: PY },
  rust: { comment: '//', keywords: RUST },
  sh: { comment: '#', keywords: SH },
  sql: { comment: '--', keywords: SQL },
  ts: { comment: '//', keywords: TS },
  yaml: { comment: '#', keywords: KW('true false null yes no on off') }
}

const ALIAS: Record<string, string> = {
  bash: 'sh',
  javascript: 'ts',
  js: 'ts',
  jsx: 'ts',
  python: 'py',
  rs: 'rust',
  shell: 'sh',
  tsx: 'ts',
  typescript: 'ts',
  yml: 'yaml',
  zsh: 'sh'
}

const resolve = (lang: string): LangSpec | null => LANGS[ALIAS[lang] ?? lang] ?? null

export const isHighlightable = (lang: string): boolean => resolve(lang) !== null

const TOKEN_RE = /'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`|\b\d+(?:\.\d+)?\b|[A-Za-z_$][\w$]*/g

export function highlightLine(line: string, lang: string, t: Theme): Token[] {
  const spec = resolve(lang)

  if (!spec) {
    return [['', line]]
  }

  if (spec.comment && line.trimStart().startsWith(spec.comment)) {
    return [[t.color.dim, line]]
  }

  const tokens: Token[] = []
  let last = 0

  for (const m of line.matchAll(TOKEN_RE)) {
    const start = m.index ?? 0

    if (start > last) {
      tokens.push(['', line.slice(last, start)])
    }

    const tok = m[0]
    const ch = tok[0]!

    if (ch === '"' || ch === "'" || ch === '`') {
      tokens.push([t.color.amber, tok])
    } else if (ch >= '0' && ch <= '9') {
      tokens.push([t.color.cornsilk, tok])
    } else if (spec.keywords.has(tok)) {
      tokens.push([t.color.bronze, tok])
    } else {
      tokens.push(['', tok])
    }

    last = start + tok.length
  }

  if (last < line.length) {
    tokens.push(['', line.slice(last)])
  }

  return tokens
}
