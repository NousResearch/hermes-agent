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

const isDigit = (ch: string): boolean => ch >= '0' && ch <= '9'
const isIdentStart = (ch: string): boolean =>
  (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch === '_' || ch === '$'
const isIdentContinue = (ch: string): boolean => isIdentStart(ch) || isDigit(ch)

const scanString = (line: string, start: number, quote: string): number => {
  let i = start + 1

  while (i < line.length) {
    const ch = line[i]!

    if (ch === '\\') {
      i += 2
      continue
    }

    i += 1
    if (ch === quote) {
      return i
    }
  }

  return line.length
}

export function highlightLine(line: string, lang: string, t: Theme): Token[] {
  const spec = resolve(lang)

  if (!spec) {
    return [['', line]]
  }

  if (spec.comment && line.trimStart().startsWith(spec.comment)) {
    return [[t.color.muted, line]]
  }

  const tokens: Token[] = []
  let i = 0

  while (i < line.length) {
    const ch = line[i]!

    if (ch === '"' || ch === "'" || ch === '`') {
      const end = scanString(line, i, ch)
      tokens.push([t.color.accent, line.slice(i, end)])
      i = end
      continue
    }

    if (isDigit(ch)) {
      let end = i + 1

      while (end < line.length && isDigit(line[end]!)) {
        end += 1
      }

      if (line[end] === '.') {
        const fracStart = end + 1
        let fracEnd = fracStart

        while (fracEnd < line.length && isDigit(line[fracEnd]!)) {
          fracEnd += 1
        }

        if (fracEnd > fracStart) {
          end = fracEnd
        }
      }

      tokens.push([t.color.text, line.slice(i, end)])
      i = end
      continue
    }

    if (isIdentStart(ch)) {
      let end = i + 1

      while (end < line.length && isIdentContinue(line[end]!)) {
        end += 1
      }

      const tok = line.slice(i, end)
      tokens.push(spec.keywords.has(tok) ? [t.color.border, tok] : ['', tok])
      i = end
      continue
    }

    let end = i + 1
    while (end < line.length) {
      const next = line[end]!
      if (next === '"' || next === "'" || next === '`' || isDigit(next) || isIdentStart(next)) {
        break
      }
      end += 1
    }

    tokens.push(['', line.slice(i, end)])
    i = end
  }

  return tokens
}
