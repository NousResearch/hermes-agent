import type { Theme } from '../theme.js'

export type Token = [string, string]

interface LangSpec {
  comment: null | string
  keywords: Set<string>
}

const KW = (s: string) => new Set(s.split(/\s+/).filter(Boolean))

const TS = KW(`
  abstract as async await break case catch class const continue debugger default delete do else enum export extends
  finally for from function get if implements import in instanceof interface is let new of package private
  protected public readonly return set static super switch throw try type typeof var void while
  with yield
`)

const PY = KW(`
  and as assert async await break class continue def del elif else except finally for from global if
  import in is lambda nonlocal not or pass raise return try while with yield
`)

const SH = KW(`
  if then else elif fi for in do done while until case esac function return break continue local export readonly
  declare typeset
`)

const GO = KW(`
  break case chan const continue default defer else fallthrough for func go goto if import interface map package range
  return select struct switch type var
`)

const RUST = KW(`
  as async await break const continue crate dyn else enum extern fn for if impl in let loop match mod move mut
  pub ref return static struct super trait type unsafe use where while yield
`)

const SQL = KW(`
  select from where and or not in is as by group order limit offset insert into values update set delete create
  table drop alter add column primary key foreign references join left right inner outer on
`)

const LANGS: Record<string, LangSpec> = {
  go: { comment: '//', keywords: GO },
  json: { comment: null, keywords: KW('') },
  py: { comment: '#', keywords: PY },
  rust: { comment: '//', keywords: RUST },
  sh: { comment: '#', keywords: SH },
  sql: { comment: '--', keywords: SQL },
  ts: { comment: '//', keywords: TS },
  yaml: { comment: '#', keywords: KW('') }
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

const TOKEN_RE = /'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`|\b\d+(?:\.\d+)?\b|[A-Za-z_$][\w$]*|=>|==={0,1}|!==|!=|<=|>=|&&|\|\||[+\-*/%=<>!?:.&|^~]+/g
const CONSTANTS = new Set(['true', 'false', 'null', 'undefined', 'True', 'False', 'None', 'nil', 'self', 'this', 'Self'])
const TYPE_PRECEDERS = new Set(['class', 'interface', 'type', 'enum', 'struct', 'trait', 'impl', 'extends', 'implements', 'new'])
const IDENT_RE = /^[A-Za-z_$][\w$]*$/
const OP_RE = /^(?:=>|==={0,1}|!==|!=|<=|>=|&&|\|\||[+\-*/%=<>!?:.&|^~]+)$/

function pushText(tokens: Token[], text: string, color = '') {
  if (!text) {
    return
  }

  tokens.push([color, text])
}

function findCommentStart(line: string, marker: string): number {
  let quote: string | null = null
  let escaped = false

  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!

    if (escaped) {
      escaped = false
      continue
    }

    if (quote) {
      if (ch === '\\') {
        escaped = true
      } else if (ch === quote) {
        quote = null
      }
      continue
    }

    if (ch === '"' || ch === "'" || ch === '`') {
      quote = ch
      continue
    }

    if (line.startsWith(marker, i)) {
      return i
    }
  }

  return -1
}

export function highlightLine(line: string, lang: string, t: Theme): Token[] {
  const spec = resolve(lang)

  if (!spec) {
    return [['', line]]
  }

  const commentStart = spec.comment ? findCommentStart(line, spec.comment) : -1
  const code = commentStart >= 0 ? line.slice(0, commentStart) : line
  const comment = commentStart >= 0 ? line.slice(commentStart) : ''

  const tokens: Token[] = []
  let last = 0
  let previousIdentifier = ''

  for (const m of code.matchAll(TOKEN_RE)) {
    const start = m.index ?? 0

    if (start > last) {
      pushText(tokens, code.slice(last, start))
    }

    const tok = m[0]
    const ch = tok[0]!
    const nextNonWhitespace = code.slice(start + tok.length).match(/\S/)?.[0]
    let color = t.color.syntaxText

    if (ch === '"' || ch === "'" || ch === '`') {
      color = t.color.syntaxString
    } else if (ch >= '0' && ch <= '9') {
      color = t.color.syntaxNumber
    } else if (OP_RE.test(tok)) {
      color = t.color.syntaxOperator
    } else if (CONSTANTS.has(tok) || /^[A-Z][A-Z0-9_]*$/.test(tok)) {
      color = t.color.syntaxConstant
    } else if (spec.keywords.has(tok)) {
      color = t.color.syntaxKeyword
    } else if (IDENT_RE.test(tok) && (TYPE_PRECEDERS.has(previousIdentifier) || /^[A-Z]/.test(tok))) {
      color = t.color.syntaxType
    } else if (IDENT_RE.test(tok) && nextNonWhitespace === '(') {
      color = t.color.syntaxFunction
    }

    pushText(tokens, tok, color)

    if (IDENT_RE.test(tok)) {
      previousIdentifier = tok
    }

    last = start + tok.length
  }

  if (last < code.length) {
    pushText(tokens, code.slice(last))
  }

  if (comment) {
    pushText(tokens, comment, t.color.syntaxComment)
  }

  return tokens
}
