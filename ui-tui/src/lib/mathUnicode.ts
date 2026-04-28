// Best-effort LaTeX → Unicode for inline / display math captured by the
// markdown renderer. The terminal can't typeset LaTeX, but Unicode covers
// most of what models actually emit: Greek letters, blackboard / fraktur /
// calligraphic capitals, set theory + logic operators, common arrows,
// sub/superscripts, and `\frac{a}{b}` collapsed to `a/b`.
//
// Design rules:
//   • Pure regex pipeline. Anything we don't recognise is preserved
//     verbatim (so a `\foo{bar}` we've never heard of still survives).
//     A real LaTeX parser would be more correct but throws on partial
//     input — terminal users would rather see the raw command than a
//     parse-error placeholder.
//   • Longest-match-first ordering on commands so `\le` doesn't shadow
//     `\leq`, `\sub` doesn't shadow `\subseteq`, etc.
//   • Word-boundary lookahead `(?![A-Za-z])` after each command so
//     `\pix` (made-up command) doesn't get partially substituted as `π`.
//   • `\mathbb{X}`, `\mathcal{X}`, `\mathfrak{X}` only handle a single
//     letter argument — multi-letter `\mathbb{NN}` is rare and would
//     need a real parser to do correctly.
//   • Sub/super scripts only convert if EVERY character has a Unicode
//     equivalent. Mixed content like `^{n+1}` falls back to the raw
//     LaTeX so we don't emit `ⁿ+¹` (which has no `+` superscript glyph
//     in some fonts and reads worse than the source).

const SYMBOLS: Record<string, string> = {
  // Greek lowercase
  '\\alpha': 'α',
  '\\beta': 'β',
  '\\gamma': 'γ',
  '\\delta': 'δ',
  '\\epsilon': 'ε',
  '\\varepsilon': 'ε',
  '\\zeta': 'ζ',
  '\\eta': 'η',
  '\\theta': 'θ',
  '\\vartheta': 'ϑ',
  '\\iota': 'ι',
  '\\kappa': 'κ',
  '\\lambda': 'λ',
  '\\mu': 'μ',
  '\\nu': 'ν',
  '\\xi': 'ξ',
  '\\pi': 'π',
  '\\varpi': 'ϖ',
  '\\rho': 'ρ',
  '\\varrho': 'ϱ',
  '\\sigma': 'σ',
  '\\varsigma': 'ς',
  '\\tau': 'τ',
  '\\upsilon': 'υ',
  '\\phi': 'φ',
  '\\varphi': 'φ',
  '\\chi': 'χ',
  '\\psi': 'ψ',
  '\\omega': 'ω',

  // Greek uppercase
  '\\Gamma': 'Γ',
  '\\Delta': 'Δ',
  '\\Theta': 'Θ',
  '\\Lambda': 'Λ',
  '\\Xi': 'Ξ',
  '\\Pi': 'Π',
  '\\Sigma': 'Σ',
  '\\Upsilon': 'Υ',
  '\\Phi': 'Φ',
  '\\Psi': 'Ψ',
  '\\Omega': 'Ω',

  // Big operators
  '\\sum': '∑',
  '\\prod': '∏',
  '\\coprod': '∐',
  '\\int': '∫',
  '\\iint': '∬',
  '\\iiint': '∭',
  '\\oint': '∮',
  '\\bigcup': '⋃',
  '\\bigcap': '⋂',
  '\\bigvee': '⋁',
  '\\bigwedge': '⋀',
  '\\bigoplus': '⨁',
  '\\bigotimes': '⨂',

  // Calculus
  '\\partial': '∂',
  '\\nabla': '∇',
  '\\sqrt': '√',

  // Sets
  '\\emptyset': '∅',
  '\\varnothing': '∅',
  '\\infty': '∞',
  '\\in': '∈',
  '\\notin': '∉',
  '\\ni': '∋',
  '\\subset': '⊂',
  '\\supset': '⊃',
  '\\subseteq': '⊆',
  '\\supseteq': '⊇',
  '\\subsetneq': '⊊',
  '\\supsetneq': '⊋',
  '\\cup': '∪',
  '\\cap': '∩',
  '\\setminus': '∖',
  '\\complement': '∁',

  // Logic
  '\\forall': '∀',
  '\\exists': '∃',
  '\\nexists': '∄',
  '\\land': '∧',
  '\\lor': '∨',
  '\\lnot': '¬',
  '\\neg': '¬',
  '\\therefore': '∴',
  '\\because': '∵',

  // Relations
  '\\le': '≤',
  '\\leq': '≤',
  '\\ge': '≥',
  '\\geq': '≥',
  '\\ne': '≠',
  '\\neq': '≠',
  '\\ll': '≪',
  '\\gg': '≫',
  '\\approx': '≈',
  '\\equiv': '≡',
  '\\cong': '≅',
  '\\sim': '∼',
  '\\simeq': '≃',
  '\\propto': '∝',
  '\\perp': '⊥',
  '\\parallel': '∥',
  '\\models': '⊨',
  '\\vdash': '⊢',
  '\\mid': '∣',

  // Brackets / fences (named delimiter commands; the `\left\X` / `\right\X`
  // unwrapping below leaves these behind for the symbol pass to resolve).
  '\\langle': '⟨',
  '\\rangle': '⟩',
  '\\lceil': '⌈',
  '\\rceil': '⌉',
  '\\lfloor': '⌊',
  '\\rfloor': '⌋',
  '\\|': '‖',

  // Arrows
  '\\to': '→',
  '\\rightarrow': '→',
  '\\leftarrow': '←',
  '\\leftrightarrow': '↔',
  '\\Rightarrow': '⇒',
  '\\Leftarrow': '⇐',
  '\\Leftrightarrow': '⇔',
  '\\implies': '⟹',
  '\\impliedby': '⟸',
  '\\iff': '⟺',
  '\\mapsto': '↦',
  '\\hookrightarrow': '↪',
  '\\hookleftarrow': '↩',
  '\\uparrow': '↑',
  '\\downarrow': '↓',
  '\\updownarrow': '↕',

  // Binary operators
  '\\cdot': '⋅',
  '\\cdots': '⋯',
  '\\ldots': '…',
  '\\dots': '…',
  '\\dotsb': '…',
  '\\dotsc': '…',
  '\\vdots': '⋮',
  '\\ddots': '⋱',
  '\\times': '×',
  '\\div': '÷',
  '\\pm': '±',
  '\\mp': '∓',
  '\\circ': '∘',
  '\\bullet': '•',
  '\\star': '⋆',
  '\\ast': '∗',
  '\\oplus': '⊕',
  '\\ominus': '⊖',
  '\\otimes': '⊗',
  '\\odot': '⊙',
  '\\diamond': '⋄',
  '\\angle': '∠',
  '\\triangle': '△',

  // Spacing — collapse to varying widths of regular space
  '\\,': ' ',
  '\\;': ' ',
  '\\:': ' ',
  '\\!': '',
  '\\ ': ' ',
  '\\quad': '  ',
  '\\qquad': '    ',

  // Functions (LaTeX renders these in roman; we just keep the name)
  '\\sin': 'sin',
  '\\cos': 'cos',
  '\\tan': 'tan',
  '\\cot': 'cot',
  '\\sec': 'sec',
  '\\csc': 'csc',
  '\\arcsin': 'arcsin',
  '\\arccos': 'arccos',
  '\\arctan': 'arctan',
  '\\sinh': 'sinh',
  '\\cosh': 'cosh',
  '\\tanh': 'tanh',
  '\\log': 'log',
  '\\ln': 'ln',
  '\\exp': 'exp',
  '\\det': 'det',
  '\\dim': 'dim',
  '\\ker': 'ker',
  '\\lim': 'lim',
  '\\liminf': 'liminf',
  '\\limsup': 'limsup',
  '\\sup': 'sup',
  '\\inf': 'inf',
  '\\max': 'max',
  '\\min': 'min',
  '\\arg': 'arg',
  '\\gcd': 'gcd',

  // Escaped literals — model occasionally emits these for display
  '\\&': '&',
  '\\%': '%',
  '\\$': '$',
  '\\#': '#',
  '\\_': '_',
  '\\{': '{',
  '\\}': '}'
}

const BB: Record<string, string> = {
  A: '𝔸',
  B: '𝔹',
  C: 'ℂ',
  D: '𝔻',
  E: '𝔼',
  F: '𝔽',
  G: '𝔾',
  H: 'ℍ',
  I: '𝕀',
  J: '𝕁',
  K: '𝕂',
  L: '𝕃',
  M: '𝕄',
  N: 'ℕ',
  O: '𝕆',
  P: 'ℙ',
  Q: 'ℚ',
  R: 'ℝ',
  S: '𝕊',
  T: '𝕋',
  U: '𝕌',
  V: '𝕍',
  W: '𝕎',
  X: '𝕏',
  Y: '𝕐',
  Z: 'ℤ'
}

const CAL: Record<string, string> = {
  A: '𝒜',
  B: 'ℬ',
  C: '𝒞',
  D: '𝒟',
  E: 'ℰ',
  F: 'ℱ',
  G: '𝒢',
  H: 'ℋ',
  I: 'ℐ',
  J: '𝒥',
  K: '𝒦',
  L: 'ℒ',
  M: 'ℳ',
  N: '𝒩',
  O: '𝒪',
  P: '𝒫',
  Q: '𝒬',
  R: 'ℛ',
  S: '𝒮',
  T: '𝒯',
  U: '𝒰',
  V: '𝒱',
  W: '𝒲',
  X: '𝒳',
  Y: '𝒴',
  Z: '𝒵'
}

const FRAK: Record<string, string> = {
  A: '𝔄',
  B: '𝔅',
  C: 'ℭ',
  D: '𝔇',
  E: '𝔈',
  F: '𝔉',
  G: '𝔊',
  H: 'ℌ',
  I: 'ℑ',
  J: '𝔍',
  K: '𝔎',
  L: '𝔏',
  M: '𝔐',
  N: '𝔑',
  O: '𝔒',
  P: '𝔓',
  Q: '𝔔',
  R: 'ℜ',
  S: '𝔖',
  T: '𝔗',
  U: '𝔘',
  V: '𝔙',
  W: '𝔚',
  X: '𝔛',
  Y: '𝔜',
  Z: 'ℨ'
}

const SUPERSCRIPT: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '+': '⁺',
  '-': '⁻',
  '=': '⁼',
  '(': '⁽',
  ')': '⁾',
  a: 'ᵃ',
  b: 'ᵇ',
  c: 'ᶜ',
  d: 'ᵈ',
  e: 'ᵉ',
  f: 'ᶠ',
  g: 'ᵍ',
  h: 'ʰ',
  i: 'ⁱ',
  j: 'ʲ',
  k: 'ᵏ',
  l: 'ˡ',
  m: 'ᵐ',
  n: 'ⁿ',
  o: 'ᵒ',
  p: 'ᵖ',
  r: 'ʳ',
  s: 'ˢ',
  t: 'ᵗ',
  u: 'ᵘ',
  v: 'ᵛ',
  w: 'ʷ',
  x: 'ˣ',
  y: 'ʸ',
  z: 'ᶻ'
}

const SUBSCRIPT: Record<string, string> = {
  '0': '₀',
  '1': '₁',
  '2': '₂',
  '3': '₃',
  '4': '₄',
  '5': '₅',
  '6': '₆',
  '7': '₇',
  '8': '₈',
  '9': '₉',
  '+': '₊',
  '-': '₋',
  '=': '₌',
  '(': '₍',
  ')': '₎',
  a: 'ₐ',
  e: 'ₑ',
  h: 'ₕ',
  i: 'ᵢ',
  j: 'ⱼ',
  k: 'ₖ',
  l: 'ₗ',
  m: 'ₘ',
  n: 'ₙ',
  o: 'ₒ',
  p: 'ₚ',
  r: 'ᵣ',
  s: 'ₛ',
  t: 'ₜ',
  u: 'ᵤ',
  v: 'ᵥ',
  x: 'ₓ'
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// Pre-compile two symbol regexes: one for letter-ending commands (`\pi`,
// `\sum`) which need a `(?![A-Za-z])` lookahead so they don't partially
// match `\pix` or `\summa`, and one for punctuation-ending commands
// (`\{`, `\,`, `\|`) which must NOT have the lookahead — otherwise
// `\{p` would refuse to substitute because `p` is a letter.
//
// Longest commands first inside each group so `\leq` beats `\le`.
const splitByEnding = (keys: string[]) => {
  const letter: string[] = []
  const punct: string[] = []

  for (const k of keys) {
    if (/[A-Za-z]$/.test(k)) {
      letter.push(k)
    } else {
      punct.push(k)
    }
  }

  return { letter, punct }
}

const buildAlt = (cmds: string[]) =>
  cmds
    .sort((a, b) => b.length - a.length)
    .map(escapeRe)
    .join('|')

const { letter: LETTER_CMDS, punct: PUNCT_CMDS } = splitByEnding(Object.keys(SYMBOLS))

const SYMBOL_LETTER_RE = new RegExp('(?:' + buildAlt(LETTER_CMDS) + ')(?![A-Za-z])', 'g')
const SYMBOL_PUNCT_RE = new RegExp('(?:' + buildAlt(PUNCT_CMDS) + ')', 'g')

const convertScript = (input: string, table: Record<string, string>, sigil: '^' | '_'): string => {
  let out = ''
  let allMapped = true

  for (const ch of input) {
    const mapped = table[ch]

    if (!mapped) {
      allMapped = false

      break
    }

    out += mapped
  }

  if (allMapped) {
    return out
  }

  // Fallback: if the body is a single visible character (e.g. `∞` after
  // earlier symbol substitution), render it without braces — `^∞` reads
  // far better than `^{∞}` in a terminal. Multi-char bodies that don't
  // fully convert use parens (`e^(iπ)`) instead of braces (`e^{iπ}`)
  // because parens are normal punctuation while braces look like
  // unrendered LaTeX.
  const trimmed = input.trim()

  if ([...trimmed].length === 1) {
    return `${sigil}${trimmed}`
  }

  return `${sigil}(${trimmed})`
}

// Wrap multi-token expressions in parens so `\frac{a+b}{c}` becomes
// `(a+b)/c` rather than `a+b/c`. We only wrap when the expression has
// loose precedence — additive operators or whitespace that would change
// meaning under inline `/`. Atomic factors like `n!`, `x^2`, `\sin x`
// don't need parens; wrapping them just clutters the output.
const wrapForFrac = (expr: string) => {
  const trimmed = expr.trim()

  if (!trimmed) {
    return trimmed
  }

  if (/^\(.*\)$/.test(trimmed)) {
    return trimmed
  }

  if (/[+\-/*]|\s/.test(trimmed)) {
    return `(${trimmed})`
  }

  return trimmed
}

export function texToUnicode(input: string): string {
  let s = input

  s = s.replace(/\\mathbb\s*\{([A-Za-z])\}/g, (raw, c: string) => BB[c] ?? raw)
  s = s.replace(/\\mathcal\s*\{([A-Za-z])\}/g, (raw, c: string) => CAL[c] ?? raw)
  s = s.replace(/\\mathfrak\s*\{([A-Za-z])\}/g, (raw, c: string) => FRAK[c] ?? raw)
  s = s.replace(/\\mathbf\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\mathit\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\mathrm\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\text\s*\{([^{}]+)\}/g, (_, c: string) => c)
  s = s.replace(/\\operatorname\s*\{([^{}]+)\}/g, (_, c: string) => c)

  s = s.replace(/\\overline\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0305`)
  s = s.replace(/\\hat\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0302`)
  s = s.replace(/\\bar\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0304`)
  s = s.replace(/\\tilde\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0303`)
  s = s.replace(/\\vec\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u20D7`)
  s = s.replace(/\\dot\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0307`)
  s = s.replace(/\\ddot\s*\{([^{}]+)\}/g, (_, c: string) => `${c}\u0308`)

  // Apply \frac repeatedly so nested fractions resolve from the inside
  // out — `\frac{1}{1+\frac{1}{x}}` collapses cleanly.
  let prev = ''
  let guard = 0

  while (s !== prev && guard++ < 8) {
    prev = s
    s = s.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, (_, num: string, den: string) => `${wrapForFrac(num)}/${wrapForFrac(den)}`)
  }

  // `\xrightarrow{label}` / `\xleftarrow{label}` collapse to an arrow with
  // the label inline. LaTeX renders the label above the arrow; in monospace
  // we put it adjacent — `─label→` is the closest readable approximation.
  // Run before the symbol pass so the label can still pick up Greek and
  // operator substitutions afterwards.
  s = s.replace(/\\xrightarrow\s*\{([^{}]*)\}/g, (_, label: string) => `─${label.trim()}→`)
  s = s.replace(/\\xleftarrow\s*\{([^{}]*)\}/g, (_, label: string) => `←${label.trim()}─`)
  s = s.replace(/\\Longrightarrow/g, '⟹')
  s = s.replace(/\\Longleftarrow/g, '⟸')
  s = s.replace(/\\Longleftrightarrow/g, '⟺')

  // `\left` and `\right` are sizing wrappers around any delimiter — bare
  // (`\left(`), escaped (`\left\{`), or named (`\left\langle`). Strip the
  // wrapper unconditionally and let the rest of the pipeline (or the
  // upcoming symbol pass) handle whatever delimiter follows. The optional
  // `.?` consumes `\left.` / `\right.` which mean "no delimiter".
  // Lookahead `(?![A-Za-z])` keeps `\leftarrow` / `\leftrightarrow` safe.
  s = s.replace(/\\left(?![A-Za-z])\.?/g, '')
  s = s.replace(/\\right(?![A-Za-z])\.?/g, '')

  // Run symbol substitution BEFORE scripts so a body like `^{\infty}`
  // becomes `^{∞}` first; convertScript can then either map ∞ to a
  // superscript (it can't — Unicode lacks one) or fall back to `^∞`
  // by stripping braces around the now-single-character body.
  //
  // Punctuation pass first — these can be followed by letters (`\{p`
  // is "open-brace then p"), so the letter pass's `(?![A-Za-z])` rule
  // would wrongly block them.
  s = s.replace(SYMBOL_PUNCT_RE, m => SYMBOLS[m] ?? m)
  s = s.replace(SYMBOL_LETTER_RE, m => SYMBOLS[m] ?? m)

  // Bare `^c` / `_c` handles ONLY alphanumerics and `+`/`-`/`=`. Parens
  // are intentionally excluded because the braced-fallback above can
  // emit `(...)` and we don't want a second pass to greedily convert
  // its opening paren into `⁽` and orphan the closing one.
  s = s.replace(/\^\s*\{([^{}]+)\}/g, (_, body: string) => convertScript(body, SUPERSCRIPT, '^'))
  s = s.replace(/\^([A-Za-z0-9+\-=])/g, (raw, ch: string) => SUPERSCRIPT[ch] ?? raw)
  s = s.replace(/_\s*\{([^{}]+)\}/g, (_, body: string) => convertScript(body, SUBSCRIPT, '_'))
  s = s.replace(/_([A-Za-z0-9+\-=])/g, (raw, ch: string) => SUBSCRIPT[ch] ?? raw)

  return s
}
