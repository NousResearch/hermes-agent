import { describe, expect, it } from 'vitest'

import { BOX_CLOSE, BOX_OPEN, BOX_RE, texToUnicode } from '../lib/mathUnicode.js'

const stripBox = (s: string) => s.replace(BOX_RE, '$1')

describe('texToUnicode — symbols', () => {
  it('substitutes lowercase Greek', () => {
    expect(texToUnicode('\\alpha + \\beta + \\pi')).toBe('α + β + π')
    expect(texToUnicode('\\omega')).toBe('ω')
  })

  it('substitutes uppercase Greek', () => {
    expect(texToUnicode('\\Sigma \\Omega \\Pi')).toBe('Σ Ω Π')
  })

  it('substitutes set theory and logic operators', () => {
    expect(texToUnicode('A \\cup B \\cap C')).toBe('A ∪ B ∩ C')
    expect(texToUnicode('\\forall x \\in \\emptyset')).toBe('∀ x ∈ ∅')
    expect(texToUnicode('p \\implies q \\iff r')).toBe('p ⟹ q ⟺ r')
  })

  it('substitutes relations and arrows', () => {
    expect(texToUnicode('a \\le b \\ge c \\ne d')).toBe('a ≤ b ≥ c ≠ d')
    expect(texToUnicode('f: A \\to B')).toBe('f: A → B')
  })

  it('uses longest-match-first so \\leq beats \\le', () => {
    expect(texToUnicode('\\leq')).toBe('≤')
  })

  it('preserves unknown commands that share a prefix with known ones', () => {
    // `\leqq` is a real LaTeX command (≦) we don't have in our table.
    // The word-boundary lookahead prevents `\le` from matching, so the
    // whole thing is preserved verbatim — much better than `≤qq`.
    expect(texToUnicode('\\leqq')).toBe('\\leqq')
  })

  it('refuses to substitute a partial command (word boundary)', () => {
    expect(texToUnicode('\\alphabet')).toBe('\\alphabet')
    expect(texToUnicode('\\pin')).toBe('\\pin')
  })
})

describe('texToUnicode — blackboard / calligraphic / fraktur', () => {
  it('renders \\mathbb capitals', () => {
    expect(texToUnicode('\\mathbb{R}')).toBe('ℝ')
    expect(texToUnicode('\\mathbb{N} \\subset \\mathbb{Z} \\subset \\mathbb{Q} \\subset \\mathbb{R}')).toBe('ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ')
  })

  it('renders \\mathcal and \\mathfrak', () => {
    expect(texToUnicode('\\mathcal{F} \\subset \\mathfrak{A}')).toBe('ℱ ⊂ 𝔄')
  })

  it('preserves \\mathbb{...} when argument is multi-letter or non-letter', () => {
    expect(texToUnicode('\\mathbb{NN}')).toBe('\\mathbb{NN}')
    expect(texToUnicode('\\mathbb{1}')).toBe('\\mathbb{1}')
  })

  it('strips \\mathbf / \\mathit / \\mathrm / \\text wrappers (no Unicode bold/italic in monospace)', () => {
    expect(texToUnicode('\\mathbf{x}')).toBe('x')
    expect(texToUnicode('\\text{if } x > 0')).toBe('if  x > 0')
    expect(texToUnicode('\\operatorname{rank}(A)')).toBe('rank(A)')
  })
})

describe('texToUnicode — sub / superscripts', () => {
  it('converts simple superscripts', () => {
    expect(texToUnicode('x^2 + y^2')).toBe('x² + y²')
    expect(texToUnicode('e^{n}')).toBe('eⁿ')
  })

  it('converts simple subscripts', () => {
    expect(texToUnicode('a_1 + a_2 + a_n')).toBe('a₁ + a₂ + aₙ')
    expect(texToUnicode('x_{0}')).toBe('x₀')
  })

  it('converts mixed-content scripts when every glyph has a Unicode form', () => {
    // `+`, digits, and lowercase letters all have superscript glyphs,
    // so `n+1` → `ⁿ⁺¹`. Comma has no subscript form, so `i,j` falls
    // back to `_(i,j)` (parens) rather than partially substituting —
    // parens read as ordinary grouping while braces look like leftover
    // unrendered LaTeX.
    expect(texToUnicode('x^{n+1}')).toBe('xⁿ⁺¹')
    expect(texToUnicode('a_{i,j}')).toBe('a_(i,j)')
  })

  it('uses parens (not braces) when the body has Greek with no superscript form', () => {
    // π has no Unicode superscript, so `e^{i\pi}` after symbol pass is
    // `e^{iπ}` and the script fallback emits `e^(iπ)` — much more
    // readable than the LaTeX-looking `e^{iπ}`.
    expect(texToUnicode('e^{i\\pi}')).toBe('e^(iπ)')
  })

  it('strips braces on script fallback when body collapses to a single char', () => {
    // `^{\infty}` → symbol pass produces `^{∞}` → convertScript can't
    // find ∞ in SUPERSCRIPT, but the body is one char so we drop the
    // braces and emit `^∞` (much more readable than `^{∞}`).
    expect(texToUnicode('e^{\\infty}')).toBe('e^∞')
  })

  it('handles a real-world sum', () => {
    expect(texToUnicode('\\sum_{n=0}^{\\infty} \\frac{1}{n!}')).toBe('∑ₙ₌₀^∞ 1/n!')
  })
})

describe('texToUnicode — fractions', () => {
  it('collapses \\frac to a/b', () => {
    expect(texToUnicode('\\frac{1}{2}')).toBe('1/2')
    expect(texToUnicode('\\frac{a}{b}')).toBe('a/b')
  })

  it('parenthesises multi-token numerator / denominator', () => {
    expect(texToUnicode('\\frac{n+1}{2}')).toBe('(n+1)/2')
    expect(texToUnicode('\\frac{a + b}{c - d}')).toBe('(a + b)/(c - d)')
  })

  it('handles nested fractions', () => {
    expect(texToUnicode('\\frac{1}{\\frac{1}{x}}')).toBe('1/(1/x)')
  })

  it('handles braces inside numerator / denominator (regression: regex \\frac couldn\'t)', () => {
    // The regex-only `\frac` matcher used `[^{}]*` for each arg, which
    // failed the moment a numerator contained its own braces (here the
    // `{p-1}` from a superscript). The balanced-brace parser handles it.
    expect(texToUnicode('\\frac{|t|^{p-1}|P(t)|^p}{(p-1)!}')).toBe('(|t|ᵖ⁻¹|P(t)|ᵖ)/((p-1)!)')
  })

  it('preserves \\frac when arguments are malformed', () => {
    expect(texToUnicode('\\frac{a}')).toBe('\\frac{a}')
    expect(texToUnicode('\\fraction{a}{b}')).toBe('\\fraction{a}{b}')
  })
})

describe('texToUnicode — typography no-ops', () => {
  it('strips \\displaystyle / \\textstyle / \\scriptstyle / \\scriptscriptstyle', () => {
    expect(texToUnicode('\\displaystyle\\sum_{i=1}^n x_i')).toBe('∑ᵢ₌₁ⁿ xᵢ')
    expect(texToUnicode('f(x) = \\displaystyle \\frac{1}{2}')).toBe('f(x) = 1/2')
    expect(texToUnicode('\\textstyle x + y')).toBe('x + y')
  })

  it('strips \\limits / \\nolimits which only affect bound positioning', () => {
    expect(texToUnicode('\\sum\\limits_{k=1}^n a_k')).toBe('∑ₖ₌₁ⁿ aₖ')
    expect(texToUnicode('\\int\\nolimits_0^1 f(x) dx')).toBe('∫₀¹ f(x) dx')
  })

  it('does not eat letter-continuation commands like \\limit_inf', () => {
    // The `(?![A-Za-z])` lookahead protects hypothetical commands that
    // start with `\limit` / `\display` / etc. The bare names are stripped
    // but anything longer is preserved verbatim.
    expect(texToUnicode('\\limitinf x')).toBe('\\limitinf x')
  })
})

describe('texToUnicode — sizing wrappers', () => {
  it('strips \\big / \\Big / \\bigg / \\Bigg before delimiters', () => {
    expect(texToUnicode('\\bigl[ x \\bigr]')).toBe('[ x ]')
    expect(texToUnicode('\\Big( y \\Big)')).toBe('( y )')
    expect(texToUnicode('\\bigg| z \\bigg|')).toBe('| z |')
    expect(texToUnicode('\\Biggl\\{ a \\Biggr\\}')).toBe('{ a }')
  })

  it('does not eat \\bigtriangleup or other letter-continuations', () => {
    expect(texToUnicode('A \\bigtriangleup B')).toBe('A \\bigtriangleup B')
  })
})

describe('texToUnicode — modular arithmetic and tags', () => {
  it('renders \\pmod{p} as " (mod p)"', () => {
    expect(texToUnicode('a \\equiv b \\pmod{p}')).toBe('a ≡ b (mod p)')
  })

  it('renders \\bmod / \\mod inline', () => {
    expect(texToUnicode('a \\bmod n')).toBe('a mod n')
  })

  it('collapses \\tag{n} to " (n)"', () => {
    expect(texToUnicode('x = y \\tag{24}')).toBe('x = y (24)')
  })
})

describe('texToUnicode — newly added symbols', () => {
  it('renders \\nmid, \\blacksquare, \\qed', () => {
    expect(texToUnicode('p \\nmid q')).toBe('p ∤ q')
    expect(texToUnicode('Therefore \\blacksquare')).toBe('Therefore ■')
    expect(texToUnicode('done \\qed')).toBe('done ∎')
  })
})

describe('texToUnicode — \\boxed / \\fbox', () => {
  // `\boxed` produces non-printable U+0001 / U+0002 sentinels around its
  // content so the markdown renderer can apply highlight styling. These
  // tests assert both the sentinel form and the human-readable
  // strip-fallback (BOX_RE).
  it('wraps simple boxed content in BOX_OPEN/BOX_CLOSE sentinels', () => {
    expect(texToUnicode('\\boxed{x = 0}')).toBe(`${BOX_OPEN}x = 0${BOX_CLOSE}`)
    expect(stripBox(texToUnicode('\\boxed{x = 0}'))).toBe('x = 0')
    expect(stripBox(texToUnicode('\\fbox{answer}'))).toBe('answer')
  })

  it('handles boxed expressions with nested braces (regression: regex couldn\'t)', () => {
    // A `[^{}]*` regex would stop at the first `{` inside the body. The
    // balanced-brace parser walks past it.
    expect(stripBox(texToUnicode('\\boxed{x^{n+1}}'))).toBe('xⁿ⁺¹')
    expect(stripBox(texToUnicode('\\boxed{\\frac{a}{b}}'))).toBe('a/b')
  })

  it('handles real-world boxed final answer', () => {
    expect(stripBox(texToUnicode('\\boxed{J = -\\sum_{k=0}^n a_k F(k)}'))).toBe('J = -∑ₖ₌₀ⁿ aₖ F(k)')
  })

  it('preserves \\boxed without a brace argument', () => {
    expect(texToUnicode('\\boxed something')).toBe('\\boxed something')
  })
})

describe('texToUnicode — combining marks', () => {
  it('applies \\overline / \\bar / \\hat / \\vec / \\tilde', () => {
    expect(texToUnicode('\\overline{x}')).toBe('x\u0305')
    expect(texToUnicode('\\hat{y}')).toBe('y\u0302')
    expect(texToUnicode('\\vec{v}')).toBe('v\u20D7')
  })
})

describe('texToUnicode — left/right delimiters', () => {
  it('strips \\left and \\right keeping the delimiter character', () => {
    expect(texToUnicode('\\left( x + y \\right)')).toBe('( x + y )')
    expect(texToUnicode('\\left| x \\right|')).toBe('| x |')
  })

  it('handles escaped delimiters \\left\\{ ... \\right\\}', () => {
    expect(texToUnicode('\\left\\{p/q \\mid q \\neq 0\\right\\}')).toBe('{p/q ∣ q ≠ 0}')
  })

  it('handles named delimiters via \\left\\langle / \\right\\rangle', () => {
    expect(texToUnicode('\\left\\langle u, v \\right\\rangle')).toBe('⟨ u, v ⟩')
  })

  it('drops \\left. and \\right. (which are explicit "no delimiter")', () => {
    expect(texToUnicode('\\left. f \\right|')).toBe(' f |')
  })

  it('preserves \\leftarrow / \\rightarrow (word boundary blocks the strip)', () => {
    expect(texToUnicode('A \\leftarrow B \\rightarrow C')).toBe('A ← B → C')
  })
})

describe('texToUnicode — labelled arrows', () => {
  it('renders \\xrightarrow{label} as ─label→', () => {
    expect(texToUnicode('a \\xrightarrow{x=1} b')).toBe('a ─x=1→ b')
  })

  it('renders \\xleftarrow{label} as ←label─', () => {
    expect(texToUnicode('a \\xleftarrow{n} b')).toBe('a ←n─ b')
  })

  it('still applies symbol substitution inside the label', () => {
    expect(texToUnicode('a \\xrightarrow{n \\to \\infty} L')).toBe('a ─n → ∞→ L')
  })
})

describe('texToUnicode — punctuation commands without lookahead', () => {
  it('substitutes \\{ even when immediately followed by a letter', () => {
    // Regression: with a global `(?![A-Za-z])` lookahead, `\{p` refused
    // to substitute (because `p` is a letter) and rendered as `\{p`.
    expect(texToUnicode('\\{p, q\\}')).toBe('{p, q}')
  })

  it('substitutes thin-space \\, before a letter', () => {
    expect(texToUnicode('a\\,b')).toBe('a b')
  })
})

describe('texToUnicode — round-trip realism', () => {
  it('renders a typical model-emitted formula', () => {
    expect(texToUnicode('\\alpha \\in \\mathbb{R}, \\alpha \\notin \\mathbb{Q}')).toBe('α ∈ ℝ, α ∉ ℚ')
  })

  it('preserves unknown commands verbatim', () => {
    expect(texToUnicode('\\bigtriangleup \\circledast')).toBe('\\bigtriangleup \\circledast')
  })

  it('handles commands without delimiters between', () => {
    // Word-boundary lookahead means `\alpha\beta` doesn't accidentally
    // match `\alphabeta` as one ungrouped token.
    expect(texToUnicode('\\alpha\\beta')).toBe('αβ')
  })

  it('leaves plain text alone', () => {
    expect(texToUnicode('hello world')).toBe('hello world')
    expect(texToUnicode('')).toBe('')
  })
})
