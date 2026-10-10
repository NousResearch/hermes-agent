/**
 * Tests for `src/domain/codeFence.ts` — pure code-fence parser.
 */

import { describe, expect, it } from 'vitest'

import { parseCodeFences } from '../domain/codeFence.js'

describe('parseCodeFences', () => {
  it('parses a simple backtick fence', () => {
    const text = '```python\nprint("hello")\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.closed).toBe(true)
    expect(fences[0]!.fenceChar).toBe('`')
    expect(fences[0]!.fenceLength).toBe(3)
    expect(fences[0]!.language).toBe('python')
    expect(fences[0]!.rawContent).toBe('print("hello")')
    expect(fences[0]!.infoString).toBe('python')
    expect(fences[0]!.endLineIndex).toBe(2)
  })

  it('parses a tilde fence', () => {
    const text = '~~~javascript\nconst x = 1\n~~~\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.fenceChar).toBe('~')
    expect(fences[0]!.fenceLength).toBe(3)
    expect(fences[0]!.language).toBe('javascript')
    expect(fences[0]!.rawContent).toBe('const x = 1')
  })

  it('requires fence length >= 3', () => {
    // `` only two backticks — not a fence opener
    const text = '``python\nprint("nope")\n``\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(0)
  })

  it('requires closer to match fence character', () => {
    // Opens with backticks, closes with tildes — never closes
    const text = '```python\nprint("hello")\n~~~\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.closed).toBe(false)
  })

  it('requires closer length >= opener length', () => {
    // Opens with ```` (4), closer is ``` (3) — not enough
    const text = '````python\ncode\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.closed).toBe(false)
  })

  it('parses a longer fence', () => {
    const text = '````ts\nconst a = 1\n````\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.fenceLength).toBe(4)
    expect(fences[0]!.language).toBe('ts')
    expect(fences[0]!.rawContent).toBe('const a = 1')
  })

  it('extracts language from info string', () => {
    const text = '```typescript linenums="1" hl_lines=[1]\nprint(1)\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.language).toBe('typescript')
    expect(fences[0]!.infoString).toBe('typescript linenums="1" hl_lines=[1]')
  })

  it('defaults language to "text" when no info string', () => {
    const text = '```\nplain content\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.language).toBe('text')
    expect(fences[0]!.infoString).toBe('')
  })

  it('defaults language to "diff" for diff content', () => {
    const text = '```\n--- old.py\n+++ new.py\n-print()\n+print(1)\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.language).toBe('diff')
  })

  it('preserves tabs and trailing spaces in rawContent', () => {
    const text = '```py\n\tindented\ntrailing  \n```\n'
    const fences = parseCodeFences(text)

    expect(fences[0]!.rawContent).toBe('\tindented\ntrailing  ')
  })

  it('preserves empty interior lines', () => {
    const text = '```js\nline one\n\nline three\n```\n'
    const fences = parseCodeFences(text)

    expect(fences[0]!.rawContent).toBe('line one\n\nline three')
  })

  it('handles unclosed fences', () => {
    const text = '```python\ncode without closer\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.closed).toBe(false)
    expect(fences[0]!.endLineIndex).toBe(-1)
    expect(fences[0]!.rawContent).toBe('code without closer\n')
  })

  it('handles empty code blocks', () => {
    const text = '```text\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.rawContent).toBe('')
  })

  it('parses multiple fences in one source', () => {
    const text = '```python\nprint(1)\n```\n\nsome text\n\n```rust\nfn main() {}\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(2)
    expect(fences[0]!.language).toBe('python')
    expect(fences[0]!.rawContent).toBe('print(1)')
    expect(fences[1]!.language).toBe('rust')
    expect(fences[1]!.rawContent).toBe('fn main() {}')
  })

  it('handles $${bait}$$ inside a code fence without breaking', () => {
    const text = '```\n// $$ looks like math $$\ncode here\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.closed).toBe(true)
    expect(fences[0]!.rawContent).toBe('// $$ looks like math $$\ncode here')
  })

  it('requires closer to use same character type', () => {
    // Backtick opener, tilde closer — no match, never closes
    const text = '````\ncode\n~~~`\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.fenceChar).toBe('`')
    expect(fences[0]!.closed).toBe(false)
  })

  it('handles tilde closer matching tilde opener', () => {
    const text = '~~~python\ncode\n~~~\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.fenceChar).toBe('~')
    expect(fences[0]!.closed).toBe(true)
  })

  it('ignores whitespace-only info string', () => {
    const text = '```   \ncode\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.language).toBe('text')
    expect(fences[0]!.infoString).toBe('')
  })

  it('preserves intentional final blank line', () => {
    const text = '```python\ncode\n\n```\n'
    const fences = parseCodeFences(text)

    expect(fences[0]!.rawContent).toBe('code\n')
  })

  it('handles mixed fence types in same source', () => {
    const text = '```backtick```\ncode1\n```\n~~~tilde~~~\ncode2\n~~~\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(2)
    expect(fences[0]!.fenceChar).toBe('`')
    expect(fences[0]!.rawContent).toBe('code1')
    expect(fences[1]!.fenceChar).toBe('~')
    expect(fences[1]!.rawContent).toBe('code2')
  })

  it('handles fenced code with complex content', () => {
    const content = [
      'import sys',
      '',
      'def main():',
      '\t# tab-indented body',
      '\tprint("tabs and quotes")',
      '\treturn 0  ',
      '',
      'if __name__ == "__main__":',
      '\tsys.exit(main())',
      ''
    ].join('\n')

    const text = '```python\n' + content + '\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.rawContent).toBe(content)
  })

  it('handles fence with leading whitespace on opener', () => {
    const text = '    ```python\n    code\n    ```\n'
    const fences = parseCodeFences(text)

    // The parser uses `/^\s*(`{3,}|~{3,})/` — leading whitespace is allowed
    // on the opener. But the content includes the leading whitespace.
    expect(fences).toHaveLength(1)
    expect(fences[0]!.language).toBe('python')
  })

  it('handles tilde fence inside backtick fence', () => {
    const text = '```\ntilde ~~~ is not a fence\n~~~\n```\n'
    const fences = parseCodeFences(text)

    expect(fences).toHaveLength(1)
    expect(fences[0]!.closed).toBe(true)
    expect(fences[0]!.rawContent).toBe('tilde ~~~ is not a fence\n~~~')
  })
})
