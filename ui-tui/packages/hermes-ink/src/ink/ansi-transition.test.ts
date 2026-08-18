import { describe, expect, it } from 'vitest'

import { transitionAnsiCodes } from './ansi-transition.js'
import { StylePool } from './screen.js'

const ESC = '\u001b'
const BOLD = { type: 'ansi' as const, code: `${ESC}[1m`, endCode: `${ESC}[22m` }
const DIM = { type: 'ansi' as const, code: `${ESC}[2m`, endCode: `${ESC}[22m` }
const FG_PINK = { type: 'ansi' as const, code: `${ESC}[38;5;204m`, endCode: `${ESC}[39m` }
const FG_GRAY = { type: 'ansi' as const, code: `${ESC}[38;5;245m`, endCode: `${ESC}[39m` }

const codes = (result: ReturnType<typeof transitionAnsiCodes>) => result.map(c => c.code)

/**
 * SGR 1 (bold) and SGR 2 (dim) are independent attributes sharing one reset
 * (SGR 22). A transition that swaps one for the other MUST pass through 22 —
 * otherwise the terminal accumulates both and every later transition is
 * computed from phantom state (the "random dimness/opacity" drift).
 */
describe('transitionAnsiCodes weight family', () => {
  it('bold → dim resets the weight family before applying dim', () => {
    expect(codes(transitionAnsiCodes([BOLD], [DIM]))).toEqual([`${ESC}[22m`, `${ESC}[2m`])
  })

  it('dim → bold resets the weight family before applying bold', () => {
    expect(codes(transitionAnsiCodes([DIM], [BOLD]))).toEqual([`${ESC}[22m`, `${ESC}[1m`])
  })

  it('carries color changes through a weight swap', () => {
    expect(codes(transitionAnsiCodes([BOLD, FG_PINK], [DIM, FG_GRAY]))).toEqual([
      `${ESC}[22m`,
      `${ESC}[38;5;245m`,
      `${ESC}[2m`
    ])
  })

  it('weight removal to plain still emits the reset', () => {
    expect(codes(transitionAnsiCodes([BOLD, FG_PINK], [FG_GRAY]))).toEqual([`${ESC}[22m`, `${ESC}[38;5;245m`])
    expect(codes(transitionAnsiCodes([DIM], []))).toEqual([`${ESC}[22m`])
  })

  it('pure additions stay minimal (no gratuitous resets)', () => {
    expect(codes(transitionAnsiCodes([], [BOLD]))).toEqual([`${ESC}[1m`])
    expect(codes(transitionAnsiCodes([FG_PINK], [FG_PINK, DIM]))).toEqual([`${ESC}[2m`])
    expect(codes(transitionAnsiCodes([BOLD], [BOLD, FG_PINK]))).toEqual([`${ESC}[38;5;204m`])
  })

  it('unchanged styles emit nothing', () => {
    expect(transitionAnsiCodes([BOLD, FG_PINK], [BOLD, FG_PINK])).toEqual([])
  })

  // Real tool output (ls/grep) ships COMPOUND sequences like `[1;31m` whose
  // endCode is `[0m` — weight detection must parse params, not endCodes.
  const compound = (params: string) => ({ type: 'ansi' as const, code: `${ESC}[${params}m`, endCode: `${ESC}[0m` })

  it('compound bold → compound dim resets the weight family', () => {
    expect(codes(transitionAnsiCodes([compound('1;31')], [compound('2;37')]))).toEqual([
      `${ESC}[22m`,
      `${ESC}[2;37m`
    ])
  })

  it('compound bold → compound bold (color change) stays minimal', () => {
    expect(codes(transitionAnsiCodes([compound('1;31')], [compound('1;32')]))).toEqual([`${ESC}[1;32m`])
  })

  it('compound weight removal to plain emits the reset', () => {
    expect(codes(transitionAnsiCodes([compound('1;31')], [FG_GRAY]))).toEqual([`${ESC}[22m`, `${ESC}[38;5;245m`])
  })

  it('extended-color arguments are not read as weight atoms', () => {
    // `38;2;r;g;b` / `38;5;N` carry literal 2/5 sub-params: not SGR atoms.
    const tc = { type: 'ansi' as const, code: `${ESC}[38;2;120;87;109m`, endCode: `${ESC}[39m` }

    expect(codes(transitionAnsiCodes([tc], [FG_GRAY]))).toEqual([`${ESC}[38;5;245m`])
    expect(codes(transitionAnsiCodes([FG_PINK], [tc]))).toEqual([`${ESC}[38;2;120;87;109m`])
  })
})

describe('StylePool.transition weight correctness', () => {
  // Minimal SGR interpreter: the ground truth a real terminal applies.
  const applySgr = (state: { bold: boolean; dim: boolean; fg: string }, str: string) => {
    const re = new RegExp(`${ESC}\\[([0-9;]*)m`, 'g')
    let match: null | RegExpExecArray

    while ((match = re.exec(str)) !== null) {
      const parts = (match[1] || '0').split(';')

      for (let i = 0; i < parts.length; i++) {
        const p = parts[i]!

        if (p === '0' || p === '') {
          state.bold = false
          state.dim = false
          state.fg = 'default'
        } else if (p === '1') {
          state.bold = true
        } else if (p === '2') {
          state.dim = true
        } else if (p === '22') {
          state.bold = false
          state.dim = false
        } else if (p === '39') {
          state.fg = 'default'
        } else if (p === '38' && parts[i + 1] === '5') {
          state.fg = `ansi${parts[i + 2]}`
          i += 2
        }
      }
    }
  }

  it('every pairwise transition lands the terminal in exactly the target state', () => {
    const pool = new StylePool()

    const ids = [
      pool.none,
      pool.intern([BOLD]),
      pool.intern([DIM]),
      pool.intern([FG_PINK]),
      pool.intern([BOLD, FG_PINK]),
      pool.intern([DIM, FG_GRAY]),
      pool.intern([DIM, FG_PINK]),
      pool.intern([BOLD, FG_GRAY])
    ]

    const expected = (id: number) => {
      const styles = pool.get(id)
      const fgCode = styles.find(s => s.endCode === `${ESC}[39m`)?.code

      return {
        bold: styles.some(s => s.code === `${ESC}[1m`),
        dim: styles.some(s => s.code === `${ESC}[2m`),
        fg: fgCode ? `ansi${fgCode.slice(7, -1)}` : 'default'
      }
    }

    for (const from of ids) {
      for (const to of ids) {
        const state = expected(from)
        applySgr(state, pool.transition(from, to))
        expect(state, `transition ${from} → ${to}`).toEqual(expected(to))
      }
    }
  })
})
