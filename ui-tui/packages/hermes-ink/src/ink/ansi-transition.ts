import { type AnsiCode, diffAnsiCodes } from '@alcalzone/ansi-tokenize'

/**
 * Bold (SGR 1) and dim (SGR 2) are INDEPENDENT terminal attributes that share
 * one reset code (SGR 22). `diffAnsiCodes` models "same endCode" as "same
 * slot, new start code overwrites" — true for fg (39) / bg (49), false here:
 * emitting `[2m` over a bold cell yields bold+dim, not dim.
 *
 * Every transition the diff renderer emits from that assumption leaves the
 * terminal's real attributes diverged from the StylePool's tracked state, and
 * because later transitions are computed FROM that tracked state, the
 * corruption compounds and sticks — visible as random spans of wrong
 * weight/brightness ("random dimness/opacity changes") that depend on which
 * cells happened to change in which order.
 *
 * Weight flags hide in two shapes: standalone `[1m`/`[2m` (endCode `[22m`),
 * and compound sequences from real tool output — `[1;31m` ls/grep style —
 * whose endCode is `[0m`, dodging any endCode-based check. Both are detected
 * by parsing the params (skipping 38/48 extended-color arguments, whose
 * literal `2`/`5` sub-params are not SGR atoms).
 */
const WEIGHT_END = '\u001b[22m'

const WEIGHT_RESET: AnsiCode = {
  type: 'ansi',
  code: WEIGHT_END,
  endCode: WEIGHT_END
}

const SGR_PARAMS_RE = /^\u001b\[([0-9;]*)m$/

/** The bold/dim atoms ('1' / '2') a single SGR sequence turns on. */
function weightAtoms(code: AnsiCode): string[] {
  const match = SGR_PARAMS_RE.exec(code.code)

  if (!match) {
    return []
  }

  const parts = (match[1] || '0').split(';')
  const atoms: string[] = []

  for (let i = 0; i < parts.length; i++) {
    const p = parts[i] || '0'

    if ((p === '38' || p === '48') && i + 1 < parts.length) {
      // Extended color: consume the argument sub-params so their literal
      // 2/5 aren't read as weight atoms.
      i += parts[i + 1] === '5' ? 2 : parts[i + 1] === '2' ? 4 : 0
      continue
    }

    if (p === '1' || p === '2') {
      atoms.push(p)
    }
  }

  return atoms
}

const carriesWeight = (code: AnsiCode): boolean => weightAtoms(code).length > 0

/**
 * Like `diffAnsiCodes`, but correct for the shared-reset weight family:
 * when the bold/dim set changes in a way that removes a flag, emit SGR 22
 * first, then re-apply every weight-carrying sequence the target style has.
 */
export function transitionAnsiCodes(from: AnsiCode[], to: AnsiCode[]): AnsiCode[] {
  const fromAtoms = new Set(from.flatMap(weightAtoms))

  if (fromAtoms.size === 0) {
    // Nothing to un-set; the library's "add what's missing" pass is correct.
    return diffAnsiCodes(from, to)
  }

  const toAtoms = new Set(to.flatMap(weightAtoms))
  let removesWeight = false

  for (const atom of fromAtoms) {
    if (!toAtoms.has(atom)) {
      removesWeight = true
      break
    }
  }

  if (!removesWeight) {
    // from's weights ⊆ to's weights — additions only, library handles it.
    return diffAnsiCodes(from, to)
  }

  // A weight flag must be dropped: SGR 22 is the only way (it clears BOTH),
  // so reset the family and re-apply the target's weight-carrying sequences
  // in full (a compound re-asserts its color too — redundant bytes, never
  // wrong). The rest of the style diffs normally with the weight carriers
  // stripped from both sides.
  const rest = diffAnsiCodes(from.filter(code => !carriesWeight(code)), to.filter(code => !carriesWeight(code)))

  return [WEIGHT_RESET, ...rest, ...to.filter(carriesWeight)]
}
