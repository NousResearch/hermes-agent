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
 */
const WEIGHT_END = '\u001b[22m'

const WEIGHT_RESET: AnsiCode = {
  type: 'ansi',
  code: WEIGHT_END,
  endCode: WEIGHT_END
}

/**
 * Like `diffAnsiCodes`, but correct for the shared-reset weight family:
 * when the bold/dim set changes in a way that removes a flag, emit SGR 22
 * first, then re-apply every weight flag the target style carries.
 */
export function transitionAnsiCodes(from: AnsiCode[], to: AnsiCode[]): AnsiCode[] {
  const fromWeights = from.filter(code => code.endCode === WEIGHT_END)
  const toWeights = to.filter(code => code.endCode === WEIGHT_END)

  if (fromWeights.length === 0) {
    // Nothing to un-set; the library's "add what's missing" pass is correct.
    return diffAnsiCodes(from, to)
  }

  const toWeightCodes = new Set(toWeights.map(code => code.code))
  const removesWeight = fromWeights.some(code => !toWeightCodes.has(code.code))

  if (!removesWeight) {
    // from's weights ⊆ to's weights — additions only, library handles it.
    return diffAnsiCodes(from, to)
  }

  // A weight flag must be dropped: SGR 22 is the only way (it clears BOTH),
  // so reset the family and re-apply the target's full weight set. The rest
  // of the style (colors, italic, underline, …) diffs normally with the
  // weight family stripped from both sides.
  const rest = diffAnsiCodes(
    from.filter(code => code.endCode !== WEIGHT_END),
    to.filter(code => code.endCode !== WEIGHT_END)
  )

  return [WEIGHT_RESET, ...rest, ...toWeights]
}
