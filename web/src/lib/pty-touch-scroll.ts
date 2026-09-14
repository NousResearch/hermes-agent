export interface TouchScrollState {
  lastY: number;
  remainderPx: number;
}

export interface TouchScrollStep {
  lines: number;
  state: TouchScrollState;
}

/**
 * Convert a one-finger vertical pan into terminal scrollback rows while
 * retaining sub-row movement for the next touch event.
 */
export function advanceTouchScroll(
  state: TouchScrollState,
  currentY: number,
  lineHeightPx: number,
): TouchScrollStep {
  if (!Number.isFinite(currentY) || !Number.isFinite(lineHeightPx) || lineHeightPx <= 0) {
    return { lines: 0, state };
  }

  const deltaPx = state.lastY - currentY + state.remainderPx;
  const lines = Math.trunc(deltaPx / lineHeightPx);
  return {
    lines,
    state: {
      lastY: currentY,
      remainderPx: deltaPx - lines * lineHeightPx,
    },
  };
}
