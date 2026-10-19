/**
 * Geometry for the `/history` pager overlay.
 *
 * The pager renders as a full-height panel of its own (see `PagerOverlay`)
 * instead of a content-sized floating box: paging advances by ROWS, so the
 * slice size must be exactly what the panel can show or a page turn silently
 * skips the rows that did not fit.
 */

/** Panel chrome, in rows: double border (2) + padding (2) + title and its gap (2) + hint and its gap (2). */
export const PAGER_CHROME_ROWS = 8

/**
 * Rows of text one page can show. The pager is a modal surface (the composer is
 * not drawn under it), so the whole terminal minus the panel chrome is usable —
 * one spare row keeps the panel off the last cell, where a trailing newline
 * would scroll it.
 */
export const pagerPageSize = (rows: number): number => Math.max(1, rows - PAGER_CHROME_ROWS - 1)

/** Panel width in cells: the terminal width minus one column of margin on each side. */
export const pagerPanelWidth = (cols: number): number => Math.max(24, cols - 2)

/** Cells available to one pager line: the panel minus its border and horizontal padding. */
export const pagerLineWidth = (cols: number): number => Math.max(12, pagerPanelWidth(cols) - 6)
