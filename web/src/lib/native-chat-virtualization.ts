export type VirtualRange = {
  start: number;
  end: number;
  offsetTop: number;
  bottomSpacer: number;
  totalHeight: number;
};

function safeHeight(value: number | undefined, estimated: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : estimated;
}

function firstRowAtOffset(prefix: readonly number[], count: number, offset: number): number {
  if (count === 0 || offset <= 0) return 0;
  if (offset >= prefix[count]) return count - 1;
  let low = 0;
  let high = count;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (prefix[middle + 1] <= offset) low = middle + 1;
    else high = middle;
  }
  return Math.min(low, count - 1);
}

/**
 * Return the visible transcript window plus spacer sizes. `end` is exclusive.
 * A measured row height is preferred; rows that have not rendered yet use the
 * estimate so the browser can paint a stable first window before measurement.
 */
export function getVirtualRange(
  count: number,
  scrollTop: number,
  viewportHeight: number,
  measuredHeights: readonly number[] = [],
  estimatedRowHeight = 120,
  overscan = 4,
): VirtualRange {
  const safeCount = Math.max(0, Math.floor(Number.isFinite(count) ? count : 0));
  if (safeCount === 0) return { start: 0, end: 0, offsetTop: 0, bottomSpacer: 0, totalHeight: 0 };

  const estimated = Math.max(1, safeHeight(estimatedRowHeight, 120));
  const safeOverscan = Math.max(0, Math.floor(Number.isFinite(overscan) ? overscan : 0));
  const prefix = new Array<number>(safeCount + 1).fill(0);
  for (let index = 0; index < safeCount; index += 1) {
    prefix[index + 1] = prefix[index] + safeHeight(measuredHeights[index], estimated);
  }

  const totalHeight = prefix[safeCount];
  const viewport = Math.max(0, Number.isFinite(viewportHeight) ? viewportHeight : 0);
  const maxScrollTop = Math.max(0, totalHeight - viewport);
  const top = Math.min(Math.max(0, Number.isFinite(scrollTop) ? scrollTop : 0), maxScrollTop);
  const bottom = Math.min(totalHeight, top + viewport);
  const first = firstRowAtOffset(prefix, safeCount, top);
  const last = firstRowAtOffset(prefix, safeCount, bottom);
  const start = Math.max(0, first - safeOverscan);
  const end = Math.min(safeCount, last + 1 + safeOverscan);

  return {
    start,
    end: Math.max(start, end),
    offsetTop: prefix[start],
    bottomSpacer: Math.max(0, totalHeight - prefix[Math.max(start, end)]),
    totalHeight,
  };
}
