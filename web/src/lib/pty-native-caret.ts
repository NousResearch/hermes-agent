import type { Terminal } from '@xterm/xterm';

interface NativeCaretState { value: string; editable: boolean }

export function gridCellFromPointer(
  clientX: number,
  clientY: number,
  screen: { left: number; top: number; width: number; height: number },
  cols: number,
  rows: number,
): { col: number; row: number } | null {
  if (cols <= 0 || rows <= 0 || screen.width <= 0 || screen.height <= 0) return null;
  const col = Math.floor((clientX - screen.left) / (screen.width / cols));
  const row = Math.floor((clientY - screen.top) / (screen.height / rows));
  if (col < 0 || row < 0 || col >= cols || row >= rows) return null;
  return { col, row };
}

export function caretOffsetInSuffix(
  cells: Array<{ text: string; row: number; col: number }>,
  row: number,
  col: number,
): number | null {
  let offset = 0;
  for (const cell of cells) {
    if (cell.row === row && cell.col === col) return offset;
    offset += cell.text.length;
  }
  return null;
}

export function caretColumnFromOffset(cursorX: number, valueLength: number, offset: number): number {
  const remaining = Math.max(0, valueLength - Math.min(offset, valueLength));
  return Math.max(0, cursorX - remaining);
}

/** Walk the rendered suffix backward from the PTY cursor. Fail closed when
 * the buffer text does not match the acknowledged native value. */
export function suffixCellsMatchingValue(
  buffer: {
    baseY: number;
    cursorY: number;
    cursorX: number;
    viewportY: number;
    getLine(y: number): { isWrapped: boolean; getCell(x: number): { getWidth(): number; getChars(): string } | undefined } | undefined;
  },
  cols: number,
  value: string,
): Array<{ text: string; row: number; col: number }> | null {
  let row = buffer.baseY + buffer.cursorY;
  let col = Math.min(buffer.cursorX, cols);
  let text = '';
  const cells: Array<{ text: string; row: number; col: number }> = [];
  while (text.length < value.length && row >= buffer.viewportY) {
    if (col === 0) {
      if (!buffer.getLine(row)?.isWrapped) break;
      row--;
      col = cols;
    }
    const cell = buffer.getLine(row)?.getCell(--col);
    if (!cell || cell.getWidth() === 0) continue;
    const chars = cell.getChars() || ' ';
    cells.unshift({ text: chars, row, col });
    text = chars + text;
  }
  return text === value ? cells : null;
}

export function caretDeltaSequence(from: number, to: number): string {
  const start = Math.max(0, Math.floor(from));
  const end = Math.max(0, Math.floor(to));
  if (end < start) return "\x1b[D".repeat(start - end);
  if (end > start) return "\x1b[C".repeat(end - start);
  return "";
}

export function moveNativeCaret(
  value: string,
  start: number,
  end: number,
  key: string,
): { start: number; end: number } | null {
  const from = Math.min(start, end);
  const to = Math.max(start, end);
  const bounds = [0, ...Array.from(
    new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(value),
    (part) => part.index + part.segment.length,
  )];
  if (key === "Home") return { start: 0, end: 0 };
  if (key === "End") return { start: value.length, end: value.length };
  if (key === "ArrowLeft") {
    const next = bounds.filter((index) => index < from).at(-1) ?? 0;
    return { start: next, end: next };
  }
  if (key === "ArrowRight") {
    const next = bounds.find((index) => index > to) ?? value.length;
    return { start: next, end: next };
  }
  return null;
}

/** Project the native caret onto the acknowledged PTY suffix, without moving
 * the PTY cursor (tail rewrites still depend on that cursor staying at the end).
 * Never guess a position before the server has rendered the matching text. */
export function installPtyNativeCaret(term: Terminal, state: () => NativeCaretState) {
  const textarea = term.textarea!;
  const doc = textarea.ownerDocument;
  const screen = term.element!.querySelector<HTMLElement>('.xterm-screen')!;
  const caret = doc.createElement('div');
  caret.className = 'pty-native-caret';
  caret.setAttribute('aria-hidden', 'true');
  caret.style.cssText = 'display:none;position:fixed;pointer-events:none;width:3px;z-index:2147483645;background:#fff;box-shadow:0 0 0 1px #000;';
  doc.body.append(caret);
  let disposed = false;
  let frame = 0;

  const show = (visible: boolean) => {
    caret.style.display = visible ? 'block' : 'none';
  };
  const update = () => {
    frame = 0;
    show(false);
  };
  const schedule = () => {
    if (!disposed && !frame) frame = requestAnimationFrame(update);
  };
  const onPointer = (event: PointerEvent) => {
    if (event.button !== 0 || event.altKey || event.shiftKey || event.ctrlKey || event.metaKey) return;
    const { value, editable } = state();
    if (!editable || !value || textarea.value !== value) return;
    const cells = suffixCellsMatchingValue(term.buffer.active, term.cols, value);
    if (!cells) return;
    const grid = gridCellFromPointer(
      event.clientX,
      event.clientY,
      screen.getBoundingClientRect(),
      term.cols,
      term.rows,
    );
    if (!grid) return;
    const offset = caretOffsetInSuffix(cells, term.buffer.active.viewportY + grid.row, grid.col);
    if (offset === null) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    textarea.focus();
    textarea.setSelectionRange(offset, offset);
    schedule();
  };
  screen.addEventListener('pointerdown', onPointer, true);
  doc.addEventListener('selectionchange', schedule);
  textarea.addEventListener('select', schedule);
  textarea.addEventListener('focus', schedule);
  textarea.addEventListener('blur', schedule);
  const listeners = [term.onRender(schedule), term.onResize(schedule), term.onScroll(schedule)];
  return {
    refresh: schedule,
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      screen.removeEventListener('pointerdown', onPointer, true);
      doc.removeEventListener('selectionchange', schedule);
      textarea.removeEventListener('select', schedule);
      textarea.removeEventListener('focus', schedule);
      textarea.removeEventListener('blur', schedule);
      listeners.forEach(listener => listener.dispose());
      show(false);
      caret.remove();
    },
  };
}
