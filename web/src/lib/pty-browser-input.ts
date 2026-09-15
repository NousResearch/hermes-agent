import type { Terminal } from '@xterm/xterm';
import { installPtyNativeCaret, moveNativeCaret, caretDeltaSequence } from './pty-native-caret';

const graphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
const modifiers = new Set(['Shift', 'Control', 'Alt', 'Meta', 'AltGraph']);
const isLineBreak = (type: string) => type === 'insertLineBreak' || type === 'insertParagraph';
const isEdit = (type: string) => /^(insert(Text|ReplacementText|CompositionText|FromComposition)|delete(ContentBackward|ContentForward|ByCut|WordBackward|WordForward))$/.test(type);
// Native text cannot smuggle terminal controls from clipboard helper contents.
// Clipboard paste deliberately goes through xterm.paste (including bracketed paste).
// eslint-disable-next-line no-control-regex -- terminal protocol boundary
const plainText = (value: string) => value.replace(/[\x00-\x1f\x7f]/g, '');

/** The PTY cursor stays at the end of our acknowledged suffix, even when iOS
 * moves the textarea caret without a keyboard event (long-press-space). */
function rewriteTail(before: string, after: string): string {
  const old = Array.from(graphemes.segment(before), part => part.segment);
  const next = Array.from(graphemes.segment(after), part => part.segment);
  let common = 0;
  while (common < old.length && common < next.length && old[common] === next[common]) common++;
  // The dashboard's Ink TextInput uses graphemeStops/prevPos for Backspace,
  // NOT prompt_toolkit's base-character erase rule. This is consumer-specific.
  const erase = old.length - common;
  return '\x7f'.repeat(erase) + next.slice(common).join('');
}

interface PendingKey { event: KeyboardEvent; released: boolean }
interface EditTransaction {
  event: InputEvent;
  value: string;
  start: number;
  end: number;
  helper: boolean;
  key?: PendingKey;
}

/** Only synthesize operations fully determined by beforeinput's snapshot. */
function intendedValue(edit: EditTransaction): string | undefined {
  const { event, value } = edit;
  let { start, end } = edit;
  if (isLineBreak(event.inputType)) return value.slice(0, start) + '\n' + value.slice(end);
  if (event.inputType.startsWith('insert') && event.data !== null) {
    return value.slice(0, start) + event.data + value.slice(end);
  }
  if (!event.inputType.startsWith('delete')) return undefined;
  if (start === end) {
    const boundaries = [0, ...Array.from(graphemes.segment(value), part => part.index + part.segment.length)];
    if (event.inputType === 'deleteContentBackward') start = boundaries.filter(index => index < start).at(-1) ?? 0;
    else if (event.inputType === 'deleteContentForward') end = boundaries.find(index => index > end) ?? value.length;
    else return undefined;
  }
  return value.slice(0, start) + value.slice(end);
}

/** Own native edits before xterm's key/composition/input listeners can emit.
 * Only public xterm API and the existing textarea; not a replacement composer.
 * Ancestor capture is essential: open() installs xterm's textarea capture
 * listeners first. Blocking them also prevents its legacy 229 diff timer.
 */
export function installPtyBrowserInput(
  term: Terminal,
  canInput: () => boolean = () => true,
  sendBytes?: (data: string) => boolean,
  syncInkCaret = true,
) {
  const textarea = term.textarea!;
  const host = term.element!;
  const preedit = host.querySelector<HTMLElement>('.composition-view');
  const showPreedit = (data: string) => {
    if (!preedit) return;
    preedit.textContent = data;
    preedit.classList.toggle('active', Boolean(data));
    preedit.style.color = "transparent";
    preedit.style.setProperty("-webkit-text-fill-color", "transparent", "important");
    preedit.style.background = 'transparent';
    preedit.style.left = textarea.style.left;
    preedit.style.top = textarea.style.top;
    preedit.style.fontFamily = term.options.fontFamily!;
    preedit.style.fontSize = `${term.options.fontSize}px`;
  };
  let acknowledged = '';
  let ptyOffset = 0;
  let helperOwned = false;
  let sending = false;
  let transaction: EditTransaction | undefined;
  let overtaken: EditTransaction | undefined;
  let composition: { prefix: string; suffix: string; data: string } | undefined;
  let finalizedComposition: { data: string; value: string } | undefined;
  let compositionTrigger: PendingKey | undefined;
  const keys: PendingKey[] = [];
  const listeners: Array<() => void> = [];
  const nativeCaret = installPtyNativeCaret(term, () => ({
    value: acknowledged,
    editable: canInput() && !helperOwned && !composition && !transaction,
  }));

  let caretSuspended = false;
  const send = (data: string) => {
    if (!data) return;
    if (sendBytes?.(data)) return;
    sending = true;
    try { term.input(data, true); } finally { sending = false; }
  };
  const syncVisibleCaret = () => {
    if (!syncInkCaret || caretSuspended || helperOwned || composition || transaction || !canInput()) return;
    const value = acknowledged || textarea.value;
    if (!value) return;
    const to = Math.max(0, Math.min(value.length, textarea.selectionStart));
    send(caretDeltaSequence(ptyOffset, to));
    ptyOffset = to;
  };
  const mirror = (value: string) => {
    textarea.value = value;
    const off = Math.max(0, Math.min(value.length, ptyOffset));
    textarea.setSelectionRange(off, off);
    nativeCaret.refresh();
  };
  const commit = (value: string) => {
    value = plainText(value);
    const toEnd = acknowledged.length - ptyOffset;
    if (toEnd > 0) send('\x1b[C'.repeat(toEnd));
    send(rewriteTail(acknowledged, value));
    acknowledged = value;
    ptyOffset = value.length;
  };
  const rebaseHelper = () => {
    if (!helperOwned) return;
    acknowledged = '';
    mirror('');
    helperOwned = false;
  };
  const reset = () => {
    acknowledged = '';
    ptyOffset = 0;
    helperOwned = false;
    composition = undefined;
    showPreedit('');
    compositionTrigger = undefined;
    transaction = undefined;
    keys.length = 0;
    mirror('');
  };
  const finishComposition = () => {
    if (!composition) return;
    const value = composition.prefix + composition.data + composition.suffix;
    const finalized = { data: composition.data, value };
    finalizedComposition = finalized;
    // Only the composition's native event chain can supply an input-only tail.
    // A later keyless edit has no identity linking it to this composition.
    setTimeout(() => { if (finalizedComposition === finalized) finalizedComposition = undefined; }, 0);
    const caret = composition.prefix.length + composition.data.length;
    composition = undefined;
    showPreedit('');
    commit(value);
    mirror(acknowledged);
    textarea.setSelectionRange(caret, caret);
  };
  const restoreKey = (edit: EditTransaction) => { if (edit.key) keys.unshift(edit.key); };
  const settleTransaction = () => {
    if (!transaction) return;
    const edit = transaction;
    transaction = undefined;
    const value = edit.event.defaultPrevented ? undefined : intendedValue(edit);
    // Install ownership before publishing: onData consumers can reenter input.
    overtaken = value === undefined ? undefined : edit;
    if (value === undefined) restoreKey(edit);
    else if (isLineBreak(edit.event.inputType)) {
      finishComposition();
      acknowledged = '';
      mirror('');
      send('\r');
    }
    else commit(value);
    mirror(acknowledged);
  };
  const flushKeys = (all = false) => {
    if (transaction) return; // A captured older key still owns the head of the FIFO.
    while (keys.length && (all || keys[0].released)) {
      const { event } = keys.shift()!;
      rebaseHelper();
      const inputType = { Backspace: 'deleteContentBackward', Delete: 'deleteContentForward' }[event.key] ?? 'insertText';
      const value = intendedValue({ event: new InputEvent('beforeinput', { inputType, data: event.key }), value: acknowledged, start: textarea.selectionStart, end: textarea.selectionEnd, helper: false });
      if (value !== undefined) commit(value);
      mirror(acknowledged);
    }
  };
  const boundary = () => { settleTransaction(); finishComposition(); flushKeys(true); reset(); };
  // onData runs after xterm has already emitted. Public producers must enter
  // here before paste clears the native composition/edit buffer.
  const pasteText = (data: string) => {
    if (!canInput()) { reset(); return; }
    boundary();
    term.paste(data);
  };
  const markHelper = () => { settleTransaction(); finishComposition(); flushKeys(true); helperOwned = true; acknowledged = ''; };
  const nativeSelection = () => !helperOwned && textarea.selectionStart !== textarea.selectionEnd;
  const contextMenu = (event: Event) => {
    if (event.target === textarea && nativeSelection()) event.stopImmediatePropagation();
    else markHelper();
  };
  host.addEventListener('contextmenu', contextMenu, true);
  listeners.push(() => host.removeEventListener('contextmenu', contextMenu, true));
  const selectionListener = term.onSelectionChange(() => { if (term.hasSelection()) markHelper(); });
  const listen = (type: string, listener: (event: Event) => void) => {
    const handler = (event: Event) => {
      if (event.target !== textarea) return;
      if (!canInput()) {
        overtaken = undefined;
        finalizedComposition = undefined;
        reset();
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      listener(event);
      nativeCaret.refresh();
    };
    host.addEventListener(type, handler, true);
    listeners.push(() => host.removeEventListener(type, handler, true));
  };
  const doc = textarea.ownerDocument;
  const onSelectionChange = () => {
    if (doc.activeElement !== textarea) return;
    syncVisibleCaret();
  };
  doc.addEventListener("selectionchange", onSelectionChange);
  listeners.push(() => doc.removeEventListener("selectionchange", onSelectionChange));
  const nativeKey = (event: KeyboardEvent) => !event.metaKey && ((!event.ctrlKey && !event.altKey) || event.getModifierState('AltGraph')) &&
    (Array.from(event.key).length === 1 || (!event.altKey && ['Backspace', 'Delete'].includes(event.key)));

  listen('keydown', event => {
    const key = event as KeyboardEvent;
    if (nativeSelection() && key.key.toLowerCase() === 'c' && (key.ctrlKey || key.metaKey) && !key.altKey) {
      event.stopImmediatePropagation(); // Preserve selection and browser copy default.
      return;
    }
    if (!key.altKey && !key.ctrlKey && !key.metaKey && !helperOwned) {
      const nav = key.key === "ArrowLeft" || key.key === "ArrowRight" || key.key === "Home" || key.key === "End";
      if (nav) {
        event.stopImmediatePropagation();
        event.preventDefault();
        const value = acknowledged || textarea.value;
        const next = moveNativeCaret(value, textarea.selectionStart, textarea.selectionEnd, key.key);
        if (!next) return;
        if (!acknowledged && textarea.value) {
          acknowledged = plainText(textarea.value);
          ptyOffset = textarea.selectionStart;
        }
        const delta = next.start - ptyOffset;
        if (delta < 0) send("\x1b[D".repeat(-delta));
        if (delta > 0) send("\x1b[C".repeat(delta));
        ptyOffset = next.start;
        textarea.setSelectionRange(next.start, next.end);
        nativeCaret.refresh();
        return;
      }
    }
    compositionTrigger = undefined;
    if (key.keyCode === 229 || key.key === 'Process' || key.key === 'Dead' || key.isComposing) {
      event.stopImmediatePropagation();
      return;
    }
    const outsideMirror = !helperOwned && textarea.selectionStart === textarea.selectionEnd &&
      ((key.key === 'Backspace' && textarea.selectionStart === 0) ||
       (key.key === 'Delete' && textarea.selectionEnd === acknowledged.length));
    if (nativeKey(key) && !outsideMirror) {
      const pending = { event: key, released: false };
      keys.push(pending);
      compositionTrigger = pending;
      event.stopImmediatePropagation();
    } else if (!modifiers.has(key.key) && key.key !== "ArrowUp" && key.key !== "ArrowDown") boundary();
  });
  listen('keypress', event => { if (nativeKey(event as KeyboardEvent) || composition) event.stopImmediatePropagation(); });
  listen('keyup', event => {
    const key = event as KeyboardEvent;
    const matches = (pending: PendingKey) => key.code ? key.code === pending.event.code : key.key === pending.event.key;
    if (transaction?.key && matches(transaction.key)) settleTransaction();
    if (overtaken?.key && matches(overtaken.key)) overtaken = undefined;
    for (const pending of keys) if (matches(pending)) pending.released = true;
    flushKeys();
    if (!modifiers.has(key.key)) { compositionTrigger = undefined; finalizedComposition = undefined; }
    if (nativeKey(key)) event.stopImmediatePropagation();
  });
  listen('compositionstart', event => {
    event.stopImmediatePropagation();
    finalizedComposition = undefined;
    if (compositionTrigger) {
      const index = keys.indexOf(compositionTrigger);
      if (index !== -1) keys.splice(index, 1);
      if (transaction?.key === compositionTrigger) transaction.key = undefined;
    }
    compositionTrigger = undefined;
    settleTransaction();
    flushKeys(true);
    rebaseHelper();
    composition = { prefix: textarea.value.slice(0, textarea.selectionStart), suffix: textarea.value.slice(textarea.selectionEnd), data: '' };
  });
  listen('compositionupdate', event => {
    event.stopImmediatePropagation();
    if (composition) {
      composition.data = (event as CompositionEvent).data;
      showPreedit(composition.data);
    }
  });
  listen('compositionend', event => {
    event.stopImmediatePropagation();
    if (composition) { composition.data = (event as CompositionEvent).data; finishComposition(); }
  });
  listen('beforeinput', event => {
    event.stopImmediatePropagation();
    const input = event as InputEvent;
    // Preedit insertion/removal belongs to composition, never the ordinary
    // beforeinput slot. Removal before reinsertion is not a terminal delete.
    if (composition && input.inputType.includes('Composition')) return;
    settleTransaction();
    overtaken = undefined; // A new semantic edit, never suppress it by equal text.
    if (!input.inputType.includes('Composition')) finalizedComposition = undefined;
    if (isLineBreak(input.inputType)) finishComposition();
    const first = keys[0]?.event;
    const keyedText = input.inputType === 'insertText' && input.data !== null && first !== undefined &&
      Array.from(graphemes.segment(input.data)).length === 1 &&
      (input.data === first.key || input.data.normalize('NFD').includes(first.key.normalize('NFD')));
    // Replacement text owns a pending key only when the operation carries its
    // delimiter. A word-only replacement is keyless even if another printable
    // key happens to be pending.
    const keyedReplacement = input.inputType === 'insertReplacementText' && input.data !== null && first !== undefined &&
      /^[\s\p{P}]$/u.test(first.key) && input.data.endsWith(first.key);
    const keyed = keyedText || keyedReplacement ||
      (input.inputType === 'deleteContentBackward' && first?.key === 'Backspace') ||
      (input.inputType === 'deleteContentForward' && first?.key === 'Delete');
    // A replacement rewrites already-acknowledged text. Unrelated pending keys
    // remain outside that transaction and settle against its committed tail.
    if (!keyed && !composition && input.inputType !== 'insertReplacementText') flushKeys(true);
    const helper = helperOwned;
    rebaseHelper();
    transaction = { event: input, value: textarea.value, start: textarea.selectionStart, end: textarea.selectionEnd, helper, key: keyed ? keys.shift() : undefined };
  });
  listen('input', event => {
    event.stopImmediatePropagation();
    const input = event as InputEvent;
    const settled = overtaken;
    overtaken = undefined; // One observation, or a newer transaction, expires the claim.
    if (settled && !transaction && input.inputType === settled.event.inputType && input.data === settled.event.data &&
        (textarea.value === intendedValue(settled) || textarea.value === (isLineBreak(input.inputType) ? '\n' : input.data))) {
      mirror(acknowledged);
      return;
    }
    const edit = transaction;
    transaction = undefined;
    const finalized = finalizedComposition;
    const compositionObservation = /^(insertText|insertCompositionText|insertFromComposition)$/.test(input.inputType);
    // A line-break transaction may finalize composition before its own input.
    // Keep that exact one-shot claim for the browser's trailing composition
    // observation; every other observation consumes or expires it.
    if (!isLineBreak(input.inputType)) finalizedComposition = undefined;
    if (finalized && compositionObservation &&
        input.data === finalized.data && (textarea.value === finalized.value || textarea.value === finalized.data)) {
      if (edit) restoreKey(edit);
      mirror(acknowledged);
      return;
    }
    if (isLineBreak(input.inputType)) { finishComposition(); flushKeys(true); send('\r'); reset(); return; }
    if (composition) return;
    if (!isEdit(input.inputType)) { if (edit) restoreKey(edit); mirror(acknowledged); return; }
    if (!edit) {
      // No pre-edit range: the DOM is not evidence that acknowledged text was
      // deleted. Accept the insertion payload only, never a destructive diff.
      // Input-owned replacement text is unsafe even when the resulting DOM
      // matches data: that does not reveal the replaced range.
      const data = input.data;
      const insertion = input.inputType.startsWith('insert') && data !== null;
      const safePayload = insertion && (helperOwned ||
        (input.inputType !== 'insertReplacementText' && (textarea.value === data || textarea.value === acknowledged + data)));
      if (safePayload) {
        const first = keys[0]?.event;
        const keyedText = input.inputType === 'insertText' && first !== undefined &&
          Array.from(graphemes.segment(data)).length === 1 &&
          (data === first.key || data.normalize('NFD').includes(first.key.normalize('NFD')));
        if (keyedText) keys.shift();
        acknowledged = '';
        helperOwned = false;
        commit(data);
        mirror(acknowledged);
      } else { acknowledged = ''; mirror(''); }
      flushKeys();
      return;
    }
    if (edit.helper || helperOwned) {
      rebaseHelper();
      mirror(plainText(input.data ?? ''));
    }
    commit(textarea.value);
    flushKeys();
  });
  listen('blur', () => {
    overtaken = undefined;
    finalizedComposition = undefined;
    composition = undefined;
    transaction = undefined;
    keys.length = 0;
    showPreedit('');
  });
  listen('copy', event => {
    if (nativeSelection()) event.stopImmediatePropagation(); // Keep the browser's native copy default.
  });
  listen('paste', event => {
    const paste = event as ClipboardEvent;
    if (!paste.clipboardData) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    pasteText(paste.clipboardData.getData('text/plain'));
  });
  const dataListener = term.onData((data: string) => {
    if (sending) return;
    if (data === "\r" || data === "\n" || data === "\x03") boundary();
  });
  return {
    paste: pasteText,
    nudge(key: "ArrowLeft" | "ArrowRight" | "Home" | "End") {
      const value = acknowledged || textarea.value;
      const from = textarea.selectionStart;
      const next = moveNativeCaret(value, from, textarea.selectionEnd, key);
      if (!next) return;
      if (!acknowledged && textarea.value) {
        acknowledged = plainText(textarea.value);
        ptyOffset = from;
      }
      if (syncInkCaret) send(caretDeltaSequence(ptyOffset, next.start));
      ptyOffset = next.start;
      textarea.setSelectionRange(next.start, next.end);
      nativeCaret.refresh();
    },
    setCaretSuspended(value: boolean) {
      caretSuspended = value;
    },
    reset() { overtaken = undefined; finalizedComposition = undefined; reset(); },
    dispose() { nativeCaret.dispose(); listeners.forEach(dispose => dispose()); dataListener.dispose(); selectionListener.dispose(); reset(); },
  };
}
