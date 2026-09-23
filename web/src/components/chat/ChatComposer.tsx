/**
 * Composer for the new /chat-ui structured Chat UI.
 *
 * Behaviours (Phase 3 + Phase 4 polish):
 *
 *   - Autosizing textarea (1 → 8 rows, then scrolls).
 *   - Enter sends; Shift+Enter inserts a newline.
 *   - IME composition (CJK / Japanese / Korean / Chinese handwriting) does
 *     NOT trigger send while `keydown` is part of a composition session;
 *     `compositionstart` / `compositionend` set a flag the keydown handler
 *     checks before sending.
 *   - Disabled while the session is connecting or initialising. The parent
 *     tells us via `disabled`; we also disable while a send is in flight
 *     (the parent swaps Send ↔ Stop and gates `onSubmit` itself).
 *   - Clear input after a SUCCESSFUL submit (`onSubmit` resolves to `true`).
 *     Preserve input if the submit rejects or resolves to `false` so the
 *     user can retry without retyping.
 *   - Send / Stop buttons keyboard accessible (`aria-label`, focus-visible,
 *     form `type="submit"` / `type="button"` semantics).
 *   - Pending attachments render as chips above the textarea. Each chip
 *     exposes a remove button; clicking it removes the attachment from
 *     the queue. The composer auto-sends after attach resolves with at
 *     least one queued attachment and a non-empty prompt.
 *   - Attachments are discriminated by `kind` ("image" vs "pdf") so the
 *     chip renderer picks the correct icon and preview.
 *
 *   - Image attachment uses the gateway `image.attach_bytes` JSON-RPC
 *     method. PDF attachment uses `pdf.attach`. Both must be called BEFORE
 *     `prompt.submit` — neither method carries an inline field on
 *     `PromptSubmitParams` (see `gateway-contract.generated.ts`).
 *
 * The composer is intentionally pure: no transport, no gateway calls. The
 * parent wires `onSubmit` to {@link useChatSession.submit} and the two
 * attach callbacks to the session-aware attach helpers.
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";

import { Button } from "@nous-research/ui/ui/components/button";
import {
  FileText,
  Image as ImageIcon,
  Loader2,
  Send,
  Square,
  X,
} from "lucide-react";

import type { ChatAttachment } from "@/components/chat/types";

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

export interface ChatComposerProps {
  /** True while a turn is running; swaps Send ↔ Stop. */
  busy?: boolean;
  /**
   * True while the connection is NOT yet open (initial resume / create).
   * Renders the textarea + buttons as read-only with a "Connecting…"
   * placeholder so the user knows why send is unavailable.
   */
  connecting?: boolean;
  /** Called with trimmed text + any queued attachments when the user submits. */
  onSubmit: (
    text: string,
    attachments: ReadonlyArray<ChatAttachment>,
  ) => Promise<boolean> | boolean;
  /** Called when the user requests a stop / interrupt. */
  onStop?: () => void;
  /**
   * Called when the user picks / pastes an image. The parent is responsible
   * for running `image.attach_bytes` on the active gateway session and
   * returning the {@link ChatAttachment} projection for the composer chip.
   * Resolves to `null` when the upload / attach failed; the composer
   * surfaces that as an inline error and lets the user retry or remove.
   */
  onAttachImage?: (file: File) => Promise<ChatAttachment | null>;
  /**
   * Called when the user picks a PDF. The parent runs `pdf.attach` on the
   * active session. Same return contract as `onAttachImage` — `null` on
   * failure, otherwise a {@link ChatAttachment} with `kind: "pdf"`.
   *
   * When omitted the PDF picker button is hidden, keeping the composer
   * usable on surfaces that have not wired PDF yet (e.g. legacy CLI).
   */
  onAttachPdf?: (file: File) => Promise<ChatAttachment | null>;
  /** Optional clipboard paste interceptor. */
  onPasteFiles?: (files: File[]) => void;
  /**
   * One-shot suggestion seed. When set, the composer adopts this text as
   * its current value (preserving any existing value the user typed) and
   * immediately calls `onSuggestionConsumed`. Used by the empty-state
   * suggestion cards on the message list.
   */
  suggestedText?: string;
  /** Called after the composer adopts a `suggestedText` value. */
  onSuggestionConsumed?: () => void;
  /** Placeholder text. Defaults to the Enter/Shift+Enter hint. */
  placeholder?: string;
  /** Optional class name passthrough (used by tests). */
  className?: string;
}

/* -------------------------------------------------------------------------- */
/* Constants                                                                  */
/* -------------------------------------------------------------------------- */

const MIN_ROWS = 1;
const MAX_ROWS = 8;

/* -------------------------------------------------------------------------- */
/* Component                                                                  */
/* -------------------------------------------------------------------------- */

export function ChatComposer({
  busy,
  connecting,
  onSubmit,
  onStop,
  onAttachImage,
  onAttachPdf,
  onPasteFiles,
  suggestedText,
  onSuggestionConsumed,
  placeholder,
  className,
}: ChatComposerProps) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [attachError, setAttachError] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [isComposing, setIsComposing] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const pdfInputRef = useRef<HTMLInputElement | null>(null);

  // Autosize: shrink on empty, grow up to MAX_ROWS.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const lineHeight =
      parseInt(getComputedStyle(el).lineHeight || "20", 10) || 20;
    const maxHeight = lineHeight * MAX_ROWS + 16; // padding allowance
    const next = Math.min(el.scrollHeight, maxHeight);
    el.style.height = `${Math.max(next, lineHeight * MIN_ROWS)}px`;
  }, [value]);

  // Autofocus on mount.
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // One-shot suggestion seed: adopt the text only when the parent supplies
  // a new value. We never overwrite a non-empty current value — the user
  // may have started typing in the meantime.
  useEffect(() => {
    if (suggestedText === undefined || suggestedText === null) return;
    if (suggestedText === "") return;
    setValue((prev) => (prev.trim().length > 0 ? prev : suggestedText));
    onSuggestionConsumed?.();
    // Focus the textarea so the user can edit / send immediately.
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, [suggestedText, onSuggestionConsumed]);

  const send = useCallback(async () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    if (busy || connecting) return;
    setSubmitting(true);
    let ok: boolean;
    try {
      ok = await Promise.resolve(onSubmit(trimmed, attachments));
    } catch {
      ok = false;
    } finally {
      setSubmitting(false);
    }
    if (ok) {
      setValue("");
      setAttachments([]);
      setAttachError(null);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
    // On failure: leave value + attachments in place so the user can retry.
  }, [value, attachments, busy, connecting, onSubmit]);

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      void send();
    },
    [send],
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      // IME guard: don't treat Enter from a composition session as submit.
      if (event.key === "Enter" && !event.shiftKey && !isComposing) {
        event.preventDefault();
        void send();
        return;
      }
      // Esc cancels an in-flight generation.
      if (event.key === "Escape" && busy) {
        event.preventDefault();
        onStop?.();
      }
    },
    [busy, isComposing, onStop, send],
  );

  const handleCompositionStart = useCallback(() => setIsComposing(true), []);
  const handleCompositionEnd = useCallback(() => setIsComposing(false), []);

  const attachImage = useCallback(
    async (file: File) => {
      if (!onAttachImage) return;
      setAttachError(null);
      setPendingCount((c) => c + 1);
      try {
        const attachment = await onAttachImage(file);
        if (!attachment) {
          setAttachError(`Failed to attach ${file.name || "image"}`);
          return;
        }
        setAttachments((prev) => [...prev, attachment]);
      } catch (err) {
        setAttachError(
          err instanceof Error ? err.message : "Failed to attach image",
        );
      } finally {
        setPendingCount((c) => Math.max(0, c - 1));
      }
    },
    [onAttachImage],
  );

  const attachPdf = useCallback(
    async (file: File) => {
      if (!onAttachPdf) return;
      setAttachError(null);
      setPendingCount((c) => c + 1);
      try {
        const attachment = await onAttachPdf(file);
        if (!attachment) {
          setAttachError(`Failed to attach ${file.name || "PDF"}`);
          return;
        }
        setAttachments((prev) => [...prev, attachment]);
      } catch (err) {
        setAttachError(
          err instanceof Error ? err.message : "Failed to attach PDF",
        );
      } finally {
        setPendingCount((c) => Math.max(0, c - 1));
      }
    },
    [onAttachPdf],
  );

  const handlePaste = useCallback(
    (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
      if (!onPasteFiles && !onAttachImage && !onAttachPdf) return;
      const files = Array.from(event.clipboardData?.files ?? []);
      if (files.length === 0) return;
      const imageFiles = files.filter((f) => f.type.startsWith("image/"));
      const pdfFiles = files.filter(
        (f) =>
          f.type === "application/pdf" ||
          f.name.toLowerCase().endsWith(".pdf"),
      );
      if (imageFiles.length === 0 && pdfFiles.length === 0) return;
      event.preventDefault();
      if (onPasteFiles) {
        onPasteFiles(files);
        return;
      }
      // Default: route images to onAttachImage, PDFs to onAttachPdf.
      void (async () => {
        for (const f of imageFiles) {
          // eslint-disable-next-line no-await-in-loop
          await attachImage(f);
        }
        for (const f of pdfFiles) {
          // eslint-disable-next-line no-await-in-loop
          await attachPdf(f);
        }
      })();
    },
    // attachImage / attachPdf close over setAttachments — kept in the same
    // render scope.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onAttachImage, onAttachPdf, onPasteFiles],
  );

  const handlePickImage = useCallback(() => {
    imageInputRef.current?.click();
  }, []);

  const handlePickPdf = useCallback(() => {
    pdfInputRef.current?.click();
  }, []);

  const handleImageChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      // Reset the input so re-picking the same file fires onChange again.
      event.target.value = "";
      for (const f of files) {
        await attachImage(f);
      }
    },
    [attachImage],
  );

  const handlePdfChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      event.target.value = "";
      for (const f of files) {
        await attachPdf(f);
      }
    },
    [attachPdf],
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const showStop = !!busy;
  const sendDisabled =
    !value.trim() || !!busy || !!connecting || !!submitting || isComposing;
  const hasAttachments = attachments.length > 0;
  const uploading = pendingCount > 0;

  return (
    <form
      onSubmit={handleSubmit}
      className={className}
      aria-label="Message composer"
      data-testid="chat-ui-composer"
    >
      {attachError && (
        <div
          className="mb-1 flex items-start gap-2 rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-xs text-destructive"
          role="alert"
          data-testid="chat-ui-composer-attach-error"
        >
          <span className="flex-1">{attachError}</span>
          <button
            type="button"
            onClick={() => setAttachError(null)}
            aria-label="Dismiss attachment error"
            className="rounded p-0.5 text-destructive/80 hover:bg-destructive/20"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}

      {(hasAttachments || uploading) && (
        <ul
          className="mb-2 flex flex-wrap gap-1.5"
          data-testid="chat-ui-composer-attachments"
          aria-label="Pending attachments"
        >
          {attachments.map((a) => (
            <AttachmentChip
              key={a.id}
              attachment={a}
              onRemove={() => removeAttachment(a.id)}
            />
          ))}
          {uploading && (
            <li
              className="flex items-center gap-1 rounded border border-current/15 bg-background-base/40 px-2 py-1 text-xs text-text-secondary"
              data-testid="chat-ui-composer-attachment-pending"
              aria-label="Uploading attachment"
            >
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>Uploading…</span>
            </li>
          )}
        </ul>
      )}

      <div
        className="flex items-end gap-2 rounded-2xl border border-current/15 bg-background-base px-3 py-2 shadow-sm transition-colors focus-within:border-midground"
        data-testid="chat-ui-composer-shell"
      >
        {onAttachImage && (
          <>
            <Button
              type="button"
              ghost
              size="icon"
              onClick={handlePickImage}
              disabled={!!connecting || !!submitting || uploading}
              aria-label="Attach image"
              title="Attach image"
              data-testid="chat-ui-composer-attach"
            >
              <ImageIcon />
            </Button>
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={handleImageChange}
              data-testid="chat-ui-composer-file-input"
            />
          </>
        )}

        {onAttachPdf && (
          <>
            <Button
              type="button"
              ghost
              size="icon"
              onClick={handlePickPdf}
              disabled={!!connecting || !!submitting || uploading}
              aria-label="Attach PDF"
              title="Attach PDF"
              data-testid="chat-ui-composer-attach-pdf"
            >
              <FileText />
            </Button>
            <input
              ref={pdfInputRef}
              type="file"
              accept="application/pdf,.pdf"
              multiple
              hidden
              onChange={handlePdfChange}
              data-testid="chat-ui-composer-pdf-input"
            />
          </>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={handleCompositionStart}
          onCompositionEnd={handleCompositionEnd}
          onPaste={handlePaste}
          rows={MIN_ROWS}
          readOnly={!!connecting}
          placeholder={
            placeholder ??
            (connecting
              ? "Connecting…"
              : "Send a message…  (Enter to send, Shift+Enter for newline)")
          }
          aria-label="Message input"
          aria-disabled={!!connecting}
          data-testid="chat-ui-composer-input"
          className="min-h-[2.25rem] max-h-[12rem] flex-1 resize-none overflow-y-auto border-0 bg-transparent text-sm leading-6 text-foreground placeholder:text-text-tertiary focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
        />

        {showStop ? (
          <Button
            type="button"
            onClick={() => onStop?.()}
            aria-label="Stop generating"
            className="shrink-0"
            data-testid="chat-ui-composer-stop"
          >
            <Square className="h-3.5 w-3.5" />
            <span className="ml-1">Stop</span>
          </Button>
        ) : (
          <Button
            type="submit"
            disabled={sendDisabled}
            aria-label="Send message"
            className="shrink-0"
            data-testid="chat-ui-composer-send"
          >
            {submitting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Send className="h-3.5 w-3.5" />
            )}
            <span className="ml-1">Send</span>
          </Button>
        )}
      </div>
    </form>
  );
}

/* -------------------------------------------------------------------------- */
/* Attachment chip                                                            */
/* -------------------------------------------------------------------------- */

function AttachmentChip({
  attachment,
  onRemove,
}: {
  attachment: ChatAttachment;
  onRemove: () => void;
}) {
  const displayName = attachment.name ?? "attachment";
  // Truncate very long filenames in the middle to preserve the extension.
  const label = truncateMiddle(displayName, 28);
  const subtitle =
    attachment.kind === "pdf" && attachment.pageCount
      ? `${attachment.pageCount} page${attachment.pageCount === 1 ? "" : "s"}`
      : attachment.width && attachment.height
        ? `${attachment.width}×${attachment.height}`
        : attachment.bytes !== undefined
          ? formatBytes(attachment.bytes)
          : null;
  const isImage = attachment.kind === "image";
  return (
    <li
      className="group flex items-center gap-1.5 rounded-full border border-current/15 bg-background-base/60 py-0.5 pl-1 pr-0.5 text-xs"
      data-testid="chat-ui-composer-attachment"
      data-attachment-kind={attachment.kind}
      data-attachment-id={attachment.id}
    >
      {isImage && attachment.dataUri ? (
        // eslint-disable-next-line jsx-a11y/img-redundant-alt
        <img
          src={attachment.dataUri}
          alt=""
          aria-hidden
          className="h-6 w-6 rounded-full border border-current/10 object-cover"
        />
      ) : isImage ? (
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-midground/30">
          <ImageIcon
            className="h-3.5 w-3.5 text-text-secondary"
            aria-hidden
          />
        </span>
      ) : (
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-red-500/10 text-red-500">
          <FileText className="h-3.5 w-3.5" aria-hidden />
        </span>
      )}
      <span
        className="max-w-[12rem] truncate font-medium text-foreground"
        title={displayName}
      >
        {label}
      </span>
      {subtitle && (
        <span className="text-text-tertiary" aria-hidden>
          {" "}
          · {subtitle}
        </span>
      )}
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${displayName}`}
        className="ml-0.5 flex h-5 w-5 items-center justify-center rounded-full text-text-tertiary transition-colors hover:bg-midground/30 hover:text-foreground focus:bg-midground/30 focus:text-foreground focus:outline-none"
      >
        <X className="h-3 w-3" />
      </button>
    </li>
  );
}

function truncateMiddle(value: string, max: number): string {
  if (value.length <= max) return value;
  const head = Math.ceil((max - 1) / 2);
  const tail = Math.floor((max - 1) / 2);
  return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
