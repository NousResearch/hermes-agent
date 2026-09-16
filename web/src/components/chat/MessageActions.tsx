import { Check, Copy as CopyIcon, MessageSquarePlus, Pencil, RotateCcw, Volume2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";

export type MessageRole = "user" | "assistant";

export interface MessageActionsProps {
  message: string;
  messageRole: MessageRole;
  onUseAsPrompt: (message: string) => void;
  onSpeak?: (message: string) => Promise<void> | void;
  onEdit?: (message: string) => void;
  editLabel?: string;
  onRegenerate?: () => void;
}

type FeedbackState = "copied" | "copy-failed" | "prompt-filled" | "spoken" | "speak-failed";

const FEEDBACK_DURATION_MS = 1800;
const iconClassName = "h-4 w-4";

export function MessageActions({ message, messageRole, onUseAsPrompt, onSpeak, onEdit, editLabel, onRegenerate }: MessageActionsProps) {
  const [feedback, setFeedback] = useState<FeedbackState | null>(null);
  const feedbackTimerRef = useRef<number | null>(null);

  const clearFeedbackTimer = useCallback(() => {
    if (feedbackTimerRef.current === null) return;
    window.clearTimeout(feedbackTimerRef.current);
    feedbackTimerRef.current = null;
  }, []);

  useEffect(() => clearFeedbackTimer, [clearFeedbackTimer]);

  const showFeedback = useCallback((next: FeedbackState) => {
    clearFeedbackTimer();
    setFeedback(next);
    feedbackTimerRef.current = window.setTimeout(() => {
      feedbackTimerRef.current = null;
      setFeedback(null);
    }, FEEDBACK_DURATION_MS);
  }, [clearFeedbackTimer]);

  const copyMessage = useCallback(async () => {
    try {
      showFeedback(await copyTextToClipboard(message) ? "copied" : "copy-failed");
    } catch {
      showFeedback("copy-failed");
    }
  }, [message, showFeedback]);

  const useAsPrompt = useCallback(() => {
    onUseAsPrompt(message);
    showFeedback("prompt-filled");
  }, [message, onUseAsPrompt, showFeedback]);

  const speakMessage = useCallback(async () => {
    if (!onSpeak) return;
    try {
      await onSpeak(message);
      showFeedback("spoken");
    } catch {
      showFeedback("speak-failed");
    }
  }, [message, onSpeak, showFeedback]);

  const feedbackText = feedback === "copied"
    ? "Copied"
    : feedback === "copy-failed"
      ? "Copy failed"
      : feedback === "prompt-filled"
        ? "Draft filled"
        : feedback === "spoken"
          ? "Spoken"
          : feedback === "speak-failed"
            ? "Speak failed"
            : "";
  const actionTextClass = messageRole === "user" ? "text-primary-foreground" : "text-foreground text-midground";
  const actionButtonClass = "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded border border-current/50 bg-current/10 p-0 text-inherit hover:bg-current/20 hover:text-inherit focus-visible:outline-2 focus-visible:outline-ring focus-visible:ring-2 focus-visible:ring-current/70 focus-visible:ring-offset-1";

  return (
    <div data-slot="message-actions" data-actions-role={messageRole} className={`mt-2 flex items-center gap-1 text-xs ${actionTextClass}`}>
      <button
        type="button"
        aria-label={`Copy ${messageRole} message`}
        title={feedback === "copied" ? "Copied" : "Copy message"}
        className={actionButtonClass}
        onClick={() => void copyMessage()}
      >
        {feedback === "copied" ? <Check aria-hidden="true" className={iconClassName} /> : <CopyIcon aria-hidden="true" className={iconClassName} />}
      </button>
      <button
        type="button"
        aria-label={`Use ${messageRole} message as prompt`}
        title="Use as prompt"
        className={actionButtonClass}
        onClick={useAsPrompt}
      >
        <MessageSquarePlus aria-hidden="true" className={iconClassName} />
      </button>
      {messageRole === "user" && onEdit && (
        <button
          type="button"
          aria-label="Edit user message"
          title={editLabel ?? "Edit message"}
          className={actionButtonClass}
          onClick={() => onEdit(message)}
        >
          <Pencil aria-hidden="true" className={iconClassName} />
        </button>
      )}
      {messageRole === "assistant" && onRegenerate && (
        <button
          type="button"
          aria-label="Run assistant message again"
          title="Run again"
          className={actionButtonClass}
          onClick={onRegenerate}
        >
          <RotateCcw aria-hidden="true" className={iconClassName} />
        </button>
      )}
      {messageRole === "assistant" && onSpeak && (
        <button
          type="button"
          aria-label="Read assistant message aloud"
          title="Read aloud"
          className={actionButtonClass}
          onClick={() => void speakMessage()}
        >
          <Volume2 aria-hidden="true" className={iconClassName} />
        </button>
      )}
      {feedbackText && (
        <span data-testid="message-action-feedback" role="status" aria-live="polite" aria-atomic="true" className="ml-1 text-[0.7rem]">
          {feedbackText}
        </span>
      )}
    </div>
  );
}
