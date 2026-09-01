import { useCallback, useEffect, useRef, useState } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";

export type MessageRole = "user" | "assistant";

export interface MessageActionsProps {
  message: string;
  messageRole: MessageRole;
  onUseAsPrompt: (message: string) => void;
  onSpeak?: (message: string) => Promise<void> | void;
}

type FeedbackState = "copied" | "copy-failed" | "prompt-filled" | "spoken" | "speak-failed";

const FEEDBACK_DURATION_MS = 1800;

export function MessageActions({ message, messageRole, onUseAsPrompt, onSpeak }: MessageActionsProps) {
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

  return (
    <div data-slot="message-actions" className="mt-2 flex items-center gap-1 text-xs text-current/70">
      <button
        type="button"
        aria-label={`Copy ${messageRole} message`}
        className="rounded px-1.5 py-0.5 hover:bg-current/10 hover:text-current focus-visible:outline-2 focus-visible:outline-ring"
        onClick={() => void copyMessage()}
      >
        {feedback === "copied" ? "Copied" : "Copy"}
      </button>
      <button
        type="button"
        aria-label={`Use ${messageRole} message as prompt`}
        className="rounded px-1.5 py-0.5 hover:bg-current/10 hover:text-current focus-visible:outline-2 focus-visible:outline-ring"
        onClick={useAsPrompt}
      >
        Use as prompt
      </button>
      {messageRole === "assistant" && onSpeak && (
        <button
          type="button"
          aria-label="Speak assistant message"
          className="rounded px-1.5 py-0.5 hover:bg-current/10 hover:text-current focus-visible:outline-2 focus-visible:outline-ring"
          onClick={() => void speakMessage()}
        >
          Speak
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
