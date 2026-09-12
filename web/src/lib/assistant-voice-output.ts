interface AssistantCompletePayload {
  status?: unknown;
  text?: unknown;
}

export function assistantFinalText(type: string, payload: unknown): string | null {
  if (type !== "message.complete" || !payload || typeof payload !== "object") return null;
  const complete = payload as AssistantCompletePayload;
  if (complete.status !== undefined && complete.status !== "complete") return null;
  if (typeof complete.text !== "string") return null;
  const text = complete.text.trim();
  return text || null;
}

interface SpeechSynthesisLike {
  speak(utterance: SpeechSynthesisUtterance): void;
  cancel(): void;
}

export function speakAssistantFinal(
  synthesis: SpeechSynthesisLike,
  text: string,
  onSettled: () => void,
): { cancel(): void } | null {
  if (typeof SpeechSynthesisUtterance !== "function") return null;
  const utterance = new SpeechSynthesisUtterance(text);
  let settled = false;
  const settle = () => {
    if (settled) return;
    settled = true;
    onSettled();
  };
  utterance.onend = settle;
  utterance.onerror = settle;
  synthesis.speak(utterance);
  return {
    cancel() {
      synthesis.cancel();
      settle();
    },
  };
}
