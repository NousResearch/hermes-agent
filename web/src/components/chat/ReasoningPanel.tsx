/**
 * Collapsible reasoning panel for the new Chat UI.
 *
 * Mirrors the existing dashboard reasoning disclosure pattern (a labelled
 * `<details>` block) and pipes the reasoning text through the shared
 * `Markdown` component. While streaming it stays open by default so the
 * user can follow the model's chain-of-thought.
 *
 * Phase 4 polish:
 *   - Collapsed by default once the turn finalises (the brief's "no raw
 *     internal protocol text" goal — users opt in to read the reasoning,
 *     it's not in their face).
 *   - Streaming badge so the user can see the model is still thinking.
 *   - A11y: the `<summary>` is labelled, the disclosure is keyboard
 *     toggleable (native `<details>` semantics).
 */

import { Brain, Loader2 } from "lucide-react";
import { useState } from "react";

import { Markdown } from "@/components/Markdown";

interface ReasoningPanelProps {
  text: string;
  /**
   * When true, defaults the disclosure open (streaming). Defaults to true
   * (open) so the model chain-of-thought stays visible while it's being
   * generated. Callers set this to false on the next render once the
   * turn finalises (the panel then starts closed).
   */
  initiallyOpen?: boolean;
}

export function ReasoningPanel({ text, initiallyOpen }: ReasoningPanelProps) {
  const [open, setOpen] = useState(Boolean(initiallyOpen));
  const trimmed = text.trim();
  if (!trimmed) return null;
  return (
    <details
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
      className="group/reasoning rounded-md border border-purple-500/20 bg-purple-500/5 px-3 py-2 text-xs"
      data-testid="chat-ui-reasoning-panel"
      data-open={open || undefined}
    >
      <summary className="flex cursor-pointer select-none items-center gap-1.5 font-mono-ui uppercase tracking-wider text-purple-500 marker:hidden">
        <Brain className="h-3.5 w-3.5" aria-hidden />
        <span className="font-medium">Reasoning</span>
        {initiallyOpen && (
          <Loader2
            className="ml-1 h-3 w-3 animate-spin text-purple-500"
            aria-label="Reasoning in progress"
          />
        )}
      </summary>
      <div className="mt-2 text-text-secondary" data-testid="chat-ui-reasoning-text">
        <Markdown content={trimmed} />
      </div>
    </details>
  );
}
