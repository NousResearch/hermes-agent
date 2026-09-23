/**
 * Renders the timeline of a Chat UI session.
 *
 * Walks {@link ChatUIStore.turns} in order and maps each row onto its
 * React component. Auto-scrolls to the bottom on new turns + on streaming
 * text updates, but only when the user has not scrolled away (a common
 * ChatGPT-style affordance).
 *
 * Phase 4 polish:
 *   - Centred empty state with prompt suggestions (suggestion clicks only
 *     populate the composer — never auto-submit).
 *   - Slightly tighter message rhythm (1px divider on streaming) and a
 *     "thinking" badge the streaming assistant renders above the caret.
 */

import {
  useEffect,
  useMemo,
  useRef,
} from "react";
import { Sparkles } from "lucide-react";

import { AssistantMessage } from "@/components/chat/AssistantMessage";
import { UserMessage } from "@/components/chat/UserMessage";
import type {
  ChatUIAssistantTurn,
  ChatUIStore,
} from "@/components/chat/types";

interface ChatMessageListProps {
  store: ChatUIStore;
  /** Optional test hook — parent can pin the scroll position. */
  testPinToBottom?: boolean;
  /**
   * Called when the user picks an empty-state suggestion. The parent
   * forwards the text to the composer (does NOT auto-submit).
   */
  onSuggestion?: (text: string) => void;
}

/** Default prompt suggestions shown when the conversation is empty. */
const DEFAULT_SUGGESTIONS: ReadonlyArray<{
  title: string;
  subtitle: string;
  prompt: string;
}> = [
  {
    title: "Summarize a document",
    subtitle: "Drop a PDF or paste text — Hermes will read and summarise it.",
    prompt: "Summarise the document I'll attach next.",
  },
  {
    title: "Plan a refactor",
    subtitle: "Share the module path and your goals; get a step-by-step plan.",
    prompt: "Help me plan a refactor — what should we tackle first?",
  },
  {
    title: "Debug a stack trace",
    subtitle: "Paste the trace and any context; Hermes will triage likely causes.",
    prompt: "Help me debug this stack trace. I'll paste it next.",
  },
  {
    title: "Brainstorm features",
    subtitle: "Describe the product or surface; get five concrete ideas.",
    prompt: "Brainstorm five concrete feature ideas for my product.",
  },
];

export function ChatMessageList({
  store,
  testPinToBottom,
  onSuggestion,
}: ChatMessageListProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pinRef = useRef(true);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      pinRef.current = distFromBottom < 64;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (testPinToBottom === false) return;
    if (!pinRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [store.turns, store.turns.length > 0 ? lastText(store) : "", testPinToBottom]);

  if (store.turns.length === 0) {
    return (
      <EmptyState
        containerRef={containerRef}
        onSuggestion={onSuggestion}
      />
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex flex-1 flex-col gap-4 overflow-y-auto px-3 py-6 sm:px-6"
      data-testid="chat-ui-message-list"
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
    >
      {store.turns.map((turn) => {
        if (turn.role === "user") {
          return <UserMessage key={turn.id} turn={turn} />;
        }
        if (turn.role === "assistant") {
          return (
            <AssistantMessage
              key={turn.id}
              turn={turn as ChatUIAssistantTurn}
            />
          );
        }
        return (
          <div
            key={turn.id}
            className="mx-auto max-w-2xl border-l-2 border-muted pl-3 text-xs text-text-tertiary"
          >
            {turn.text}
          </div>
        );
      })}
    </div>
  );
}

function EmptyState({
  containerRef,
  onSuggestion,
}: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  onSuggestion?: (text: string) => void;
}) {
  const suggestions = useMemo(() => DEFAULT_SUGGESTIONS, []);
  return (
    <div
      ref={containerRef}
      className="flex flex-1 items-center justify-center overflow-y-auto px-4 py-12"
      data-testid="chat-ui-empty"
    >
      <div className="w-full max-w-xl space-y-8 text-center">
        <div className="space-y-2">
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Sparkles className="h-5 w-5" aria-hidden />
          </span>
          <h2
            className="text-2xl font-semibold text-foreground"
            data-testid="chat-ui-empty-title"
          >
            How can Hermes help you?
          </h2>
          <p className="text-sm text-text-secondary">
            Pick a starter or send a message below to begin.
          </p>
        </div>

        <ul
          className="grid grid-cols-1 gap-2 sm:grid-cols-2"
          data-testid="chat-ui-empty-suggestions"
        >
          {suggestions.map((s) => (
            <li key={s.title}>
              <button
                type="button"
                onClick={() => onSuggestion?.(s.prompt)}
                className="group w-full rounded-lg border border-current/15 bg-card/40 p-3 text-left transition-colors hover:border-midground hover:bg-card"
                data-testid="chat-ui-empty-suggestion"
                data-suggestion-title={s.title}
              >
                <div className="text-sm font-medium text-foreground group-hover:text-primary">
                  {s.title}
                </div>
                <div className="mt-0.5 text-xs text-text-tertiary">
                  {s.subtitle}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function lastText(store: ChatUIStore): string {
  for (let i = store.turns.length - 1; i >= 0; i -= 1) {
    const t = store.turns[i];
    if (t && (t.role === "assistant" || t.role === "user")) return t.text;
  }
  return "";
}
