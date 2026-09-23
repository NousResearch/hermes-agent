/**
 * ChatUIPage — the new structured /chat-ui Chat surface.
 *
 * Reuses the same Hermes gateway contract as the embedded TUI Chat (no
 * PTY, no xterm). One {@link useChatSession} per page mount owns the
 * connection, the resume bootstrap, and the imperative actions
 * (newSession, resume, submit, interrupt).
 *
 * The page handles the URL contract:
 *
 *   /chat-ui                → start a fresh session on mount
 *   /chat-ui?resume=<id>    → attach to the stored session
 *
 * Resume semantics survive reload, browser navigation, and switching
 * from the conversation list. Selecting another row in the
 * ConversationList fires `navigate(/chat-ui?resume=<id>)` which causes
 * the hook to attach to the new session.
 *
 * Phase 4 additions:
 *   - Empty-state prompt suggestions: when the user clicks a suggestion
 *     the page passes the text down to ChatComposer as `suggestedText`;
 *     the composer seeds its textarea and reports back via
 *     `onSuggestionConsumed` so the same suggestion isn't replayed twice.
 *   - A new "New chat" button sits at the top of the sidebar (above the
 *     search input) for discoverability on every screen size.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { Plus } from "lucide-react";

import { Button } from "@nous-research/ui/ui/components/button";
import { ChatSessionList } from "@/components/ChatSessionList";
import { ChatUIShell } from "@/components/chat/ChatUIShell";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useChatSession } from "@/hooks/useChatSession";

/* -------------------------------------------------------------------------- */
/* Sidebar primitives                                                         */
/* -------------------------------------------------------------------------- */

function ConversationSearch({
  value,
  onChange,
}: {
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <div className="border-b border-current/15 px-2 py-2">
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.currentTarget.value)}
        placeholder="Search conversations"
        aria-label="Search conversations"
        className="w-full rounded-md border border-current/15 bg-background-base px-2 py-1.5 text-xs text-foreground placeholder:text-text-tertiary focus:outline-none focus:border-midground"
      />
    </div>
  );
}

function NewChatButton({ onClick }: { onClick: () => void }) {
  return (
    <div className="border-b border-current/10 px-2 py-2">
      <Button
        ghost
        size="sm"
        onClick={onClick}
        className="w-full justify-start gap-1.5 normal-case tracking-normal"
        aria-label="Start a new chat"
        data-testid="chat-ui-new-chat"
      >
        <Plus className="h-3.5 w-3.5" />
        New chat
      </Button>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Page                                                                       */
/* -------------------------------------------------------------------------- */

export default function ChatUIPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const rawResume = searchParams.get("resume");
  const resumeSessionId = rawResume ?? null;
  const { profile } = useProfileScope();
  const [searchTerm, setSearchTerm] = useState("");
  const [suggestedText, setSuggestedText] = useState<string | null>(null);

  // The hook should connect on mount and disconnect on unmount; pass
  // `enabled` so it resets when the page toggles (e.g. user switches
  // between this and the CLI surface).
  const controller = useChatSession({
    resumeSessionId,
    profile,
    enabled: true,
  });

  // When the page lands on `/chat-ui` (no resume param) and the hook has
  // resolved an active sessionId, mirror it to the URL so reloading /
  // re-sharing the page keeps the same context. Skip for placeholder /
  // pending ids.
  const seededRef = useRef<string | null>(null);
  useEffect(() => {
    if (resumeSessionId) return;
    const sid = controller.state.sessionId;
    if (!sid || seededRef.current === sid) return;
    seededRef.current = sid;
    const next = new URLSearchParams(searchParams);
    next.set("resume", sid);
    setSearchParams(next, { replace: true });
  }, [controller.state.sessionId, resumeSessionId, searchParams, setSearchParams]);

  const activeSessionId = controller.state.sessionId ?? resumeSessionId;

  const startFresh = useCallback(() => {
    void controller.newSession();
  }, [controller]);

  const handleSuggestion = useCallback((text: string) => {
    setSuggestedText(text);
  }, []);

  const handleSuggestionConsumed = useCallback(() => {
    setSuggestedText(null);
  }, []);

  const sidebar = useMemo(
    () => (
      <div className="flex h-full flex-col">
        <NewChatButton onClick={startFresh} />
        <ConversationSearch value={searchTerm} onChange={setSearchTerm} />
        <div className="min-h-0 flex-1 overflow-y-auto">
          <ChatSessionList
            activeSessionId={activeSessionId}
            profile={profile ?? undefined}
            path="/chat-ui"
            searchTerm={searchTerm}
            onNewChat={startFresh}
            onPicked={() => setSearchTerm("")}
          />
        </div>
      </div>
    ),
    [activeSessionId, profile, searchTerm, startFresh],
  );

  return (
    <div className="flex h-full min-h-0 w-full" data-testid="chat-ui-page">
      <ChatUIShell
        controller={controller}
        profile={profile}
        sidebar={sidebar}
        onSuggestion={handleSuggestion}
        suggestedText={suggestedText}
        onSuggestionConsumed={handleSuggestionConsumed}
      />
    </div>
  );
}
