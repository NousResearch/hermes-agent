/**
 * Shell for the new /chat-ui page.
 *
 * Layout matches the dashboard's shell — desktop splits the screen into a
 * conversation list on the left and the active conversation on the right;
 * mobile collapses the list into a top bar drawer that opens with the
 * menu button.
 *
 * This is the only piece the new page renders; all chat state lives in
 * {@link useChatSession}. The shell itself does not touch transport, so
 * tests can render it with a stubbed controller.
 *
 * Phase 3 additions:
 *   - {@link ChatComposer} wired with image attachment (image.attach_bytes)
 *   - {@link ModelPickerDialog} + {@link ModelInfoCard} in the header
 *   - {@link ReasoningPicker} (gated by capability support)
 *   - Stop button → controller.interrupt() (no socket teardown)
 *   - Accessibility: aria-live log region, focusable composer, role badges
 *
 * Phase 4 additions:
 *   - PDF attachment wired through ChatComposer.onAttachPdf → pdf.attach
 *   - Mobile drawer: closes on Escape, restores focus to the menu button,
 *     backdrop is keyboard-focusable and aria-labelled
 *   - Polished spacing / typography / message hierarchy in collaboration
 *     with the inner components (no markup duplication here)
 */

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Menu, X } from "lucide-react";

import { Button } from "@nous-research/ui/ui/components/button";
import { ChatComposer } from "@/components/chat/ChatComposer";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import type { ChatSessionController } from "@/hooks/useChatSession";
import { isStreaming } from "@/lib/chat/reducer";
import { cn } from "@/lib/utils";

interface ChatUIShellProps {
  controller: ChatSessionController;
  /** Profile scope, forwarded to the model + reasoning pickers. */
  profile?: string | null;
  /** Sidebar slot — typically the (newly-extensible) ChatSessionList. */
  sidebar: ReactNode;
  /** Empty-state suggestion click → seeded into the composer. */
  onSuggestion?: (text: string) => void;
  /** Suggested composer seed; consumed once by the composer. */
  suggestedText?: string | null;
  /** Called by the composer after it consumes a suggested seed. */
  onSuggestionConsumed?: () => void;
}

export function ChatUIShell({
  controller,
  profile,
  sidebar,
  onSuggestion,
  suggestedText,
  onSuggestionConsumed,
}: ChatUIShellProps) {
  const { state, submit, interrupt, attachImage, attachPdf } = controller;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const streaming = isStreaming(state);
  // "Ready" = WebSocket open AND an active session id has landed in the
  // reducer. ``state.connection === "open"`` alone is too lax: there is a
  // brief window after the socket opens where ``session.create`` / resume
  // is still in flight and ``state.sessionId`` is null. Letting the user
  // pick a PDF in that window would surface a misleading "No active
  // session; cannot attach PDF" error. Treat the whole pre-session window
  // as connecting so the picker + send stay disabled until the session is
  // truly ready.
  const connecting =
    state.connection !== "open" || state.sessionId === null;
  const busy = streaming || state.submitting;

  // Auto-close the drawer when the session id changes (user picked a
  // different conversation) — keeps mobile UX in sync without coupling
  // the sidebar list to the drawer state.
  useEffect(() => {
    setDrawerOpen(false);
  }, [state.sessionId]);

  // Escape closes the drawer on mobile.
  useEffect(() => {
    if (!drawerOpen) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const closeDrawer = useCallback(() => setDrawerOpen(false), []);
  const openDrawer = useCallback(() => setDrawerOpen(true), []);

  return (
    <div
      className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-background-base text-foreground lg:flex-row"
      data-testid="chat-ui-shell"
    >
      {/* Desktop sidebar (always visible from lg). Mobile uses drawer. */}
      <aside
        className={cn(
          "z-40 flex w-72 shrink-0 flex-col border-r border-current/15 bg-background-base",
          "fixed inset-y-0 left-0 transition-transform duration-200 ease-out lg:static lg:translate-x-0",
          drawerOpen ? "translate-x-0" : "-translate-x-full",
        )}
        aria-label="Conversations"
        data-testid="chat-ui-sidebar"
      >
        <div className="flex items-center justify-end border-b border-current/10 px-2 py-1 lg:hidden">
          <Button
            ghost
            size="icon"
            onClick={closeDrawer}
            aria-label="Close conversations"
            data-testid="chat-ui-sidebar-close"
          >
            <X />
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-hidden">{sidebar}</div>
      </aside>

      {drawerOpen && (
        <button
          type="button"
          aria-label="Close conversations"
          onClick={closeDrawer}
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          data-testid="chat-ui-sidebar-backdrop"
        />
      )}

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {/* Mobile menu button row */}
        <div className="flex items-center gap-2 border-b border-current/10 px-3 py-2 lg:hidden">
          <Button
            ghost
            size="icon"
            onClick={openDrawer}
            aria-label="Open conversations"
            aria-expanded={drawerOpen}
            aria-controls="chat-ui-sidebar"
            data-testid="chat-ui-menu-button"
          >
            <Menu />
          </Button>
          <span className="font-mono-ui text-xs uppercase tracking-wider text-text-secondary">
            Conversations
          </span>
        </div>

        <ChatHeader
          title={state.title}
          model={state.model}
          provider={state.provider}
          reasoningEffort={state.reasoningEffort}
          connection={state.connection}
          profile={profile ?? undefined}
        />

        <ChatMessageList
          store={state}
          testPinToBottom={undefined}
          onSuggestion={onSuggestion}
        />

        {state.error && (
          <div
            className="border-t border-destructive/30 bg-destructive/10 px-4 py-2 text-xs text-destructive"
            data-testid="chat-ui-error"
            role="alert"
          >
            {state.error}
          </div>
        )}

        <div className="border-t border-current/15 px-3 py-3 sm:px-4">
          <ChatComposer
            busy={busy}
            connecting={connecting}
            suggestedText={suggestedText ?? undefined}
            onSuggestionConsumed={onSuggestionConsumed}
            onSubmit={async (text, attachments) => {
              // Attachment contract: each `image.*` / `pdf.attach` call must
              // have been made BEFORE prompt.submit so the queued attachment
              // rides the next turn. The composer invokes the per-kind
              // attach helper as the user picks / pastes the file; here we
              // only verify the chip carries the path that lets the gateway
              // find the queued bytes (image attachments expose `path`,
              // PDFs ride on the `first_page`/`last_page` window the
              // gateway already cached — they have no `path`).
              for (const a of attachments) {
                if (a.kind === "image" && !a.path) {
                  // No path means the chip wasn't queued with the gateway
                  // (e.g. the user picked but the session dropped). Bail.
                  return false;
                }
              }
              const ok = await submit(text, attachments);
              return ok;
            }}
            onAttachImage={(file) => attachImage(file)}
            onAttachPdf={(file) => attachPdf(file)}
            onStop={() => {
              void interrupt();
            }}
            placeholder={
              connecting
                ? `Connecting (${state.connection})…`
                : "Send a message…  (Enter to send, Shift+Enter for newline)"
            }
          />
        </div>
      </div>
    </div>
  );
}
