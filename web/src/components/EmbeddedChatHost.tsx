import { lazy, Suspense, useCallback, useEffect, useMemo } from "react";

import type {
  DashboardEmbedEvent,
  DashboardEmbedRequest,
} from "@/lib/dashboard-embed";
import { postDashboardEmbedEvent } from "@/lib/dashboard-embed";
import type { PtyConnectionState } from "@/lib/pty-reconnect";
import { ProfileContext } from "@/contexts/profile-context";

const ChatPage = lazy(() => import("@/pages/ChatPage"));

function embedEventForPtyState(state: PtyConnectionState): DashboardEmbedEvent {
  if (state === "open") return "ready";
  if (state === "closed") return "disconnected";
  return state;
}

export function EmbeddedChatHost({ request }: { request: DashboardEmbedRequest }) {
  const profileScope = useMemo(
    () => ({
      profile: request.profile,
      currentProfile: request.profile || "default",
      profiles: [request.profile || "default"],
      setProfile: () => {},
    }),
    [request.profile],
  );

  useEffect(() => {
    if (!request.authBridge) return;
    postDashboardEmbedEvent(
      window.opener,
      request.parentOrigin,
      "authenticated",
      request.embedId,
    );
    window.close();
  }, [request.authBridge, request.embedId, request.parentOrigin]);

  const announcePtyState = useCallback(
    (state: PtyConnectionState) => {
      postDashboardEmbedEvent(
        window.parent,
        request.parentOrigin,
        embedEventForPtyState(state),
        request.embedId,
      );
    },
    [request.embedId, request.parentOrigin],
  );

  if (request.authBridge) {
    return (
      <div className="flex h-dvh items-center justify-center bg-black text-sm text-white/75">
        Sign-in complete. You can close this window.
      </div>
    );
  }

  return (
    <ProfileContext.Provider value={profileScope}>
      <main
        className="flex h-dvh max-h-dvh min-h-0 min-w-0 overflow-hidden bg-black"
        data-dashboard-embed={request.embedId}
      >
        <Suspense fallback={<div className="h-full w-full bg-black" aria-label="Loading chat" />}>
          <ChatPage
            isActive
            embedded
            embedId={request.embedId}
            onPtyStateChange={announcePtyState}
          />
        </Suspense>
      </main>
    </ProfileContext.Provider>
  );
}
